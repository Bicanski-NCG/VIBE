"""Extract tiered audio features from video files using the Omar‑RQ model.

2025‑07‑10 • *stability patch*
---------------------------------
* Uses **layers 5 / 14 / 22** (one forward‑pass per channel).
* Audio I/O via **soundfile** (no SoX dependency).
* **Robust resampling** with *torchaudio.functional.resample*.
* **Scaled‑Dot‑Product Attention fallback** – we monkey‑patch
  ``torch.nn.functional.scaled_dot_product_attention`` to a plain‑math version
  that always works, even when Flash/efficient kernels are absent or disabled.
"""

# -----------------------------------------------------------------------------
# IMPORTS & GLOBAL PATCHES
# -----------------------------------------------------------------------------

import os, math, argparse, random
from typing import Dict, List, Tuple

import numpy as np
import torch
import soundfile as sf  # robust WAV/FLAC/OGG reader
import torchaudio.functional as AF
from moviepy import VideoFileClip
from tqdm import tqdm

from omar_rq import get_model

# ── Attention‑kernel fallback ────────────────────────────────────────────────
# Some CUDA builds (e.g. certain A100 clusters) ship PyTorch without *any*
# compatible Scaled‑Dot‑Product kernels. We: 1) disable Flash/efficient paths;
# 2) replace F.scaled_dot_product_attention with a safe math implementation.
# This guarantees the model can run everywhere, albeit a bit slower.

import torch.nn.functional as F
try:  # disable flash/efficient kernels if the API is available
    torch.backends.cuda.enable_flash_sdp(False)  # type: ignore[attr-defined]
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # type: ignore[attr-defined]
    torch.backends.cuda.enable_math_sdp(True)  # type: ignore[attr-defined]
except AttributeError:
    os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
    os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"

if not getattr(F, "_orig_sdp", None):
    F._orig_sdp = F.scaled_dot_product_attention  # type: ignore[attr-defined]

    def _safe_sdp(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (..., Lq, Lk)
        if attn_mask is not None:
            scores = scores + attn_mask
        if is_causal:
            idx = torch.triu_indices(scores.size(-2), scores.size(-1), 1, device=scores.device)
            scores[..., idx[0], idx[1]] = float("-inf")
        probs = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0 and probs.requires_grad:
            probs = torch.nn.functional.dropout(probs, p=dropout_p)
        return torch.matmul(probs, v)

    F.scaled_dot_product_attention = _safe_sdp  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

TIERS = ["low", "mid", "high"]
TIER_LAYERS = {"low": 5, "mid": 14, "high": 22}
TARGET_SR = 16_000  # Omar‑RQ training rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# AUDIO HELPERS
# -----------------------------------------------------------------------------

def extract_audio_from_video(src: str, dst: str) -> None:
    """Extract 16‑kHz WAV from *src* to *dst* using FFmpeg via MoviePy."""
    try:
        VideoFileClip(src).audio.write_audiofile(dst, fps=TARGET_SR, logger=None)
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed for {src}: {e}")


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """Load audio as (C, S) float32 tensor plus sample‑rate."""
    wav_np, sr = sf.read(path, dtype="float32", always_2d=True)  # (S, C)
    return torch.from_numpy(wav_np.T), sr

# -----------------------------------------------------------------------------
# WINDOW SCHEDULING
# -----------------------------------------------------------------------------

def make_windows(total: float, window: float, tr: float) -> List[Tuple[float, float]]:
    """Return (start, end) windows advancing by *tr* seconds."""
    ends, e = [], tr
    while e < total:
        ends.append(e)
        e += tr
    ends.append(total)
    return [(max(0.0, e - window), e) for e in ends]

# -----------------------------------------------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------------------------------------------

# Helper: mean over the **time** axis (0) – feature dim is always last.

def _roi_mean(emb: torch.Tensor, roi_start: int) -> torch.Tensor:
    """Average `emb` over frames `[roi_start:, :]` → (feature_dim,)"""
    return emb.squeeze()[roi_start:, :].mean(dim=0).squeeze()

# -----------------------------------------------------------------------------

# def _roi_mean(emb: torch.Tensor, roi_start: int) -> torch.Tensor:
#     """Average *emb* over frames from *roi_start* to end."""
#     return emb[roi_start:].mean(dim=0).squeeze()


def extract_omar_features(
    video_path: str,
    output_base: str,
    rel: str,
    model,
    tr: float,
    window: float,
    stereo: bool = False,
):
    tmp_wav = video_path.rsplit(".", 1)[0] + "_tmp.wav"
    if os.path.exists(tmp_wav):
        return
    extract_audio_from_video(video_path, tmp_wav)

    wav, sr = load_audio(tmp_wav)
    if sr != TARGET_SR:
        wav = torch.stack([AF.resample(ch, sr, TARGET_SR) for ch in wav])
        sr = TARGET_SR

    if not stereo or wav.shape[0] == 1:
        wav = wav.mean(dim=0, keepdim=True)

    dur = wav.shape[1] / sr
    windows = make_windows(dur, window, tr)

    layer_ids = list(TIER_LAYERS.values())
    tiers_out: Dict[str, List[np.ndarray]] = {k: [] for k in TIERS}

    for w_start, w_end in windows:
        s = int(w_start * sr)
        e = int(w_end * sr)
        chunk = wav[..., s:e]  # (C, S)
        chunk_len = w_end - w_start

        chans_feat = {k: [] for k in TIERS}
        for ch in range(chunk.shape[0]):
            x = chunk[ch].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                all_embs = model.extract_embeddings(x, layers=layer_ids).squeeze(0)
                layer_map = {lid: all_embs[i] for i, lid in enumerate(layer_ids)}

            seq_len = all_embs.shape[-1]
            frames_per_sec = seq_len / chunk_len
            roi_frames = int(min(tr, chunk_len) * frames_per_sec + 0.5)
            roi_start = max(0, seq_len - roi_frames)

            for tier, lid in TIER_LAYERS.items():
                chans_feat[tier].append(_roi_mean(layer_map[lid], roi_start))

        for tier in TIERS:
            vec = (
                torch.cat(chans_feat[tier], dim=-1)
                if stereo and len(chans_feat[tier]) == 2
                else chans_feat[tier][0]
            )
            tiers_out[tier].append(vec.cpu().numpy())

    base_npy = os.path.basename(video_path).replace(".mkv", ".npy")
    for tier, mats in tiers_out.items():
        tier_dir = os.path.join(output_base, tier, rel)
        os.makedirs(tier_dir, exist_ok=True)
        np.save(os.path.join(tier_dir, base_npy), np.stack(mats))

    os.remove(tmp_wav)

# -----------------------------------------------------------------------------
# DATASET DRIVER
# -----------------------------------------------------------------------------

def process_dataset(
    inp_root: str,
    out_root: str,
    model,
    tr: float,
    window: float,
    stereo: bool = False,
):
    ood_skip = {
        "task-passepartoutS02E08_video.mkv",
        "task-passepartoutS02E07_video.mkv",
        "task-chaplin_video.mkv",
        "task-pulpfiction_video.mkv",
        "task-mononoke_video.mkv",
        "task-planetearth_video.mkv",
    }

    vids = [
        os.path.join(r, f)
        for r, _, fs in os.walk(inp_root)
        if not any(p.startswith(".") or p.startswith("colored") for p in r.split(os.sep))
        for f in fs
        if f.endswith(".mkv") and f not in ood_skip
    ]
    random.shuffle(vids)

    for vid in tqdm(vids, desc="Videos"):
        rel = os.path.relpath(os.path.dirname(vid), inp_root)
        base = os.path.basename(vid).replace(".mkv", ".npy")
        if all(os.path.exists(os.path.join(out_root, t, rel, base)) for t in TIERS):
            continue
        extract_omar_features(vid, out_root, rel, model, tr, window, stereo)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Tiered Omar‑RQ features (moving window)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder", default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies/")
    parser.add_argument("--output_folder", default="/u/shdixit/MultimodalBrainModel/Features/Audio/OmarRQ")
    parser.add_argument("--model_id", default="mtg-upf/omar-rq-multifeature-25hz-fsq")
    parser.add_argument("--chunk_duration", type=float, default=1.49)
    parser.add_argument("--chunk_length", type=float, default=20.0)
    parser.add_argument("--stereo", action="store_true")
    args = parser.parse_args()

    model = get_model(model_id=args.model_id, device=DEVICE).eval()
    out_root = f"{args.output_folder}-ctx{int(args.chunk_length)}-stereo{args.stereo}"

    process_dataset(
        args.input_folder,
        out_root,
        model,
        args.chunk_duration,
        args.chunk_length,
        args.stereo,
    )
