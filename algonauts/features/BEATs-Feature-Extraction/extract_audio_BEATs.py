# ---------------------------------------------------------- #
# 0.  pip install fairseq torchaudio moviepy soundfile tqdm
# ---------------------------------------------------------- #
"""Chunk‑wise BEATs feature extraction with adaptive layer grouping.

Highlights
----------
* **Sliding context window** (default 4 s) keeps VRAM usage bounded while a
  **TR‑sized step** (default 1.49 s) yields one feature vector per fMRI volume.
* We replicate BEATs’ internal forward pass so we can keep **all encoder layer
  outputs**.  Some checkpoints use 12 layers, others 24 – the script adapts at
  runtime and divides them into low/mid/high thirds automatically.
* Hidden states sometimes come out as `(T, 2, 768)` (two stereo tokens).  We
  collapse that second dimension by averaging before any further pooling.
"""

import argparse, math, os, random, tempfile, warnings
from typing import List, Optional, Tuple

import numpy as np
import torch, torchaudio
from moviepy.editor import VideoFileClip
from BEATs import BEATs, BEATsConfig
from tqdm import tqdm

MODEL_CKPT = "BEATs_iter3_plus_AS2M_finetuned.pt"  # path or HF id
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------  audio helper  -------------------------------------------------- #

def extract_audio(video_path: str, sr: int) -> Tuple[torch.Tensor, int]:
    """Extract **stereo** waveform at `sr` Hz from *video_path*."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(tmp_name, fps=sr, verbose=False, logger=None)
        wav, _ = torchaudio.load(tmp_name)  # (2, samples)
    finally:
        os.remove(tmp_name)
    return wav, sr

# ----------  BEATs loader  --------------------------------------------------- #

def load_beats(checkpoint: str):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE).eval(), 16_000  # BEATs’ native sample rate

# ----------  BEATs forward (keep hiddens)  ---------------------------------- #

@torch.no_grad()
def forward_beats_with_hiddens(
    model: BEATs,
    source: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    fbank_mean: float = 15.41663,
    fbank_std: float = 6.55582,
):
    """Return list of encoder hidden states `[L × (T, …, D)]`.

    If the checkpoint disables `return_all_hiddens`, we flip it *on* for the
    call and restore afterwards.  A fallback hook covers even older fairseq
    versions.
    """
    # 1. FBANK front‑end ---------------------------------------------------- #
    fbank = model.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(fbank, padding_mask)

    # 2. Patch embedding ---------------------------------------------------- #
    fbank = fbank.unsqueeze(1)
    features = model.patch_embedding(fbank)  # (B, C, T', D')
    features = features.reshape(features.shape[0], features.shape[1], -1)
    features = features.transpose(1, 2)      # (B, T, D)
    features = model.layer_norm(features)

    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(features, padding_mask)

    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)

    x = model.dropout_input(features)

    # 3. Transformer encoder ---------------------------------------------- #
    if hasattr(model.encoder, "return_all_hiddens"):
        old_flag = model.encoder.return_all_hiddens
        model.encoder.return_all_hiddens = True
        x, layer_results = model.encoder(x, padding_mask=padding_mask)
        model.encoder.return_all_hiddens = old_flag
    else:
        # Fairseq < 0.12 fallback: forward hooks
        hidden_list: List[torch.Tensor] = []

        def _cap(_, __, output):
            hidden_list.append(output[0] if isinstance(output, tuple) else output)

        hooks = [blk.register_forward_hook(_cap) for blk in model.encoder.layers]
        x, _ = model.encoder(x, padding_mask=padding_mask)
        for h in hooks:
            h.remove()
        layer_results = [(h, None) for h in hidden_list]

    hiddens = [h[0] for h in layer_results]  # strip attn weights
    return hiddens  # length L (12 or 24 depending on checkpoint)

# ----------  core extraction (chunked)  ------------------------------------- #

@torch.no_grad()

def extract_features(
    video_path: str,
    out_dirs: dict,  # {"low": …, "mid": …, "high": …}
    basename: str,
    model: BEATs,
    target_sr: int,
    tr_seconds: float = 1.49,
    context_seconds: float = 4.0,
):
    wav, _ = extract_audio(video_path, target_sr)  # (2, T)
    T = wav.shape[1]

    step_samples = int(round(tr_seconds * target_sr))
    context_samples = int(round(context_seconds * target_sr))
    tr_frames = int(round(tr_seconds / 0.02))  # BEATs hop = 20 ms ⇒ 50 Hz

    n_tr = int(math.ceil(T / step_samples))
    pooled_by_tag = {tag: [] for tag in ("low", "mid", "high")}

    # --------------------------------------------------------------------- #
    for tr_idx in range(n_tr):
        # Audio chunk boundaries
        win_end = min((tr_idx + 1) * step_samples, T)
        win_start = max(0, win_end - context_samples)
        wav_slice = wav[:, win_start:win_end]

        # Forward through BEATs
        hiddens = forward_beats_with_hiddens(model, wav_slice.to(DEVICE))
        n_layers = len(hiddens)

        # Adaptive thirds (works for 12 or 24 layers)
        g = math.ceil(n_layers / 3)
        layer_groups = (
            range(0, g),
            range(g, min(2 * g, n_layers)),
            range(min(2 * g, n_layers), n_layers),
        )

        # Crop to current TR only
        seq_len = hiddens[0].shape[0]
        slice_start = max(0, seq_len - tr_frames)

        for tag, idxs in zip(("low", "mid", "high"), layer_groups):
            selected = []
            for i in idxs:
                h = hiddens[i]  # (T, 2, 768) or (T, 768)
                if h.dim() == 3:  # collapse stereo tokens
                    h = h.view(-1, 768*2)
                selected.append(h)
            stacked = torch.stack(selected).mean(0)          # (T, D)
            tr_part = stacked[slice_start:]                  # ≤ tr_frames rows
            pooled = torch.cat(
                [tr_part.mean(0), tr_part.std(0, unbiased=False)], dim=0
            )
            pooled_by_tag[tag].append(pooled.cpu().numpy().astype(np.float32))

        # housekeeping
        del hiddens
        torch.cuda.empty_cache()

    # Save outputs --------------------------------------------------------- #
    for tag in ("low", "mid", "high"):
        np.save(
            os.path.join(out_dirs[tag], basename),
            np.stack(pooled_by_tag[tag], axis=0),  # (n_tr, 2*D)
        )

# ----------  folder driver  -------------------------------------------------- #

def process_folder(
    in_root: str,
    out_root: str,
    model: BEATs,
    sr: int,
    tr: float,
    context: float,
):
    videos = []
    for root, _, files in os.walk(in_root):
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for f in files:
            if f.endswith(".mkv"):
                videos.append((root, f))
    random.shuffle(videos)

    for root, fname in tqdm(videos, desc="BEATs feats"):
        rel_dir = os.path.relpath(root, in_root)
        out_dirs = {tag: os.path.join(out_root, tag, rel_dir) for tag in ("low", "mid", "high")}
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        basename = fname.replace(".mkv", ".npy")
        if all(os.path.exists(os.path.join(out_dirs[t], basename)) for t in out_dirs):
            continue  # skip if already done

        extract_features(
            os.path.join(root, fname),
            out_dirs,
            basename,
            model,
            sr,
            tr,
            context,
        )

# ----------  CLI  ------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_folder",
        default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies/",
        help="Root folder containing movies",
    )
    p.add_argument(
        "--output_folder",
        default="/u/shdixit/MultimodalBrainModel/Features/Audio/BEATS_Full",
        help="Root folder where .npy features will be stored",
    )
    p.add_argument("--tr", type=float, default=1.49, help="Repetition time in seconds (default: 1.49)")
    p.add_argument(
        "--context", type=float, default=10.0, help="Context window length in seconds (default: 4.0)"
    )
    args = p.parse_args()

    model, sr = load_beats(MODEL_CKPT)
    out_root = f"{args.output_folder}_tr{args.tr}_ctx{args.context}"

    process_folder(args.input_folder, out_root, model, sr, args.tr, args.context)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
