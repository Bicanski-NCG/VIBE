from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import time, argparse, os
from functools import lru_cache
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from audio_io import stream_clips
from nnAudio.Spectrogram import Gammatonegram


# ────────────────────────────────────────────────────────────────
# 1. Gammatone cochleagram  (Santoro‑style, 128 ERB bands)
# ────────────────────────────────────────────────────────────────
def cochleagram(
    wav: np.ndarray,
    sr: int,
    *,
    n_filters: int = 128,
    fmin: float = 50.0,
    fmax: float = 8_000.0,
    hop_len: int = 235,
) -> torch.Tensor:
    """
    Log‑power gammatone spectrogram (F, T) – float32.
    Uses nnAudio (GPU if input is CUDA).
    """
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # (1,T)
    gt = Gammatonegram(
        sr=sr,
        hop_length=hop_len,
        n_bins=n_filters,
        fmin=fmin,
        fmax=fmax,
        verbose=False,
        power_spectrogram=True,   # linear power
    ).to(wav_t.device)
    spec = gt(wav_t).squeeze(0)         # (F,T)
    spec = spec.clamp_min(1e-9)
    return torch.log1p(spec).float()    # log‑power


# ────────────────────────────────────────────────────────────────
# 2. modulation-filter bank  (Ω × ω × direction × phase)
# ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def build_mtf_bank(
    frame_hz: float,
    F_bins: int,
    Ω: tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8),
    ω: tuple[float, ...] = (0.5, 1, 2, 4, 8, 16),
    directions: tuple[int, int] = (+1, -1),   # +1 = upward, -1 = downward
    phis: tuple[float, float] = (0.0, np.pi/2),  # quadrature
    k_f: int = 31,
    k_t: int = 81,
    gamma_f: float = 0.6,
    gamma_t: float = 0.6,
) -> torch.Tensor:
    """
    Returns tensor (N_filters, 1, F_k, T_k)  FP16 on CUDA.

    One filter per (Ω, ω, dir, phase).
    """
    F_axis = np.linspace(-0.5, 0.5, k_f) * np.log2(F_bins / 2)   # ± spectral range
    T_axis = np.linspace(-1, 1, k_t)                # ±1 s (scaled later)
    Fg, Tg = np.meshgrid(F_axis, T_axis, indexing="ij")   # (F_k,T_k)

    kernels: List[np.ndarray] = []
    for om_s in Ω:
        for om_t in ω:
            for d in directions:
                for phi in phis:
                    carrier = np.cos(
                        2*np.pi*(om_s*Fg + d*om_t*Tg*frame_hz) + phi
                    )
                    env = np.exp( -0.5*( (Fg/gamma_f)**2 + (Tg/gamma_t)**2 ) )
                    k = carrier * env
                    k -= k.mean()
                    kernels.append(k.astype(np.float32))

    bank = torch.from_numpy(np.stack(kernels)[:, None])   # (N,1,F_k,T_k)
    return bank.cuda()


# ────────────────────────────────────────────────────────────────
# 3. per-TR feature extractor
# ────────────────────────────────────────────────────────────────
def mtf_features_for_clip(
    clip_wav: torch.Tensor,  # (1, T_samples) float32 CPU
    sr: int,
    *,
    hop_len: int = 235,
    Ω: tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8),
    ω: tuple[float, ...] = (0.5, 1, 2, 4, 8, 16),
) -> torch.Tensor:
    """
    Returns 1-D feature tensor (F_bins × len(Ω) × len(ω) × directions × 2 + 2) float32 CPU.
    """
    # 1. cochleagram   -----------------------------------------------------
    coch = cochleagram(
        clip_wav.squeeze(0).numpy(), sr,
        hop_len=hop_len
    ).cuda()                 # (F,T)

    # 5. frequency-wise whitening
    coch = (coch - coch.mean(0, keepdim=True)) / (coch.std(0, keepdim=True) + 1e-6)

    # 6. onset envelope
    onset_env = torch.relu(coch[:, 1:] - coch[:, :-1]).mean(0, keepdim=True)  # (1,T)

    F_bins, T_frames = coch.shape
    coch = coch.unsqueeze(0).unsqueeze(0)              # (1,1,F,T)

    # 7. FP16 conv & cube-root NL
    frame_hz = sr / hop_len
    coch = coch
    bank = build_mtf_bank(frame_hz, F_bins, Ω, ω)
    resp = F.conv2d(coch, bank,
                    padding=(bank.shape[2]//2, bank.shape[3]//2))
        # resp: (1, N_filters, F, T)

    # 3. quadrature energy   (dir,phase)  -------------------------------
    D = len(Ω)*len(ω)*2        # directional pairs
    resp = resp.view(1, D, 2, F_bins, T_frames)  # (1,D,2,F,T)
    energy = resp.pow(2).sum(2).sqrt()           # (1,D,F,T)

    # 7. cube-root NL
    energy = F.relu(energy).pow(0.33)      # cube‑root NL

    # 8. keep direction channels, mean+max pooling
    # Remove averaging over direction
    mean_T = energy.mean(-1)           # (1,D,F)
    max_T  = energy.amax(-1)           # (1,D,F)
    feats_mod = torch.cat([mean_T, max_T], dim=1)   # (1,2D,F)

    # Append onset mean/max
    onset_mean = onset_env.mean(-1)    # (1,1)
    onset_max  = onset_env.amax(-1)    # (1,1)
    feats_mod = feats_mod.squeeze(0).transpose(0,1)  # (F,2D)
    feats = torch.cat([feats_mod.flatten(), onset_mean.flatten(), onset_max.flatten()])
    return feats.cpu()


# ────────────────────────────────────────────────────────────────
# 4. wrapper for a full file (uses audio_io stream_clips)
# ────────────────────────────────────────────────────────────────
def mtf_features_for_file(
    path: str | Path,
    *,
    tr_sec: float = 1.49,
    target_sr: int = 16_000,
    hop_len: int = 235,
    Ω: tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8),
    ω: tuple[float, ...] = (0.5, 1, 2, 4, 8, 16),
    load_full: bool = True,
) -> torch.Tensor:
    """
    Returns tensor (n_TRs, F_bins × len(Ω) × len(ω)) float32.
    """
    clips = []
    for clip in stream_clips(
        path,
        tr_sec=tr_sec,
        chunk_sec=None if load_full else 60.0,
        target_sr=target_sr,
        mono=True,
        load_full=load_full,
    ):
        clips.append(
            mtf_features_for_clip(
                clip, sr=target_sr,
                hop_len=hop_len, Ω=Ω, ω=ω
            )
        )
    return torch.stack(clips)     # (n_TRs, features)


# ────────────────────────────────────────────────────────────────
# 5.  Folder walker  (matches visual pipeline)
# ────────────────────────────────────────────────────────────────
def process_audio(
    in_path: Path,
    out_path: Path,
    *,
    tr_sec: float,
    target_sr: int,
    hop_len: int,
) -> None:
    start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return
    else:
        out_path.touch()

    feats = mtf_features_for_file(
        in_path,
        tr_sec=tr_sec,
        target_sr=target_sr,
        hop_len=hop_len,
    )
    torch.save(feats, out_path)
    print(f"[ok]\t{feats.shape[0]} TRs \t{in_path.name} \t{time.time()-start:.1f}s")

def walk_folder(root_in: Path, root_out: Path,
                *, tr_sec: float, target_sr:int, hop_len:int) -> None:
    audio_paths = []
    for dirpath, _, filenames in os.walk(root_in):
        if any(part.startswith(".") for part in dirpath.split(os.sep)):
            continue
        for f in filenames:
            if f.lower().endswith((".mkv", ".mp4", ".mov")):
                audio_paths.append(Path(dirpath) / f)

    print(f"Found {len(audio_paths)} audio files under {root_in}")
    for p in tqdm(audio_paths, desc="Processing audio", unit="audio", ncols=80):
        rel = p.relative_to(root_in).with_suffix(".pt")
        out = root_out / rel
        if os.path.exists(out):
            print(f"[skip]\t\t{out.name} already exists")
            continue
        else:
            print(f"[process]\t{p.name} → {out.name}")
            process_audio(p, out,
                        tr_sec=tr_sec,
                        target_sr=target_sr,
                        hop_len=hop_len)


# ────────────────────────────────────────────────────────────────
# 6. CLI
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--tr", type=float, default=1.49)
    parser.add_argument("--target_sr", type=int, default=16_000)
    parser.add_argument("--hop_len", type=int, default=235,
                        help="samples per CQT frame (235 ⇒ 68 Hz)")
    args = parser.parse_args()

    print("\n".join([
        f"Input   : {args.input_folder}",
        f"Output  : {args.output_folder}",
        f"TR      : {args.tr} s",
        f"sr      : {args.target_sr} Hz",
        f"hop_len : {args.hop_len} samples",
        "--------------------------------"]))
    walk_folder(Path(args.input_folder),
                Path(args.output_folder),
                tr_sec=args.tr,
                target_sr=args.target_sr,
                hop_len=args.hop_len)