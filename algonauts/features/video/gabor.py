from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --- colour helpers (self‑contained, no external deps) ------------------
_SRGB_TO_XYZ = torch.tensor(
    [[0.41239080, 0.35758434, 0.18048079],
     [0.21263901, 0.71516868, 0.07219232],
     [0.01933082, 0.11919478, 0.95053215]],
    dtype=torch.float32
)  # sRGB (D65) to CIE XYZ

_XYZ_TO_LMS = torch.tensor(   # Stockman & Sharpe (2000) cone fundamentals
    [[ 0.4002, 0.7075, -0.0807],
     [-0.2263, 1.1653,  0.0457],
     [ 0.0000, 0.0000,  0.9182]],
    dtype=torch.float32
)

def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse‑gamma: sRGB uint8 [0,1] → linear RGB."""
    threshold = 0.04045
    below = rgb <= threshold
    rgb_linear = torch.empty_like(rgb)
    rgb_linear[below] = rgb[below] / 12.92
    rgb_linear[~below] = ((rgb[~below] + 0.055) / 1.055) ** 2.4
    return rgb_linear

def _rgb_to_lms(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb : Tensor (3,T,H,W), linear‑RGB
    returns LMS tensor (3,T,H,W) on same device
    """
    XYZ = torch.einsum("ij,jthw->ithw", _SRGB_TO_XYZ.to(rgb.device), rgb)
    LMS = torch.einsum("ij,jthw->ithw", _XYZ_TO_LMS.to(rgb.device), XYZ)
    return LMS
# -----------------------------------------------------------------------

from video_io import stream_clips


# ───────────────────────────────────────────────────────────────────────
# 1.  Build a spatiotemporal Gabor filter bank (Nishimoto & Gallant 2011)
# ───────────────────────────────────────────────────────────────────────
def build_spatiotemporal_gabor_bank(
    *,
    fps: float = 29.94,
    ksize: int = 31,
    ksize_t: int | None = None,
    spatial_lambdas: list[int] | None = None,
    temporal_freqs: list[float] | None = None,
    thetas: list[float] | None = None,
    gamma: float = 0.5,
) -> torch.Tensor:
    """
    Implements the Gabor basis described in Nishimoto & Gallant (2011, J. Neurosci. 31:14551–64):
        • 12 motion directions (θ)
        • 5 spatial scales (λ) log‑spaced ∈ [7, 10, 14, 20, 28]  px  (≈ 0–6 cyc/° wrt their CRT)
        • 6 temporal frequencies log‑spaced ∈ [1, 2, 4, 8, 16, 30]  Hz
        • 2 quadrature phases (0°, 90°)
    Returns
    -------
    Tensor  (2·N, 1, T, H, W)
        Cosine phase followed by sine phase for every (λ, θ, f_t) combination.
    """
    import itertools

    if spatial_lambdas is None:
        spatial_lambdas = [7, 10, 14, 20, 28, 40]   # + coarser 40 px scale
    # --- temporal parameters tied to the actual movie frame‑rate ---
    if temporal_freqs is None:
        base = [0.25, 0.5, 1, 2, 4, 8, 12]          # include slower 0.25 & 0.5 Hz
        temporal_freqs = base + [-f for f in base]  # signed → opposite motion
    if ksize_t is None:
        # keep Gaussian envelope spanning ≈108 ms (as in the paper)
        ksize_t = max(3, int(round(0.108 * fps)))
        if ksize_t % 2 == 0:
            ksize_t += 1                            # force odd length
    if thetas is None:
        thetas = np.linspace(0, np.pi, 18, endpoint=False)   # 18 directions (20° steps)
    phis = [0.0, np.pi / 2]                                  # quadrature

    sigma_t = ksize_t / 4
    kernels: list[np.ndarray] = []

    t_axis = np.linspace(-(ksize_t // 2), ksize_t // 2, ksize_t)
    yx_axis = np.linspace(-(ksize // 2), ksize // 2, ksize)

    for lam, theta, ft, phi in itertools.product(spatial_lambdas, thetas, temporal_freqs, phis):
        T, Y, X = np.meshgrid(t_axis, yx_axis, yx_axis, indexing="ij")
        # rotate spatial coords
        X_theta =  X *  np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)

        sigma_s = lam                                        # σ ∝ λ exactly as in Methods
        exp_env = np.exp(
            -0.5 * (
                (X_theta**2 + (gamma**2) * (Y_theta**2)) / (sigma_s**2) +
                (T**2) / (sigma_t**2)
            )
        )
        carrier = np.cos(2 * np.pi * (X_theta / lam + (ft / fps) * T) + phi)
        k = exp_env * carrier
        k -= k.mean()
        kernels.append(k.astype(np.float32))

    return torch.from_numpy(np.stack(kernels)[:, None]).cuda()  # (2·N,1,T,H,W)


# Build the spatiotemporal Gabor filter bank and a convenience constant
ST_GABOR_BANK = build_spatiotemporal_gabor_bank(fps=29.94, ksize=61)
KSIZE_T = ST_GABOR_BANK.shape[2]
KSIZE_S = ST_GABOR_BANK.shape[3]    # spatial kernel size (H=W)
N_ST_PAIRS = ST_GABOR_BANK.shape[0] // 2   # number of cosine–sine pairs


def divisive_normalization(resp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Divisive normalization across filters (channels).
    """
    denom = resp.norm(p=2, dim=1, keepdim=True) + eps
    return resp / denom


@torch.no_grad()
def gabor_features_st(clip_u8: torch.Tensor, colour_opponent: bool = False) -> torch.Tensor:
    """
    Spatiotemporal Gabor features (Nishimoto & Gallant 2011).
    clip_u8 : uint8 Tensor (C, T, W, H) on CPU or GPU
    returns  : float32 Tensor (N_pairs,)
    """
    # Convert uint8 RGB → float [0,1] → linear‐RGB Tensor on CUDA
    rgb = clip_u8.float().div_(255).to("cuda", non_blocking=True)

    # Prepare list of channels to filter
    if colour_opponent:
        rgb_lin = _srgb_to_linear(rgb)
        lms = _rgb_to_lms(rgb_lin)              # (3,T,H,W)

        Y_achrom    = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
        L_minus_M   = lms[0] - lms[1]
        S_minus_LM  = lms[2] - 0.5 * (lms[0] + lms[1])
        channel_list = [Y_achrom, L_minus_M, S_minus_LM]
    else:
        channel_list = [0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]]

    feats_all = []
    for chan in channel_list:
        Y = (chan - chan.mean()) / (chan.std() + 1e-6)
        Y = Y.unsqueeze(0).unsqueeze(0)                        # (1,1,T,H,W)
        resp = F.conv3d(Y, ST_GABOR_BANK,
                        stride=(1, 2, 2),         # finer spatial stride
                        padding=(KSIZE_T // 2, KSIZE_S // 2, KSIZE_S // 2))
        resp = resp.view(1, N_ST_PAIRS, 2, *resp.shape[-3:]).pow(2).sum(dim=2).sqrt()
        resp = F.relu(resp).pow(0.5)
        resp = divisive_normalization(resp)
        feats_all.append(resp.mean(dim=(2, 3, 4)))             # (1,N)
    return torch.cat(feats_all, dim=1).squeeze(0).cpu()        # (N*channels,)


# ───────────────────────────────────────────────────────────────────────
# 3.  Folder walker
# ───────────────────────────────────────────────────────────────────────
def process_video(
    path: Path,
    out_path: Path,
    tr_sec: float,
    chunk_sec: float,
    decode_w: int | None,
    decode_h: int | None,
    colour_opponent: bool = False,
) -> None:
    start_time = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return
    else:
        out_path.touch()

    feats: List[torch.Tensor] = []
    for clip in stream_clips(
        path,
        tr_sec=tr_sec,
        chunk_sec=chunk_sec,
        width=decode_w,
        height=decode_h,
    ):
        feats.append(gabor_features_st(clip, colour_opponent=colour_opponent))

    if not feats:
        print(f"[warn] {path.name} produced 0 clips")
        return
    
    feats = torch.stack(feats)  # (n_TRs, N)
    
    # Remove all NaN values


    torch.save(feats, out_path)        # (n_TRs, N)
    print(f"[ok]\t\t{feats.shape[0]} TRs extracted from {path.name} in {time.time() - start_time:.2f} sec")


def walk_folder(
    root_in: Path,
    root_out: Path,
    tr_sec: float,
    chunk_sec: float,
    decode_w: int | None,
    decode_h: int | None,
    colour_opponent: bool = False,
):
    video_paths = []
    for dirpath, _, filenames in os.walk(root_in):
        if any(part.startswith(".") for part in dirpath.split(os.sep)):
            continue
        for f in filenames:
            if f.lower().endswith((".mkv", ".mp4", ".mov")):
                video_paths.append(Path(dirpath) / f)

    print(f"Found {len(video_paths)} videos under {root_in}")
    for vp in tqdm(video_paths, desc="Processing videos", unit="video", ncols=80):
        rel = vp.relative_to(root_in).with_suffix(".pt")
        out = root_out / rel
        if os.path.exists(out):
            print(f"[skip]\t\t{out.name} already exists")
            continue
        else:
            print(f"[process]\t{vp.name} → {out.name}")
            process_video(vp, out, tr_sec, chunk_sec, decode_w, decode_h, colour_opponent=colour_opponent)


# ───────────────────────────────────────────────────────────────────────
# 4.  CLI
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--tr", type=float, default=1.49, help="seconds per TR")
    parser.add_argument("--chunk_sec", type=float, default=60.0, help="decode window")
    parser.add_argument("--decode_width",  type=int, default=-1)
    parser.add_argument("--decode_height", type=int, default=256)
    parser.add_argument("--colour_opponent", action="store_true",
                        help="If set, extract L–M and S–(L+M) opponent motion-energy channels in addition to luminance.")
    args = parser.parse_args()

    print(f"Input folder: {args.input_folder}"
          f"\nOutput folder: {args.output_folder}"
          f"\nTR: {args.tr} sec"
          f"\nChunk size: {args.chunk_sec} sec"
          f"\nDecode width: {args.decode_width}"
          f"\nDecode height: {args.decode_height}"
          f"\nColour‑opponent: {args.colour_opponent}"
          f"\n--------------------------------")
    
    print(f"Gabor filter bank size: {N_ST_PAIRS if not args.colour_opponent else 3 * N_ST_PAIRS} filters,"
          f" memory footprint: {sys.getsizeof(ST_GABOR_BANK) / 1e6:.2f} MB")

    walk_folder(
        Path(args.input_folder),
        Path(args.output_folder),
        tr_sec=args.tr,
        chunk_sec=args.chunk_sec,
        decode_w=args.decode_width,
        decode_h=args.decode_height,
        colour_opponent=args.colour_opponent,
    )