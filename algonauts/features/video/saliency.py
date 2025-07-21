#!/usr/bin/env python3
"""
Itti–Koch–Niebur (1998) Saliency Feature Extractor
==================================================

Implements the purely bottom-up saliency model described in

    Itti, L., Koch, C., & Niebur, E.  *A model of saliency-based visual
    attention for rapid scene analysis*.  IEEE TPAMI **20**(11),
    1254-1259 (1998) — IEEE Doc 730558.

For every TR-length video clip we compute the final input **6(s=4)** to the
saliency map (Eq. 8 in the paper) and store it as a flattened vector.
Output tensor shape → *(n_TR, 60 × 60 = 3 600)* when the decode height is
240 px (scale-4 size = H/4).

Key steps faithfully reproduced
-------------------------------
1. **Nine-scale dyadic Gaussian pyramids** for intensity and RGB ⇒
   scales 0..8.
2. **Center–surround feature maps**:
   * Intensity contrast |I(c) – I(s)| → 6 maps.
   * Color RG & BY double-opponent (Eqs 2–3) → 12 maps.
   * Orientation contrast using Gabor pyramids at 0°,45°,90°,135° → 24 maps.
3. **Across-scale combination** at scale 4, yielding conspicuity maps
   , (intensity), & (color), 2 (orientation).
4. **Normalization operator ℵ(.)** (paper’s Fig. 2):
   * linear scale to [0,1]; compute global max M and mean of *other local
     maxima* m; multiply map by (M – m)².
5. **Final feature vector** is the flattened
   ℵ(,)+ℵ(&)+ℵ(2)  (no WTA dynamics – we only need the bottom-up map).

Dependencies:  `torch ≥ 2.0`, `scipy` (for local-max detection).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List

import tqdm
import os

import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

from gabor import stream_clips  # reuse existing video decoding helper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
TORCH_COMPLEX = torch.complex64

aRGB2I = torch.tensor([0.2126, 0.7152, 0.0722], device=DEVICE)

# ----------------------------------------------------------------------
# Gaussian pyramid utilities
# ----------------------------------------------------------------------

def gaussian_blur(img: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    ksize = int(2 * math.ceil(3 * sigma) + 1)
    grid = torch.arange(ksize, device=img.device) - ksize // 2
    g = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g = g.to(img.dtype)
    # separable blur
    kern_h = g.view(1, 1, 1, -1)     # 1×1×1×K
    kern_v = g.view(1, 1, -1, 1)     # 1×1×K×1
    img = F.conv2d(img, kern_h, padding=(0, ksize // 2))
    img = F.conv2d(img, kern_v, padding=(ksize // 2, 0))
    return img

def build_pyramid(frame: torch.Tensor, levels: int = 9) -> list[torch.Tensor]:
    pyr = [frame]
    cur = frame
    for _ in range(1, levels):
        cur = gaussian_blur(cur, sigma=1.0)
        cur = F.interpolate(cur, scale_factor=0.5, mode="bilinear", align_corners=False)
        pyr.append(cur)
    return pyr  # scale 0..8

# ----------------------------------------------------------------------
# Orientation Gabor pyramid (steerable-like but lightweight)
# ----------------------------------------------------------------------

def gabor_kernel(theta_deg: float, size: int = 9, lambda_pix: float = 4.0) -> torch.Tensor:
    theta = math.radians(theta_deg)
    sigma = size / 3
    grid = torch.arange(size, device=DEVICE) - size // 2
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xr = xx * math.cos(theta) + yy * math.sin(theta)
    yr = -xx * math.sin(theta) + yy * math.cos(theta)
    env = torch.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2))
    carrier = torch.cos(2 * math.pi * xr / lambda_pix)
    return (env * carrier).unsqueeze(0).unsqueeze(0)

GABOR_KERNELS = {th: gabor_kernel(th) for th in (0, 45, 90, 135)}


def orientation_pyramid(intensity_pyr: list[torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
    """Returns {(scale, theta_idx): map}."""
    maps = {}
    for s, img in enumerate(intensity_pyr):
        for th_idx, th in enumerate((0, 45, 90, 135)):
            k = GABOR_KERNELS[th]
            resp = F.conv2d(img, k, padding=k.shape[-1] // 2)
            maps[(s, th_idx)] = resp.squeeze(0)
    return maps

# ----------------------------------------------------------------------
# Normalization operator ℵ (call per map)
# ----------------------------------------------------------------------

def normalize_map(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    # local maxima (8-neighb)
    mx = torch.as_tensor(maximum_filter(x.cpu().numpy(), size=3, mode="nearest"), device=x.device)
    local_max = x[(x == mx) & (x > 0)]
    if local_max.numel() < 2:
        return x * 0  # no salient peak
    M = local_max.max()
    m = local_max[local_max != M].mean()
    return x * ((M - m) ** 2)

# ----------------------------------------------------------------------
# Feature extraction per frame
# ----------------------------------------------------------------------

def saliency_features(frame_u8: torch.Tensor) -> torch.Tensor:
    """frame_u8: uint8 RGB C×H×W on CPU → returns flat saliency at scale 4."""
    frame = frame_u8.float().to(DEVICE) / 255.0
    I = (frame * aRGB2I[:, None, None]).sum(0, keepdim=True)  # 1×H×W

    # color channels R,G,B,Y
    r, g, b = frame
    R = (r - (g + b) / 2).clamp(min=0, max=1)[None]
    G = (g - (r + b) / 2).clamp(min=0, max=1)[None]
    B = (b - (r + g) / 2).clamp(min=0, max=1)[None]
    Y = ((r + g) / 2 - (r - g).abs() / 2 - b).clamp(min=0, max=1)[None]

    # pyramids
    I_pyr = build_pyramid(I)
    R_pyr = build_pyramid(R)
    G_pyr = build_pyramid(G)
    B_pyr = build_pyramid(B)
    Y_pyr = build_pyramid(Y)

    # intensity feature maps (|I(c) − I(s)|)
    IMaps = []
    for c in (2, 3, 4):
        for d in (3, 4):
            s = c + d
            diff = (I_pyr[c] - F.interpolate(I_pyr[s], size=I_pyr[c].shape[-2:], mode="bilinear", align_corners=False)).abs()
            IMaps.append(diff)

    # color maps RG and BY
    CMaps = []
    for c in (2, 3, 4):
        for d in (3, 4):
            s = c + d
            RG = (R_pyr[c] - G_pyr[c]).abs()
            GR = (G_pyr[c] - R_pyr[c]).abs()
            rg_map = (RG - F.interpolate(GR, size=RG.shape[-2:], mode="bilinear", align_corners=False)).abs()
            BY = (B_pyr[c] - Y_pyr[c]).abs()
            YB = (Y_pyr[c] - B_pyr[c]).abs()
            by_map = (BY - F.interpolate(YB, size=BY.shape[-2:], mode="bilinear", align_corners=False)).abs()
            CMaps.extend([rg_map, by_map])

    # orientation maps
    OrientMaps = []
    O_pyr = orientation_pyramid(I_pyr)
    for th_idx in range(4):
        for c in (2, 3, 4):
            for d in (3, 4):
                s = c + d
                Oc = O_pyr[(c, th_idx)]
                Os = F.interpolate(O_pyr[(s, th_idx)].unsqueeze(0), size=Oc.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                OrientMaps.append((Oc - Os).abs())

    # normalize and combine to conspicuity maps at scale 4
    def to_scale4(m: torch.Tensor) -> torch.Tensor:
        return F.interpolate(m.unsqueeze(0), size=I_pyr[4].shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

    C_intensity = normalize_map(sum(to_scale4(m) for m in IMaps))
    C_color     = normalize_map(sum(to_scale4(m) for m in CMaps))
    # orientation summed over orientations then norm
    C_orient = torch.zeros_like(I_pyr[4])
    for m in OrientMaps:
        C_orient += to_scale4(m)
    C_orient = normalize_map(C_orient)

    saliency = C_intensity + C_color + C_orient  # Eq 8 (three maps equally weighted)
    return saliency.flatten().cpu()

# ----------------------------------------------------------------------
# Video processing pipeline
# ----------------------------------------------------------------------

def process_video(path: Path, out_path: Path, tr_sec: float, chunk_sec: float,
                  decode_w: int, decode_h: int):
    start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return

    vecs: List[torch.Tensor] = []
    for clip in stream_clips(path, tr_sec=tr_sec, chunk_sec=chunk_sec,
                              width=decode_w, height=decode_h):
        running: torch.Tensor | None = None
        cnt = 0
        for t in range(clip.shape[1]):
            f = saliency_features(clip[:, t])
            cnt += 1
            running = f if running is None else running + (f - running) / cnt
        vecs.append(running)

    if not vecs:
        print(f"[warn] {path.name} produced 0 clips")
        return

    out = torch.stack(vecs)  # (n_TR, 3600)
    torch.save(out, out_path)
    print(f"[ok] {out.shape[0]} TRs × {out.shape[1]}  ← {path.name} ({time.time()-start:.1f}s)")

# ----------------------------------------------------------------------
# Folder traversal & CLI
# ----------------------------------------------------------------------

def walk_folder(
    root_in: Path,
    root_out: Path,
    tr_sec: float,
    chunk_sec: float,
    decode_w: int | None,
    decode_h: int | None,
):
    video_paths = []
    for dirpath, _, filenames in os.walk(root_in):
        if any(part.startswith(".") for part in dirpath.split(os.sep)):
            continue
        for f in filenames:
            if f.lower().endswith((".mkv", ".mp4", ".mov")):
                video_paths.append(Path(dirpath) / f)

    print(f"Found {len(video_paths)} videos under {root_in}")
    for vp in tqdm.tqdm(video_paths, desc="Processing videos", unit="video", ncols=80):
        rel = vp.relative_to(root_in).with_suffix(".pt")
        out = root_out / rel
        if os.path.exists(out):
            print(f"[skip]\t\t{out.name} already exists")
            continue
        else:
            print(f"[process]\t{vp.name} → {out.name}", flush=True)
            process_video(vp, out, tr_sec, chunk_sec, decode_w, decode_h)


def main() -> None:
    ap = argparse.ArgumentParser("Itti-Koch saliency map extractor (rectangular)")
    ap.add_argument("--input_folder", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--tr", type=float, default=1.49)
    ap.add_argument("--chunk_sec", type=float, default=60.0)
    ap.add_argument("--decode_width", type=int, default=-1,
                    help="Width passed to ffmpeg; −1 keeps aspect ratio")
    ap.add_argument("--decode_height", type=int, default=240,
                    help="Height passed to ffmpeg; −1 keeps aspect ratio")
    args = ap.parse_args()

    walk_folder(
        Path(args.input_folder),
        Path(args.output_folder),
        tr_sec=args.tr,
        chunk_sec=args.chunk_sec,
        decode_w=args.decode_width,
        decode_h=args.decode_height,
    )


if __name__ == "__main__":
    main()