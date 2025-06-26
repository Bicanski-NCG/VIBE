#!/usr/bin/env python3
"""
Portilla-Simoncelli texture-statistics extractor. Based on Henderson *et al.* (2023, *J. Neurosci.* 43:4144-4161).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

import tqdm
import os

# ------------------------------------------------------------------
# Global device & model constants (declared early so helpers can use them)
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

BT709_COEFFS = torch.tensor([0.2126, 0.7152, 0.0722], device=DEVICE)  # R,G,B → L
HEIGHT = 4   # number of radial scales
ORDER  = 3   # steerable-pyramid order → K = ORDER+1 = 4 orientations
SHIFT_RADIUS = {3: 3, 2: 3, 1: 2, 0: 1}  # px per scale for autocorr windows


# ----------------- fast cross‑correlation helpers ------------------
_mask_cache: dict[tuple[int,int], torch.Tensor] = {}
_idx_cache: dict[tuple[int,int,int], torch.Tensor] = {}

def _gauss_mask_cached(h: int, w: int, device=DEVICE):
    key = (h, w)
    if key not in _mask_cache:
        _mask_cache[key] = _gaussian_mask(h, w, device=device)
    return _mask_cache[key]

def _sample_idx(h: int, w: int, k: int, device=DEVICE):
    key = (h, w, k)
    if key not in _idx_cache:
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        d = ((y - h//2)**2 + (x - w//2)**2).float()
        _idx_cache[key] = torch.argsort(d.flatten())[:k]
    return _idx_cache[key].to(device)

def _cross_corr_fft(a: torch.Tensor, b: torch.Tensor, k: int):
    """
    FFT‑based valid cross‑correlation. Returns k smallest‑shift coeffs.
    a, b : (H,W) float32/complex on DEVICE
    """
    H, W = a.shape
    Fa = torch.fft.rfftn(a, dim=(-2,-1))
    Fb = torch.fft.rfftn(b, dim=(-2,-1)).conj()
    corr = torch.fft.irfftn(Fa*Fb, s=(H,W), dim=(-2,-1)).real
    corr *= _gauss_mask_cached(H, W, a.device)
    idx = _sample_idx(H, W, k, a.device)
    return corr.flatten()[idx]

# ------------------------------------------------------------------
# 0b.  Home-rolled Fourier steerable pyramid (only if pyrtools missing)
# ------------------------------------------------------------------
from functools import lru_cache

def _raised_cosine(rho: torch.Tensor) -> torch.Tensor:
    """Simoncelli raised-cosine radial window."""
    return torch.where(rho < 1,
                       torch.cos(math.pi/2 * torch.log2(rho.clip(min=1e-6)))**2,
                       torch.zeros_like(rho))

def _radial_windows(h: int, w: int, S=HEIGHT, device=DEVICE):
    """Return list of S band-pass radial windows + final low-pass."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device),
                            indexing='ij')
    rho = torch.hypot(xx, yy)
    windows = []
    wl = _raised_cosine(rho)
    for _ in range(S):
        wh = torch.sqrt(1 - wl**2)
        windows.append(wh)
        # next octave
        rho = rho * 2
        wl = _raised_cosine(rho)
    return windows, wl  # band-pass list, low-pass residual

def _angle_windows(h: int, w: int, K=ORDER+1, device=DEVICE):
    """Return tensor [K, H, W] whose squared sum is 1 everywhere."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device),
                            indexing='ij')
    theta = torch.atan2(yy, xx)
    wins = []
    for k in range(K):
        ang = theta - k * math.pi / K
        wins.append(torch.cos(ang).clamp(0) ** (K - 1))
    stack = torch.stack(wins)
    return stack / torch.sqrt((stack**2).sum(0, keepdim=True) + 1e-8)

@lru_cache(maxsize=8)
def _fourier_filters(size: tuple[int, int], S=HEIGHT, K=ORDER+1):
    """Pre-compute complex Fourier filters for given spatial size."""
    h, w = size
    radials, low = _radial_windows(h, w, S, DEVICE)
    ang = _angle_windows(h, w, K, DEVICE)
    # Use list comprehension to populate filters as complex tensors
    filters = [(rad * ang[k]).to(torch.complex64)
               for s, rad in enumerate(radials)
               for k in range(K)]
    return filters, low


class _TorchPyr:
    """Minimal container mimicking pyrtools API (`pyr_coeffs`, `highpass_residual`)."""
    def __init__(self, img: torch.Tensor, S=HEIGHT, K=ORDER+1):
        h, w = img.shape
        fft = torch.fft.fft2(img)
        filts, low = _fourier_filters((h, w), S, K)
        coeffs: dict = {}
        # band-pass
        idx = 0
        for s in range(S):
            for k in range(K):
                coeffs[(s, k)] = torch.fft.ifft2(fft * filts[idx])
                idx += 1
        # residuals
        coeffs['residual_highpass'] = torch.fft.ifft2(fft * (1 - low))
        coeffs['residual_lowpass_0'] = torch.fft.ifft2(fft * low)
        self.pyr_coeffs = coeffs
        self.highpass_residual = coeffs['residual_highpass']

# ------------------------------------------------------------------
# Batched steerable pyramid: one FFT for the whole clip
# ------------------------------------------------------------------
class _TorchPyrBatch:
    """
    Vectorised version of `_TorchPyr` for a stack of frames.
    `frames` shape (T, H, W) float32 on DEVICE.
    Produces pyr_coeffs[(s,o)] -> (T, Hs, Ws) complex64
    plus .highpass_residual (T, H, W) float32.
    """
    def __init__(self, frames: torch.Tensor, S=HEIGHT, K=ORDER+1):
        assert frames.dim() == 3, "frames must be (T,H,W)"
        T, H, W = frames.shape
        fft = torch.fft.fft2(frames)                      # single batched FFT
        filts, low = _fourier_filters((H, W), S, K)
        coeffs: dict = {}
        idx = 0
        # band-pass coeffs
        for s in range(S):
            for k in range(K):
                band = torch.fft.ifft2(fft * filts[idx])   # (T,H,W) complex
                coeffs[(s, k)] = band.to(torch.complex64)
                idx += 1
        # residuals (keep real part only to save VRAM)
        coeffs['residual_highpass'] = torch.fft.ifft2(fft * (1 - low)).real.to(torch.float32)
        coeffs['residual_lowpass_0'] = torch.fft.ifft2(fft * low).real.to(torch.float32)
        self.pyr_coeffs = coeffs
        self.highpass_residual = coeffs['residual_highpass']


# ----------------------------------------------------------------------
# 1a.  Colour-space & weighting helpers (NEW)
# ----------------------------------------------------------------------
def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse-gamma: sRGB 0-1 → linear RGB."""
    threshold = 0.04045
    below = rgb <= threshold
    out = torch.empty_like(rgb)
    out[below] = rgb[below] / 12.92
    out[~below] = ((rgb[~below] + 0.055) / 1.055).pow(2.4)
    return out

def _gaussian_mask(h: int, w: int, *, sigma: float | None = None,
                   device: torch.device = DEVICE) -> torch.Tensor:
    """
    Return a centred 2-D Gaussian mask (H×W).  σ defaults to half the shorter edge,
    matching the broad pRF weighting used by Henderson et al. (2023).
    """
    if sigma is None:
        sigma = 0.5 * min(h, w)
    y = torch.arange(h, device=device) - (h - 1) / 2
    x = torch.arange(w, device=device) - (w - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.exp(-(yy**2 + xx**2) / (2 * sigma**2))

# ————————————————————————————————————————————————————————————
# 2.  Pyramid helpers
# ————————————————————————————————————————————————————————————


def _build_pyr(img: torch.Tensor):
    """
    Return a steerable-pyramid object (`_TorchPyr`) that lives on DEVICE.
    No external dependencies; always uses the home-rolled Fourier implementation.
    """
    if img.device != DEVICE:
        img = img.to(DEVICE)
    return _TorchPyr(img)

def _build_pyr_batch(frames: torch.Tensor):
    """Always returns `_TorchPyrBatch` on DEVICE."""
    if frames.device != DEVICE:
        frames = frames.to(DEVICE)
    return _TorchPyrBatch(frames)

# ————————————————————————————————————————————————————————————
# 3.  Statistics helpers (unchanged)
# ————————————————————————————————————————————————————————————

def _stats_1d(vec: torch.Tensor):
    """
    First four central moments of a 1-D tensor.  Returns mean, variance,
    skewness, and (Pearson) kurtosis.
    """
    vec = vec.flatten().float()
    n   = vec.numel()
    mean = vec.mean()
    var  = vec.var(unbiased=False) + 1e-12  # avoid zero-div
    centred = vec - mean
    skew = (centred.pow(3).mean()) / var.sqrt().pow(3)
    kurt = (centred.pow(4).mean()) / var.pow(2)
    return float(mean), float(var), float(skew), float(kurt)


def _autocorr_feats(arr: torch.Tensor, max_shift: int):
    """
    Gaussian-windowed 2-D autocorrelation.  Returns the first 17 unique
    upper-triangle values centred on zero-lag.
    """
    h, w = arr.shape
    pad = (h - 1, w - 1)
    kernel = arr.flip(0, 1).unsqueeze(0).unsqueeze(0)          # 1×1×H×W
    corr = F.conv2d(arr.unsqueeze(0).unsqueeze(0), kernel,      # full corr
                    padding=pad).squeeze()                     # H*2-1 × W*2-1

    c0, c1 = h - 1, w - 1
    patch = corr[c0 - max_shift : c0 + max_shift + 1,
                 c1 - max_shift : c1 + max_shift + 1]

    # Gaussian window (σ half patch-size) – matches Henderson repo
    g = _gaussian_mask(patch.shape[0], patch.shape[1], device=arr.device)
    patch = patch * g

    tri_idx = torch.triu_indices(patch.shape[0], patch.shape[1], device=arr.device)
    tri = patch[tri_idx[0], tri_idx[1]]   # 1-D vector of upper-triangular values
    out = torch.zeros(17, dtype=patch.dtype, device=arr.device)
    out[: min(17, tri.numel())] = tri[:17]
    return out.cpu().tolist()


def _cross_corr_feats(a: torch.Tensor, b: torch.Tensor, k: int):
    """
    Gaussian-windowed cross-correlation (valid mode).  Returns the k
    coefficients with the smallest spatial shift magnitude.
    """
    pad = (b.shape[0] - 1, b.shape[1] - 1)
    kernel = b.flip(0, 1).unsqueeze(0).unsqueeze(0)
    corr = F.conv2d(a.unsqueeze(0).unsqueeze(0), kernel, padding=pad).squeeze()

    # Gaussian weighting
    g_full = _gaussian_mask(corr.shape[0], corr.shape[1], device=a.device)
    corr = corr * g_full

    # sort coordinates by Euclidean distance from centre
    y, x = torch.meshgrid(torch.arange(corr.shape[0]),
                          torch.arange(corr.shape[1]), indexing="ij")
    d = ((y - (corr.shape[0] - 1) / 2) ** 2 +
         (x - (corr.shape[1] - 1) / 2) ** 2).sqrt()
    flat = corr.flatten()
    _, idx = torch.sort(d.flatten())
    return flat[idx][:k].cpu().tolist()

# ------------------------------------------------------------------
# Phase compensation (Portilla–Simoncelli Appendix A.3)
# ------------------------------------------------------------------
def _phase_compensate(cmplx: torch.Tensor) -> torch.Tensor:
    """
    Remove local phase: x → Re( x / |x| ).  For zero-magnitude elements we
    keep them at 0 to avoid NaNs.  Result is a *real* tensor on DEVICE.
    """
    mag = cmplx.abs().clamp(min=1e-8)
    return (cmplx / mag).real

# ————————————————————————————————————————————————————————————
# 4.  Frame-level feature extractor
# ————————————————————————————————————————————————————————————

def frame_features(frame_u8: torch.Tensor) -> torch.Tensor:
    """Return 641-D feature vector for a single RGB frame."""
    rgb = frame_u8.to(DEVICE, dtype=torch.float32).div(255.0)                    # 0-1 float
    rgb_lin = _srgb_to_linear(rgb)                       # undo sRGB gamma
    lum = (rgb_lin * BT709_COEFFS[:, None, None]).sum(0) # linear luminance
    lum = (lum - lum.mean()) / (lum.std() + 1e-8)        # per-frame z-score

    # pixel stats
    p_min, p_max = float(lum.min()), float(lum.max())
    p_mean, p_var, p_skew, p_kurt = _stats_1d(lum.flatten())
    feats: List[float] = [p_min, p_max, p_mean, p_var, p_skew, p_kurt]

    pyr = _build_pyr(lum)
    mag_maps, real_maps, comp_maps, lowpass_recons = {}, {}, {}, []

    mask_cache = {}  # shape -> Gaussian mask (torch)
    for key, arr in pyr.pyr_coeffs.items():
        # band-pass keys in our home-rolled pyramid are 2-tuples (scale, orient)
        if isinstance(key, tuple) and len(key) == 2:
            s, b = key
        elif isinstance(key, str):
            # residual strings → handle after if/elif cascade
            b = -99  # sentinel
        else:
            continue  # unknown key format

        if b >= 0:  # oriented band-pass
            arr_c = torch.as_tensor(arr, dtype=torch.complex64, device=DEVICE)
            h_arr, w_arr = arr_c.shape
            m = mask_cache.setdefault((h_arr, w_arr), _gaussian_mask(h_arr, w_arr))
            mag_maps[(s, b)]  = arr_c.abs() * m
            real_maps[(s, b)] = arr_c.real.float() * m
            comp_maps[(s, b)] = arr_c * m
        elif isinstance(key, str):
            if key.startswith("residual_lowpass"):
                arr_t = torch.as_tensor(arr.real, dtype=torch.float32, device=DEVICE)
                h_arr, w_arr = arr_t.shape
                m = mask_cache.setdefault((h_arr, w_arr), _gaussian_mask(h_arr, w_arr))
                lowpass_recons.append(arr_t * m)
            elif key == "residual_highpass":
                hp_resid = torch.as_tensor(arr.real, dtype=torch.float32, device=DEVICE)

    lowpass_recons = lowpass_recons[:HEIGHT]

    # hp_resid = torch.from_numpy(hp_resid).float()
    hp_resid *= _gaussian_mask(*hp_resid.shape)

    # energy-mean & linear-mean
    feats.extend([m.mean().item() for (s, o), m in sorted(mag_maps.items())])
    feats.extend([r.mean().item() for (s, o), r in sorted(real_maps.items())])

    # marginal
    for lp in lowpass_recons:
        mu, _, sk, ku = _stats_1d(lp.flatten())
        feats.extend([mu, sk, ku])
    hp_mu, hp_var, hp_sk, hp_ku = _stats_1d(hp_resid.flatten())
    feats.extend([hp_mu, hp_var, hp_sk, hp_ku])

    # energy-auto & linear-auto
    for (s, o), m in sorted(mag_maps.items()):
        feats.extend(_autocorr_feats(m, SHIFT_RADIUS[s]))
    for lp in lowpass_recons:
        feats.extend(_autocorr_feats(lp, 3))
    feats.extend(_autocorr_feats(hp_resid, 3))

    # energy-cross-orient & linear-cross-orient
    for s in range(HEIGHT):
        for o1 in range(4):
            for o2 in range(o1 + 1, 4):
                feats.extend(_cross_corr_feats(mag_maps[(s, o1)], mag_maps[(s, o2)], 6))
                # linear (phase-compensated)
                a_lin = _phase_compensate(comp_maps[(s, o1)])
                b_lin = _phase_compensate(comp_maps[(s, o2)])
                feats.extend(_cross_corr_feats(a_lin, b_lin, 8))

    # energy-cross-scale & linear-cross-scale
    for o in range(4):
        for s1 in range(HEIGHT):
            for s2 in range(s1 + 1, HEIGHT):
                feats.extend(_cross_corr_feats(mag_maps[(s1, o)], mag_maps[(s2, o)], 4))
                a_lin = _phase_compensate(comp_maps[(s1, o)])
                b_lin = _phase_compensate(comp_maps[(s2, o)])
                feats.extend(_cross_corr_feats(a_lin, b_lin, 8))

    return torch.tensor(feats, dtype=torch.float32)

# ————————————————————————————————————————————————————————————
# 5.  Video helpers (loader unchanged)
# ————————————————————————————————————————————————————————————



# ------------------------------------------------------------------
# Re‑usable frame‑statistics from a shared batch pyramid
# ------------------------------------------------------------------
def _features_from_pyr(pyr: _TorchPyrBatch,
                       lum_frame: torch.Tensor,
                       t: int) -> torch.Tensor:
    """Compute 641‑D PS stats for frame *t* using pre‑built pyramid."""
    timings = {}
    t0 = time.perf_counter()
    # pixel stats
    p_min, p_max = float(lum_frame.min()), float(lum_frame.max())
    p_mean, p_var, p_skew, p_kurt = _stats_1d(lum_frame.flatten())
    feats: List[float] = [p_min, p_max, p_mean, p_var, p_skew, p_kurt]

    timings['pixel'] = time.perf_counter() - t0
    t0 = time.perf_counter()

    mag_maps, comp_maps, lowpass_recons = {}, {}, []
    mask_cache = {}

    for key, arr in pyr.pyr_coeffs.items():
        if isinstance(key, tuple) and len(key) == 2:
            s, o = key
            arr_c = arr[t]                               # (Hs,Ws) complex
            h_arr, w_arr = arr_c.shape
            m = mask_cache.setdefault((h_arr, w_arr), _gaussian_mask(h_arr, w_arr))
            mag_maps[(s, o)]  = arr_c.abs().float() * m
            comp_maps[(s, o)] = arr_c * m
        elif key.startswith('residual_lowpass'):
            lp = arr[t].float()
            h_arr, w_arr = lp.shape
            m = mask_cache.setdefault((h_arr, w_arr), _gaussian_mask(h_arr, w_arr))
            lowpass_recons.append(lp * m)
        elif key == 'residual_highpass':
            hp_resid = pyr.highpass_residual[t]

    timings['map_build'] = time.perf_counter() - t0
    t0 = time.perf_counter()

    lowpass_recons = lowpass_recons[:HEIGHT]
    hp_resid *= _gaussian_mask(*hp_resid.shape)

    # energy & linear means
    feats.extend([mag_maps[k].mean().item()  for k in sorted(mag_maps)])
    feats.extend([comp_maps[k].real.float().mean().item() for k in sorted(comp_maps)])

    timings['means'] = time.perf_counter() - t0
    t0 = time.perf_counter()

    # marginal
    for lp in lowpass_recons:
        mu, _, sk, ku = _stats_1d(lp.flatten())
        feats.extend([mu, sk, ku])
    hp_mu, hp_var, hp_sk, hp_ku = _stats_1d(hp_resid.flatten())
    feats.extend([hp_mu, hp_var, hp_sk, hp_ku])

    timings['marginal'] = time.perf_counter() - t0
    t0 = time.perf_counter()

    # autocorrs
    for (s,o), m in sorted(mag_maps.items()):
        feats.extend(_autocorr_feats(m, SHIFT_RADIUS[s]))
    for lp in lowpass_recons:
        feats.extend(_autocorr_feats(lp, 3))
    feats.extend(_autocorr_feats(hp_resid, 3))

    timings['auto'] = time.perf_counter() - t0
    t0 = time.perf_counter()

    # ---------- fast cross‑orientation using FFT ----------
    for s in range(HEIGHT):
        # precompute FFTs per orientation for this scale
        F_mag = []
        F_lin = []
        for o in range(4):
            m = mag_maps[(s, o)]
            F_mag.append(torch.fft.rfftn(m, dim=(-2,-1)))
            a_lin = _phase_compensate(comp_maps[(s, o)])
            F_lin.append(torch.fft.rfftn(a_lin, dim=(-2,-1)))
        # iterate over unique pairs
        for o1 in range(4):
            for o2 in range(o1+1, 4):
                # energy correlations
                corr_mag = torch.fft.irfftn(F_mag[o1]*F_mag[o2].conj(),
                                            s=mag_maps[(s,o1)].shape,
                                            dim=(-2,-1)).real
                corr_mag *= _gauss_mask_cached(*corr_mag.shape, corr_mag.device)
                feats.extend(corr_mag.flatten()[_sample_idx(*corr_mag.shape, 6)].cpu().tolist())
                # linear correlations (phase‑compensated)
                corr_lin = torch.fft.irfftn(F_lin[o1]*F_lin[o2].conj(),
                                            s=mag_maps[(s,o1)].shape,
                                            dim=(-2,-1)).real
                corr_lin *= _gauss_mask_cached(*corr_lin.shape, corr_lin.device)
                feats.extend(corr_lin.flatten()[_sample_idx(*corr_lin.shape, 8)].cpu().tolist())
    timings['cross_orient'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    # cross‑scale
    for o in range(4):
        for s1 in range(HEIGHT):
            for s2 in range(s1+1, HEIGHT):
                feats.extend(_cross_corr_feats(mag_maps[(s1,o)], mag_maps[(s2,o)], 4))
                a_lin = _phase_compensate(comp_maps[(s1,o)])
                b_lin = _phase_compensate(comp_maps[(s2,o)])
                feats.extend(_cross_corr_feats(a_lin, b_lin, 8))

    timings['cross_scale'] = time.perf_counter() - t0
    print(f"[perf frame] pixel {timings['pixel']:.3f}s, "
          f"maps {timings['map_build']:.3f}s, means {timings['means']:.3f}s, "
          f"marg {timings['marginal']:.3f}s, auto {timings['auto']:.3f}s, "
          f"co {timings['cross_orient']:.3f}s, cs {timings['cross_scale']:.3f}s",
          flush=True)

    return torch.tensor(feats, dtype=torch.float32, device=lum_frame.device)


def process_video(
    path: Path,
    out_path: Path,
    tr_sec: float,
    chunk_sec: float,
    decode_w: int,
    decode_h: int,
) -> None:
    from gabor import stream_clips

    start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return

    tr_features: List[torch.Tensor] = []
    for clip in stream_clips(
        path,
        tr_sec=tr_sec,
        chunk_sec=chunk_sec,
        width=decode_w,
        height=decode_h,
    ):
        clip_start = time.perf_counter()
        # --- batched luminance pre‑processing ---
        rgb = clip.float().to(DEVICE) / 255.0
        rgb_lin = _srgb_to_linear(rgb)
        lum = (rgb_lin * BT709_COEFFS[:, None, None, None]).sum(0)  # (T,H,W)
        lum = (lum - lum.mean(dim=(1,2), keepdim=True)) / (lum.std(dim=(1,2), keepdim=True)+1e-8)
        t_pre  = time.perf_counter()
        print(f"[timer] {Path(path).name}: preprocess {t_pre - clip_start:.3f}s", flush=True)

        pyr_batch = _build_pyr_batch(lum)                          # one FFT
        t_pyr = time.perf_counter()
        print(f"[timer] {Path(path).name}: pyramid   {t_pyr - t_pre:.3f}s", flush=True)

        T_frames = lum.size(0)
        running: torch.Tensor | None = None
        for t in range(T_frames):
            feat = _features_from_pyr(pyr_batch, lum[t], t)
            running = feat if running is None else running + feat
            print(f"[timer] {Path(path).name}: frame {t+1:03d}/{T_frames:03d}  {time.perf_counter() - t_pyr:.3f}s", flush=True)
            t_pyr = time.perf_counter()

        tr_features.append((running / T_frames).cpu())
        t_feat = time.perf_counter()
        print(f"[timer] {Path(path).name}: stats     {t_feat - t_pyr:.3f}s  (clip total {t_feat - clip_start:.3f}s)", flush=True)

    if not tr_features:
        print(f"[warn] {path.name} produced 0 clips")
        return

    out = torch.stack(tr_features)           # shape (n_TR, n_feat)
    torch.save(out, out_path)
    dur = time.time() - start
    print(f"[ok] {out.shape[0]} TRs, {out.shape[1]} feats  ← {path.name} ({dur:.1f}s)")


# ————————————————————————————————————————————————————————————
# 6.  Folder walker & CLI
# ————————————————————————————————————————————————————————————

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
    ap = argparse.ArgumentParser("Portilla–Simoncelli extractor (rectangular)")
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
