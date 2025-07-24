from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import torch
import librosa

__all__ = [
    "open_audio",
    "audio_tensor",
    "stream_chunks",
    "stream_clips",
]

# ---------------------------------------------------------------------------
# 1.  Reader
# ---------------------------------------------------------------------------
def open_audio(
    path: str | Path,
    *,
    target_sr: int = 16_000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Extract the audio track from an .mkv (or any FFmpeg‑readable) container
    and resample to *target_sr* (Hz).

    Returns
    -------
    y : ndarray float32, shape (C, N) in [-1, 1]
    sr: int – effective sampling‑rate (equals target_sr)
    """
    y, sr = librosa.load(str(path), sr=None, mono=False)  # native rate
    if y.ndim == 1:
        y = y[np.newaxis, :]          # (1, N)
    elif y.shape[0] > y.shape[1]:     # librosa sometimes returns (N, C)
        y = y.T

    y = y.astype(np.float32, copy=False)

    if mono and y.shape[0] > 1:
        y = y.mean(axis=0, keepdims=True)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, axis=-1)
        sr = target_sr

    return y, sr


# ---------------------------------------------------------------------------
# 2.  Load entire waveform (use for short files)
# ---------------------------------------------------------------------------
def audio_tensor(
    path: str | Path,
    *,
    target_sr: int = 16_000,
    mono: bool = True,
) -> torch.Tensor:
    """
    Return tensor **(C, T_samples)** float32.
    """
    y, _ = open_audio(path, target_sr=target_sr, mono=mono)
    return torch.from_numpy(y).contiguous()


# ---------------------------------------------------------------------------
# 3.  Chunk generator (≤ chunk_sec)
# ---------------------------------------------------------------------------
def stream_chunks(
    path: str | Path,
    *,
    chunk_sec: float = 60.0,
    target_sr: int = 16_000,
    mono: bool = True,
) -> Generator[Tuple[torch.Tensor, int], None, None]:
    """
    Yield successive *(wave, sr)* pairs where wave is (C, T_chunk) float32.
    """
    y, sr = open_audio(path, target_sr=target_sr, mono=mono)
    chunk_samples = int(round(chunk_sec * sr))
    N = y.shape[-1]
    for start in range(0, N, chunk_samples):
        chunk = y[..., start : start + chunk_samples]
        yield torch.from_numpy(chunk).contiguous(), sr


# ---------------------------------------------------------------------------
# 4.  TR‑aligned clip generator
# ---------------------------------------------------------------------------
def stream_clips(
    path: str | Path,
    *,
    tr_sec: float,
    chunk_sec: float | None = 60.0,
    target_sr: int = 16_000,
    mono: bool = True,
    load_full: bool = True,
) -> Generator[torch.Tensor, None, None]:
    """
    Yield consecutive **TR‑length** waveform clips, shape (C, T_TR).
    Uses the same fractional‑accumulator logic as `video_io.stream_clips`.
    """
    if load_full or chunk_sec is None:
        # read entire audio into RAM once
        full, sr_ref = open_audio(path, target_sr=target_sr, mono=mono)
        gen_iter = [(torch.from_numpy(full).contiguous(), sr_ref)]
    else:
        gen_iter = stream_chunks(
            path,
            chunk_sec=chunk_sec,
            target_sr=target_sr,
            mono=mono,
        )

    carry: torch.Tensor | None = None
    sr_ref: int | None = None
    acc         = 0.0           # fraction accumulator initialised later
    base_len    = 0
    frac_per_tr = 0.0

    for chunk, sr in gen_iter:
        if sr_ref is None:
            sr_ref = sr
            samples_per_tr_f = tr_sec * sr_ref
            base_len    = int(samples_per_tr_f)
            frac_per_tr = samples_per_tr_f - base_len
            if base_len < 1:
                raise ValueError(f"tr_sec={tr_sec} too small for sr={sr_ref}")
        else:
            if sr != sr_ref:
                raise RuntimeError("sample‑rate drift")

        if carry is not None and carry.numel() > 0:
            chunk = torch.cat((carry, chunk), dim=-1)

        T = chunk.shape[-1]
        start = 0
        while True:
            tr_len = base_len
            acc += frac_per_tr
            if acc >= 1.0 - 1e-6:
                tr_len += 1
                acc -= 1.0

            if start + tr_len <= T:
                yield chunk[..., start : start + tr_len]
                start += tr_len
            else:
                break

        carry = chunk[..., start:] if start < T else None

    if carry is not None and carry.numel() > 0:
        yield carry