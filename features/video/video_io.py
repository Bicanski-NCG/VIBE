from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple, Union

import torch
from decord import VideoReader, cpu

__all__ = [
    "open_reader",
    "video_tensor",
    "stream_chunks",
    "stream_clips",
]

# ---------------------------------------------------------------------------
# 1.  Reader factory
# ---------------------------------------------------------------------------
def open_reader(
    path: Union[str, Path],
    *,
    width: int | None = None,
    height: int | None = None,
    num_threads: int = 8,
) -> VideoReader:
    """
    Create a Decord `VideoReader` for the given video file.

    Parameters
    ----------
    path        : video file
    width/height: decode-time scaling; keep one of them -1 to preserve aspect
    num_threads : C++ worker threads
    """
    opts = {}
    if width is not None:
        opts["width"] = width
    if height is not None:
        opts["height"] = height

    return VideoReader(str(path), ctx=cpu(0), num_threads=num_threads, **opts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2.  Decode the *entire* movie (use only for short / down-scaled clips)
# ---------------------------------------------------------------------------
def video_tensor(
    path: Union[str, Path] | VideoReader,
    *,
    width: int | None = 256,
    height: int | None = -1,
    num_threads: int = 8,
) -> torch.Tensor:
    """
    Return one tensor shaped **(C, T, W, H)** uint8 containing all frames.
    """
    vr = (
        open_reader(path, width=width, height=height, num_threads=num_threads)
        if not isinstance(path, VideoReader)
        else path
    )

    frames = vr.get_batch(range(len(vr))).asnumpy()          # (T,H,W,3)
    return torch.as_tensor(frames).permute(3, 0, 2, 1).contiguous()  # (C,T,W,H)


# ---------------------------------------------------------------------------
# 3.  Generator that yields ≤ chunk_sec windows
# ---------------------------------------------------------------------------
def stream_chunks(
    path: Union[str, Path] | VideoReader,
    *,
    chunk_sec: float = 30.0,
    width: int | None = None,
    height: int | None = None,
    num_threads: int = 8,
) -> Generator[Tuple[torch.Tensor, float], None, None]:
    """
    Yield successive *(tensor, fps)* pairs where tensor is (C, T, W, H) uint8.

    Each tensor spans at most *chunk_sec* seconds, letting you bound RAM while
    still amortising the decoder startup cost.
    """
    vr = (
        open_reader(path, width=width, height=height, num_threads=num_threads)
        if not isinstance(path, VideoReader)
        else path
    )

    fps = float(vr.get_avg_fps())
    chunk_frames = int(round(chunk_sec * fps))

    for frm0 in range(0, len(vr), chunk_frames):
        idx = range(frm0, min(len(vr), frm0 + chunk_frames))
        batch = vr.get_batch(idx).asnumpy()                  # (T,H,W,3)
        yield torch.as_tensor(batch).permute(3, 0, 2, 1).contiguous(), fps


# ---------------------------------------------------------------------------
# 4.  Generator that yields **TR-aligned** clips directly
# ---------------------------------------------------------------------------
def stream_clips(
    path: Union[str, Path] | VideoReader,
    *,
    tr_sec: float,                  # duration of one clip
    chunk_sec: float = 30.0,        # decode window size
    width: int | None = None,
    height: int | None = None,
    num_threads: int = 8,
) -> Generator[torch.Tensor, None, None]:
    """
    Yield consecutive TR-sized tensors (C, T_TR, W, H) uint8.

    Internally decodes ≤ *chunk_sec* at a time, so memory stays bounded.
    """
    # -----------------------------------------------------------------------
    # Buffered implementation that preserves remainder frames across chunks
    # -----------------------------------------------------------------------
    carry: torch.Tensor | None = None          # leftover (C, T_rem, W, H)
    fps_ref: float | None = None
    chunk_iter = stream_chunks(
        path,
        chunk_sec=chunk_sec,
        width=width,
        height=height,
        num_threads=num_threads,
    )
    for chunk, fps in chunk_iter:
        # ── 1.  Initialise TR parameters once ─────────────────────────────
        if fps_ref is None:
            fps_ref = fps
            frames_per_tr_f = tr_sec * fps_ref          # non‑integer (e.g. 44.61)
            base_len       = int(frames_per_tr_f)       # 44
            frac_per_tr    = frames_per_tr_f - base_len # 0.61
            acc            = 0.0                        # residual accumulator
            if base_len < 1:
                raise ValueError(f"tr_sec={tr_sec} too small for fps={fps_ref}")
        else:
            if abs(fps - fps_ref) > 1e-2:
                raise RuntimeError("fps drift between chunks")

        # ── 2.  Prepend carry‑over frames ────────────────────────────────
        if carry is not None and carry.numel() > 0:
            chunk = torch.cat((carry, chunk), dim=1)    # concat on time

        # ── 3.  Emit TR windows using the residual‑accumulation rule ────
        T = chunk.shape[1]
        start = 0
        while True:
            # Decide how many frames this TR should have (44 or 45)
            tr_len = base_len
            acc += frac_per_tr
            if acc >= 1.0 - 1e-6:       # numeric guard
                tr_len += 1
                acc -= 1.0

            if start + tr_len <= T:
                yield chunk[:, start : start + tr_len]
                start += tr_len
            else:
                break  # not enough frames left for a full TR

        # ── 4.  Save remainder for next chunk ────────────────────────────
        carry = chunk[:, start:] if start < T else None

    # Emit any remaining frames
    if carry is not None and carry.numel() > 0:
        yield carry