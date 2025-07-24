import argparse
import gc
import math
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm
import random

# ----------------------------------------------------------------------------
# Model initialisation
# ----------------------------------------------------------------------------
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

# Increase CUDA allocator flexibility (optional but mirrors your SlowFast env)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_movie_info(movie_path: str):
    """Return `(fps, duration_seconds)` for *movie_path*."""
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, num_frames / fps


def register_hooks(mapping: Dict[str, torch.nn.Module]):
    """Register forward‑hooks and return (store, handles)."""
    store: Dict[str, List[torch.Tensor]] = {k: [] for k in mapping}
    handles = []

    def make_hook(name):
        def _hook(_, __, out):
            # keep on CPU → free GPU immediately
            x = pool_proj_out(out)
            store[name].append(x)
        return _hook

    for name, module in mapping.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    return store, handles


def pool_proj_out(x: torch.Tensor, *, frames=32) -> torch.Tensor:
    """
    x : Tensor(B·T, N_tokens, 1024)  from the forward hook

    Returns
    -------
    Tensor(B, 1024)  – one vector per clip
    """
    # 1) average over tokens
    x = x.mean(dim=1)                      # (B·T, 1024)

    # 2) restore (B, T, 1024) and average over time
    if x.shape[0] % frames != 0:
        raise ValueError("Unexpected number of frames in hook output")
    x = x.view(-1, frames, x.shape[-1]).mean(dim=1)   # (B, 1024)

    return x 


def uniform_sample(frames: List[np.ndarray], num_samples: int = 32):
    """Uniformly pick *num_samples* indices from *frames* (with repeat if needed)."""
    if len(frames) == num_samples:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=num_samples, dtype=int)
    return [frames[i] for i in idx]


# ---------------------------------------------------------------------------
# Core extraction routine
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_video_features(
    video_path: str,
    video_file: str,
    output_folder: str,
    relative_path: str,
    model: VideoDepthAnything,
    tr: float,
):
    """Stream *video_path* → save pooled proj_out features to disk."""
    fps, _ = get_movie_info(video_path)
    frames_per_window = int(math.ceil(tr * fps))

    # Hook registration (one layer only)
    mapping = {
        "proj_out": model.head.motion_modules[0].temporal_transformer.proj_out,
    }
    post_transform = {"proj_out": pool_proj_out}

    store, handles = register_hooks(mapping)

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    frames_buf: List[np.ndarray] = []

    # NEW ─────────────────────────────────────────
    clip_buffer: List[np.ndarray] = []        # keeps 32-frame clips until we hit BATCH_SIZE
    # ────────────────────────────────────────────

    with tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        desc=f"Processing {os.path.basename(video_path)}",
        leave=False,
    ) as pbar:
        while success:
            frames_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frames_buf) >= frames_per_window:
                # ---------------- Clip completed ----------------
                sampled   = uniform_sample(frames_buf, 32)
                frames_np = np.stack(sampled, axis=0).astype(np.uint8)
                clip_buffer.append(frames_np)
                frames_buf.clear()

                # ---------- NEW: flush when we reach a mini-batch ----------
                if len(clip_buffer) == 4:
                    batch_np = np.stack(clip_buffer, axis=0)  # (B,T,H,W,3)
                    _ = model.infer_video_depth(batch_np, target_fps=fps, fp32=False)
                    clip_buffer.clear()
                    torch.cuda.empty_cache()
                # -----------------------------------------------------------

            success, frame = cap.read()
            pbar.update(1)

    # Process any residual frames that never closed a full TR window
    if frames_buf:
        sampled = uniform_sample(frames_buf, 32)
        clip_buffer.append(np.stack(sampled, axis=0).astype(np.uint8))

    # ---------------- NEW: final flush (fp32=True to preserve past behaviour) -------
    if clip_buffer:
        batch_np = np.stack(clip_buffer, axis=0)
        _ = model.infer_video_depth(batch_np, target_fps=fps, fp32=False)
        torch.cuda.empty_cache()
    # -------------------------------------------------------------------------------

    cap.release()

    # ------------------------------------------------------------------
    # Post-process collected activations → single feature vector per video
    # ------------------------------------------------------------------
    features: Dict[str, torch.Tensor] = {}
    for k, tensor_list in store.items():
        concat = torch.cat(tensor_list, dim=0)  # (total_T, N_tokens, 1024)
        features[k] = concat  # (1024,)

    for h in handles:
        h.remove()
    store.clear()
    gc.collect()

    # --------------------------------------------------------------
    # Persist to disk (mirrors SlowFast directory layout)
    # --------------------------------------------------------------
    for k, v in features.items():
        save_path = Path(output_folder, k, relative_path)
        save_path.mkdir(parents=True, exist_ok=True)
        print(v.shape, save_path / f"{Path(video_file).stem}.pt")
        torch.save(v, save_path / f"{Path(video_file).stem}.pt")

# ---------------------------------------------------------------------------
# Folder‑level driver
# ---------------------------------------------------------------------------

def process_video_folder(
    input_folder: str,
    output_folder: str,
    model: VideoDepthAnything,
    tr: float,
):
    video_files = []
    for root, _, files in os.walk(input_folder):
        # Skip hidden dirs (e.g. .ipynb_checkpoints)
        if any(part.startswith(".") for part in Path(root).parts):
            continue
        for file in files:
            if file.endswith(".mkv"):
                video_files.append((root, file))

    random.shuffle(video_files)

    for root, file in tqdm(video_files, desc="Processing videos"):
        relative_path = os.path.relpath(root, input_folder)
        out_file = Path(output_folder, "proj_out", relative_path, f"{Path(file).stem}.pt")
        if not out_file.is_file():
            extract_video_features(
                video_path=os.path.join(root, file),
                video_file=file,
                output_folder=output_folder,
                relative_path=relative_path,
                model=model,
                tr=tr,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract VideoDepthAnything motion features (proj_out) from videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies",
        help="Root folder containing .mkv videos",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/u/shdixit/MultimodalBrainModel/Features/Visual/VideoDepthAnything",
        help="Where to save *.pt feature files",
    )
    parser.add_argument("--tr", type=float, default=1.49, help="Window length in seconds")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/video_depth_anything_vitl.pth",
        help="Path to VideoDepthAnything checkpoint",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vitl", "vits"],
        default="vitl",
        help="Model encoder size",
    )

    args = parser.parse_args()

    # --------------------------- Model ---------------------------
    config = {"encoder": args.encoder, "features": 256, "out_channels": [256, 512, 1024, 1024]}
    model = VideoDepthAnything(**config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    model = torch.compile(model, mode="reduce-overhead")

    # Create base output directory
    Path(args.output_folder, "proj_out").mkdir(parents=True, exist_ok=True)

    process_video_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model=model,
        tr=args.tr,
    )
