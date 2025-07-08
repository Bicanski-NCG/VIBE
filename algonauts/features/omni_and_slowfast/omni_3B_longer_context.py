from collections import defaultdict
import contextlib
import glob
import subprocess
import sys
import cv2
import torch
import os
import argparse
from tqdm import tqdm
import random
import soundfile as sf
from itertools import islice
import warnings
import logging

# -----------------------------------------------------------------------------
# Environment & logging --------------------------------------------------------
# -----------------------------------------------------------------------------

device = "auto" if torch.cuda.is_available() else "cpu"

# Suppress warnings that spam the console
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Transformers & Qwen‑Omni ------------------------------------------------------
# -----------------------------------------------------------------------------

from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info  # util supplied by official repo

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stdout():
    """Silence all stdout inside the context (for ffmpeg, etc.)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_movie_info(movie_path: str):
    """Return (fps, duration[s])."""
    cap = cv2.VideoCapture(movie_path)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, frames / fps


def cut_clip(src: str,
             t0: float,
             t1: float,
             target_width: int = 720,
             crf: int = 18,
             preset: str = "ultrafast") -> str:
    """
    Losslessly* cut [t0, t1) from *src* into ./clips/,
    resize so the width is *target_width* while keeping aspect ratio,
    and return the resulting path.

    *Video is re-encoded at CRF quality; audio is copied bit-exactly.
    """
    os.makedirs("clips", exist_ok=True)
    out = os.path.join(
        "clips",
        f"{random.randint(1, 1_000_000_000)}_{t0:.2f}_{t1:.2f}.mp4"
    )

    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",           # overwrite silently
        "-ss", str(t0), "-to", str(t1), "-i", src,      # trim first
        "-vf", f"scale={target_width}:-2",              # resize, keep AR
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),  # encode video
        "-c:a", "copy",                                 # copy audio stream
        out,
    ]
    subprocess.run(cmd, check=True)
    return out


def safe_mean_std(slice_: torch.Tensor):
    """Mean & std that never produce NaNs (handles empty/single‑token slices)."""
    if slice_.numel() == 0:  # empty → duplicate last token
        slice_ = slice_[-1:].detach()
    mean = slice_.mean(dim=0)
    std = slice_.std(dim=0) if slice_.shape[0] > 1 else torch.zeros_like(mean)
    return mean, std


def crop_by_time(tensor: torch.Tensor,
                 rel_start: float,
                 rel_end: float,
                 clip_seconds: float):
    """Return tokens whose timestamps fall in [rel_start, rel_end)."""
    if tensor.dim() == 3:  # (1, T, D) → (T, D)
        tensor = tensor[0]
    T = tensor.size(0)
    tok0 = min(T - 1, int(round(rel_start / clip_seconds * T)))
    tok1 = min(T, max(tok0 + 1, int(round(rel_end / clip_seconds * T))))
    return tensor[tok0:tok1]

# -----------------------------------------------------------------------------
# Chunk builder (120‑s window / 60‑s stride) -----------------------------------
# -----------------------------------------------------------------------------

def build_chunks_with_roi(movie_path: str,
                          tr: float,
                          window_len: float = 120.0,
                          stride: float = 60.0):
    """Generate overlapping windows + within‑window TR regions of interest."""
    _, video_duration = get_movie_info(movie_path)
    overlap = window_len - stride  # section reused from previous window

    chunks, roi_lists = [], []
    first = True
    t0 = 0.0  # absolute start of current window

    while t0 < video_duration:
        t1 = min(t0 + window_len, video_duration)
        rel_roi_start = 0.0 if first else overlap  # keep all TRs in first win
        first = False

        # Build list of (rel_start, rel_end) for every TR we need from window
        roi = []
        t = t0 + rel_roi_start  # absolute start of first TR in ROI
        while t < t1:
            rel_start = t - t0
            rel_end = min(rel_start + tr, t1 - t0)
            roi.append((rel_start, rel_end))
            t += tr

        chunks.append((t0, t1))
        roi_lists.append(roi)
        t0 += stride  # slide window forward

    return chunks, roi_lists

# -----------------------------------------------------------------------------
# Forward‑hook machinery -------------------------------------------------------
# -----------------------------------------------------------------------------

def register_hooks(mapping):
    """Attach forward hooks and return (store_dict, handles_list)."""
    store, handles = {}, []

    def _save(name):
        def hook(_, __, out):
            store[name] = out.detach()
        return hook

    for name, module in mapping.items():
        handles.append(module.register_forward_hook(_save(name)))
    return store, handles

# -----------------------------------------------------------------------------
# Load model & set up hooks ----------------------------------------------------
# -----------------------------------------------------------------------------

model_name = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
).thinker.eval()

# Where to tap hidden states – adapt freely
mapping = {
    # video tower
    "conv3d_features": model.visual.patch_embed,
    "vis_block5": model.visual.blocks[4].norm1,
    "vis_block8": model.visual.blocks[7].norm2,
    "vis_block12": model.visual.blocks[11].norm2,
    "vis_merged": model.visual.merger,
    # audio tower
    "audio_ln_post": model.audio_tower.ln_post,
    "aud_last": model.audio_tower.layers[-1].final_layer_norm,
    # thinker (text) tower
    "thinker_12": model.model.layers[11].post_attention_layernorm,
    "thinker_24": model.model.layers[23].post_attention_layernorm,
    "thinker_36": model.model.layers[35].post_attention_layernorm,
}
store, handles = register_hooks(mapping)  # persistent across calls

# -----------------------------------------------------------------------------
# Core feature‑extraction routine ---------------------------------------------
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_video_features(video_path: str,
                           video_file: str,
                           output_folder: str,
                           relative_path: str,
                           tr: float,
                           window_len: float,
                           stride: float):
    """Run the model with large windows & save mean|std for every TR."""

    chunks, roi_lists = build_chunks_with_roi(video_path, tr, window_len, stride)
    features = defaultdict(list)

    for (c_start, c_end), rois in tqdm(
        zip(chunks, roi_lists),
        total=len(chunks),
        desc=f"{os.path.basename(video_path)}"):

        # ---- 1. cut 2‑minute window ------------------------------------------------
        clip_len = c_end - c_start
        if clip_len < tr:          # no room even for one TR ⇒ skip
            continue
        clip_path = cut_clip(video_path, c_start, c_end)

        # ---- 2. run model once ------------------------------------------------------
        convo = [{"role": "user", "content": [{"type": "video", "video": clip_path}]}]
        text = processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        aud, imgs, vids = process_mm_info(convo, use_audio_in_video=True)
        inputs = processor(
            text=text, audio=aud, images=imgs, videos=vids,
            return_tensors="pt", padding=True, do_rescale=False,
            use_audio_in_video=True
        ).to(model.device, model.dtype)

        _ = model(**inputs, use_audio_in_video=True)  # fills `store`

        # ---- 3. slice once per TR of interest --------------------------------------
        for rel_start, rel_end in rois:
            for k, tensors in store.items():
                slice_ = crop_by_time(tensors, rel_start, rel_end, clip_len)
                mu, sigma = safe_mean_std(slice_)
                features[k].append(torch.cat((mu, sigma), dim=0))

        store.clear()  # IMPORTANT: avoid mixing chunks
        os.remove(clip_path)

    # ---- 4. stack & save -----------------------------------------------------------
    for k, tensors in features.items():
        out_dir = os.path.join(output_folder, k, relative_path)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(torch.stack(tensors, dim=0),
                   os.path.join(out_dir, video_file.rsplit(".", 1)[0] + ".pt"))

# -----------------------------------------------------------------------------
# Folder‑level orchestration ---------------------------------------------------
# -----------------------------------------------------------------------------

def process_video_folder(input_folder: str,
                         output_folder: str,
                         tr: float,
                         window_len: float,
                         stride: float):
    """Walk *input_folder*, process every .mkv and save features."""
    ood_skip_files = ["task-passepartoutS02E08_video.mkv", "task-passepartoutS02E07_video.mkv", "task-chaplin_video.mkv", "task-pulpfiction_video.mkv", "task-mononoke_video.mkv",  "task-planetearth_video.mkv"]
    video_files = []
    for root, _, files in os.walk(input_folder):
        # skip hidden folders (., .., .ipynb_checkpoints, etc.)
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue
        for file in files:
            if file in ood_skip_files:
                continue
            if file.endswith(".mkv"):
                video_files.append((root, file))

    random.shuffle(video_files)  # avoids GPU idling if one video is huge

    for root, file in tqdm(video_files, desc="Processing videos"):
        relative_path = os.path.relpath(root, input_folder)
        video_path = os.path.join(root, file)
        for k in mapping.keys():
            os.makedirs(os.path.join(output_folder, k, relative_path), exist_ok=True)
            output_path = os.path.join(output_folder, k, relative_path, file.split(".")[0] + ".pt")

        if not os.path.exists(output_path):
            try:
                extract_video_features(
                    video_path=video_path,
                    video_file=file,
                    output_folder=output_folder,
                    relative_path=relative_path,
                    tr=tr,
                    window_len=window_len,
                    stride=stride,
                )
            except Exception as e:
                print(f"Failed to process {video_path}: {e}")

# -----------------------------------------------------------------------------
# Entry‑point ------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract TR‑level multimodal features with windowed context")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies",
        help="Path to the input folder containing videos",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/u/shdixit/MultimodalBrainModel/Features/Omni/Qwen2.5_3B_LongerContext/features",
        help="Path to the output folder where features will be stored",
    )
    parser.add_argument("--tr", type=float, default=1.49, help="TR in seconds")
    parser.add_argument("--window_len", type=float, default=60.0,
                        help="Length of the context window (seconds)")
    parser.add_argument("--stride", type=float, default=30.0,
                        help="Stride between windows (seconds)")
    args = parser.parse_args()

    # Create a descriptive sub‑folder name (optional)
    output_folder = (
        f"{args.output_folder}_tr{args.tr}_win{args.window_len}_stride{args.stride}"
    )

    process_video_folder(
        input_folder=args.input_folder,
        output_folder=output_folder,
        tr=args.tr,
        window_len=args.window_len,
        stride=args.stride,
    )

