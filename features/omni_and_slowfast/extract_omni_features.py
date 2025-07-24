from collections import defaultdict
import contextlib
import subprocess
import sys
import cv2
import torch
import os
import argparse
from tqdm import tqdm
import random
from itertools import islice 

device = "auto" if torch.cuda.is_available() else "cpu"

from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info  # comes with the official repo

import warnings
import logging

# Suppress UserWarnings and FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress the 'root' logger (Qwen's audio warning)
logging.getLogger().setLevel(logging.ERROR)

# Also suppress torchvision deprecation logger
logging.getLogger("torchvision").setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_movie_info(movie_path):
    """
    Extracts the frame rate (FPS) and total duration of a movie.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """
    cap = cv2.VideoCapture(movie_path)
    fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, frame_count / fps


# ------------------------------------------------------------------
# 1) split the movie (your original function, unchanged)
# ------------------------------------------------------------------
def split_movie_into_chunks(movie_path, tr, chunk_length, seconds_before_chunk):
    assert seconds_before_chunk < chunk_length, \
        "seconds_before_chunk must be shorter than chunk_length"
    assert seconds_before_chunk + tr <= chunk_length, \
        "chunk must be long enough to hold the whole TR window"

    _, video_duration = get_movie_info(movie_path)  # <-- your helper
    chunks, chunk_of_interests = [], []
    start_time = 0.0
    while start_time < video_duration:
        chunk_start = max(0, start_time - seconds_before_chunk)
        chunk_end = min(chunk_start + chunk_length, video_duration)

        rel_start = start_time - chunk_start
        rel_end = min(rel_start + tr, chunk_end - chunk_start)

        chunks.append((chunk_start, chunk_end))
        chunk_of_interests.append((rel_start, rel_end))
        start_time += tr
    return chunks, chunk_of_interests


# ------------------------------------------------------------------
# 2) trim a sub-clip on disk (makes Qwen ingestion easy)
# ------------------------------------------------------------------
def cut_clip(src, t0, t1):
    os.makedirs("clips", exist_ok=True)
    random_number = random.randint(1, 1_000_000_000)
    out = os.path.join("clips", f"{random_number}_chunk_{t0:.2f}_{t1:.2f}.mp4")
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",      # overwrite silently
        "-ss", str(t0), "-to", str(t1), "-i", src, # seek
        "-c", "copy",                              # NO re-encoding
        out,
    ]
    subprocess.run(cmd, check=True)
    return out

def safe_mean_std(slice_: torch.Tensor):
    """
    Return mean and std that are *never* NaN.

    • if the slice has ≥2 tokens → behave exactly like before  
    • if the slice has 1 token   → std = 0 (mean unchanged)  
    • if the slice is empty      → use the very last token of the clip
    """
    if slice_.numel() == 0:                 # empty → steal the last token
        slice_ = slice_[-1:].detach()       # shape (1, D)

    mean = slice_.mean(dim=0)

    if slice_.shape[0] > 1:                 # ≥ 2 tokens → old behaviour
        std  = slice_.std(dim=0)            # unbiased=True (default)
    else:                                   # single token → std = 0
        std  = torch.zeros_like(mean)

    return mean, std

def grouper(iterable, n):
    "chunk iterable into fixed-size lists (last one may be shorter)"
    it = iter(iterable)
    while (chunk := list(islice(it, n))):
        yield chunk

# ------------------------------------------------------------------
# 3) register forward hooks to capture hidden-states
# ------------------------------------------------------------------
def register_hooks(mapping):
    store, handles = {}, []

    def save(name):
        def _hook(_, __, out):
            store[name] = out.detach()

        return _hook

    # map layer names to actual modules -----------------------------
    for k in mapping.keys():
        handles.append(mapping[k].register_forward_hook(save(k)))
    return store, handles

def crop_by_time(tensor, rel_start, rel_end, clip_seconds):
    """
    tensor      – hidden state, shape (T, D) or (1, T, D)
    rel_start   – seconds from the beginning of this clip
    rel_end     – seconds from the beginning of this clip
    clip_seconds– total duration of this clip in seconds
    returns     – the slice of tokens that fall inside [rel_start, rel_end)
    """
    if tensor.dim() == 3:          # (1, T, D) → (T, D)
        tensor = tensor[0]

    T = tensor.size(0)
    tok0 = min(T - 1, int(round(rel_start / clip_seconds * T)))
    tok1 = min(T,     max(tok0 + 1,
                          int(round(rel_end / clip_seconds * T))))
    return tensor[tok0:tok1]       # shape (tok1-tok0, D)


model_name = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
).thinker.eval()

mapping = {
    # video
    "conv3d_features": model.visual.patch_embed,
    "vis_block5": model.visual.blocks[4].norm1,
    "vis_block8": model.visual.blocks[7].norm2,
    "vis_block12": model.visual.blocks[11].norm2,
    "vis_merged": model.visual.merger,
    # audio
    "audio_ln_post": model.audio_tower.ln_post,
    "aud_last": model.audio_tower.layers[-1].final_layer_norm,
    # thinker
    "thinker_12": model.model.layers[11].post_attention_layernorm,
    "thinker_24": model.model.layers[23].post_attention_layernorm,
    "thinker_36": model.model.layers[35].post_attention_layernorm,
}
store, handles = register_hooks(mapping)     # only once!


def process_video_folder(
    input_folder,
    output_folder,
    model,
    processor,
    mapping,
    tr,
    chunk_length,
    seconds_before_chunk,
):
    video_files = []
    for root, _, files in os.walk(input_folder):
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for file in files:
            if file.endswith(".mkv"):
                video_files.append((root, file))

    random.shuffle(video_files)

    for root, file in tqdm(video_files, desc="Processing videos"):
        relative_path = os.path.relpath(root, input_folder)
        video_path = os.path.join(root, file)
        for k in mapping.keys():
            os.makedirs(os.path.join(output_folder, k, relative_path), exist_ok=True)
            output_path = os.path.join(output_folder, k, relative_path, file.split(".")[0] + ".pt")
        
        if not os.path.isfile(output_path):
            extract_video_features(
                video_path,
                file,
                output_folder,
                relative_path,
                model,
                processor,
                mapping,
                tr,
                chunk_length,
                seconds_before_chunk,
            )


@torch.no_grad()
def extract_video_features(
    video_path,
    video_file,
    output_folder,
    relative_path,
    model,
    processor,
    mapping,
    tr,
    chunk_length,
    seconds_before_chunk,
):
    chunks, chunk_of_interests = split_movie_into_chunks(
        video_path, tr, chunk_length, seconds_before_chunk
    )

    features = defaultdict(list)

    for (c_start, c_end), (i_start, i_end) in tqdm(zip(chunks, chunk_of_interests), desc=f"Processing {os.path.basename(video_path)}"):                    # parallel meta info
        # 4.1 cut the clip -----------------------------------------
        clip_path = cut_clip(video_path, c_start, c_end)

        # 4.2 grab hidden-states -----------------------------------
        convo = [{"role":"user","content":[{"type":"video","video":clip_path}]}]
        text  = processor.apply_chat_template(convo, tokenize=False,
                                            add_generation_prompt=False)
        aud, imgs, vids = process_mm_info(convo, use_audio_in_video=True)
        inputs = processor(text=text, audio=aud, images=imgs, videos=vids,
                        return_tensors="pt", padding=True, do_rescale=False,
                        use_audio_in_video=True).to(model.device, model.dtype)
        _ = model(**inputs, use_audio_in_video=True)
        for k, tensors in store.items():
            slice_of_interest = crop_by_time(tensors, i_start, i_end, c_end - c_start)
            mean_of_interest, std_of_interest = safe_mean_std(slice_of_interest)
            tensor_concat = torch.cat((mean_of_interest, std_of_interest), dim=0)
            assert not torch.isnan(tensor_concat).any(), f"Got NaN at chunk {c_start}-{c_end}"
            features[k].append(tensor_concat)
        os.remove(clip_path)
        store.clear() 
    for k, v in features.items():
        features[k] = torch.stack(v, dim=0)

    for k, v in features.items():
        torch.save(v, os.path.join(output_folder, k, relative_path, video_file.split(".")[0] + ".pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and extract features")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies",
        help="Path to the input folder containing videos",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/u/shdixit/MultimodalBrainModel/Features/Omni/Qwen2.5_3B/features",
        help="Path to the output folder where features will be stored",
    )
    parser.add_argument("--tr", type=float, default=1.49, help="TR")
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=30,
        help="Total duration of each chunk in seconds",
    )
    parser.add_argument(
        "--seconds_before_chunk",
        type=int,
        default=28.5,
        help="Length of video included before the chunk's start time",
    )

    args = parser.parse_args()

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16  # Set False for 8-bit
    # )
    output_folder = f"{args.output_folder}_tr{args.tr}_len{args.chunk_length}_before{args.seconds_before_chunk}"
    process_video_folder(
        args.input_folder,
        output_folder,
        model,
        processor,
        mapping,
        args.tr,
        args.chunk_length,
        args.seconds_before_chunk,
    )
