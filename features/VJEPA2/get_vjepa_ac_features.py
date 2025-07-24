from collections import defaultdict
import contextlib
import time
import subprocess
import sys
import cv2
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
from transformers import AutoVideoProcessor, AutoModel

# do before: git clone git@github.com:facebookresearch/vjepa2.git
from src.models.attentive_pooler import AttentiveClassifier

import warnings
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append('vjepa2')


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

# Suppress UserWarnings and FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress the 'root' logger (Qwen's audio warning)
logging.getLogger().setLevel(logging.ERROR)
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
    out = os.path.join("clips", f"{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}_{random_number}_chunk_{t0:.2f}_{t1:.2f}.mp4")
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",      # overwrite silently
        "-ss", str(t0), "-to", str(t1), "-i", src, # seek
        "-c", "copy",                              # NO re-encoding
        out,
    ]
    subprocess.run(cmd, check=True)

    for _ in range(10):

        if os.path.isfile(out) and os.path.getsize(out) > 0:
            cap = cv2.VideoCapture(out)
            ret, _ = cap.read()
            cap.release()
            if ret:
                break
        else:
            print("FAIL")
        time.sleep(0.2)  # wait 200 ms

    return out


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
    if tensor.dim() < 3:
        return tensor.squeeze()

    tensor = tensor.reshape(-1, 16, 16, tensor.shape[-1]) # T H W D
    T, H, W, D = tensor.shape
    
    features_pooled = F.adaptive_avg_pool2d(
        tensor.permute(0, 3, 1, 2),  # [T, D, H, W]
        (3, 3)
    ).permute(0, 2, 3, 1) # T H W D

    T = tensor.size(0)
    tok0 = min(T - 1, int(round(rel_start / clip_seconds * T)))
    tok1 = min(T,     max(tok0 + 1,
                          int(round(rel_end / clip_seconds * T))))
    tensor = features_pooled[tok0:tok1].mean(0).flatten()
    return tensor     # shape (tok1-tok0, D)


hf_model_name = (
    "facebook/vjepa2-vitl-fpc64-256"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384 # Updated to giant model
)
hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)

hf_model = AutoModel.from_pretrained(hf_model_name)
hf_model.cuda().eval()

# wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt
classifier_model_path = "ssv2-vitl-16x2x3.pt"
classifier = (
    AttentiveClassifier(embed_dim=1024, num_heads=16, depth=4, num_classes=174).cuda().eval()
)
load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

mapping = {
    "v-jepa2-vitl-enc-10layer-fc2": hf_model.encoder.layer[10].mlp.fc2,
    "v-jepa2-vitl-enc-18layer-norm2": hf_model.encoder.layer[18].norm2,
    "v-jepa2-vitl-enc-20layer-fc2": hf_model.encoder.layer[20].mlp.fc2,
    "v-jepa2-vitl-enc-last-ln": hf_model.encoder.layernorm,

    "v-jepa2-vitl-predictor-5-layer-norm": hf_model.predictor.layer[5].norm1,
    "v-jepa2-vitl-pred-fc1": hf_model.predictor.layer[11].mlp.fc1,

    "v-jepa2-vitl-final-features": hf_model.predictor.proj,

    "v-jepa2-ac-classifier-block0-fc2": classifier.pooler.blocks[0].mlp.fc2,
    "v-jepa2-ac-classifier-block-last-fc2": classifier.pooler.blocks[-1].mlp.fc2,
    "v-jepa2-ac-classifier-out": classifier.linear
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
    num_chunks,
    chunk_id
):
    video_files = []
    for root, _, files in os.walk(input_folder):
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for file in files:
            if file.endswith(".mkv"):
                video_files.append((root, file))


    random.seed(0)
    random.shuffle(video_files)
    chunks = np.array_split(video_files, num_chunks)

    video_files = chunks[chunk_id]

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


def get_video_from_path(video_path):
    """
    Load video from path and return as numpy array
    Returns video in T x H x W x C format
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")

    return np.stack(frames, axis=0)  

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
        video = get_video_from_path(clip_path)  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W

        with torch.inference_mode():
            x_hf = processor(video, return_tensors="pt")["pixel_values_videos"].to(device)

            patch_feats = hf_model.get_vision_features(x_hf)
            _ = classifier(patch_feats)


            for k, tensors in store.items():
                slice_of_interest = crop_by_time(tensors, i_start, i_end, c_end - c_start)
                features[k].append(slice_of_interest)

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
        default="/path/to/input/stimuli",
        help="Path to the input folder containing videos",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/path/to/output/features",
        help="Path to the output folder where features will be stored",
    )
    parser.add_argument("--tr", type=float, default=1.49, help="TR")
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=8,
        help="Total duration of each chunk in seconds",
    )
    parser.add_argument(
        "--seconds_before_chunk",
        type=int,
        default=6,
        help="Length of video included before the chunk's start time",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="Length of video included before the chunk's start time",
    )

    args = parser.parse_args()

    output_folder = f"{args.output_folder}_tr{args.tr}_len{args.chunk_length}_before{args.seconds_before_chunk}"
    process_video_folder(
        args.input_folder,
        output_folder,
        hf_model,
        hf_transform,
        mapping,
        args.tr,
        args.chunk_length,
        args.seconds_before_chunk,
        args.num_chunks,
        args.chunk_id
    )
