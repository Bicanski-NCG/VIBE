from collections import defaultdict
import os

import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import torch
import torch.nn.functional as F            # deep-learning ops (pooling, conv, etc.)

import argparse
from tqdm import tqdm
import random

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import sys
import types
import torchvision.transforms.functional as TF 
mod = types.ModuleType("torchvision.transforms.functional_tensor") 
mod.__dict__.update(TF.__dict__) 
sys.modules["torchvision.transforms.functional_tensor"] = mod
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def register_hooks(mapping):
    """
    Forward hooks that collect the requested activations.
    Instead of overwriting the tensor on every forward pass, we append
    to a list so we can run the model in several smaller mini-batches.
    """
    store   = {k: [] for k in mapping.keys()}
    handles = []

    def make_hook(name):
        def _hook(_, __, out):
            # keep the result on CPU to free GPU memory immediately
            store[name].append(out.detach().cpu())
        return _hook

    for k, m in mapping.items():
        handles.append(m.register_forward_hook(make_hook(k)))

    return store, handles

def pool_and_stats(x, h_out, w_out):
    """
    Keeps a spatial grid, pools *only* time.
    Args
    ----
    x : Tensor  (B, C, T, H, W)
    h_out, w_out : int  – target spatial grid size (after adaptive pooling).
    Returns
    -------
    Tensor (B, 2*C*h_out*w_out) :  [µ_time  ||  σ_time] per grid cell.
    """
    B, C, T, _, _ = x.shape

    # 1) Spatial pooling to the requested grid
    x = F.adaptive_avg_pool3d(x, output_size=(T, h_out, w_out))   # (B,C,T,h,w)

    # 2) Time-only moments (preserve grid)
    mu_t  = x.mean(dim=2)            # (B, C, h, w)
    sigma_t = x.std(dim=2)           # (B, C, h, w)

    # 3) Concatenate μ and σ on channel axis, then flatten spatial grid
    feats = torch.cat([mu_t, sigma_t], dim=1)        # (B, 2C, h, w)
    return feats.view(B, -1)

def global_stats(x, h_out=1, w_out=1):
    """
    Optional h_out/w_out let you squeeze spatially to a fixed size
    before taking μ/σ over time.
    """
    B, C, T, _, _ = x.shape
    if (h_out, w_out) != (x.shape[-2], x.shape[-1]):      # only if needed
        x = F.adaptive_avg_pool3d(x, output_size=(T, h_out, w_out))
    mu_t  = x.mean(dim=2)
    return torch.cat([mu_t], dim=1).view(B, -1)

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
model = model.eval()
model = model.to(device)

mapping = {
    # 1) Very-early edges & color (Slow stem, after ReLU)
    "slow_stem_act":  model.blocks[0].multipathway_blocks[0].activation,

    # 2) High-temporal motion energy (Fast stem, after ReLU)
    "fast_stem_act":  model.blocks[0].multipathway_blocks[1].activation,

    # 3) Mid-level form/action (Slow Res3 – last block’s ReLU)
    "slow_res3_act":  model.blocks[2].multipathway_blocks[0].res_blocks[-1].activation,

    # 4) High-level object/scene semantics (Slow Res5 – last block’s ReLU)
    "slow_res5_act":  model.blocks[4].multipathway_blocks[0].res_blocks[-1].activation,

    # 5) Global scene gist (Slow+Fast pooled concat just before the head)
    "pool_concat":    model.blocks[5],   # PoolConcatPathway forward output

    # 6) OPTIONAL – extra motion-specific layer (Fast Res3 – last block’s ReLU)
    "fast_res3_act":  model.blocks[2].multipathway_blocks[1].res_blocks[-1].activation,
}

post_transform = {
    # 1) Slow stem  -> keep fine retinotopy, 8×8 grid
    "slow_stem_act": lambda x: pool_and_stats(x, 8, 8),

    # 2) Fast stem  -> same 8×8 grid for symmetry
    "fast_stem_act": lambda x: pool_and_stats(x, 8, 8),

    # 3) Slow Res3  -> coarser 4×4 grid
    "slow_res3_act": lambda x: pool_and_stats(x, 2, 2),

    # 4) Fast Res3  -> coarser 4×4 grid
    "fast_res3_act": lambda x: pool_and_stats(x, 4, 4),

    # 5) Slow Res5  -> global spatial stats (7×7 already small)
    "slow_res5_act": lambda x: pool_and_stats(x, 1, 1),

    # 6) Pooled Slow+Fast concat -> global stats (shape (B,2304,1,2,2))
    "pool_concat":  lambda x: torch.flatten(x, start_dim=1) if center_crop else global_stats(x, 2, 2),
}
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32


def process_video_folder(
    input_folder,
    output_folder,
    model,
    tr,
    center_crop=False
):
    video_files = []
    ood_skip_files = ["task-passepartoutS02E08_video.mkv", "task-passepartoutS02E07_video.mkv", "task-chaplin_video.mkv", "task-pulpfiction_video.mkv", "task-mononoke_video.mkv",  "task-planetearth_video.mkv"]
    for root, _, files in os.walk(input_folder):
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for file in files:
            if file in ood_skip_files:
                continue
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
                tr,
                center_crop=center_crop
            )

@torch.no_grad()
def extract_video_features(
    video_path,
    video_file,
    output_folder,
    relative_path,
    model,
    tr,
    center_crop=False,
):
    features = defaultdict(list)
    _, video_duration = get_movie_info(video_path)

    transform_list = [
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size),
        PackPathway(),
    ]

    if not center_crop:
        transform_list.pop(-2)

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(transform_list),
    )

    video = EncodedVideo.from_path(video_path, decode_audio=False, decoder="pyav")

    inputs_slow, inputs_fast = [], []

    # --------------------------------------------------------------
    # 1) Decode clips and keep them on CPU (lists of tensors)
    # --------------------------------------------------------------
    for clip_idx, t0 in tqdm(
        enumerate(np.arange(0, video_duration, tr)),
        desc=f"Processing {os.path.basename(video_path)}",
    ):
        try:
            clip = video.get_clip(t0, min(video_duration, t0 + tr))
            clip = transform(clip)["video"]
        except:
            print(f"Error during Processing {os.path.basename(video_path)} {t0} {video_duration}")
            break

        inputs_slow.append(clip[0][None, ...])  # [1, C, T_slow, H, W]
        inputs_fast.append(clip[1][None, ...])  # [1, C, T_fast, H, W]

    # --------------------------------------------------------------
    # 2) Run the network in mini-batches
    # --------------------------------------------------------------
    store, handles = register_hooks(mapping)
    del video
    batch_size = 64
    for start in range(0, len(inputs_slow), batch_size):
        end = start + batch_size
        batch_slow = torch.cat(inputs_slow[start:end], dim=0).to(device, non_blocking=True)
        batch_fast = torch.cat(inputs_fast[start:end], dim=0).to(device, non_blocking=True)

        _ = model([batch_slow, batch_fast])     # forward pass

        # free GPU ASAP
        del batch_slow, batch_fast
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # 3) Post-process: concatenate every layer’s activations and
    #    collapse them to a single video-level feature vector
    # --------------------------------------------------------------
    features = {}
    for k, tensor_list in store.items():
        concat = torch.cat(tensor_list, dim=0)        # shape: (TotalClips, …)
        features[k] = post_transform[k](concat)        # flattened 1-D vector

    # remove hooks
    for h in handles:
        h.remove()
    store.clear()

    # --------------------------------------------------------------
    # 4) Persist to disk
    # --------------------------------------------------------------
    for k, v in features.items():
        torch.save(
            v,
            os.path.join(
                output_folder,
                k,
                relative_path,
                video_file.split(".")[0] + ".pt",
            ),
        )


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
        default="/u/shdixit/MultimodalBrainModel/Features/Visual/SlowFast_R101",
        help="Path to the output folder where features will be stored",
    )
    parser.add_argument("--tr", type=float, default=1.49, help="TR")
    parser.add_argument("--center_crop", action="store_true", help="Center crop video")
    


    args = parser.parse_args()

    global center_crop
    center_crop = args.center_crop

    output_folder = f"{args.output_folder}_tr{args.tr}"

    output_folder = f"{output_folder}_no_center_crop" if not args.center_crop else output_folder
    process_video_folder(
        args.input_folder,
        output_folder,
        model,
        args.tr,
        center_crop=args.center_crop
    )