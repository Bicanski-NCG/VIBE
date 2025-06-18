import argparse
import os
import random
import tempfile
from typing import Tuple

import numpy as np
import torch
import torchaudio
from moviepy import VideoFileClip
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ------------------------------------------------------------------------ #
TR_SECONDS = 1.49          # adjust if your fMRI TR differs
FEATURE_STRIDE = 0.02      # seconds per Wav2Vec2 time-step (≈ 20 ms)
MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
# ------------------------------------------------------------------------ #

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ helpers ---------------------------------- #
def extract_audio(video_path: str, sr: int) -> Tuple[torch.Tensor, int]:
    """
    Returns mono waveform @ sr Hz from an .mkv file.
    Uses MoviePy because it's easier to install than ffmpeg-python.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_name = tmp_wav.name

    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(
            tmp_name, fps=sr
        )
        waveform, _ = torchaudio.load(tmp_name)    # (channels, samples)
        waveform = waveform.mean(0, keepdim=True)  # mono
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return waveform, sr


def pool_per_tr(
    hidden: torch.Tensor, tr_steps: int
) -> np.ndarray:
    """
    hidden : Tensor (seq_len, hidden_dim)
    Returns ndarray (n_tr, 2*hidden_dim) where cols = [mean | std].
    """
    seq_len, dim = hidden.shape
    n_tr = int(np.ceil(seq_len / tr_steps))
    out = np.zeros((n_tr, dim * 2), dtype=np.float32)

    for tr in range(n_tr):
        start = tr * tr_steps
        end = min(start + tr_steps, seq_len)
        slice_ = hidden[start:end]

        if slice_.numel() == 0:  # should not happen
            continue

        mean = slice_.mean(0)
        std = slice_.std(0, unbiased=False)
        out[tr, :dim] = mean.cpu().numpy()
        out[tr, dim:] = std.cpu().numpy()

    return out


# --------------------------- core routine -------------------------------- #
@torch.no_grad()
def extract_features_for_video(
    video_path: str,
    output_path: str,
    processor,
    model,
):
    # 1. load audio at model sampling-rate (16 kHz)
    target_sr = processor.feature_extractor.sampling_rate
    waveform, _ = extract_audio(video_path, target_sr)  # (1, samples)

    # 2. Wav2Vec2 forward
    inputs = processor(
        waveform.squeeze(), sampling_rate=target_sr,
        return_tensors="pt", padding=False
    ).to(device)
    hidden = model(**inputs).last_hidden_state.squeeze(0)  # (T, 1024)

    # 3. pool every TR
    tr_steps = int(round(TR_SECONDS / FEATURE_STRIDE))
    features = pool_per_tr(hidden, tr_steps)              # (n_tr, 2048)

    np.save(output_path, features)


# --------------------------- folder driver ------------------------------- #
def process_video_folder(
    input_folder: str,
    output_folder: str,
    processor,
    model,
):
    videos = []
    for root, _, files in os.walk(input_folder):
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue
        for f in files:
            if f.endswith(".mkv"):
                videos.append((root, f))

    random.shuffle(videos)

    for root, file in tqdm(videos, desc="Audio feature extraction"):
        rel_dir = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        video_path = os.path.join(root, file)
        out_path = os.path.join(out_dir, file.replace(".mkv", ".npy"))

        if not os.path.isfile(out_path):
            extract_features_for_video(video_path, out_path, processor, model)


# ----------------------------- CLI --------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Extract Wav2Vec2 features per TR from video soundtracks"
    )
    p.add_argument("--input_folder", default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies/",
                        help="Root folder containing .tsv transcripts")
    p.add_argument("--output_folder", default="/u/shdixit/MultimodalBrainModel/Features/Audio/Wav2Vec2Full",
                        help="Root folder where .npy features will be stored")
    p.add_argument("--tr", type=float, default=1.49,
                        help="Repetition time in seconds (default: 1.49)")
    args = p.parse_args()

    global TR_SECONDS
    TR_SECONDS = args.tr          # allow override from CLI

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = (
        Wav2Vec2Model
        .from_pretrained(MODEL_NAME)
        .to(device)
        .eval()
    )

    out_root = f"{args.output_folder}_tr{TR_SECONDS}"
    process_video_folder(args.input_folder, out_root, processor, model)


if __name__ == "__main__":
    main()
