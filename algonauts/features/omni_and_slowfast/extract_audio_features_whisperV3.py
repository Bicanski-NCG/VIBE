import torchaudio
import torch
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
from moviepy import VideoFileClip
from transformers import AutoProcessor, WhisperModel

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
TIERS = ["low", "mid", "high"]

# -----------------------------------------------------------------------------
# AUDIO HELPERS
# -----------------------------------------------------------------------------

def extract_audio_from_video(src: str, dst: str) -> None:
    """Dump 16‑kHz WAV from the input .mkv/.mp4/etc."""
    try:
        VideoFileClip(src).audio.write_audiofile(dst, fps=16000)
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed for {src}: {e}")


def wav_duration(path: str) -> float:
    """Return duration in seconds for a WAV file (no context‑manager needed)."""
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate

# -----------------------------------------------------------------------------
# CHUNK SCHEDULE
# -----------------------------------------------------------------------------

def make_windows(total: float, window: float, tr: float):
    """Return [(start, end)] where *end* steps forward by tr seconds each time.

    The window covers the *last* ≤window seconds before *end* (never before t=0).
    """
    ends = []
    e = tr
    while e < total:
        ends.append(e)
        e += tr
    ends.append(total)  # ensure we cover the tail

    return [(max(0.0, e - window), e) for e in ends]

# -----------------------------------------------------------------------------
# FEATURE AGGREGATION
# -----------------------------------------------------------------------------

def _roi_mean(h, roi_start):
    """Mean over time frames starting from roi_start."""
    return h[:, roi_start:, :].mean(dim=1).squeeze()


def _tier_feature(states, idxs, roi_start):
    return torch.stack([_roi_mean(states[i], roi_start) for i in idxs]).mean(dim=0)

# -----------------------------------------------------------------------------
# MAIN EXTRACTION PER VIDEO
# -----------------------------------------------------------------------------

def extract_whisper(video_path: str,
                    output_base: str,
                    rel: str,
                    processor,
                    model,
                    tr: float,
                    window: float,
                    stereo: bool = False):
    tmp = video_path.rsplit('.', 1)[0] + '_tmp.wav'
    # if os.path.exists(tmp):
    #     return
    extract_audio_from_video(video_path, tmp)

    dur = wav_duration(tmp)
    windows = make_windows(dur, window, tr)  # list of (start, end)

    wav, sr = torchaudio.load(tmp)
    target_sr = processor.feature_extractor.sampling_rate
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    if not stereo or wav.shape[0] == 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono

    tiers_out = {k: [] for k in TIERS}
    n_layers = model.config.num_hidden_layers  # 32 for large‑v3
    third = n_layers // 3
    layer_sel = {
        'low': range(0, third),
        'mid': range(third, 2 * third),
        'high': range(2 * third, n_layers)
    }

    for w_start, w_end in tqdm(windows, desc=os.path.basename(video_path)):
        s = int(w_start * target_sr)
        e = int(w_end * target_sr)
        chunk = wav[..., s:e]  # (C, S)
        chunk_len = w_end - w_start

        chans_feat = {k: [] for k in TIERS}
        for ch in range(chunk.shape[0]):
            inp = processor(chunk[ch], sampling_rate=target_sr, return_tensors='pt').input_features.to(device)
            with torch.no_grad():
                enc_out = model.encoder(input_features=inp, output_hidden_states=True, return_dict=True)
            enc = enc_out.hidden_states[1:]  # drop conv layer

            seq_len = enc[0].shape[1]
            frames_per_sec = seq_len / chunk_len
            roi_frames = int(min(tr, chunk_len) * frames_per_sec + 0.5)
            roi_start = max(0, seq_len - roi_frames)

            for tier in TIERS:
                feat = _tier_feature(enc, layer_sel[tier], roi_start)
                chans_feat[tier].append(feat)

        # Aggregate channels
        for tier in TIERS:
            vec = torch.cat(chans_feat[tier], dim=-1) if stereo and len(chans_feat[tier]) == 2 else chans_feat[tier][0]
            tiers_out[tier].append(vec.cpu().numpy())

    # ---------- save ----------
    base_npy = os.path.basename(video_path).replace('.mkv', '.npy')
    for tier, mats in tiers_out.items():
        tier_dir = os.path.join(output_base, tier, rel)
        os.makedirs(tier_dir, exist_ok=True)
        np.save(os.path.join(tier_dir, base_npy), np.stack(mats))

    
    os.remove(tmp)

# -----------------------------------------------------------------------------
# DATASET DRIVER
# -----------------------------------------------------------------------------

def process_dataset(inp_root: str,
                    out_root: str,
                    processor,
                    model,
                    tr: float,
                    window: float,
                    stereo: bool = False):
    ood_skip_files = ["task-passepartoutS02E08_video.mkv", "task-passepartoutS02E07_video.mkv", "task-chaplin_video.mkv", "task-pulpfiction_video.mkv", "task-mononoke_video.mkv",  "task-planetearth_video.mkv"]
    vids = [os.path.join(r, f)
            for r, _, fs in os.walk(inp_root)
            if not any(p.startswith('.') for p in r.split(os.sep))
            for f in fs if f.endswith('.mkv') and f not in ood_skip_files]

    random.shuffle(vids)

    for vid in tqdm(vids, desc='Videos'):
        rel = os.path.relpath(os.path.dirname(vid), inp_root)
        base = os.path.basename(vid).replace('.mkv', '.npy')

        # skip if all tiers done
        if all(os.path.exists(os.path.join(out_root, t, rel, base)) for t in TIERS):
            continue

        extract_whisper(vid,
                        out_root,
                        rel,
                        processor,
                        model,
                        tr,
                        window,
                        stereo)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser('Whisper tiered features (moving window)')
    p.add_argument('--input_folder', default="/u/shdixit/MultimodalBrainModel/data/algonauts_2025.competitors/stimuli/movies/")
    p.add_argument('--output_folder', default="/u/shdixit/MultimodalBrainModel/Features/Audio/WhisperV3-Large")
    p.add_argument('--chunk_duration', type=float, default=1.49, help='tr (s) stride & ROI length')
    p.add_argument('--chunk_length', type=float, default=30.0, help='max context window (s)')
    p.add_argument('--stereo', action='store_true')
    args = p.parse_args()

    proc = AutoProcessor.from_pretrained('openai/whisper-large-v3')
    mdl = WhisperModel.from_pretrained('openai/whisper-large-v3').to(device)
    mdl.eval()

    output_folder = f"{args.output_folder}-ctx{int(args.chunk_length)}-stereo{args.stereo}"
    process_dataset(args.input_folder,
                    output_folder,
                    proc,
                    mdl,
                    args.chunk_duration,
                    args.chunk_length,
                    args.stereo)
