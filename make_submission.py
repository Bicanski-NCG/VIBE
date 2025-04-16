import os
import glob
import numpy as np
import torch
from fastprogress import progress_bar
from model import FMRIModel
import zipfile


model = FMRIModel({'audio': 2048, 'video': 512, 'text': 2048}, 1000, subject_count=4, max_len=600)

model.load_state_dict(torch.load('final_model.pt'))
model.eval();

def pad_to_length(x, target_len):
    if x.shape[0] >= target_len:
        return x[:target_len]
    repeat_count = target_len - x.shape[0]
    pad = x[-1:].repeat(repeat_count, 1)
    return torch.cat([x, pad], dim=0)

def load_features_for_episode(episode_id, audio_root, video_root, text_root):
    def find_feature_file(root, name):
        matches = glob.glob(os.path.join(root, '**', f"*{name}.npy"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"{name}.npy not found in {root}")
        return matches[0]

    audio = torch.tensor(np.load(find_feature_file(audio_root, episode_id)), dtype=torch.float32)
    video = torch.tensor(np.load(find_feature_file(video_root, episode_id)), dtype=torch.float32)
    text = torch.tensor(np.load(find_feature_file(text_root, episode_id)), dtype=torch.float32)
    
    return audio.squeeze(), video.squeeze(), text.squeeze()  # ensure video is 2D

def predict_fmri_for_test_set(model, audio_root, video_root, text_root, sample_counts_root, device='cuda'):
    model.eval()
    model.to(device)

    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
    subject_name_id_dict = {
            "sub-01": 0,
            "sub-02": 1,
            "sub-03": 2,
            "sub-05": 3
        }
    episodes = [f's07e{str(i).zfill(2)}{suffix}' for i in range(1, 24) for suffix in 'abcd' if not (i == 23 and suffix > 'd')]
    output_dict = {}

    for subj in subjects:
        output_dict[subj] = {}
        subj_id = subject_name_id_dict[subj]  # sub-01 → 0, etc.
        sample_dict_path = os.path.join(sample_counts_root, subj, 'target_sample_number', f'{subj}_friends-s7_fmri_samples.npy')
        sample_counts = np.load(sample_dict_path, allow_pickle=True).item()  # a dict: {episode: count}

        for episode in progress_bar(sample_counts.keys()):
            n_samples = sample_counts[episode]
            
            try:
                audio, video, text = load_features_for_episode(episode, audio_root, video_root, text_root)
            except FileNotFoundError as e:
                print(f"Skipping {episode}: {e}")
                continue

            audio = pad_to_length(audio, n_samples)
            video = pad_to_length(video, n_samples)
            text = pad_to_length(text, n_samples)

            # Trim to expected length
            audio = audio[:n_samples].unsqueeze(0).to(device)      # (1, T, D)
            video = video[:n_samples].unsqueeze(0).to(device)
            text = text[:n_samples].unsqueeze(0).to(device)
            attention_mask = torch.ones((1, n_samples), dtype=torch.bool).to(device)
            subj_ids = torch.tensor([subj_id]).to(device)

            with torch.no_grad():
                preds = model(audio, video, text, subj_ids, attention_mask)  # (1, T, 1000)

            output_dict[subj][episode] = preds.squeeze(0).cpu().numpy().astype(np.float32)

    return output_dict


predictions = predict_fmri_for_test_set(
    model=model,
    audio_root="Features/Audio/Wave2Vec2/features_chunk1.49_len60_before50/friends/s7",
    video_root="Features/Visual/InternVideo/features_chunk1.49_len60_before50_frames120_imgsize224/friends/s7",
    text_root="Features/Text/Qwen3B_tr1.49_len60_before50/friends/s7",
    sample_counts_root="fmri",
    device='cuda:1'
)

output_file = "fmri_predictions_friends_s7.npy"
np.save(output_file, predictions, allow_pickle=True)

zip_file = "fmri_predictions_friends_s7.zip"
with zipfile.ZipFile(zip_file, 'w') as zipf:
    zipf.write("fmri_predictions_friends_s7.npy")