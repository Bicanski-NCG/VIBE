import os
import glob
import numpy as np
import torch
from fastprogress import progress_bar
from model import FMRIModel
import zipfile
from model import FMRIModel


def normalize_feature(x, mean, std):
    return (x - mean) / std


def pad_to_length(x, target_len):
    if x.shape[0] >= target_len:
        return x[:target_len]
    repeat_count = target_len - x.shape[0]
    pad = x[-1:].repeat(repeat_count, 1)
    return torch.cat([x, pad], dim=0)


def load_features_for_episode(episode_id, feature_paths, normalization_stats=None):
    def find_feature_file(root, name):
        matches = glob.glob(os.path.join(root, "**", f"*{name}.npy"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"{name}.npy not found in {root}")
        return matches[0]

    features = {}
    for modality, root in feature_paths.items():
        path = find_feature_file(root, episode_id)
        feat = torch.tensor(np.load(path), dtype=torch.float32).squeeze()
        if normalization_stats and f"{modality}_mean" in normalization_stats:
            feat = normalize_feature(
                feat,
                normalization_stats[f"{modality}_mean"],
                normalization_stats[f"{modality}_std"],
            )
        features[modality] = feat

    return features


def predict_fmri_for_test_set(
    model, feature_paths, sample_counts_root, normalization_stats=None, device="cuda"
):
    model.eval()
    model.to(device)
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]
    subject_name_id_dict = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}

    output_dict = {}
    for subj in subjects:
        output_dict[subj] = {}
        subj_id = subject_name_id_dict[subj]

        sample_dict_path = os.path.join(
            sample_counts_root,
            subj,
            "target_sample_number",
            f"{subj}_friends-s7_fmri_samples.npy",
        )
        sample_counts = np.load(sample_dict_path, allow_pickle=True).item()

        for episode in progress_bar(sample_counts.keys()):
            n_samples = sample_counts[episode]
            try:
                features = load_features_for_episode(
                    episode, feature_paths, normalization_stats
                )
            except FileNotFoundError as e:
                print(f"Skipping {episode}: {e}")
                continue

            for key in features:
                features[key] = (
                    pad_to_length(features[key], n_samples)[:n_samples]
                    .unsqueeze(0)
                    .to(device)
                )

            attention_mask = torch.ones((1, n_samples), dtype=torch.bool).to(device)
            subj_ids = torch.tensor([subj_id]).to(device)

            with torch.no_grad():
                preds = model(features, subj_ids, attention_mask)

            output_dict[subj][episode] = (
                preds.squeeze(0).cpu().numpy().astype(np.float32)
            )

    return output_dict


input_dims = {
    "audio": 2048,
    "video_low_level": 8192,
    "video_high_level": 512,
    "text": 2048,
}
model = FMRIModel(
    input_dims, 1000, hidden_dim=256, fuse_mode="concat", subject_count=4, max_len=600
)

model.load_state_dict(torch.load("final_model.pt"))
model.eval()

feature_paths = {
    "audio": "Features/Audio",
    "video_low_level": "Features/Visual/SlowR50",
    "video_high_level": "Features/Visual/InternVideo/features_chunk1.49_len60_before50_frames120_imgsize224",
    "text": "Features/Text",
}
predictions = predict_fmri_for_test_set(
    model=model,
    feature_paths=feature_paths,
    sample_counts_root="fmri",
    normalization_stats=None,
    device="cuda:1",
)

output_file = "fmri_predictions_friends_s7.npy"
np.save(output_file, predictions, allow_pickle=True)

zip_file = "fmri_predictions_friends_s7.zip"
with zipfile.ZipFile(zip_file, "w") as zipf:
    zipf.write("fmri_predictions_friends_s7.npy")
