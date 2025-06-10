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
        matches = glob.glob(os.path.join(root, "**", f"*{name}.*"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"{name}.npy not found in {root}")
        return matches[0]

    features = {}
    for modality, root in feature_paths.items():
        path = find_feature_file(root, episode_id)
        if path.endswith(".npy"):
            feat = torch.tensor(np.load(path), dtype=torch.float32).squeeze()
        elif path.endswith(".pt"):
            feat = torch.load(path, map_location="cpu").squeeze().float()
        else:
            raise ValueError(f"Unknown feature file extension: {path}")
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
                preds = model(features, subj_ids, [0],attention_mask)

            output_dict[subj][episode] = (
                preds.squeeze(0).cpu().numpy().astype(np.float32)
            )

    return output_dict

def predict_fmri_for_season7(
    model,
    feature_paths,
    sample_counts_root,
    normalization_stats=None,
    device="cuda",
):
    """
    Predict fMRI responses for all season-7 episodes in one forward pass
    per subject, returning the same nested dict structure that the
    submission checker expects:

        { "sub-01": { "s07e01a": np.array, ... }, ... }
    """

    model.eval()
    model.to(device)

    subjects           = ["sub-01", "sub-02", "sub-03", "sub-05"]
    subject_name2idx   = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}

    output_dict = {}

    for subj in subjects:
        print(f"\n→  Processing {subj}")
        output_dict[subj] = {}
        subj_id_t = torch.tensor([subject_name2idx[subj]], device=device)

        # ---------------------------------------------------------------------
        # 1) Gather sample counts for every season-7 episode for this subject
        # ---------------------------------------------------------------------
        counts_path = os.path.join(
            sample_counts_root,
            subj,
            "target_sample_number",
            f"{subj}_friends-s7_fmri_samples.npy",
        )
        sample_counts = np.load(counts_path, allow_pickle=True).item()

        # Sort keys to ensure chronological order: s07e01a, s07e01b, …
        # A lexicographic sort is already correct for this naming scheme:
        episodes = sorted(sample_counts.keys())

        # ---------------------------------------------------------------------
        # 2) Load & pad every modality for every episode, keep split indices
        # ---------------------------------------------------------------------
        concat_feats = {m: [] for m in feature_paths}   # lists of tensors
        split_lengths = []                               # n_samples per episode

        for epi_name in episodes:
            n_samples = sample_counts[epi_name]

            try:
                features = load_features_for_episode(
                    epi_name, feature_paths, normalization_stats
                )
            except FileNotFoundError as e:
                print(f"  ⚠️  Skipping {epi_name}: {e}")
                continue

            for m in feature_paths:
                # pad so every modality matches n_samples for this episode
                x = pad_to_length(features[m], n_samples)[:n_samples]
                concat_feats[m].append(x)

            split_lengths.append(n_samples)

        # If we skipped every episode, continue gracefully
        if not split_lengths:
            continue

        # ---------------------------------------------------------------------
        # 3) Concatenate over the *time* dimension -> shape [T_total, feat_dim]
        # ---------------------------------------------------------------------
        for m in concat_feats:
            concat_feats[m] = torch.cat(concat_feats[m], dim=0).unsqueeze(0).to(device)

        total_len = sum(split_lengths)

        attention_mask = torch.ones((1, total_len), dtype=torch.bool, device=device)

        # ---------------------------------------------------------------------
        # 4) Forward pass & split prediction back into episodes
        # ---------------------------------------------------------------------
        with torch.no_grad():
            preds = model(concat_feats, subj_id_t, attention_mask)      # [1,T,V]
        preds = preds.squeeze(0).cpu().numpy().astype(np.float32)       # [T,V]

        # Split along time-axis
        start = 0
        for epi_name, n in zip(episodes, split_lengths):
            output_dict[subj][epi_name] = preds[start : start + n]
            start += n

    return output_dict


feature_paths = {
    "aud_last": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/aud_last", #torch.Size([102, 1280])
    "aud_ln_post": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/audio_ln_post", #torch.Size([102, 1280])
    "conv3d_features": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/conv3d_features", #torch.Size([3536, 1280])
    "vis_block5": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/vis_block5", #torch.Size([3536, 1280])
    "vis_block8": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/vis_block8", #torch.Size([3536, 1280])
    "vis_block12": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/vis_block12", #torch.Size([3536, 1280])
    "vis_merged": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/vis_merged", #torch.Size([884, 2048])
    "thinker_12": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/thinker_12", #torch.Size([1, 984, 2048])
    "thinker_24": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/thinker_24", #torch.Size([1, 984, 2048])
    "thinker_36": "Features/Omni/Qwen2.5_3B/features_tr1.49_len8_before6/thinker_36", #torch.Size([1, 984, 2048])
    "text": "Features/Text/Qwen2.5_7B_Full_tr1.49",
    "fast_res3_act": "Features/Visual/SlowFast_R101_tr1.49/fast_res3_act",
    "fast_stem_act": "Features/Visual/SlowFast_R101_tr1.49/fast_stem_act",
    "pool_concat": "Features/Visual/SlowFast_R101_tr1.49/pool_concat",
    "slow_res3_act": "Features/Visual/SlowFast_R101_tr1.49/slow_res3_act",
    "slow_res5_act": "Features/Visual/SlowFast_R101_tr1.49/slow_res5_act",
    "slow_stem_act": "Features/Visual/SlowFast_R101_tr1.49/slow_stem_act",
    "audio_long_context": "Features/Audio/Wave2Vec2/features_chunk1.49_len60_before50",
    # "audio_mfcc_mono": "Features/Audio/LowLevel/_chunk1.49_len4.0_before2.0_nmfcc32_nstats4/mono/movies/",
    # "audio_mfcc_stereo": "Features/Audio/LowLevel/_chunk1.49_len4.0_before2.0_nmfcc32_nstats4/stereo/movies/",

}

input_dims = {
    "aud_last": 1280 * 2,
    #"aud_ln_post": 1280 * 2,
    #"conv3d_features": 1280 * 2,
    #"vis_block5": 1280 * 2,
    #"vis_block8": 1280 * 2,
    #"vis_block12": 1280 * 2,
    "vis_merged": 2048 * 2,
    #"thinker_12": 2048 * 2,
   # "thinker_24": 2048 * 2,
    "thinker_36": 2048 * 2,
    "text": 3584,
    #"fast_res3_act": 2048,
    #"fast_stem_act": 1024,
    "pool_concat": 9216,
    "slow_res3_act": 4096,
    #"slow_res5_act": 4096,
    #"slow_stem_act": 8192,
    #"audio_long_context": 2048,
    # "audio_mfcc_mono":int(4*32),
    # "audio_mfcc_stereo":int(4*32)

}


model = FMRIModel(
    input_dims, 1000, hidden_dim=256, fuse_mode="concat", subject_count=4,
)

model.load_state_dict(torch.load("checkpoints/f08qah39/final_model.pt"))
model.eval()

predictions = predict_fmri_for_test_set(
    model=model,
    feature_paths=feature_paths,
    sample_counts_root="fmri",
    normalization_stats=None,
    device="cuda:4",
)

output_file = "fmri_predictions_friends_s7.npy"
np.save(output_file, predictions, allow_pickle=True)

zip_file = "fmri_predictions_friends_s7.zip"
with zipfile.ZipFile(zip_file, "w") as zipf:
    zipf.write("fmri_predictions_friends_s7.npy")
