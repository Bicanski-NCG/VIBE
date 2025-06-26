import argparse
import os
from pathlib import Path
import glob
import numpy as np
import torch
import zipfile
from algonauts.models import load_model_from_ckpt
from algonauts.models.ensemble import EnsembleAverager, ROIAdaptiveEnsemble
from algonauts.utils import ensure_paths_exist


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

        for episode in sample_counts.keys():
            print(f"→  Processing {subj} - {episode}")
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
                preds = model(features, subj_ids, [0], attention_mask)

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


def main():
    args = argparse.ArgumentParser(description="Make submission for fMRI predictions")
    group = args.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Checkpoint to load")
    group.add_argument("--ensemble", type=str, nargs="+",
                       help="List of checkpoints to load for ensemble averaging")
    args.add_argument("--name", type=str, default=None, help="Name of output file")
    args.add_argument("--output_dir", default=None, type=str,
                      help="Root directory for outputs & checkpoints "
                           "(default $OUTPUT_DIR or data/outputs)")
    args.add_argument("--roi_ensemble", action="store_true",
                      help="Use ROIAdaptiveEnsemble to select best models per ROI")
    args = args.parse_args()

    if args.name is None:
        name = "submission"
    else:
        name = args.name

    # Attach checkpoint or ensemble name to the submission name
    if args.checkpoint:
        name = f"{name}_{args.checkpoint}"
    elif args.ensemble:
        name = f"{name}_ensemble"

    if not args.checkpoint and not args.ensemble:
        raise ValueError("Please provide a checkpoint to load or an ensemble list.")
    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")
    else:
        print(f"Using ensemble checkpoints: {args.ensemble}")

    output_root = Path(args.output_dir or os.getenv("OUTPUT_DIR", "data/outputs"))
    submission_dir = output_root / "submissions"
    checkpoint_dir = output_root / "checkpoints" / args.checkpoint if args.checkpoint else None
    final_model_path = checkpoint_dir / "final_model.pt" if checkpoint_dir else None
    config_path = checkpoint_dir / "config.yaml" if checkpoint_dir else None

    ensure_paths_exist(
        (output_root, "output_dir"),
        (submission_dir, "submission_dir"),
        *(([(checkpoint_dir, "checkpoint_dir")] if checkpoint_dir else [])),
        *(([(final_model_path, "final_model.pt")] if final_model_path else [])),
        *(([(config_path, "config.yaml")] if config_path else [])),
    )

    # Build model according to --ensemble or single checkpoint, with optional ROI wrap
    device = "cuda"
    if args.ensemble:
        # Ensemble averaging over provided run IDs
        checkpoint_names = args.ensemble
        # Load config from the first checkpoint
        first_ckpt_dir = output_root / "checkpoints" / checkpoint_names[0]
        _, config = load_model_from_ckpt(
            model_ckpt_path=str(first_ckpt_dir / "final_model.pt"),
            params_path=str(first_ckpt_dir / "config.yaml"),
        )
        # Load each model and collect
        models = []
        for chk in checkpoint_names:
            print(f"Loading model from checkpoint: {chk}")
            ckpt_dir = output_root / "checkpoints" / chk
            if args.roi_ensemble:
                m = ROIAdaptiveEnsemble(
                    roi_labels=torch.load(ckpt_dir / "roi_names.pt", weights_only=False),
                    roi_to_epoch=torch.load(ckpt_dir / "roi_to_epoch.pt", weights_only=False),
                    ckpt_dir=ckpt_dir,
                    device=device,
                )
            else:
                m, _ = load_model_from_ckpt(
                    model_ckpt_path=str(ckpt_dir / "final_model.pt"),
                    params_path=str(ckpt_dir / "config.yaml"),
                )
            m.to(device).eval()
            models.append(m)
        model = EnsembleAverager(models=models, device=device, normalize=True)
    else:
        # Single checkpoint path
        print(f"Loading model from checkpoint: {args.checkpoint}")
        checkpoint = args.checkpoint
        checkpoint_dir = output_root / "checkpoints" / checkpoint
        model, config = load_model_from_ckpt(
            model_ckpt_path=str(checkpoint_dir / "final_model.pt"),
            params_path=str(checkpoint_dir / "config.yaml"),
        )
        model.to(device)
        if args.roi_ensemble:
            # Wrap in ROIAdaptiveEnsemble for per-ROI best epoch
            roi_labels = torch.load(checkpoint_dir / "roi_names.pt", weights_only=False)
            roi_to_epoch = torch.load(checkpoint_dir / "roi_to_epoch.pt", weights_only=False)
            model = ROIAdaptiveEnsemble(
                roi_labels=roi_labels,
                roi_to_epoch=roi_to_epoch,
                ckpt_dir=checkpoint_dir,
                device=device,
            )
    model.eval()

    feature_paths = {name: Path(config.features_dir) / path for name, path in config.features.items()}

    print("Starting predictions for fMRI season 7 episodes...")
    predictions = predict_fmri_for_test_set(
        model=model,
        feature_paths=feature_paths,
        sample_counts_root=config.data_dir,
        normalization_stats=None,
        device=device,
    )

    output_file = submission_dir / f"{name}.npy"
    np.save(output_file, predictions, allow_pickle=True)

    zip_file = submission_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(output_file, f"{name}.npy")
    print(f"Saved predictions to {zip_file}")

if __name__ == "__main__":
    main()