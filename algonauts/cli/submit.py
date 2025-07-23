import argparse
import os
from pathlib import Path
import glob
import numpy as np
import torch
import zipfile
from algonauts.models import load_model_from_ckpt
from algonauts.models.ensemble import ROIAdaptiveEnsemble
from algonauts.utils import ensure_paths_exist
from algonauts.utils import logger


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
        matches = glob.glob(os.path.join(root, "**", f"*{name}*"), recursive=True)
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


        if normalization_stats and normalization_stats.get(modality) and normalization_stats.get(modality).get("mean"):
            feat = normalize_feature(
                feat,
                normalization_stats[modality]["mean"],
                normalization_stats[modality]["std"]
            )
        elif normalization_stats and f"{modality}_mean" in normalization_stats: #left this in for possible backward compatibility
            feat = normalize_feature(
                feat,
                normalization_stats[f"{modality}_mean"],
                normalization_stats[f"{modality}_std"],
            )
        features[modality] = feat

    return features


def predict_fmri_for_test_set(
    model, feature_paths, sample_counts_root, test_names, normalization_stats=None, device="cuda"
):
    model.eval()
    model.to(device)
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]
    subject_name_id_dict = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}

    if "s07" in test_names and len(test_names) > 1:
        raise ValueError("Only one OOD dataset can be specified for season 7 predictions.")
    elif "s07" in test_names:
        samples_file_postfix = "_friends-s7_fmri_samples.npy"
    else:
        samples_file_postfix = "_ood_fmri_samples.npy" 
    
    output_dict = {}
    for subj in subjects:
        output_dict[subj] = {}
        subj_id = subject_name_id_dict[subj]

        sample_dict_path = os.path.join(
            sample_counts_root,
            subj,
            "target_sample_number",
            str(subj) + samples_file_postfix,
        )
        sample_counts = np.load(sample_dict_path, allow_pickle=True).item()

        for clip in sample_counts.keys():
            if not any([ood in clip for ood in test_names]):
                continue
            logger.info(f"→  Processing {subj} - {clip}")
            n_samples = sample_counts[clip]
            try:
                features = load_features_for_episode(
                    clip, feature_paths, normalization_stats
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping {clip}: {e}")
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

            output_dict[subj][clip] = (
                preds.squeeze(0).cpu().numpy().astype(np.float32)
            )

    return output_dict


def main():
    parser = argparse.ArgumentParser(description="Make submission for fMRI predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, nargs="+", default=None,
                        help="Checkpoint(s) to load, single or multiple for ensemble averaging")
    group.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing the checkpoint(s) to load")
    parser.add_argument("--name", type=str, default=None, help="Name of output file")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Root directory for outputs & checkpoints (default $OUTPUT_DIR or runs)")
    parser.add_argument("--roi_ensemble", action="store_true",
                        help="Use ROIAdaptiveEnsemble to select best models per ROI")
    parser.add_argument("--test_names", type=str, default=None, nargs="+",
                        help="Name of dataset for which to make predictions")
    args = parser.parse_args()

    if args.checkpoint_dir:
        logger.info(f"Loading checkpoints from directory: {args.checkpoint_dir}")

        args.checkpoint = [
            os.path.basename(os.path.dirname(p))          # parent dir’s name
            for p in glob.glob(os.path.join(args.checkpoint_dir, "*", "final_model.pt"))
        ]

        if not args.checkpoint:
            raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")

    if args.name is None:
        name = "submission"
    else:
        name = args.name

    if args.test_names is None:
        args.test_names = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
        name = f"{name}_ood_all"
    else:
        name = f"{name}_{'_'.join(args.test_names)}"

    # Attach checkpoint(s) to submission name
    name = f"{name}_ensemble" if len(args.checkpoint) > 1 else f"{name}_{args.checkpoint[0]}"

    logger.info(f"Using checkpoint(s): {args.checkpoint}")

    output_root = Path(args.output_dir or os.getenv("OUTPUT_DIR", "runs"))
    submission_dir = output_root / "submissions"
    checkpoint_dir = output_root / "checkpoints" / args.checkpoint[0]
    final_model_path = checkpoint_dir / "final_model.pt" if checkpoint_dir else None
    config_path = checkpoint_dir / "config.yaml" if checkpoint_dir else None

    ensure_paths_exist(
        (output_root, "output_dir"),
        *(([(checkpoint_dir, "checkpoint_dir")] if checkpoint_dir else [])),
        *(([(final_model_path, "final_model.pt")] if final_model_path else [])),
        *(([(config_path, "config.yaml")] if config_path else [])),
    )

    try:
        ensure_paths_exist(
            (submission_dir, "submission_dir"),
        )
    except FileNotFoundError:
        os.makedirs(submission_dir, exist_ok=True)

    device = "cuda"
    # Determine checkpoints to process (single or ensemble)
    checkpoint_list = args.checkpoint
    # Load config from first checkpoint
    first_ckpt_dir = output_root / "checkpoints" / checkpoint_list[0]
    _, config = load_model_from_ckpt(
        model_ckpt_path=str(first_ckpt_dir / "final_model.pt"),
        params_path=str(first_ckpt_dir / "config.yaml"),
    )
    # Prepare feature paths
    feature_paths = {
        name: Path(config.features_dir) / path
        for name, path in config.features.items()
        if name in config.input_dims
    }
    num_models = len(checkpoint_list)
    predictions_sum = None

    for i, chk in enumerate(checkpoint_list):
        with logger.step(f"Loading and predicting with checkpoint ({i+1}/{len(checkpoint_list)}): {chk}"):
            ckpt_dir = output_root / "checkpoints" / chk
            # Load model (with optional ROI ensemble wrapper)
            model_instance, _ = load_model_from_ckpt(
                model_ckpt_path=str(ckpt_dir / "final_model.pt"),
                params_path=str(ckpt_dir / "config.yaml"),
            )
            if args.roi_ensemble:
                roi_labels = torch.load(ckpt_dir / "roi_names.pt", weights_only=False)
                roi_to_epoch = torch.load(ckpt_dir / "roi_to_epoch.pt", weights_only=False)
                model_instance = ROIAdaptiveEnsemble(
                    roi_labels=roi_labels,
                    roi_to_epoch=roi_to_epoch,
                    ckpt_dir=ckpt_dir,
                    device=device,
                )
            model_instance.to(device).eval()
            current_preds = predict_fmri_for_test_set(
                model=model_instance,
                feature_paths=feature_paths,
                sample_counts_root=config.data_dir,
                test_names=args.test_names,
                normalization_stats=None,
                device=device,
            )
            if predictions_sum is None:
                # Deep copy first model's predictions
                predictions_sum = {
                    subj: {clip: pred.copy() for clip, pred in clips.items()}
                    for subj, clips in current_preds.items()
                }
            else:
                for subj, clips in current_preds.items():
                    for clip, pred in clips.items():
                        predictions_sum[subj][clip] += pred
            # Clean up GPU memory
            del model_instance
            torch.cuda.empty_cache()

    # Average predictions across models
    for subj, clips in predictions_sum.items():
        for clip in clips:
            predictions_sum[subj][clip] /= num_models
    predictions = predictions_sum

    # Some basic checks
    for subj, clips in predictions.items():
        for clip, pred in clips.items():
            if not isinstance(pred, np.ndarray):
                raise ValueError(f"Prediction for {subj} - {clip} is not a numpy array.")
            if pred.ndim != 2:
                raise ValueError(f"Prediction for {subj} - {clip} should be 2D, got {pred.ndim}D.")
            if not np.issubdtype(pred.dtype, np.floating):
                raise ValueError(f"Prediction for {subj} - {clip} should be a float array, got {pred.dtype}.")
            if not pred.dtype == np.float32:
                print("Warning: Prediction for {subj} - {clip} is not float32, it is {pred.dtype}.")
            if not np.isfinite(pred).all():
                raise ValueError(f"Prediction for {subj} - {clip} contains non-finite values.")

    # random_number = random.randint(0, 1000)
    # name = f"{name}_{random_number}"
    output_file = submission_dir / f"{name}.npy"
    np.save(output_file, predictions, allow_pickle=True)

    zip_file = submission_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(output_file, f"{name}.npy")
    logger.info(f"Saved predictions to {zip_file}")

if __name__ == "__main__":
    main()