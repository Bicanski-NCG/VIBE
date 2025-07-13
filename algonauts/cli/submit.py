import argparse
import os
from pathlib import Path
import glob
import random
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
    model, feature_paths, sample_counts_root, ood_names, normalization_stats=None, device="cuda"
):
    model.eval()
    #model.to(device)
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
            f"{subj}_ood_fmri_samples.npy",
        )
        sample_counts = np.load(sample_dict_path, allow_pickle=True).item()

        for clip in sample_counts.keys():
            if not any([ood in clip for ood in ood_names]):
                continue
            print(f"→  Processing {subj} - {clip}", flush=True)
            n_samples = sample_counts[clip]
            try:
                features = load_features_for_episode(
                    clip, feature_paths, normalization_stats
                )
            except FileNotFoundError as e:
                print(f"Skipping {clip}: {e}", flush=True)
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
    args.add_argument("--ood_names", type=str, default=None, nargs="+",
                      help="Name of OOD dataset for which to make predictions")
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
        print(f"Using checkpoint: {args.checkpoint}", flush=True)
    else:
        print(f"Using ensemble checkpoints: {args.ensemble}", flush=True)

    output_root = Path(args.output_dir or os.getenv("OUTPUT_DIR", "data/outputs"))
    submission_dir = output_root / "submissions"
    checkpoint_dir = output_root / "checkpoints" / args.checkpoint if args.checkpoint else None
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
    except:
        os.mkdir(submission_dir)

    # Build model according to --ensemble or single checkpoint, with optional ROI wrap
    device = "cuda"
    if args.ensemble:
        load_device = "cpu" if len(args.ensemble) > 10 else device
        # Ensemble averaging over provided run IDs
        checkpoint_names = args.ensemble
        # Load config from the first checkpoint
        first_ckpt_dir = output_root / "checkpoints" / checkpoint_names[0]
        _, config = load_model_from_ckpt(
            model_ckpt_path=str(first_ckpt_dir / "final_model.pt"),
            params_path=str(first_ckpt_dir / "config.yaml"),device = load_device
        )
        # Load each model and collect
        models = []
        for chk in checkpoint_names:
            print(f"Loading model from checkpoint: {chk}", flush=True)
            ckpt_dir = output_root / "checkpoints" / chk
            if args.roi_ensemble:
                m = ROIAdaptiveEnsemble(
                    roi_labels=torch.load(ckpt_dir / "roi_names.pt", weights_only=False),
                    roi_to_epoch=torch.load(ckpt_dir / "roi_to_epoch.pt", weights_only=False),
                    ckpt_dir=ckpt_dir,
                    device=load_device,
                )
            else:
                m, _ = load_model_from_ckpt(
                    model_ckpt_path=str(ckpt_dir / "final_model.pt"),
                    params_path=str(ckpt_dir / "config.yaml"),device = load_device
                )
            #m.to(load_device).eval()
            device = torch.device(f"cuda:{random.randrange(torch.cuda.device_count())}") if torch.cuda.is_available() else torch.device("cpu")
            m.to(device).eval()
            models.append(m)
        model = EnsembleAverager(models=models, device = device, normalize=True)
    else:
        # Single checkpoint path
        print(f"Loading model from checkpoint: {args.checkpoint}", flush=True)
        checkpoint = args.checkpoint
        checkpoint_dir = output_root / "checkpoints" / checkpoint
        model, config = load_model_from_ckpt(
            model_ckpt_path=str(checkpoint_dir / "final_model.pt"),
            params_path=str(checkpoint_dir / "config.yaml"),
        )
        model.to(device)
        if args.roi_ensemble:
            # Wrap in ROIAdaptiveEnsemble for per-ROI best iters
            roi_labels = torch.load(checkpoint_dir / "roi_names.pt", weights_only=False)
            roi_to_epoch = torch.load(checkpoint_dir / "roi_to_epoch.pt", weights_only=False)
            model = ROIAdaptiveEnsemble(
                roi_labels=roi_labels,
                roi_to_epoch=roi_to_epoch,
                ckpt_dir=checkpoint_dir,
                device=device,
            )
    model.eval()

    feature_paths = {name: Path(config.features_dir) / path for name, path in config.features.items() if name in config.input_dims}

    print("Starting predictions for fMRI season 7 episodes...", flush=True)
    predictions = predict_fmri_for_test_set(
        model=model,
        feature_paths=feature_paths,
        sample_counts_root=config.data_dir,
        ood_names=args.ood_names,
        normalization_stats=None,
        device=device,
    )
    random_number = random.randint(0, 1000)
    name = f"{name}_{'_'.join(args.ood_names)}_{random_number}"
    output_file = submission_dir / f"{name}.npy"
    np.save(output_file, predictions, allow_pickle=True)

    zip_file = submission_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(output_file, f"{name}.npy")
    print(f"Saved predictions to {zip_file}", flush=True)

if __name__ == "__main__":
    main()