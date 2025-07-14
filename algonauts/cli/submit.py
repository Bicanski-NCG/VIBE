import argparse
import os
from pathlib import Path
import glob
import random
import numpy as np
import torch
import zipfile
from algonauts.models import load_model_from_ckpt
from algonauts.models.ensemble import ROIAdaptiveEnsemble
from algonauts.utils import ensure_paths_exist


def normalize_feature(x, mean, std):
    return (x - mean) / std


def pad_to_length(x, target_len):
    if x.shape[0] >= target_len:
        return x[:target_len]
    repeat_count = target_len - x.shape[0]
    pad = x[-1:].repeat(repeat_count, 1)
    return torch.cat([x, pad], dim=0)


# Helper to average prediction dictionaries
def average_prediction_dicts(pred_list):
    """
    Average a list of nested prediction dictionaries returned by
    `predict_fmri_for_test_set`.  Assumes all dicts share identical
    subject/clip keys.
    """
    out = {}
    for subj in pred_list[0]:
        out[subj] = {}
        for clip in pred_list[0][subj]:
            stacked = np.stack([p[subj][clip] for p in pred_list], axis=0)
            out[subj][clip] = stacked.mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
# High‑level helpers
# --------------------------------------------------------------------------- #
def load_single_model(checkpoint_name: str, output_root: Path, *,
                      device: str = "cuda", roi_ensemble: bool = False):
    """
    Load one checkpoint and return (model, config).  If `roi_ensemble` is
    True the checkpoint is wrapped in ROIAdaptiveEnsemble.
    """
    ckpt_dir = output_root / "checkpoints" / checkpoint_name
    model, config = load_model_from_ckpt(
        model_ckpt_path=str(ckpt_dir / "final_model.pt"),
        params_path=str(ckpt_dir / "config.yaml"),
        device=device,
    )
    if roi_ensemble:
        roi_labels = torch.load(ckpt_dir / "roi_names.pt", weights_only=False)
        roi_to_epoch = torch.load(ckpt_dir / "roi_to_epoch.pt", weights_only=False)
        model = ROIAdaptiveEnsemble(
            roi_labels=roi_labels,
            roi_to_epoch=roi_to_epoch,
            ckpt_dir=ckpt_dir,
            device=device,
        )
    model.to(device).eval()
    return model, config


def build_feature_paths(config):
    """Return a {modality: path} dict restricted to the model input dims."""
    return {
        name: Path(config.features_dir) / path
        for name, path in config.features.items()
        if name in config.input_dims
    }


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
            print(f"    →  Processing {subj} - {clip}", flush=True)
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
    parser = argparse.ArgumentParser(description="Make submission for fMRI predictions")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=str, help="Single checkpoint to load")
    g.add_argument("--ensemble",   type=str, nargs="+",
                   help="List of checkpoints to average")
    parser.add_argument("--name", type=str, default="submission",
                        help="Base name of the output files")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Root directory for outputs & checkpoints "
                             "(default $OUTPUT_DIR or data/outputs)")
    parser.add_argument("--roi_ensemble", action="store_true",
                        help="Use ROIAdaptiveEnsemble per checkpoint")
    parser.add_argument("--ood_names", type=str, required=True, nargs="+",
                        help="OOD dataset names for which to make predictions")
    args = parser.parse_args()

    # ------------------------------------------------------------------ paths
    output_root = Path(args.output_dir or os.getenv("OUTPUT_DIR", "data/outputs"))
    submission_dir = output_root / "submissions"
    ensure_paths_exist((output_root, "output_dir"))
    submission_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------- load models
    checkpoint_names = args.ensemble if args.ensemble else [args.checkpoint]
    device       = "cuda"
    load_device  = "cpu" if len(checkpoint_names) > 10 else device

    models   = []
    config   = None
    for ckpt in checkpoint_names:
        print(f"Loading model from checkpoint: {ckpt}", flush=True)
        m, cfg = load_single_model(
            ckpt,
            output_root=output_root,
            device=load_device,
            roi_ensemble=args.roi_ensemble,
        )
        models.append(m)
        if config is None:
            config = cfg

    feature_paths = build_feature_paths(config)

    # --------------------------------------------------------- predictions
    print("Starting predictions...", flush=True)
    preds_per_model = []
    for idx, model in enumerate(models):
        print(f"  ↳ running model {idx + 1}/{len(models)}", flush=True)
        model.to(device)
        preds = predict_fmri_for_test_set(
            model=model,
            feature_paths=feature_paths,
            sample_counts_root=config.data_dir,
            ood_names=args.ood_names,
            normalization_stats=None,
            device=device,
        )
        preds_per_model.append(preds)
        model.to(load_device)  # free GPU

    predictions = (preds_per_model[0] if len(preds_per_model) == 1
                   else average_prediction_dicts(preds_per_model))

    # -------------------------------------------------------------- saving
    out_name = (
        f"{args.name}_{'ensemble' if len(checkpoint_names) > 1 else checkpoint_names[0]}"
        f"_{'_'.join(args.ood_names)}_{random.randint(0, 1000)}"
    )
    npy_path  = submission_dir / f"{out_name}.npy"
    np.save(npy_path, predictions, allow_pickle=True)

    zip_path = submission_dir / f"{out_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(npy_path, f"{out_name}.npy")

    print(f"Saved predictions to {zip_path}", flush=True)


if __name__ == "__main__":
    main()