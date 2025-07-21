import argparse
from functools import cache
import os
from pathlib import Path
import random
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from algonauts.data.loader import get_train_val_loaders
from algonauts.models import load_model_from_ckpt
from algonauts.models.ensemble import EnsembleAverager, ROIAdaptiveEnsemble
from algonauts.utils import ensure_paths_exist
from algonauts.utils.utils import evaluate_corr
from msapy import msa

from algonauts.utils.viz import load_and_label_atlas, plot_voxel_contributions

def get_objective_function(model, loader, device, max_batches=-1):
    @cache
    def objective_function(lesioned_modalities):
        return evaluate_corr(model, loader, lesioned_modalities, device=device, max_batches=max_batches)

    return objective_function

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
    args.add_argument("--num_permutations", type=int, default=128,
                      help="Number of permutations for MSA analysis")
    args.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for loader")
    args.add_argument("--max_batches", type=int, default=-1,
                      help="Maximum number of batches to process")
    args = args.parse_args()

    if args.name is None:
        name = ""
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
    shapley_dir = output_root / f"shapley_{args.num_permutations}_permutations_{args.max_batches}_batches_{args.batch_size}_batchsize{name}"
    checkpoint_dir = output_root / "checkpoints" / args.checkpoint if args.checkpoint else None
    best_model_path = checkpoint_dir / "best_model.pt" if checkpoint_dir else None
    config_path = checkpoint_dir / "config.yaml" if checkpoint_dir else None

    ensure_paths_exist(
        (output_root, "output_dir"),
        *(([(checkpoint_dir, "checkpoint_dir")] if checkpoint_dir else [])),
        *(([(best_model_path, "best_model.pt")] if best_model_path else [])),
        *(([(config_path, "config.yaml")] if config_path else [])),
    )

    os.makedirs(shapley_dir, exist_ok=True)

    # Build model according to --ensemble or single checkpoint, with optional ROI wrap
    device = "cuda"
    if args.ensemble:
        load_device = "cpu" if len(args.ensemble) > 25 else device
        # Ensemble averaging over provided run IDs
        checkpoint_names = args.ensemble
        # Load config from the first checkpoint
        first_ckpt_dir = output_root / "checkpoints" / checkpoint_names[0]
        _, config = load_model_from_ckpt(
            model_ckpt_path=str(first_ckpt_dir / "best_model.pt"),
            params_path=str(first_ckpt_dir / "config.yaml"),
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
                    model_ckpt_path=str(ckpt_dir / "best_model.pt"),
                    params_path=str(ckpt_dir / "config.yaml"),
                )
            #m.to(load_device).eval()
            device = torch.device(f"cuda:{random.randrange(torch.cuda.device_count())}") if torch.cuda.is_available() else torch.device("cpu")
            m.to(device).eval()
            models.append(m)
        model = EnsembleAverager(models=models, device=device, normalize=True)
    else:
        # Single checkpoint path
        print(f"Loading model from checkpoint: {args.checkpoint}", flush=True)
        checkpoint = args.checkpoint
        checkpoint_dir = output_root / "checkpoints" / checkpoint
        model, config = load_model_from_ckpt(
            model_ckpt_path=str(checkpoint_dir / "best_model.pt"),
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
    config.batch_size = args.batch_size
    _, valid_loader = get_train_val_loaders(config, use_wandb=False)

    objective_function = get_objective_function(model, valid_loader, device=device, max_batches=args.max_batches)


    if os.path.exists(shapley_dir / "shapley_modes.csv"):
        shapley_modes = pd.read_csv(shapley_dir / "shapley_modes.csv")
    else:
        shapley_modes = msa.interface(
            n_permutations=args.num_permutations,
            elements=list(config.input_dims.keys()),
            objective_function=objective_function,
            n_parallel_games=-1,
        )

    #shaples_modes = pd.read_csv("shap_ood1/shapley_2_permutations_None/shapley_modes.csv")

    shapley_modes_files = shapley_dir / f"shapley_modes.csv"
    shapley_modes.to_csv(shapley_modes_files, index=False)
    print(f"Saved shapley modes to {shapley_modes_files}", flush=True)
    atlas_path = valid_loader.dataset.samples[0]["subject_atlas"]
    masker = load_and_label_atlas(atlas_path,
                                  yeo_networks=config.yeo_networks,
                                  anatomical=False)
    absmax = np.nanmax(np.abs(shapley_modes.values))
    for modality in config.input_dims.keys():
        contrib_vec = shapley_modes[modality]
        avg_contrib = contrib_vec.mean()
        cmap = sns.blend_palette(("#2166ac", "white", "#b2182b"), as_cmap=True)  # neg blue → pos red

        plot_voxel_contributions(
            contrib_vec,
            masker,
            cmap=cmap,        # <-- assuming you added cmap param; see patched fn below
            vmin=-absmax,
            vmax=absmax,
            title=f"{modality} contributions. Avg. Contrib {avg_contrib:.5f}",
            out_dir=shapley_dir,
            filename=modality,
        )

if __name__ == "__main__":
    main()