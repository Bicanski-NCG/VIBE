import argparse
import random
import wandb
from pathlib import Path
import pickle, gzip
import glob
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.stats import pearsonr

from data import FMRI_Dataset, split_dataset_by_name, collate_fn, make_group_weights
from model import FMRIModel
from losses import (
    masked_negative_pearson_loss,
    sample_similarity_loss,
    roi_similarity_loss
)
from utils import save_initial_state, load_initial_state, set_seed, log_model_params, collect_predictions
from config import Config
from viz import (
    load_and_label_atlas,
    voxelwise_pearsonr,
    plot_glass_brain,
    plot_corr_histogram,
    roi_table,
    plot_glass_bads,
    plot_time_correlation,
    plot_residual_glass,
    plot_pred_vs_true_scatter,
    plot_residual_psd
)


def parse_args():
    """Parse CLI arguments and load YAML configuration files."""
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--features", default=None, type=str, help="Path to features YAML file")
    parser.add_argument("--features_dir", default=None, type=str, help="Directory for features")
    parser.add_argument("--data_dir", default=None, type=str, help="Directory for fMRI data")
    parser.add_argument("--params", default=None, type=str, help="Path to training parameters YAML file")
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility")
    parser.add_argument("--name", default=None, type=str, help="Run name for W&B")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use for training (default: cuda)")
    parser.add_argument("--wandb_project", default="fmri-model", type=str, help="W&B project name")
    return parser.parse_known_args()[0]


def get_data_loaders(config):
    """Instantiate datasets and DataLoaders for training, validation, and full retraining."""
    # Assume normalization stats have been precomputed
    norm_stats = torch.load("normalization_stats.pt")

    # Prepend the features directory to each feature path
    features_dir = Path(config.features_dir)
    config.features = {n: str(features_dir / p) for n, p in config.features.items()}

    ds = FMRI_Dataset(config.data_dir,
                      feature_paths=config.features,
                      input_dims=config.input_dims,
                      modalities=config.modalities,
                      noise_std=config.train_noise_std,
                      normalization_stats=norm_stats if config.use_normalization else None,
                      oversample_factor=config.oversample_factor)

    if config.filter_name is not None:
        filter_fn = lambda sample: sample["name"] not in config.filter_name
        ds = ds.filter_samples(filter_fn)
    
    print(f"Dataset size: {len(ds)} samples")

    train_ds, valid_ds = split_dataset_by_name(
        ds,
        val_name=config.val_name,
        val_run=config.val_run,
        train_noise_std=config.train_noise_std,
        normalize_validation_bold=config.normalize_validation_bold,
    )

    if config.stratification_variable:
        train_weights = make_group_weights(train_ds, filter_on=config.stratification_variable)
        print(f"Using stratification variable: {config.stratification_variable}")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        shuffle = False
    else:
        train_weights = torch.ones(len(train_ds), dtype=torch.float32)
        sampler = None
        shuffle = True

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
    )
    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
    )

    full_loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, valid_loader, full_loader


def build_model(config):
    """Instantiate the FMRIModel and move to device."""
    model = FMRIModel(
        config.input_dims,
        config.output_dim,
        # fusion-stage hyper-params
        fusion_hidden_dim=config.fusion_hidden_dim,
        fusion_layers=config.fusion_layers,
        fusion_heads=config.fusion_heads,
        fusion_dropout=config.fusion_dropout,
        subject_dropout_prob=config.subject_dropout_prob,
        use_fusion_transformer=config.use_fusion_transformer,
        proj_layers=config.proj_layers,
        fuse_mode=config.fuse_mode,
        subject_count=config.subject_count,
        # temporal predictor hyper-params
        pred_layers=config.pred_layers,
        pred_heads=config.pred_heads,
        pred_dropout=config.pred_dropout,
        rope_pct=config.rope_pct,
        # HRF-related
        use_hrf_conv=config.use_hrf_conv,
        learn_hrf=config.learn_hrf,
        hrf_size=config.hrf_size,
        tr_sec=config.tr_sec,
        # training
        mask_prob=config.mask_prob,
    )
    model.to(config.device)
    return model


def create_optimizer_and_scheduler(model, config):
    """Create optimizer with param groups and scheduler, handling HRF conv if enabled."""
    if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
        hrf_params = [model.hrf_conv.weight]
        other_params = [p for n, p in model.named_parameters() if n != "hrf_conv.weight"]
        param_groups = [
            {"params": hrf_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": config.weight_decay},
        ]
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": config.weight_decay}]

    optimizer = optim.AdamW(param_groups, lr=config.lr)

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs
    )
    
    if config.warmup_epochs > 0:
        warm_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.warmup_start_lr_factor, 
            end_factor=1.0, total_iters=config.warmup_epochs
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_scheduler, main_scheduler],
            milestones=[config.warmup_epochs]
        )
    else:
        scheduler = main_scheduler

    return optimizer, scheduler


def run_epoch(loader, model, optimizer, device, is_train, global_step, config):
    """Run one epoch; return tuple of losses and updated global_step."""
    epoch_negative_corr_loss = 0.0
    epoch_sample_loss = 0.0
    epoch_roi_loss = 0.0
    epoch_mse_loss = 0.0
    model.train() if is_train else model.eval()

    for batch in loader:
        features = {k: batch[k].to(device) for k in loader.dataset.modalities}

        # ── Modality dropout: randomly zero‑out entire modalities ──────────
        drop_prob = config.modality_dropout_prob  # e.g. 0.15 in params.yaml
        if is_train and drop_prob > 0:
            for mod in loader.dataset.modalities:
                if float(torch.rand(1)) < drop_prob:
                    # Replace with zeros; keeps tensor shape & device
                    features[mod] = torch.zeros_like(features[mod])

        subject_ids = batch["subject_ids"]
        run_ids = batch["run_ids"]
        fmri = batch["fmri"].to(device)
        attn_mask = batch["attention_masks"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(features, subject_ids, run_ids,attn_mask)
            negative_corr_loss = masked_negative_pearson_loss(pred, fmri, attn_mask)
            sample_loss = sample_similarity_loss(pred, fmri, attn_mask)
            roi_loss = roi_similarity_loss(pred, fmri, attn_mask)
            mse_loss = nn.functional.mse_loss(pred, fmri)
            loss = (
                negative_corr_loss
                + config.lambda_sample * sample_loss
                + config.lambda_roi * roi_loss
                + config.lambda_mse * mse_loss
            )
            if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
                hrf_dev = model.hrf_conv.weight - model.hrf_prior
                loss += config.lambda_hrf * hrf_dev.norm(p=2)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if global_step is not None:
                    wandb.log(
                        {
                            "train_neg_corr_loss": negative_corr_loss.item(),
                            "train_sample_loss": sample_loss.item(),
                            "train_roi_loss": roi_loss.item(),
                            "train_mse_loss": mse_loss.item(),
                        },
                        step=global_step,
                    )
                    global_step += 1

        epoch_negative_corr_loss += negative_corr_loss.item()
        epoch_sample_loss += sample_loss.item()
        epoch_roi_loss += roi_loss.item()
        epoch_mse_loss += mse_loss.item()

    total_loss = (
        epoch_negative_corr_loss / len(loader)
        + config.lambda_sample * (epoch_sample_loss / len(loader))
        + config.lambda_roi * (epoch_roi_loss / len(loader))
        + config.lambda_mse * (epoch_mse_loss / len(loader))
    )
    return (
        total_loss,
        epoch_negative_corr_loss / len(loader),
        epoch_sample_loss / len(loader),
        epoch_roi_loss / len(loader),
        epoch_mse_loss / len(loader),
    ), global_step


def train_loop(model, optimizer, scheduler, ckpt_dir, config):
    """Full training pipeline including early stopping. Returns best_val_epoch, model, and full_loader."""
    # Prepare DataLoaders
    train_loader, valid_loader, full_loader = get_data_loaders(config)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    best_val_epoch = 0

    # Group-level masker for visual diagnostics
    group_masker = None

    for epoch in range(1, config.epochs + 1):
        train_losses, global_step = run_epoch(
            train_loader,
            model,
            optimizer,
            config.device,
            is_train=True,
            global_step=global_step,
            config=config,
        )
        val_losses, _ = run_epoch(
            valid_loader,
            model,
            optimizer,
            config.device,
            is_train=False,
            global_step=None,
            config=config,
        )

        wandb.log(
            {
                # TRAIN
                "epoch": epoch,
                "train/loss": train_losses[0],
                "train/neg_corr": train_losses[1],
                "train/sample": train_losses[2],
                "train/roi": train_losses[3],
                "train/mse": train_losses[4],

                # VAL
                "val/loss":  val_losses[0],
                "val/neg_corr": val_losses[1],
                "val/sample": val_losses[2],
                "val/roi": val_losses[3],
                "val/mse": val_losses[4],

                # shared LR
                "train/lr": optimizer.param_groups[0]["lr"]
                    if len(optimizer.param_groups) == 1
                    else optimizer.param_groups[1]["lr"],
            },
            step=global_step,
        )

        # ── Optional per‑ROI validation correlations ───────────────────────
        # Set roi_log_interval in params.yaml (e.g. 5) to control frequency; 0 disables
        roi_interval = config.roi_log_interval
        if roi_interval and epoch % roi_interval == 0:
            with torch.no_grad():
                fmri_true, fmri_pred, subj_ids, atlas_paths = collect_predictions(
                    valid_loader, model, config.device
                )
                # Load atlas once
                if not group_masker:
                    group_masker = load_and_label_atlas(atlas_paths[0])
                # Group‑level mean voxel‑wise r
                group_mean_r = np.mean(
                    [voxelwise_pearsonr(t, p) for t, p in zip(fmri_true, fmri_pred)],
                    axis=0,
                )
                df_group_roi = roi_table(group_mean_r, "group", group_masker, out_dir=None)
                # Flatten into a wandb metrics dict: roi/ROI_NAME : mean_r
                roi_metrics = {f"roi/{row['label']}": row["mean_r"] for _, row in df_group_roi.iterrows()}
                wandb.log(roi_metrics, step=global_step)

        scheduler.step()
        current_val = val_losses[1]  # negative correlation as primary metric
        print(f"Epoch {epoch}: Train NegCorr = {train_losses[1]:.4f}, Val NegCorr = {current_val:.4f}")

        if current_val < best_val_loss:
            best_val_loss = current_val
            best_val_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            wandb.log({"best_model_path": str(ckpt_dir / "best_model.pt")})
            print(f"✅ Saved new best model at epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{config.early_stop_patience}")
            if patience_counter >= config.early_stop_patience:
                print("⏹️ Early stopping triggered")
                break
            
    # ── Generate visual diagnostics on validation set ──────────────
    print("📊 Generating visualisations on validation set …")
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt"))
    fmri_true, fmri_pred, subj_ids, atlas_paths = collect_predictions(
        valid_loader, model, config.device
    )

    # Persist validation predictions for later analysis
    pred_path = ckpt_dir / "val_predictions.pkl.gz"
    with gzip.open(pred_path, "wb") as f:
        pickle.dump(
            {
                "subjects": subj_ids,
                "atlas_paths": atlas_paths,
                "fmri_true": fmri_true,   # lists of (T, V) ndarrays
                "fmri_pred": fmri_pred,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Per subject visual diagnostics
    for true, pred, sid, atlas_path in zip(fmri_true, fmri_pred, subj_ids, atlas_paths):
        # 1 – voxel-wise correlations
        r = voxelwise_pearsonr(true, pred)

        # 2 - load atlas and create masker
        masker = load_and_label_atlas(atlas_path)

        # (1) glass-brain
        plot_glass_brain(r, sid, masker, out_dir=str(ckpt_dir))

        # (2) correlation histogram
        plot_corr_histogram(r, sid, out_dir=str(ckpt_dir))

        # (3) ROI table and bar chart
        df_roi = roi_table(r, sid, masker, out_dir=str(ckpt_dir))
        table_roi = wandb.Table(dataframe=df_roi.astype({"mean_r": float}))
        bar_chart = wandb.plot.bar(
            table_roi,
            "label",      # x‑axis
            "mean_r",     # y‑axis
            title=f"ROI mean Pearson r – {sid}",
        )
        wandb.log({f"viz/roi_bar_{sid}": bar_chart})

        # (4) time correlation
        r_t = np.array([pearsonr(true[t], pred[t])[0] for t in range(true.shape[0])])
        plot_time_correlation(r_t, sid, out_dir=str(ckpt_dir))

        # (5) glass-brain of worst timepoints
        plot_glass_bads(true, pred, sid, masker, out_dir=str(ckpt_dir), pct_bads=config.pct_bads)

        # (6) residual glass-brain
        plot_residual_glass(true, pred, sid, masker, out_dir=str(ckpt_dir))

        # (7) scatter
        plot_pred_vs_true_scatter(true, pred, sid, out_dir=str(ckpt_dir))

        # (8) PSD of residuals
        plot_residual_psd(true, pred, sid, out_dir=str(ckpt_dir), fs=1/1.49)

    # Group visual diagnostics
    group_mean_r = np.mean([voxelwise_pearsonr(true, pred) for true, pred in zip(fmri_true, fmri_pred)], axis=0)
    group_masker = load_and_label_atlas(atlas_paths[0])  # use first atlas for group

    # (1) group glass-brain
    plot_glass_brain(group_mean_r, "group", group_masker, out_dir=str(ckpt_dir))

    # (2) group correlation histogram
    plot_corr_histogram(group_mean_r, "group", out_dir=str(ckpt_dir))

    # (3) group ROI bar chart
    df_group_roi = roi_table(group_mean_r, "group", group_masker, out_dir=str(ckpt_dir))
    table_roi = wandb.Table(dataframe=df_group_roi.astype({"mean_r": float}))
    bar_chart = wandb.plot.bar(
        table_roi,
        "label",       # x‑axis
        "mean_r",      # y‑axis
        title="Group ROI mean Pearson r",
    )
    wandb.log({"viz/roi_bar_group": bar_chart})

    # (4) group time correlation
    r_t_list = [
        np.array([pearsonr(pred[t], true[t])[0]
                for t in range(true.shape[0])])
        for true, pred in zip(fmri_true, fmri_pred)
    ]
    max_T   = max(arr.size for arr in r_t_list)
    r_t_mat = np.full((len(r_t_list), max_T), np.nan)
    for i, arr in enumerate(r_t_list):
        r_t_mat[i, :arr.size] = arr

    group_r_t = np.nanmean(r_t_mat, axis=0)   # (max_T,)
    plot_time_correlation(group_r_t, "group", out_dir=str(ckpt_dir))


    # (5) group residual glass-brain
    group_res_true = np.concatenate([t for t in fmri_true], 0)
    group_res_pred = np.concatenate([p for p in fmri_pred], 0)
    plot_residual_glass(group_res_true, group_res_pred, "group", group_masker, out_dir=str(ckpt_dir))

    # (6) group glass-brain of worst TRs (“bads”)
    plot_glass_bads(
        group_res_true,
        group_res_pred,
        "group",
        group_masker,
        out_dir=str(ckpt_dir),
        pct_bads=config.pct_bads
    )


    # (7) group scatter
    plot_pred_vs_true_scatter(
        group_res_true, group_res_pred, "group", out_dir=str(ckpt_dir), max_points=config.max_scatter_points
    )

    # (8) group PSD of residuals
    plot_residual_psd(group_res_true, group_res_pred, "group", out_dir=str(ckpt_dir), fs=1/1.49)


    # Log every png
    for png in glob.glob(str(ckpt_dir / "*.png")):
        wandb.log({f"viz/{Path(png).name}": wandb.Image(png)})

    wandb.run.summary["best_val_pearson"] = best_val_loss
    return best_val_epoch, model, full_loader, global_step


def retrain_full(model, full_loader, config, best_val_epoch, ckpt_dir, start_step):
    """Retrain the model from initial state on the full dataset for best_val_epoch epochs."""
    print("🔁 Reloading initial model and retraining on full dataset...")
    load_initial_state(
        model,
        ckpt_dir / "initial_model.pt",
        ckpt_dir / "initial_random_state.pt",
    )

    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    global_step = start_step

    for epoch in range(1, best_val_epoch + 1):
        full_losses, global_step = run_epoch(
            full_loader,
            model,
            optimizer,
            config.device,
            is_train=True,
            global_step=global_step,
            config=config,
        )

        wandb.log(
            {
                "retrain_epoch": epoch,
                "full/loss": full_losses[0],
                "full/neg_corr": full_losses[1],
                "full/sample": full_losses[2],
                "full/roi": full_losses[3],
                "full/mse": full_losses[4],
                "full/lr": optimizer.param_groups[0]["lr"]
                    if len(optimizer.param_groups) == 1
                    else optimizer.param_groups[1]["lr"],
            },
            step=global_step,
        )

        scheduler.step()
        print(f"Epoch {epoch}: Full NegCorr = {full_losses[1]:.4f}")

    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    wandb.log({"final_model_path": str(ckpt_dir / "final_model.pt")})
    print("✅ Final model trained on full dataset and saved.")


def main():

    args = parse_args()

    # Set seed
    if args.seed is None:
        chosen_seed = random.SystemRandom().randint(0, 2**32 - 1)
        print(f"No --seed provided; using generated seed={chosen_seed}")
    else:
        chosen_seed = args.seed
        print(f"Using user-provided seed={chosen_seed}")
    set_seed(chosen_seed)

    # Get features and params paths
    features_path = args.features or os.getenv("FEATURES_PATH", "config/features.yaml")
    params_path = args.params or os.getenv("PARAMS_PATH", "config/params.yaml")

    # Same for dirs 
    features_dir = args.features_dir or os.getenv("FEATURES_DIR", "Features")
    data_dir = args.data_dir or os.getenv("DATA_DIR", "fmri")

    # Assert paths exist
    features_path = Path(features_path)
    params_path = Path(params_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features YAML file not found: {features_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters YAML file not found: {params_path}")
    if not Path(features_dir).exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")    

    config = Config.from_yaml(features_path, params_path, chosen_seed, args.name,
                              features_dir, data_dir, args.device)

    # Initialize W&B run
    wandb.init(project="fmri-model", config=vars(config), name=config.run_name)

    # If running a W&B sweep, wandb.config contains overridden hyperparameters.
    # Merge them back into our local config dict so later code picks them up.
    config = Config(**wandb.config)

    # Build model outside the train loop
    model = build_model(config)
    log_model_params(model)

    # Define W&B metrics
    wandb.define_metric("epoch")
    wandb.define_metric("retrain_epoch")
    wandb.define_metric("val/loss", step_metric="epoch", summary="min")
    wandb.define_metric("val/neg_corr", step_metric="epoch", summary="min")

    # Prepare checkpoint directory
    ckpt_dir = Path("checkpoints") / wandb.run.id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save the model’s initial state
    save_initial_state(
        model,
        ckpt_dir / "initial_model.pt",
        ckpt_dir / "initial_random_state.pt",
    )

    # Watch the model
    wandb.watch(model, "val/neg_corr", log_graph=True, log_freq=100)

    # Optimizer & scheduler created outside the loop
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Persist the config YAML to checkpoints directory
    config_path = Path("checkpoints") / wandb.run.id / "config.yaml"
    config.save(config_path)

    # ── Train and validate ──
    best_val_epoch, model, full_loader, final_step = train_loop(
        model, optimizer, scheduler, ckpt_dir, config
    )

    # Retrain the model on the full dataset for the best validation epoch
    retrain_full(model, full_loader, config, best_val_epoch, ckpt_dir, final_step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        wandb.alert(title="Run crashed", text=str(e))
        raise
