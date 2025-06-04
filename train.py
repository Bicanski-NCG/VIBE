import argparse
import yaml
import random
import wandb
from pathlib import Path
import json

def _deep_update(a: dict, b: dict):
    "Recursively merge b into a and return the result."
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import FMRI_Dataset, split_dataset_by_season, collate_fn
from model import FMRIModel
from losses import (
    masked_negative_pearson_loss,
    sample_similarity_loss,
    roi_similarity_loss,
)
from utils import save_initial_state, load_initial_state, set_seed, log_model_params


def load_config():
    """
    Parse CLI arguments and load YAML configuration files.
    Precedence:
        1) CLI --features / --params / --local override default paths
        2) params.yaml provides defaults
        3) local.yaml (optional & git‑ignored) can override any key
    Returns
    -------
    features        : dict, modality → path
    input_dims      : dict, modality → dim
    modality_keys   : list[str]
    cfg             : merged dict with keys: data, model, optim, train, paths, wandb
    """
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-f", "--features", default="config/features.yaml")
    parser.add_argument("-p", "--params",   default="config/params.yaml")
    parser.add_argument("-l", "--local",    default="config/local.yaml")
    args = parser.parse_args()

    # choose / set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
    set_seed(seed)

    # ------- load YAML files -------
    with open(args.features) as f:
        feat_yaml = yaml.safe_load(f)
    with open(args.params) as f:
        cfg = yaml.safe_load(f)            # base config
    if Path(args.local).exists():
        with open(args.local) as f:
            _deep_update(cfg, yaml.safe_load(f))   # local overrides

    # unpack feature dict
    features      = feat_yaml["features"]
    input_dims    = feat_yaml["input_dims"]
    modality_keys = list(input_dims.keys())

    # inject the seed so W&B sees it
    cfg["seed"] = seed
    return features, input_dims, modality_keys, cfg


def get_data_loaders(features, input_dims, modality_keys, cfg):
    """Instantiate datasets and DataLoaders for training, validation, and full retraining."""
    # Assume normalization stats have been precomputed
    norm_stats = torch.load(cfg["paths"]["normalization_stats"])

    data_dir = cfg["paths"]["fmri"]
    batch_size = cfg["data"]["batch_size"]
    noise_std  = cfg["data"].get("train_noise_std", 0.0)
    use_normalization = cfg["data"].get("use_normalization", False)
    oversample_factor = cfg["data"].get("oversample_factor", 1)
    num_workers = cfg["data"].get("num_workers", 8)
    val_name   = cfg["data"]["val_name"]

    ds = FMRI_Dataset(
        data_dir,
        feature_paths=features,
        input_dims=input_dims,
        modalities=modality_keys,
        noise_std=noise_std,
        normalization_stats=norm_stats if use_normalization else None,
        oversample_factor=oversample_factor
    )
    print(f"Dataset size: {len(ds)} samples")
    # val_name like "s06" → val_season="6"
    val_season = val_name[1:] if val_name.startswith("s") else val_name
    train_ds, valid_ds = split_dataset_by_season(
        ds, val_season=val_season, train_noise_std=0.0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    full_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, full_loader


def build_model(input_dims, cfg, device):
    """Instantiate the FMRIModel and move to device."""
    model = FMRIModel(
        input_dims,
        1000,
        fuse_mode=cfg["model"]["fuse_mode"],
        hidden_dim=cfg["model"]["hidden_dim"],
        subject_count=4,
        use_hrf_conv=cfg["model"].get("use_hrf_conv", False),
        learn_hrf=cfg["model"].get("learn_hrf", False),
    )
    model.to(device)
    return model


def create_optimizer_and_scheduler(model, cfg):
    """Create optimizer with param groups and scheduler, handling HRF conv if enabled."""
    optim_cfg = cfg["optim"]
    if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
        hrf_params = [model.hrf_conv.weight]
        other_params = [p for n, p in model.named_parameters() if n != "hrf_conv.weight"]
        param_groups = [
            {"params": hrf_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": optim_cfg["weight_decay"]},
        ]
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": optim_cfg["weight_decay"]}]

    optimizer = optim.AdamW(param_groups, lr=optim_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optim_cfg["epochs"]
    )
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
        subject_ids = batch["subject_ids"]
        fmri = batch["fmri"].to(device)
        attn_mask = batch["attention_masks"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(features, subject_ids, attn_mask)
            negative_corr_loss = masked_negative_pearson_loss(pred, fmri, attn_mask)
            sample_loss = sample_similarity_loss(pred, fmri, attn_mask)
            roi_loss = roi_similarity_loss(pred, fmri, attn_mask)
            mse_loss = nn.functional.mse_loss(pred, fmri)
            loss = (
                negative_corr_loss
                + config["data"]["lambda_sample"] * sample_loss
                + config["data"]["lambda_roi"] * roi_loss
                + config["data"]["lambda_mse"] * mse_loss
            )
            if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
                hrf_dev = model.hrf_conv.weight - model.hrf_prior
                loss += config["model"].get("lambda_hrf", 0.0) * hrf_dev.norm(p=2)

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
        + config["data"]["lambda_sample"] * (epoch_sample_loss / len(loader))
        + config["data"]["lambda_roi"] * (epoch_roi_loss / len(loader))
        + config["data"]["lambda_mse"] * (epoch_mse_loss / len(loader))
    )
    return (
        total_loss,
        epoch_negative_corr_loss / len(loader),
        epoch_sample_loss / len(loader),
        epoch_roi_loss / len(loader),
        epoch_mse_loss / len(loader),
    ), global_step


def train_loop(features, input_dims, modality_keys, cfg):
    """Full training pipeline including early stopping. Returns best_val_epoch, ckpt_dir, model, and full_loader."""
    # Initialize W&B
    wandb.init(project=cfg["wandb"]["project"], config=cfg)
    run_id = wandb.run.id
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    device = cfg.get("device", "cuda:0") if cfg.get("device","auto")!="auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare DataLoaders
    train_loader, valid_loader, full_loader = get_data_loaders(
        features, input_dims, modality_keys, cfg
    )

    # Build model and save initial state
    model = build_model(input_dims, cfg, device)
    log_model_params(model)
    save_initial_state(
        model,
        ckpt_dir / "initial_model.pt",
        ckpt_dir / "initial_random_state.pt",
    )

    # Optimizer & scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    best_val_epoch = 0

    for epoch in range(1, cfg["optim"]["epochs"] + 1):
        train_losses, global_step = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            is_train=True,
            global_step=global_step,
            config=cfg,
        )
        val_losses, _ = run_epoch(
            valid_loader,
            model,
            optimizer,
            device,
            is_train=False,
            global_step=None,
            config=cfg,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_losses[0],
                "val_loss": val_losses[0],
                "train_neg_corr": train_losses[1],
                "val_neg_corr": val_losses[1],
                "train_sample": train_losses[2],
                "val_sample": val_losses[2],
                "train_roi": train_losses[3],
                "val_roi": val_losses[3],
                "train_mse": train_losses[4],
                "val_mse": val_losses[4],
                "lr": optimizer.param_groups[0]["lr"]
                if len(optimizer.param_groups) == 1
                else optimizer.param_groups[1]["lr"],
            },
            step=global_step,
        )

        scheduler.step()
        current_val = val_losses[1]  # negative correlation as primary metric
        print(f"Epoch {epoch}: Train NegCorr = {train_losses[1]:.4f}, Val NegCorr = {current_val:.4f}")

        if current_val < best_val_loss:
            best_val_loss = current_val
            best_val_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            wandb.save(str(ckpt_dir / "best_model.pt"))
            print(f"✅ Saved new best model at epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{cfg['optim']['early_stop_patience']}")
            if patience_counter >= cfg["optim"]["early_stop_patience"]:
                print("⏹️ Early stopping triggered")
                break

    wandb.run.summary["best_val_pearson"] = best_val_loss
    return best_val_epoch, ckpt_dir, model, full_loader, global_step


def retrain_full(model, full_loader, cfg, best_val_epoch, ckpt_dir, start_step):
    """Retrain the model from initial state on the full dataset for best_val_epoch epochs."""
    print("🔁 Reloading initial model and retraining on full dataset...")
    load_initial_state(
        model,
        ckpt_dir / "initial_model.pt",
        ckpt_dir / "initial_random_state.pt",
    )

    device = cfg.get("device", "cuda:0") if cfg.get("device","auto")!="auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)
    global_step = start_step

    for epoch in range(1, best_val_epoch + 1):
        full_losses, global_step = run_epoch(
            full_loader,
            model,
            optimizer,
            device,
            is_train=True,
            global_step=global_step,
            config=cfg,
        )

        wandb.log(
            {
                "retrain_epoch": epoch,
                "full_loss": full_losses[0],
                "full_neg_corr": full_losses[1],
                "full_sample": full_losses[2],
                "full_roi": full_losses[3],
                "full_mse": full_losses[4],
                "lr": optimizer.param_groups[0]["lr"]
                if len(optimizer.param_groups) == 1
                else optimizer.param_groups[1]["lr"],
            },
            step=global_step,
        )

        scheduler.step()
        print(f"Epoch {epoch}: Full NegCorr = {full_losses[1]:.4f}")

    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    wandb.save(str(ckpt_dir / "final_model.pt"))
    print("✅ Final model trained on full dataset and saved.")


def main():
    features, input_dims, modality_keys, cfg = load_config()
    best_val_epoch, ckpt_dir, model, full_loader, final_step = train_loop(
        features, input_dims, modality_keys, cfg
    )
    retrain_full(model, full_loader, cfg, best_val_epoch, ckpt_dir, final_step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        wandb.alert(title="Run crashed", text=str(e))
        raise
