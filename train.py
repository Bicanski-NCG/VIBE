import argparse
import yaml
import random
import wandb
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import FMRI_Dataset, split_dataset_by_name, collate_fn, make_group_weights
from model import FMRIModel
from losses import (
    masked_negative_pearson_loss,
    sample_similarity_loss,
    roi_similarity_loss
)
from utils import save_initial_state, load_initial_state, set_seed, log_model_params


def load_config():
    """Parse CLI arguments and load YAML configuration files."""
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, a random seed will be chosen.",
    )
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        default="config/features.yaml",
        help="Path to the YAML file with feature paths",
    )
    parser.add_argument(
        "--params",
        "-p",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML file with training configuration",
    )
    args = parser.parse_args()

    # Determine final seed: use provided one or generate a new random seed
    if args.seed is None:
        chosen_seed = random.SystemRandom().randint(0, 2**32 - 1)
        print(f"No --seed provided; using generated seed={chosen_seed}")
    else:
        chosen_seed = args.seed
        print(f"Using user-provided seed={chosen_seed}")
    set_seed(chosen_seed)

    # Load feature paths and input dimensions
    with open(args.features, "r") as f:
        feature_dict = yaml.safe_load(f)
        features = feature_dict["features"]
        input_dims = feature_dict["input_dims"]
        data_dir = feature_dict.get("data_dir", "data/fmri")
        modality_keys = list(input_dims.keys())

    # Load training hyperparameters
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)
        train_params = params["train"]

    # Log the chosen seed in the config for W&B metadata
    train_params["seed"] = chosen_seed

    return features, input_dims, modality_keys, train_params, data_dir


def get_data_loaders(features, input_dims, modality_keys, config, data_dir):
    """Instantiate datasets and DataLoaders for training, validation, and full retraining."""
    # Assume normalization stats have been precomputed
    norm_stats = torch.load("normalization_stats.pt")

    ds = FMRI_Dataset(data_dir,
                      feature_paths=features,
                      input_dims=input_dims,
                      modalities=modality_keys,
                      noise_std=config.get("train_noise_std", 0.0),
                      normalization_stats=norm_stats if config.get("use_normalization", False) else None,
                      oversample_factor=config.get("oversample_factor", 1))
    
    print(f"Dataset size: {len(ds)} samples")

    train_ds, valid_ds = split_dataset_by_name(
        ds, val_name=config.get("val_name", "s06"), train_noise_std=0.0,
        normalize_validation_bold=config.get("normalize_validation_bold", False),
    )

    if config.get("stratification_variable", False):
        train_weights = make_group_weights(train_ds, filter_on=config["stratification_variable"])
        print(f"Using stratification variable: {config['stratification_variable']}")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
    else:
        train_weights = torch.ones(len(train_ds), dtype=torch.float32)

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=sampler if config.get("stratification_variable", False) else None,
        shuffle=False if config.get("stratification_variable", False) else True,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 8),
        prefetch_factor=config.get("prefetch_factor", 4),
        persistent_workers=config.get("persistent_workers", True),
        pin_memory=config.get("pin_memory", True),
    )
    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 8),
        prefetch_factor=config.get("prefetch_factor", 4),
        persistent_workers=config.get("persistent_workers", True),
        pin_memory=config.get("pin_memory", True),
    )

    full_loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 8),
        prefetch_factor=config.get("prefetch_factor", 4),
        persistent_workers=config.get("persistent_workers", True),
        pin_memory=config.get("pin_memory", True),
    )

    return train_loader, valid_loader, full_loader


def build_model(input_dims, config, device):
    """Instantiate the FMRIModel and move to device."""
    model = FMRIModel(
        input_dims,
        1000,
        fuse_mode=config["fuse_mode"],
        hidden_dim=config["hidden_dim"],
        subject_count=4,
        use_hrf_conv=config.get("use_hrf_conv", False),
        learn_hrf=config.get("learn_hrf", False),
    )
    model.to(device)
    return model


def create_optimizer_and_scheduler(model, config):
    """Create optimizer with param groups and scheduler, handling HRF conv if enabled."""
    if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
        hrf_params = [model.hrf_conv.weight]
        other_params = [p for n, p in model.named_parameters() if n != "hrf_conv.weight"]
        param_groups = [
            {"params": hrf_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": config["weight_decay"]},
        ]
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": config["weight_decay"]}]

    optimizer = optim.AdamW(param_groups, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
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
                + config["lambda_sample"] * sample_loss
                + config["lambda_roi"] * roi_loss
                + config["lambda_mse"] * mse_loss
            )
            if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
                hrf_dev = model.hrf_conv.weight - model.hrf_prior
                loss += config.get("lambda_hrf", 0.0) * hrf_dev.norm(p=2)

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
        + config["lambda_sample"] * (epoch_sample_loss / len(loader))
        + config["lambda_roi"] * (epoch_roi_loss / len(loader))
        + config["lambda_mse"] * (epoch_mse_loss / len(loader))
    )
    return (
        total_loss,
        epoch_negative_corr_loss / len(loader),
        epoch_sample_loss / len(loader),
        epoch_roi_loss / len(loader),
        epoch_mse_loss / len(loader),
    ), global_step


def train_loop(features, input_dims, modality_keys, config, data_dir):
    """Full training pipeline including early stopping. Returns best_val_epoch, ckpt_dir, model, and full_loader."""
    # Initialize W&B
    wandb.init(project="fmri-model", config=config)
    run_id = wandb.run.id
    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Prepare DataLoaders
    train_loader, valid_loader, full_loader = get_data_loaders(
        features, input_dims, modality_keys, config, data_dir
    )

    # Build model and save initial state
    model = build_model(input_dims, config, config["device"])
    log_model_params(model)
    save_initial_state(
        model,
        ckpt_dir / "initial_model.pt",
        ckpt_dir / "initial_random_state.pt",
    )

    # Optimizer & scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    best_val_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        train_losses, global_step = run_epoch(
            train_loader,
            model,
            optimizer,
            config["device"],
            is_train=True,
            global_step=global_step,
            config=config,
        )
        val_losses, _ = run_epoch(
            valid_loader,
            model,
            optimizer,
            config["device"],
            is_train=False,
            global_step=None,
            config=config,
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
            print(f"Patience {patience_counter}/{config['early_stop_patience']}")
            if patience_counter >= config["early_stop_patience"]:
                print("⏹️ Early stopping triggered")
                break

    wandb.run.summary["best_val_pearson"] = best_val_loss
    return best_val_epoch, ckpt_dir, model, full_loader, global_step


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
            config["device"],
            is_train=True,
            global_step=global_step,
            config=config,
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
    features, input_dims, modality_keys, train_params, data_dir = load_config()
    best_val_epoch, ckpt_dir, model, full_loader, final_step = train_loop(
        features, input_dims, modality_keys, train_params, data_dir
    )
    retrain_full(model, full_loader, train_params, best_val_epoch, ckpt_dir, final_step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        wandb.alert(title="Run crashed", text=str(e))
        raise
