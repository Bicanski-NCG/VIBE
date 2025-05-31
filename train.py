import argparse, yaml, random, wandb
from pathlib import Path
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


# ------------------- Training & Evaluation -------------------


def run_epoch(
    loader,
    model,
    optimizer,
    device,
    is_train=True,
    global_step=None,
    lambda_sample=1.0,
    lambda_roi=1.0,
    lambda_mse=1.0,
    lambda_hrf=1e-3,
):
    epoch_negative_corr_loss = 0.0
    epoch_sample_loss = 0.0
    epoch_roi_loss = 0.0
    epoch_mse_loss = 0.0
    model.train() if is_train else model.eval()

    for batch in loader:
        features = {k: batch[k].to(device) for k in MODALITY_KEYS}
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
                + lambda_sample * sample_loss
                + lambda_roi * roi_loss
                + lambda_mse * mse_loss
            )
            if model.use_hrf_conv and model.learn_hrf:
                hrf_dev = model.hrf_conv.weight - model.hrf_prior
                loss += lambda_hrf * hrf_dev.norm(p=2)

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

    epoch_negative_corr_loss /= len(loader)
    epoch_sample_loss /= len(loader)
    epoch_roi_loss /= len(loader)
    epoch_mse_loss /= len(loader)

    epoch_loss = (
        epoch_negative_corr_loss
        + lambda_sample * epoch_sample_loss
        + lambda_roi * epoch_roi_loss
        + lambda_mse * epoch_mse_loss
    )
    return (
        epoch_loss,
        epoch_negative_corr_loss,
        epoch_sample_loss,
        epoch_roi_loss,
        epoch_mse_loss,
    ), global_step


def train(features, input_dims, modality_keys, train_params, data_dir):
    global MODALITY_KEYS
    MODALITY_KEYS = modality_keys
    # --- WandB Init ---
    wandb.init(
        project="fmri-model",
        config=train_params,
    )
    config = wandb.config
    run_id   = wandb.run.id            # e.g. “soft-armadillo-18”
    ckpt_dir = Path("checkpoints")/run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    norm_stats = torch.load("normalization_stats.pt")
    ds = FMRI_Dataset(
        data_dir,
        feature_paths=features,
    )
    train_ds, valid_ds = split_dataset_by_season(
        ds, val_season="6", train_noise_std=0.0
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
    )

    # --- Model ---
    model = FMRIModel(
        input_dims,
        1000,
        fuse_mode=config.fuse_mode,
        hidden_dim=config.hidden_dim,
        subject_count=4,
        use_hrf_conv=config.use_hrf_conv,
        learn_hrf=config.learn_hrf,
    )
    model.to(config.device)
    log_model_params(model)
    save_initial_state(model, f"{ckpt_dir}/initial_model.pt", f"{ckpt_dir}/initial_random_state.pt")

    if config.use_hrf_conv:
        hrf_params   = [model.hrf_conv.weight]                       # keep it as a list
        other_params = [p for n, p in model.named_parameters()
                        if n != "hrf_conv.weight"]
    
        param_groups = [
            {"params": hrf_params,   "weight_decay": 0.0},
            {"params": other_params, "weight_decay": config.weight_decay},
        ]
    else:
        param_groups = [
            {
                "params": model.parameters(),
                "weight_decay": config.weight_decay,
            }
        ]

    optimizer = optim.AdamW(
        param_groups, lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    best_val_epoch = 0

    for epoch in range(config.epochs):
        train_loss_tuple, global_step = run_epoch(
            train_loader,
            model,
            optimizer,
            config.device,
            is_train=True,
            global_step=global_step,
            lambda_sample=config.lambda_sample,
            lambda_roi=config.lambda_roi,
            lambda_mse=config.lambda_mse,
        )

        val_loss_tuple, _ = run_epoch(
            valid_loader,
            model,
            optimizer,
            config.device,
            is_train=False,
            lambda_sample=config.lambda_sample,
            lambda_roi=config.lambda_roi,
            lambda_mse=config.lambda_mse,
        )

        wandb.log(
            {
                "epoch_loss_train": train_loss_tuple[0],
                "epoch_loss_valid": val_loss_tuple[0],
                "epoch_loss_train_neg_corr": train_loss_tuple[1],
                "epoch_loss_valid_neg_corr": val_loss_tuple[1],
                "epoch_loss_train_sample": train_loss_tuple[2],
                "epoch_loss_valid_sample": val_loss_tuple[2],
                "epoch_loss_train_roi": train_loss_tuple[3],
                "epoch_loss_valid_roi": val_loss_tuple[3],
                "epoch_loss_train_mse": train_loss_tuple[4],
                "epoch_loss_valid_mse": val_loss_tuple[4],
            },
            step=global_step,
        )

        scheduler.step()
        val_loss = val_loss_tuple[1]
        print(
            f"Epoch {epoch + 1}: Train Neg Corr Loss = {train_loss_tuple[1]:.4f}, Val Neg Corr Loss = {val_loss_tuple[1]:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pt")
            wandb.save(f"{ckpt_dir}/best_model.pt")
            print(f"✅ Saved new best model at epoch {epoch + 1}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Early stopping patience: {patience_counter}/{config.early_stop_patience}"
            )

        if patience_counter >= config.early_stop_patience:
            print("⏹️ Early stopping triggered")
            break

    wandb.run.summary["best_val_pearson"] = best_val_loss

    # --- Retraining on full dataset ---
    print("🔁 Reloading initial model and retraining from scratch on full dataset...")
    model = FMRIModel(
        input_dims,
        1000,
        fuse_mode=config.fuse_mode,
        hidden_dim=config.hidden_dim,
        subject_count=4,
        use_hrf_conv=config.use_hrf_conv,
        learn_hrf=config.learn_hrf,
    )
    model.to(config.device)
    load_initial_state(model, f"{ckpt_dir}/initial_model.pt", f"{ckpt_dir}/initial_random_state.pt")

    full_loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    if config.use_hrf_conv:
        hrf_params   = [model.hrf_conv.weight]                       # keep it as a list
        other_params = [p for n, p in model.named_parameters()
                        if n != "hrf_conv.weight"]
    
        param_groups = [
            {"params": hrf_params,   "weight_decay": 0.0},
            {"params": other_params, "weight_decay": config.weight_decay},
        ]
    else:
        param_groups = [
            {
                "params": model.parameters(),
                "weight_decay": config.weight_decay,
            }
        ]
    optimizer = optim.AdamW(
        param_groups, lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=best_val_epoch
    )
    global_step = 0
    for epoch in range(best_val_epoch):
        full_loss_tuple, _ = run_epoch(
            full_loader,
            model,
            optimizer,
            config.device,
            is_train=True,
            lambda_sample=config.lambda_sample,
            lambda_roi=config.lambda_roi,
            lambda_mse=config.lambda_mse,
        )

        wandb.log(
            {
                "full_dataset_loss": full_loss_tuple[0],
                "full_dataset_loss_neg_corr": full_loss_tuple[1],
                "full_dataset_loss_sample": full_loss_tuple[2],
                "full_dataset_loss_roi": full_loss_tuple[3],
                "full_dataset_loss_mse": full_loss_tuple[4],
                "retrain_epoch": epoch + 1,
            }
        )

        scheduler.step()
        print(
            f"Epoch {epoch + 1}: Full Dataset Neg Corr Loss = {full_loss_tuple[1]:.4f}"
        )

    torch.save(model.state_dict(), f"{ckpt_dir}/final_model.pt")
    wandb.save(f"{ckpt_dir}/final_model.pt")
    print("✅ Final model trained on full dataset and saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training enrypoint")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--features", "-f", type=str, default="config/features.yaml",
                        help="Path to the YAML file with feature paths")
    parser.add_argument(
        "--params", "-p", type=str, default="config/params.yaml",
        help="Path to the YAML file with training configuration"
    )
    args = parser.parse_args()
    with open(args.features, "r") as f:
        feature_dict = yaml.safe_load(f)
        features = feature_dict["features"]
        input_dims = feature_dict["input_dims"]
        data_dir = feature_dict.get("data_dir", "data/fmri")
        modality_keys = list(features.keys())
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)
        train_params = params["train"]
        

    try:
        train(features=features, input_dims=input_dims, modality_keys=modality_keys, 
              train_params=train_params, data_dir=data_dir)
    except Exception as e:
        wandb.alert(title="Run crashed", text=str(e))
        raise
