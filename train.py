import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import random
import numpy as np

from data import FMRI_Dataset, split_dataset_by_name, collate_fn, make_balanced_weights
from model import FMRIModel
from losses import (
    masked_negative_pearson_loss,
    sample_similarity_loss,
    roi_similarity_loss
)
from torch import nn

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
    "text": "Features/Text/Qwen3B_tr1.49_len60_before50",
    "fast_res3_act": "Features/Visual/SlowFast_R101_tr1.49/fast_res3_act",
    "fast_stem_act": "Features/Visual/SlowFast_R101_tr1.49/fast_stem_act",
    "pool_concat": "Features/Visual/SlowFast_R101_tr1.49/pool_concat",
    "slow_res3_act": "Features/Visual/SlowFast_R101_tr1.49/slow_res3_act",
    "slow_res5_act": "Features/Visual/SlowFast_R101_tr1.49/slow_res5_act",
    "slow_stem_act": "Features/Visual/SlowFast_R101_tr1.49/slow_stem_act",
    "audio_long_contrext": "Features/Audio/Wave2Vec2/features_chunk1.49_len60_before50",
    # "audio_mfcc_mono": "Features/Audio/LowLevel/_chunk1.49_len4.0_before2.0_nmfcc32_nstats4/mono/movies/",
    # "audio_mfcc_stereo": "Features/Audio/LowLevel/_chunk1.49_len4.0_before2.0_nmfcc32_nstats4/stereo/movies/",

}

input_dims = {
    "aud_last": 1280 * 2,
    "aud_ln_post": 1280 * 2,
    #"conv3d_features": 1280 * 2,
    #"vis_block5": 1280 * 2,
    "vis_block8": 1280 * 2,
    #"vis_block12": 1280 * 2,
    "vis_merged": 2048 * 2,
    #"thinker_12": 2048 * 2,
    "thinker_24": 2048 * 2,
    #"thinker_36": 2048 * 2,
    "text": 2048,
    #"fast_res3_act": 2048,
    #"fast_stem_act": 1024,
    "pool_concat": 9216,
    "slow_res3_act": 4096,
    #"slow_res5_act": 4096,
    #"slow_stem_act": 8192,
    "audio_long_contrext": 2048,
    # "audio_mfcc_mono":int(4*32),
    # "audio_mfcc_stereo":int(4*32)

}

modality_keys = list(input_dims.keys())


# ------------------- Utility Functions -------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_model_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"model/total_params": total_params})


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
):
    epoch_negative_corr_loss = 0.0
    epoch_sample_loss = 0.0
    epoch_roi_loss = 0.0
    epoch_mse_loss = 0.0
    model.train() if is_train else model.eval()

    for batch in loader:
        features = {k: batch[k].to(device) for k in modality_keys}
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


def train():
    # --- WandB Init ---
    wandb.init(
        project="fmri-model",
        config={
            "epochs": 150,
            "batch_size": 4,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "device": "cuda:0",
            "early_stop_patience": 1,
            "lambda_sample": 0,
            "lambda_roi": 0,
            "lambda_mse": 0.01,
            "fuse_mode": "concat",
            "hidden_dim": 256,
        },
    )
    config = wandb.config

    # --- Dataset ---
    norm_stats = torch.load("normalization_stats.pt")
    ds = FMRI_Dataset(
        "fmri",
        feature_paths=feature_paths,
        normalize_bold=True
    )
    train_ds, valid_ds = split_dataset_by_name(
        ds, val_name="bourne", train_noise_std=0.0,
        normalize_validation_bold=True
    )

    train_weights = make_balanced_weights(train_ds)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=8,
        prefetch_factor=4,
        #persistent_workers=True,
        #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        prefetch_factor=4,
        #persistent_workers=True,
        #pin_memory=True
    )

    # --- Model ---
    model = FMRIModel(
        input_dims,
        1000,
        fuse_mode=config.fuse_mode,
        hidden_dim=config.hidden_dim,
        subject_count=4,
    )
    model.to(config.device)
    log_model_params(model)
    save_initial_state(model)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
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
            torch.save(model.state_dict(), "best_model.pt")
            wandb.save("best_model.pt")
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
    )
    model.to(config.device)
    load_initial_state(model)

    full_loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
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

    torch.save(model.state_dict(), "final_model.pt")
    wandb.save("final_model.pt")
    print("✅ Final model trained on full dataset and saved.")


def save_initial_state(model, path="initial_model.pt"):
    torch.save(model.state_dict(), path)
    random_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(random_state, "initial_random_state.pt")


def load_initial_state(model, path="initial_model.pt"):
    model.load_state_dict(torch.load(path))
    random_state = torch.load("initial_random_state.pt", weights_only=False)
    random.setstate(random_state["random"])
    np.random.set_state(random_state["numpy"])
    torch.set_rng_state(random_state["torch"])
    if torch.cuda.is_available() and random_state["cuda"] is not None:
        torch.cuda.set_rng_state_all(random_state["cuda"])


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        wandb.alert(title="Run crashed", text=str(e))
        raise
