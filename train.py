import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import random
import numpy as np

from data import FMRI_Dataset, split_dataset_by_season, collate_fn
from model import FMRIModel


# ------------------- Utility Functions -------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_negative_pearson_loss(pred, target, mask, eps=1e-8):
    mask = mask.unsqueeze(-1)
    pred = pred * mask
    target = target * mask

    pred_mean = pred.sum(dim=1) / (mask.sum(dim=1) + eps)
    target_mean = target.sum(dim=1) / (mask.sum(dim=1) + eps)

    pred_centered = pred - pred_mean.unsqueeze(1)
    target_centered = target - target_mean.unsqueeze(1)

    numerator = (pred_centered * target_centered * mask).sum(dim=1)
    denominator = torch.sqrt(((pred_centered**2 * mask).sum(dim=1)) *
                             ((target_centered**2 * mask).sum(dim=1)) + eps)

    corr = numerator / (denominator + eps)
    return -corr.mean()


def log_model_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({'model/total_params': total_params})


# ------------------- Training & Evaluation -------------------

def run_epoch(loader, model, optimizer, loss_fn, device, is_train=True, global_step=None):
    epoch_loss = 0.0
    model.train() if is_train else model.eval()

    for batch in loader:
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        text = batch['text'].to(device)
        subj_ids = batch['subject_ids']
        fmri = batch['fmri'].to(device)
        attn_mask = batch['attention_masks'].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(audio, video, text, subj_ids, attn_mask)
            loss = loss_fn(pred, fmri, attn_mask)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if global_step is not None:
                    wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, step=global_step)
                    global_step += 1

        epoch_loss += loss.item()

    return epoch_loss / len(loader), global_step


def train():
    # --- WandB Init ---
    wandb.init(project="fmri-model", config={
        "epochs": 150,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": "cuda:1",
        "early_stop_patience": 3
    })
    config = wandb.config

    # --- Dataset ---
    norm_stats = torch.load("normalization_stats.pt")
    ds = FMRI_Dataset("fmri", "Features/Audio",
                      "Features/Visual/InternVideo/features_chunk1.49_len6_before6_frames120_imgsize224",
                      "Features/Text",
                      )
    train_ds, valid_ds = split_dataset_by_season(ds, val_season="6", train_noise_std=0.0)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=8)

    # --- Model ---
    model = FMRIModel({'audio': 2048, 'video': 512, 'text': 2048}, 1000, subject_count=4, max_len=600)
    model.to(config.device)
    log_model_params(model)
    save_initial_state(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    best_val_epoch = 0

    for epoch in range(config.epochs):
        train_loss, global_step = run_epoch(train_loader, model, optimizer,
                                            masked_negative_pearson_loss,
                                            config.device, is_train=True, global_step=global_step)

        val_loss, _ = run_epoch(valid_loader, model, optimizer,
                                masked_negative_pearson_loss,
                                config.device, is_train=False)

        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": train_loss,
            "avg_val_loss": val_loss,
        })

        scheduler.step()

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pt')
            wandb.save('best_model.pt')
            print(f"✅ Saved new best model at epoch {epoch + 1}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{config.early_stop_patience}")

        if patience_counter >= config.early_stop_patience:
            print("⏹️ Early stopping triggered")
            break

    wandb.run.summary["best_val_pearson"] = best_val_loss

    # --- Retraining on full dataset ---
    print("🔁 Reloading initial model and retraining from scratch on full dataset...")
    model = FMRIModel({'audio': 2048, 'video': 512, 'text': 2048}, 1000, subject_count=4, max_len=600)
    model.to(config.device)
    load_initial_state(model)

    full_loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=8)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_val_epoch)
    global_step = 0
    for epoch in range(best_val_epoch):
        full_loss, _ = run_epoch(full_loader, model, optimizer,
                                 masked_negative_pearson_loss, config.device, is_train=True)
        wandb.log({"full_dataset_loss": full_loss, "retrain_epoch": epoch + 1})
        scheduler.step()
        print(f"🔁 Retrain Epoch {epoch + 1}: Full Dataset Loss = {full_loss:.4f}")

    torch.save(model.state_dict(), 'final_model.pt')
    wandb.save('final_model.pt')
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
