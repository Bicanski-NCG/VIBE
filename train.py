import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from data import FMRI_Dataset, split_dataset_by_season, compute_mean_std, collate_fn
from model import FMRIModel

# --- WandB Init ---
wandb.init(project="fmri-model", config={
    "epochs": 150,
    "batch_size": 4,
    "lr": 1e-4,
    "device": "cuda:1"
})
config = wandb.config

# --- Dataset Split ---
# Initial dataset for access
ds = FMRI_Dataset("fmri", "Features/Audio", "Features/Visual/InternVideo/features_chunk1.49_len9_before6_frames120_imgsize224", "Features/Text")

# Split datasets
train_ds, valid_ds = split_dataset_by_season(ds, val_season="6")
train_samples = train_ds.samples
val_samples = valid_ds.samples


# Compute normalization statistics
norm_stats = torch.load('normalization_stats.pt')
# Reinitialize with normalization
train_ds = FMRI_Dataset(ds.root_folder, ds.audio_feature_path, ds.video_feature_path, ds.text_feature_path,
                        noise_std=0.0,)

valid_ds = FMRI_Dataset(ds.root_folder, ds.audio_feature_path, ds.video_feature_path, ds.text_feature_path,
                        noise_std=0.0,)

train_ds.samples = train_samples
valid_ds.samples = val_samples
# Loaders
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

# --- Loss ---
def masked_negative_pearson_loss(pred, target, mask, eps=1e-8):
    """
    pred:   (B, T, V)
    target: (B, T, V)
    mask:   (B, T) - boolean tensor (True = valid, False = pad)
    """
    mask = mask.unsqueeze(-1)  # (B, T, 1)
    pred = pred * mask
    target = target * mask

    pred_mean = pred.sum(dim=1) / (mask.sum(dim=1) + eps)
    target_mean = target.sum(dim=1) / (mask.sum(dim=1) + eps)

    pred_centered = pred - pred_mean.unsqueeze(1)
    target_centered = target - target_mean.unsqueeze(1)

    numerator = (pred_centered * target_centered * mask).sum(dim=1)
    denominator = torch.sqrt(((pred_centered**2 * mask).sum(dim=1)) * ((target_centered**2 * mask).sum(dim=1)) + eps)

    corr = numerator / (denominator + eps)
    return -corr.mean()



# --- Model ---
model = FMRIModel({'audio': 2048, 'video': 512, 'text': 2048}, 1000, subject_count=4, max_len=600)
device = config.device
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)


# --- Additional Early Stopping Config ---
patience = 3
patience_counter = 0

# --- Training ---
global_step = 0
best_val_loss = float('inf')

for epoch in range(config.epochs):
    model.train()
    train_loss_epoch = 0

    for batch in train_loader:
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        text = batch['text'].to(device)
        subj_ids = batch['subject_ids']
        fmri = batch['fmri'].to(device)
        attn_mask = batch['attention_masks'].to(device)

        optimizer.zero_grad()
        pred = model(audio, video, text, subj_ids, attn_mask)
        loss = masked_negative_pearson_loss(pred, fmri, attn_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        train_loss_epoch += loss.item()
        wandb.log({"train_loss": loss.item()}, step=global_step)
        global_step += 1

    # --- Validation ---
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for batch in valid_loader:
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            text = batch['text'].to(device)
            subj_ids = batch['subject_ids']
            fmri = batch['fmri'].to(device)
            attn_mask = batch['attention_masks'].to(device)

            pred = model(audio, video, text, subj_ids, attn_mask)
            val_loss = masked_negative_pearson_loss(pred, fmri, attn_mask)
            val_loss_epoch += val_loss.item()

    avg_train_loss = train_loss_epoch / len(train_loader)
    avg_val_loss = val_loss_epoch / len(valid_loader)


    wandb.log({
        "epoch": epoch + 1,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
    })

    scheduler.step()

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # --- Save best model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        wandb.save('best_model.pt')
        print(f"✅ Saved new best model at epoch {epoch + 1} with val loss: {best_val_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Early stopping patience: {patience_counter}/{patience}")

    # --- Early stopping check ---
    if patience_counter >= patience:
        print("⏹️ Early stopping triggered")
        break

# --- Save best val loss ---
wandb.run.summary["best_val_pearson"] = best_val_loss

# --- Retrain Best Model on Full Dataset ---
print("🔁 Retraining best model on the full dataset...")
model.load_state_dict(torch.load('best_model.pt'))

# Combine train and test sets
full_loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

# Reinitialize optimizer if needed
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

model.train()
for epoch in range(2):  # Feel free to tune this (1–5 epochs is usually good)
    full_loss = 0
    for batch in full_loader:
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        text = batch['text'].to(device)
        subj_ids = batch['subject_ids']
        fmri = batch['fmri'].to(device)
        attn_mask = batch['attention_masks'].to(device)

        optimizer.zero_grad()
        pred = model(audio, video, text, subj_ids, attn_mask)
        loss = masked_negative_pearson_loss(pred, fmri, attn_mask)
        loss.backward()
        optimizer.step()

        full_loss += loss.item()

    avg_full_loss = full_loss / len(full_loader)
    print(f"🔁 Retrain Epoch {epoch+1}: Full Dataset Loss = {avg_full_loss:.4f}")


torch.save(model.state_dict(), 'final_model.pt')
wandb.save('final_model.pt')
print("✅ Final model trained on full dataset and saved.")
