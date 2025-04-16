import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import wandb
from torch import nn

from data import FMRI_Dataset, split_dataset_by_season
from model import FMRIModel

# --- WandB Init ---
wandb.init(project="fmri-model", config={
    "epochs": 150,
    "batch_size": 2,
    "lr": 1e-4,
    "device": "cuda:4"
})
config = wandb.config

# --- Data & Collation ---
def collate_fn(batch):
    subject_ids, audio_feats, video_feats, text_feats, fmri_responses = zip(*batch)

    audio_padded = pad_sequence(audio_feats, batch_first=True, padding_value=0)
    video_padded = pad_sequence(video_feats, batch_first=True, padding_value=0)
    text_padded = pad_sequence(text_feats, batch_first=True, padding_value=0)
    fmri_padded = pad_sequence(fmri_responses, batch_first=True, padding_value=0)

    attention_masks = torch.zeros(audio_padded.shape[:2], dtype=torch.bool)
    for i, length in enumerate([af.shape[0] for af in audio_feats]):
        attention_masks[i, :length] = 1

    return {
        'subject_ids': subject_ids,
        'audio': audio_padded,
        'video': video_padded,
        'text': text_padded,
        'fmri': fmri_padded,
        'attention_masks': attention_masks
    }

# --- Dataset Split ---
ds = FMRI_Dataset("fmri", "Features/Audio", "Features/Visual", "Features/Text")

train_ds, valid_ds = split_dataset_by_season(ds, val_season="6")

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

# --- Loss ---
def negative_pearson_loss(pred, target, eps=1e-8):
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (pred * target).sum(dim=1)
    denominator = torch.sqrt((pred ** 2).sum(dim=1) * (target ** 2).sum(dim=1) + eps)
    corr = numerator / denominator
    return -corr.mean()

def hybrid_loss(pred, target, alpha=0.9):
    pearson = negative_pearson_loss(pred, target)
    mse = nn.functional.mse_loss(pred, target)
    return alpha * pearson + (1 - alpha) * mse, pearson


# --- Model ---
model = FMRIModel({'audio': 2048, 'video': 512, 'text': 2048}, 1000, subject_count=4, max_len=600)
device = config.device
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)


# --- Additional Early Stopping Config ---
patience = 3
patience_counter = 0

# --- Training ---
global_step = 0
best_val_pearson = float('inf')

for epoch in range(config.epochs):
    model.train()
    train_loss_epoch = 0
    train_pearson_epoch = 0

    for batch in train_loader:
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        text = batch['text'].to(device)
        subj_ids = batch['subject_ids']
        fmri = batch['fmri'].to(device)
        attn_mask = batch['attention_masks'].to(device)

        optimizer.zero_grad()
        pred = model(audio, video, text, subj_ids, attn_mask)
        loss, pearson = hybrid_loss(pred, fmri)
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()
        train_pearson_epoch += pearson.item()
        wandb.log({"train_loss": loss.item(), "train_pearson": pearson}, step=global_step)
        global_step += 1

    # --- Validation ---
    model.eval()
    val_loss_epoch = 0
    val_pearson_epoch = 0
    with torch.no_grad():
        for batch in valid_loader:
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            text = batch['text'].to(device)
            subj_ids = batch['subject_ids']
            fmri = batch['fmri'].to(device)
            attn_mask = batch['attention_masks'].to(device)

            pred = model(audio, video, text, subj_ids, attn_mask)
            val_loss, pearson_val = hybrid_loss(pred, fmri)
            val_loss_epoch += val_loss.item()
            val_pearson_epoch += pearson_val.item()

    avg_train_loss = train_loss_epoch / len(train_loader)
    avg_val_loss = val_loss_epoch / len(valid_loader)

    avg_train_pearson = train_pearson_epoch / len(train_loader)
    avg_val_pearson = val_pearson_epoch / len(valid_loader)

    wandb.log({
        "epoch": epoch + 1,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "avg_train_pearson": avg_train_pearson,
        "avg_val_pearson": avg_val_pearson
    })

    scheduler.step()

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # --- Save best model ---
    if avg_val_pearson < best_val_pearson:
        best_val_pearson = avg_val_pearson
        torch.save(model.state_dict(), 'best_model.pt')
        wandb.save('best_model.pt')
        print(f"✅ Saved new best model at epoch {epoch + 1} with val loss: {best_val_pearson:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Early stopping patience: {patience_counter}/{patience}")

    # --- Early stopping check ---
    if patience_counter >= patience:
        print("⏹️ Early stopping triggered")
        break

# --- Save best val loss ---
wandb.run.summary["best_val_pearson"] = best_val_pearson

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
        loss = negative_pearson_loss(pred, fmri)
        loss.backward()
        optimizer.step()

        full_loss += loss.item()

    avg_full_loss = full_loss / len(full_loader)
    print(f"🔁 Retrain Epoch {epoch+1}: Full Dataset Loss = {avg_full_loss:.4f}")


torch.save(model.state_dict(), 'final_model.pt')
wandb.save('final_model.pt')
print("✅ Final model trained on full dataset and saved.")
