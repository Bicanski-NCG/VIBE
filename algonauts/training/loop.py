import json
import torch
import torch.nn as nn
import wandb
import numpy as np

from algonauts.utils import logger
from algonauts.training.losses import (
    masked_negative_pearson_loss,
    sample_similarity_loss,
    roi_similarity_loss,
    spatial_regularizer_loss,masked_negative_rv_loss
)
from algonauts.utils.adjacency_matrices import get_laplacians,get_all_network_masks
from algonauts.utils.viz import load_and_label_atlas, roi_table, voxelwise_pearsonr
from algonauts.utils import collect_predictions

network_masks  = get_all_network_masks()

network_schedule = {
1: network_masks['Vis'],
2: network_masks['Vis'],

4: network_masks['SomMot'],
5: network_masks['SomMot'],
7: network_masks['Vis'],
8: network_masks['SomMot'],

9: network_masks['SalVentAttn'],
10: network_masks['DorsAttn'],
11: network_masks['Cont'],
12: network_masks['Default'],

13: network_masks['Limbic'],
14: network_masks['Vis'] + network_masks['SomMot'],
15: network_masks['SomMot']+ network_masks['SalVentAttn'],
16: network_masks['SalVentAttn'] + network_masks['DorsAttn'],
17:  network_masks['DorsAttn'] + network_masks['Cont'],
18: network_masks['Default'],

20: network_masks['Cont'] + network_masks['Limbic'],
21: network_masks['Cont'] + network_masks['Limbic'],
22: network_masks['Cont'] + network_masks['Limbic'],
23: network_masks['Vis'] + network_masks['Cont'] ,
24: network_masks['SomMot'] + network_masks['Limbic'],
26: network_masks['SalVentAttn'] + network_masks['DorsAttn'],
27: network_masks['Vis'],
28: network_masks['SomMot'],

30: network_masks['Vis'],
31: network_masks['SomMot'],

33: network_masks['SalVentAttn'],
34: network_masks['DorsAttn'],
35: network_masks['Cont'],
36: network_masks['Vis'] + network_masks['Cont'] ,

38: network_masks['Default'],
39: network_masks['Limbic'],
40: network_masks['Cont'],
41: network_masks['Vis'],
42: network_masks['SomMot']+ network_masks['Limbic'],

43: network_masks['Vis'],
44: network_masks['SomMot'],

46: network_masks['SalVentAttn'],
46: network_masks['DorsAttn'],
48: network_masks['Cont'],
49: network_masks['Default']+network_masks['SalVentAttn'],
50: network_masks['Limbic'],
51: network_masks['Cont'],

}
stupid = {
1: network_masks['Vis'],
2: network_masks['SomMot'],

3: network_masks['Vis'],
4: network_masks['SomMot'],
15: network_masks['Cont'] + network_masks['Limbic'],

5: network_masks['SalVentAttn'],
8: network_masks['Default'],
10: network_masks['Vis'] + network_masks['Cont'],

6: network_masks['DorsAttn'],
9: network_masks['Limbic'],
10: network_masks['Vis'] + network_masks['SomMot'],
12: network_masks['SalVentAttn'] + network_masks['DorsAttn'] +  network_masks['Vis'],

12: network_masks['SalVentAttn'] + network_masks['Limbic'] +  network_masks['SomMot'],

12: network_masks['Cont'] + network_masks['Limbic'] +  network_masks['Default'],
3: network_masks['Vis'],
4: network_masks['SomMot'],


7: network_masks['Cont'],
8: network_masks['Default'],

10: network_masks['Vis'] + network_masks['SomMot'],
11: network_masks['SomMot']+ network_masks['SalVentAttn'],
12: network_masks['SalVentAttn'] + network_masks['DorsAttn'],
13:  network_masks['DorsAttn'] + network_masks['Cont'],
14: network_masks['Default'],

15: network_masks['Cont'] + network_masks['Limbic'],
16: network_masks['Cont'] + network_masks['Limbic'],
17: network_masks['Cont'] + network_masks['Limbic'],
18: network_masks['Vis'] + network_masks['Cont'] ,
19: network_masks['SomMot'] + network_masks['Limbic'],
20: network_masks['SalVentAttn'] + network_masks['DorsAttn'],
}


def run_epoch(loader, model, optimizer, device, is_train, laplacians, config,network_mask = None):
    """Run one training or validation epoch and return loss components."""
    if is_train:
        logger.info("📈 Training...")
    else:
        logger.info("📉 Validation epoch...")

    spatial_laplacian, network_laplacian = laplacians
    spatial_laplacian = spatial_laplacian.to(config.device)
    network_laplacian = network_laplacian.to(config.device)

    epoch_negative_corr_loss = 0.0
    epoch_sample_loss = 0.0
    epoch_roi_loss = 0.0
    epoch_mse_loss = 0.0
    epoch_spatial_adjacency_loss = 0.0
    epoch_network_adjacency_loss = 0.0

    model.train() if is_train else model.eval()

    all_preds, all_true = [], []
    for batch in loader:
        features = {k: batch[k].to(device, non_blocking=True) for k in loader.dataset.modalities}

        subject_ids = batch["subject_ids"]
        run_ids = batch["run_ids"]
        fmri = batch["fmri"].to(device)
        attn_mask = batch["attention_masks"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(features, subject_ids, run_ids,attn_mask)
            # pred (B,T,V)
            # fmri (B,T,V)
            
            if is_train:
                
                #network_mask = np.ones(1000)
                #for network,network_ids in network_masks.items():

                   # if float(torch.rand(1)) < 0.15:
                   #     network_mask -= network_ids

               #     pred = pred
               #     fmri = fmri
                negative_corr_loss = masked_negative_pearson_loss(pred, fmri, attn_mask,network_mask = network_mask)
            else:
                negative_corr_loss = masked_negative_pearson_loss(pred, fmri, attn_mask)  #& loss_mask
            sample_loss = sample_similarity_loss(pred, fmri, attn_mask)
            roi_loss = roi_similarity_loss(pred, fmri, attn_mask)
            if config.normalize_pred_for_spatial_regularizer:
                normalized_pred = pred/torch.linalg.norm(pred,dim=-2,keepdim= True)
            else:
                normalized_pred = pred

            spatial_adjacency_loss = spatial_regularizer_loss(normalized_pred,spatial_laplacian)
            network_adjacency_loss = spatial_regularizer_loss(normalized_pred,network_laplacian)
            mse_loss = nn.functional.mse_loss(pred[...,network_mask], fmri[...,network_mask])
            loss = (
                negative_corr_loss
                + config.lambda_sample * sample_loss
                + config.lambda_roi * roi_loss
                + config.lambda_mse * mse_loss
                + config.lambda_sp_adj*spatial_adjacency_loss 
                + config.lambda_net_adj*network_adjacency_loss
            )
            if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
                hrf_dev = model.hrf_conv.weight - model.hrf_prior
                loss += config.lambda_hrf * hrf_dev.norm(p=2)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                wandb.log(
                    {
                        "train_neg_corr_loss": negative_corr_loss.item(),
                        "train_sample_loss": sample_loss.item(),
                        "train_roi_loss": roi_loss.item(),
                        "train_mse_loss": mse_loss.item(),
                    },
                )
        
        if not is_train:
            mask = attn_mask.bool()
            all_preds.append((pred[mask]).detach().cpu().numpy())
            all_true.append((fmri[mask]).detach().cpu().numpy())

        epoch_negative_corr_loss += negative_corr_loss.item()
        epoch_sample_loss += sample_loss.item()
        epoch_roi_loss += roi_loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_spatial_adjacency_loss+= spatial_adjacency_loss.item()
        epoch_network_adjacency_loss+= network_adjacency_loss.item()

    total_loss = (
        epoch_negative_corr_loss / len(loader)
        + config.lambda_sample * (epoch_sample_loss / len(loader))
        + config.lambda_roi * (epoch_roi_loss / len(loader))
        + config.lambda_mse * (epoch_mse_loss / len(loader))
        + config.lambda_sp_adj*(epoch_spatial_adjacency_loss/len(loader)) 
        + config.lambda_net_adj*(epoch_network_adjacency_loss/len(loader))
    )
    return (
        total_loss,
        epoch_negative_corr_loss / len(loader),
        epoch_sample_loss / len(loader),
        epoch_roi_loss / len(loader),
        epoch_mse_loss / len(loader),
        all_preds, all_true
    )


def train_val_loop(model, optimizer, scheduler, train_loader, valid_loader, ckpt_dir, config):
    """Full training pipeline including early stopping. Returns best_val_epoch."""

    best_val_loss = float("inf")
    patience_counter = 0
    best_val_epoch = 0
    laplacians = get_laplacians(config.spatial_sigma)

    # Group-level masker for visual diagnostics
    group_masker = load_and_label_atlas(valid_loader.dataset.samples[0]["subject_atlas"], yeo_networks=7)

    # Track ROI validation correlations
    roi_to_scores = {}
    roi_to_epoch = {}
    for epoch in range(1, config.epochs + 1):
        network_mask = network_schedule.get(epoch,None)

        with logger.step(f"🚀 Epoch {epoch}/{config.epochs} …"):
            *train_losses, _, _ = run_epoch(
                train_loader,
                model,
                optimizer,
                config.device,
                is_train=True,
                laplacians=laplacians,
                config=config,
                network_mask = network_mask
            )
            *val_losses, fmri_pred, fmri_true = run_epoch(
                valid_loader,
                model,
                optimizer,
                config.device,
                is_train=False,
                laplacians=laplacians,
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
                    "val/best_val_loss":best_val_loss,
                    # shared LR
                    "train/lr": optimizer.param_groups[0]["lr"]
                        if len(optimizer.param_groups) == 1
                        else optimizer.param_groups[1]["lr"],
                },
            )

            scheduler.step()
            logger.info(f"🔄 LR stepped → {optimizer.param_groups[0]['lr']:.2e}")
            current_val = val_losses[1]  # negative correlation as primary metric
            logger.info(f"🔎 Train NegCorr = {train_losses[1]:.4f}, Val NegCorr = {current_val:.4f}")

            # Go through each ROI and log the best validation correlation (flatten across batch and time)
            fmri_true = [x.reshape(-1, x.shape[-1]) for x in fmri_true]
            fmri_pred = [x.reshape(-1, x.shape[-1]) for x in fmri_pred]
            voxelwise_r = voxelwise_pearsonr(
                np.concatenate(fmri_true, axis=0),
                np.concatenate(fmri_pred, axis=0),
            )
            labels = np.array(group_masker.labels[1:])
            roi_idxs = {roi: np.argwhere(labels == roi) for roi in labels} # 1: skip background
            for roi_name, roi_idx in roi_idxs.items():
                roi_r = np.mean(voxelwise_r[roi_idx])

                # Track best validation scores per ROI
                if roi_name not in roi_to_scores or roi_r > roi_to_scores[roi_name]:
                    roi_to_scores[roi_name] = roi_r
                    roi_to_epoch[roi_name] = epoch
                    logger.info(f"🏆 New best {roi_name} at epoch {epoch}: {roi_r:.4f}")

            wandb.log({"val/roi_scores": roi_to_scores}, commit=False)

            # If any ROI has best validation score this epoch, save the model
            if epoch in roi_to_epoch.values():
                epoch_patience_counter = 0
            else:
                epoch_patience_counter += 1

            # Log group-level Pearson correlation
            if current_val < best_val_loss:
                best_val_loss = current_val
                best_val_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                wandb.log({"best_model_path": str(ckpt_dir / "best_model.pt")}, commit=False)
                logger.info(f"💾 Saved new best (global) model at epoch {epoch}")
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info(f"Global patience: {patience_counter}/{config.early_stop_patience}, "
                        f"ROI patience: {epoch_patience_counter}/{config.early_stop_patience}")
            if patience_counter >= config.early_stop_patience and epoch_patience_counter >= config.early_stop_patience:
                logger.info("🛑 Early stopping: patience exhausted.")
                break
            

    roi_names = np.array(group_masker.labels[1:])
    torch.save(roi_names, ckpt_dir / "roi_names.pt")
    torch.save(roi_to_epoch, ckpt_dir / "roi_to_epoch.pt")

    # Log best validation epoch and scores
    wandb.run.summary["best_val_pearson"] = best_val_loss
    wandb.run.summary["best_val_epoch"] = best_val_epoch
    return best_val_epoch, max(roi_to_epoch.values(), default=0) # Return the global best epoch and max ROI epoch


def full_loop(model, optimizer, scheduler, full_loader, ckpt_dir, config, best_val_epoch):
    """Retrain the model from initial state on the full dataset for best_val_epoch epochs."""

    laplacians = get_laplacians(config.spatial_sigma)

    try:
        roi_to_epoch = torch.load(ckpt_dir / "roi_to_epoch.pt", weights_only=False, map_location="cpu")
    except FileNotFoundError:
        roi_to_epoch = {}
    else:
        logger.info(f"ROI to best epoch mapping: {roi_to_epoch}")

    n_epochs = max(best_val_epoch, *[epoch for epoch in roi_to_epoch.values()], 0)

    for epoch in range(1, n_epochs + 1):
        logger.open_step(f"🚀 Full‑train Epoch {epoch}/{n_epochs} …")

        *full_losses, _, _ = run_epoch(
            full_loader,
            model,
            optimizer,
            config.device,
            is_train=True,
            laplacians=laplacians,
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
        )

        scheduler.step()
        logger.info(f"🔄 LR stepped → {optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"🔎 NegCorr = {full_losses[1]:.4f}")

        for roi_name, best_epoch in roi_to_epoch.items():
            if best_epoch == epoch:
                logger.info(f"💾 Saved {roi_name} final model.")
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}_final_model.pt")

        if epoch == best_val_epoch:
            torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
            logger.info(f"💾 Saved retrained model.")
            wandb.log({"final_model_path": str(ckpt_dir / "final_model.pt")}, commit=False)

        logger.close_step()
