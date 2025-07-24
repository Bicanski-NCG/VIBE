import torch
import torch.nn as nn
import wandb
import numpy as np

from algonauts.utils import logger
from algonauts.training.losses import (
    masked_negative_pearson_loss,
    sample_similarity_loss,
    roi_similarity_loss,
    spatial_regularizer_loss
)
from algonauts.utils.adjacency_matrices import get_laplacians
from algonauts.utils.viz import load_and_label_atlas, voxelwise_pearsonr


def get_network_mask (target_networks, roi_masks):
    mask = torch.zeros(1000)
    for net in target_networks:
        mcop = roi_masks[net].copy().astype(int)
        mask+= mcop

    return mask.bool()


def run_epoch(loader, model, optimizer, device, is_train, laplacians, config, network_mask=None):
    """Run one training or validation epoch and return loss components."""
    if is_train:
        logger.info("📈 Training...")
    else:
        logger.info("📉 Validation epoch...")

    spatial_laplacian, network_laplacian = laplacians
    spatial_laplacian = spatial_laplacian.to(config.device)
    network_laplacian = network_laplacian.to(config.device)
    network_mask = network_mask.to(config.device) if network_mask is not None else None

    epoch_negative_corr_loss = 0.0
    epoch_sample_loss = 0.0
    epoch_roi_loss = 0.0
    epoch_mse_loss = 0.0
    epoch_spatial_adjacency_loss = 0.0
    epoch_network_adjacency_loss = 0.0

    model.train() if is_train else model.eval()

    all_preds, all_true = [], []
    counter = 0
    for batch in loader:
        counter += 1
        features = {k: batch[k].to(device, non_blocking=True) for k in loader.dataset.modalities}

        subject_ids = batch["subject_ids"]
        run_ids = batch["run_ids"]
        fmri = batch["fmri"].to(device)
        attn_mask = batch["attention_masks"].to(device)
    
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(features, subject_ids, run_ids,attn_mask)
            # pred (B,T,V)
            # fmri (B,T,V)
            
                
            negative_corr_loss = masked_negative_pearson_loss(pred, fmri, attn_mask, network_mask=network_mask)
            
            sample_loss = sample_similarity_loss(pred, fmri, attn_mask)
            roi_loss = roi_similarity_loss(pred, fmri, attn_mask)
            if config.normalize_pred_for_spatial_regularizer:
                normalized_pred = pred/torch.linalg.norm(pred,dim=-2,keepdim= True)
            else:
                normalized_pred = pred

            spatial_adjacency_loss = spatial_regularizer_loss(normalized_pred, spatial_laplacian, mask=network_mask)
            network_adjacency_loss = spatial_regularizer_loss(normalized_pred, network_laplacian, mask=network_mask)
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
                if counter % 10:
                    wandb.log(
                        {
                            "train_neg_corr_loss": negative_corr_loss.item(),
                            "train_sample_loss": sample_loss.item(),
                            "train_roi_loss": roi_loss.item(),
                            "train_mse_loss": mse_loss.item(),
                            "train_spatial_adjacency_loss": spatial_adjacency_loss.item(),
                            "train_network_adjacency_loss": network_adjacency_loss.item(),
                            "train_loss": loss.item(),
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
        epoch_spatial_adjacency_loss / len(loader),
        epoch_network_adjacency_loss / len(loader),
        all_preds, all_true
    )


def train_val_loop(model, optimizer, scheduler, train_loader, valid_loader, ckpt_dir, config):
    """Full training pipeline including early stopping. Returns best_val_epoch."""

    best_val_loss = float("inf")
    global_patience_counter = 0
    best_val_epoch = 0
    laplacians = get_laplacians(config.spatial_sigma)

    group_masker = load_and_label_atlas(valid_loader.dataset.samples[0]["subject_atlas"], yeo_networks=7)

    roi_to_scores = {}
    roi_to_epoch = {}
    labels = np.array(group_masker.labels[1:])
    roi_idxs = {roi: np.argwhere(labels == roi) for roi in labels} # 1: skip background
    roi_masks = {roi: (labels == roi).copy() for roi in labels} # 1: skip background

    network_mask = get_network_mask(config.target_networks, roi_masks)

    for epoch in range(1, config.epochs + 1):

        with logger.step(f"🚀 Epoch {epoch}/{config.epochs} …"):
            *train_losses, _, _ = run_epoch(
                train_loader,
                model,
                optimizer,
                config.device,
                is_train=True,
                laplacians=laplacians,
                config=config,
                network_mask=network_mask
            )
            *val_losses, fmri_pred, fmri_true = run_epoch(
                valid_loader,
                model,
                optimizer,
                config.device,
                is_train=False,
                laplacians=laplacians,
                config=config,
                network_mask=network_mask
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
                    "train/spatial_adjacency_loss": train_losses[5],
                    "train/network_adjacency_loss": train_losses[6],

                    # VAL
                    "val/loss":  val_losses[0],
                    "val/neg_corr": val_losses[1],
                    "val/sample": val_losses[2],
                    "val/roi": val_losses[3],
                    "val/mse": val_losses[4],
                    "val/best_val_loss":best_val_loss,
                    "val/spatial_adjacency_loss": val_losses[5],
                    "val/network_adjacency_loss": val_losses[6],

                    # shared LR
                    "train/lr": optimizer.param_groups[0]["lr"]
                        if len(optimizer.param_groups) == 1
                        else optimizer.param_groups[1]["lr"],
                },
            )

            scheduler.step()
            logger.info(f"🔄 LR stepped → {optimizer.param_groups[0]['lr']:.2e}")
            current_val = val_losses[1]
            logger.info(f"🔎 Train NegCorr = {train_losses[1]:.4f}, Val NegCorr = {current_val:.4f}")

            fmri_true = [x.reshape(-1, x.shape[-1]) for x in fmri_true]
            fmri_pred = [x.reshape(-1, x.shape[-1]) for x in fmri_pred]
            voxelwise_r = voxelwise_pearsonr(
                np.concatenate(fmri_true, axis=0),
                np.concatenate(fmri_pred, axis=0),
            )
           
            roi_scores = {}
            for roi_name, roi_idx in [(name, idx) for name, idx in roi_idxs.items() if name in config.target_networks]:
                roi_r = np.mean(voxelwise_r[roi_idx])
                roi_scores[roi_name] = roi_r

                if roi_name not in roi_to_scores or roi_r > roi_to_scores[roi_name]:
                    roi_to_scores[roi_name] = roi_r
                    roi_to_epoch[roi_name] = epoch
                    logger.info(f"🏆 New best {roi_name} at epoch {epoch}: {roi_r:.4f}")

            wandb.log({"val/roi_scores": roi_scores}, commit=False)

            if epoch in roi_to_epoch.values():
                roi_patience_counter = 0
            else:
                roi_patience_counter += 1
            if current_val < best_val_loss:
                best_val_loss = current_val
                best_val_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                wandb.log({"best_model_path": str(ckpt_dir / "best_model.pt")}, commit=False)
                logger.info(f"💾 Saved new best (global) model at epoch {epoch}")
                global_patience_counter = 0
            else:
                global_patience_counter += 1

            if global_patience_counter >= config.early_stop_patience and (roi_patience_counter >= config.early_stop_patience or not config.save_rois):
                logger.info(f"🛑 Early stopping: patience exhausted at epoch {epoch}.")
                break
            else:
                logger.info(f"Global patience: {global_patience_counter}/{config.early_stop_patience}, "
                            f"ROI patience: {roi_patience_counter}/{config.early_stop_patience}")
                wandb.log({
                    "global_patience": global_patience_counter,
                    "roi_patience": roi_patience_counter,
                }, commit=False)
            
    if config.save_rois:
        roi_names = np.array(group_masker.labels[1:])
        torch.save(roi_names, ckpt_dir / "roi_names.pt")
        torch.save(roi_to_epoch, ckpt_dir / "roi_to_epoch.pt")

    wandb.run.summary["best_val_pearson"] = best_val_loss
    wandb.run.summary["best_val_epoch"] = best_val_epoch
    return best_val_epoch, max(roi_to_epoch.values(), default=0)


def full_loop(model, optimizer, scheduler, full_loader, ckpt_dir, config, best_val_epoch):
    """Retrain the model from initial state on the full dataset for best_val_epoch epochs."""

    laplacians = get_laplacians(config.spatial_sigma)
    group_masker = load_and_label_atlas(full_loader.dataset.samples[0]["subject_atlas"], yeo_networks=7)
    labels = np.array(group_masker.labels[1:])
    roi_idxs = {roi: np.argwhere(labels == roi) for roi in labels} # 1: skip background
    roi_masks = {roi: (labels == roi).copy() for roi in labels} # 1: skip background

    network_mask = get_network_mask(config.target_networks,roi_masks)

    if config.save_rois:
        try:
            roi_to_epoch = torch.load(ckpt_dir / "roi_to_epoch.pt", weights_only=False, map_location="cpu")
        except FileNotFoundError:
            roi_to_epoch = {}
            logger.warning("No ROI to epoch mapping found, will not save per-ROI models.")
        else:
            logger.info(f"ROI to best epoch mapping: {roi_to_epoch}")
    else:
        roi_to_epoch = {}
        logger.info("Not saving per-ROI models, skipping ROI to epoch mapping.")

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
            network_mask=network_mask
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

        if config.save_rois:
            for roi_name, best_epoch in [(name, idx) for name, idx in roi_idxs.items() if name in config.target_networks]:
                if best_epoch == epoch:
                    logger.info(f"💾  Saved {roi_name} final model.......NOT (Borat)")
                    torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}_final_model.pt")

        if epoch == best_val_epoch:
            torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
            logger.info("💾 Saved retrained model.")
            wandb.log({"final_model_path": str(ckpt_dir / "final_model.pt")}, commit=False)

        logger.close_step()
