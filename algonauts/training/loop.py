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
    spatial_regularizer_loss,
)
from algonauts.utils.adjacency_matrices import get_laplacians
from algonauts.utils.viz import load_and_label_atlas, roi_table, voxelwise_pearsonr
from algonauts.utils import collect_predictions


def train_step(model, batch, optimizer, laplacians, config):
    """Forward + backward + optimiser step; returns neg-corr for logging."""
    spatial_L, network_L = laplacians
    spatial_L = spatial_L.to(config.device)
    network_L = network_L.to(config.device)
    device = config.device

    # --- unpack ------------------------------------------------------------
    fmri = batch["fmri"].to(device)
    mask = batch["attention_masks"].to(device)
    subj = batch["subject_ids"]
    runid = batch["run_ids"]
    feats = {
        k: batch[k].to(device, non_blocking=True)
        for k in batch
        if k not in ("fmri", "attention_masks", "subject_ids", "run_ids")
    }
    # ­–– forward -----------------------------------------------------------
    pred = model(feats, subj, runid, mask)

    negative_corr_loss = masked_negative_pearson_loss(pred, fmri, mask)
    sample_loss = sample_similarity_loss(pred, fmri, mask)
    roi_loss = roi_similarity_loss(pred, fmri, mask)
    mse_loss = nn.functional.mse_loss(pred, fmri)

    if config.normalize_pred_for_spatial_regularizer:
        pred_for_reg = pred / torch.linalg.norm(pred, dim=-2, keepdim=True)
    else:
        pred_for_reg = pred

    sp_adj = spatial_regularizer_loss(pred_for_reg, spatial_L)
    net_adj = spatial_regularizer_loss(pred_for_reg, network_L)

    loss = (
        negative_corr_loss
        + config.lambda_sample * sample_loss
        + config.lambda_roi * roi_loss
        + config.lambda_mse * mse_loss
        + config.lambda_sp_adj * sp_adj
        + config.lambda_net_adj * net_adj
    )

    # optional HRF L2
    if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
        loss += config.lambda_hrf * (model.hrf_conv.weight - model.hrf_prior).norm()

    # --- backward ----------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    wandb.log(
        {
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss.item(),
            "train_neg_corr_loss": negative_corr_loss.item(),
            "train_sample_loss": sample_loss.item(),
            "train_roi_loss": roi_loss.item(),
            "train_mse_loss": mse_loss.item(),
        },
    )

    return (
        loss.item(),
        negative_corr_loss.item(),
        sample_loss.item(),
        roi_loss.item(),
        mse_loss.item(),
    )


def run_epoch_val(loader, model, device, laplacians, config):
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

    model.eval()

    all_preds, all_true = [], []
    for batch in loader:
        features = {
            k: batch[k].to(device, non_blocking=True) for k in loader.dataset.modalities
        }

        subject_ids = batch["subject_ids"]
        run_ids = batch["run_ids"]
        fmri = batch["fmri"].to(device)
        attn_mask = batch["attention_masks"].to(device)

        with torch.set_grad_enabled(False):
            pred = model(features, subject_ids, run_ids, attn_mask)
            negative_corr_loss = masked_negative_pearson_loss(pred, fmri, attn_mask)
            sample_loss = sample_similarity_loss(pred, fmri, attn_mask)
            roi_loss = roi_similarity_loss(pred, fmri, attn_mask)
            if config.normalize_pred_for_spatial_regularizer:
                normalized_pred = pred / torch.linalg.norm(pred, dim=-2, keepdim=True)
            else:
                normalized_pred = pred

            spatial_adjacency_loss = spatial_regularizer_loss(
                normalized_pred, spatial_laplacian
            )
            network_adjacency_loss = spatial_regularizer_loss(
                normalized_pred, network_laplacian
            )
            mse_loss = nn.functional.mse_loss(pred, fmri)
            loss = (
                negative_corr_loss
                + config.lambda_sample * sample_loss
                + config.lambda_roi * roi_loss
                + config.lambda_mse * mse_loss
                + config.lambda_sp_adj * spatial_adjacency_loss
                + config.lambda_net_adj * network_adjacency_loss
            )
            if getattr(model, "use_hrf_conv", False) and getattr(
                model, "learn_hrf", False
            ):
                hrf_dev = model.hrf_conv.weight - model.hrf_prior
                loss += config.lambda_hrf * hrf_dev.norm(p=2)

        mask = attn_mask.bool()
        all_preds.append((pred[mask]).detach().cpu().numpy())
        all_true.append((fmri[mask]).detach().cpu().numpy())

        epoch_negative_corr_loss += negative_corr_loss.item()
        epoch_sample_loss += sample_loss.item()
        epoch_roi_loss += roi_loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_spatial_adjacency_loss += spatial_adjacency_loss.item()
        epoch_network_adjacency_loss += network_adjacency_loss.item()

    total_loss = (
        epoch_negative_corr_loss / len(loader)
        + config.lambda_sample * (epoch_sample_loss / len(loader))
        + config.lambda_roi * (epoch_roi_loss / len(loader))
        + config.lambda_mse * (epoch_mse_loss / len(loader))
        + config.lambda_sp_adj * (epoch_spatial_adjacency_loss / len(loader))
        + config.lambda_net_adj * (epoch_network_adjacency_loss / len(loader))
    )
    model.train()
    return (
        total_loss,
        epoch_negative_corr_loss / len(loader),
        epoch_sample_loss / len(loader),
        epoch_roi_loss / len(loader),
        epoch_mse_loss / len(loader),
        all_preds,
        all_true,
    )


def train_val_loop(
    model, optimizer, scheduler, train_loader, valid_loader, ckpt_dir, config
):
    laplacians = get_laplacians(config.spatial_sigma)
    global_step = 0
    best_val_loss = float("inf")
    best_val_iter = 0
    patience_counter = 0

    # Group-level masker for visual diagnostics
    group_masker = load_and_label_atlas(
        valid_loader.dataset.samples[0]["subject_atlas"], yeo_networks=7
    )

    # Track ROI validation correlations
    roi_to_scores = {}
    roi_to_iter = {}
    train_losses_total = [0, 0, 0, 0, 0]
    stopping = False
    for epoch in range(1, config.epochs + 1):  # loop until early-stop or hard cap
        if stopping:
            break
        for batch in train_loader:

            # ---------- training step -------------------------------------
            global_step += 1
            train_losses = train_step(model, batch, optimizer, laplacians, config)
            scheduler.step()
            train_losses_total = [
                x + y for x, y in zip(train_losses, train_losses_total)
            ]

            # ---------- periodic validation -------------------------------
            if global_step % config.val_iter_freq == 0:
                *val_losses, fmri_pred, fmri_true = run_epoch_val(
                    valid_loader,
                    model,
                    config.device,
                    laplacians=laplacians,
                    config=config,
                )

                train_losses_total = [
                    x / config.val_iter_freq for x in train_losses_total
                ]

                # logging (shortened)
                wandb.log(
                    {
                        # TRAIN
                        "global_step": global_step,
                        "train/loss": train_losses_total[0],
                        "train/neg_corr": train_losses_total[1],
                        "train/sample": train_losses_total[2],
                        "train/roi": train_losses_total[3],
                        "train/mse": train_losses_total[4],
                        # VAL
                        "val/loss": val_losses[0],
                        "val/neg_corr": val_losses[1],
                        "val/sample": val_losses[2],
                        "val/roi": val_losses[3],
                        "val/mse": val_losses[4],
                        # shared LR
                        "train/lr": (
                            optimizer.param_groups[0]["lr"]
                            if len(optimizer.param_groups) == 1
                            else optimizer.param_groups[1]["lr"]
                        ),
                    },
                )
                current_val = val_losses[1]  # negative correlation as primary metric
                logger.info(
                    f"🔎 Train NegCorr = {train_losses_total[1]:.4f}, Val NegCorr = {current_val:.4f}"
                )
                train_losses_total = [0, 0, 0, 0, 0]
                # Go through each ROI and log the best validation correlation (flatten across batch and time)
                fmri_true = [x.reshape(-1, x.shape[-1]) for x in fmri_true]
                fmri_pred = [x.reshape(-1, x.shape[-1]) for x in fmri_pred]
                voxelwise_r = voxelwise_pearsonr(
                    np.concatenate(fmri_true, axis=0),
                    np.concatenate(fmri_pred, axis=0),
                )
                labels = np.array(group_masker.labels[1:])
                roi_idxs = {
                    roi: np.argwhere(labels == roi) for roi in labels
                }  # 1: skip background
                for roi_name, roi_idx in roi_idxs.items():
                    print(f"Evaluating ROI: {roi_name} with {len(roi_idx)} voxels")
                    roi_r = np.mean(voxelwise_r[roi_idx])

                    # Track best validation scores per ROI
                    if roi_name not in roi_to_scores or roi_r > roi_to_scores[roi_name]:
                        roi_to_scores[roi_name] = roi_r
                        roi_to_iter[roi_name] = global_step
                        logger.info(
                            f"🏆 New best {roi_name} at global step {global_step}: {roi_r:.4f}"
                        )

                wandb.log({"val/roi_scores": roi_to_scores}, commit=False)
                if global_step in roi_to_iter.values():
                    roi_patience_counter = 0
                else:
                    roi_patience_counter += 1
                # -------- early-stopping logic ----------------------------
                if current_val < best_val_loss:
                    best_val_loss = current_val
                    best_val_iter = global_step
                    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                    wandb.log(
                        {"best_model_path": str(ckpt_dir / "best_model.pt")},
                        commit=False,
                    )
                    logger.info(
                        f"💾 Saved new best (global) model at iter {global_step}"
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.info(
                    f"Global patience: {patience_counter}/{config.early_stop_patience}"
                    f"ROI patience: {roi_patience_counter}/{config.early_stop_patience}"
                )

                if (
                    patience_counter >= config.early_stop_patience
                    and roi_patience_counter >= config.early_stop_patience
                ):
                    logger.info("🛑 Early stopping triggered")
                    stopping = True
                    break

    roi_names = np.array(group_masker.labels[1:])
    torch.save(roi_names, ckpt_dir / "roi_names.pt")
    torch.save(roi_to_iter, ckpt_dir / "roi_to_iters.pt")

    # Log best validation epoch and scores
    wandb.run.summary["best_val_pearson"] = best_val_loss
    wandb.run.summary["best_val_iter"] = best_val_iter
    return best_val_iter, max(roi_to_iter.values(), default=0)


def full_loop(
    model, optimizer, scheduler, full_loader, ckpt_dir, config, best_val_iter
):
    """Retrain the model from initial state on the full dataset for best_val_epoch epochs."""

    laplacians = get_laplacians(config.spatial_sigma)
    global_step = 0
    epoch = 0

    try:
        roi_to_iter = torch.load(
            ckpt_dir / "roi_to_iters.pt", weights_only=False, map_location="cpu"
        )
    except FileNotFoundError:
        roi_to_iter = {}
    else:
        logger.info(f"ROI to best iter mapping: {roi_to_iter}")

    n_iters = max(best_val_iter, *[iters for iters in roi_to_iter.values()], 0)
    full_losses_total = [0, 0, 0, 0, 0]
    while global_step < n_iters:
        epoch += 1
        logger.open_step(f"🚀 Full-train Epoch {epoch} for {n_iters} iters")

        for batch in full_loader:
            if global_step >= n_iters:
                break
            global_step += 1
            full_losses = train_step(model, batch, optimizer, laplacians, config)
            full_losses_total = [x + y for x, y in zip(full_losses, full_losses_total)]

            scheduler.step()
            if global_step % config.val_iter_freq == 0:
                full_losses_total = [x / config.val_iter_freq for x in full_losses_total]
                wandb.log(
                    {
                        "retrain_epoch": epoch,
                        "full/loss": full_losses_total[0],
                        "full/neg_corr": full_losses_total[1],
                        "full/sample": full_losses_total[2],
                        "full/roi": full_losses_total[3],
                        "full/mse": full_losses_total[4],
                        "full/lr": (
                            optimizer.param_groups[0]["lr"]
                            if len(optimizer.param_groups) == 1
                            else optimizer.param_groups[1]["lr"]
                        ),
                    },
                )
                logger.info(f"🔄 LR stepped → {optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"🔎 NegCorr = {full_losses_total[1]:.4f}")
                full_losses_total = [0, 0, 0, 0, 0]
                logger.close_step()

            for roi_name, best_iter in roi_to_iter.items():
                if best_iter == global_step:
                    logger.info(f"💾 Saved {roi_name} final model.")
                    torch.save(
                        model.state_dict(),
                        ckpt_dir / f"iter_{global_step}_final_model.pt",
                    )

            if global_step == best_val_iter:
                torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
                logger.info(f"💾 Saved retrained model.")
                wandb.log(
                    {"final_model_path": str(ckpt_dir / "final_model.pt")}, commit=False
                )
