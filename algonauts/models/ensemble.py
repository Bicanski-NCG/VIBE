import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from algonauts.models.utils import load_model_from_ckpt


class EnsembleAverager(nn.Module):
    """
    Ensemble averaging for FMRIModel-compatible models.
    Loads multiple checkpoints and averages their predictions.
    """
    def __init__(self, models, device="cuda"):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()
        for model in models:
            # Load the model from checkpoint, move to device, set eval mode
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def forward(self, features, subject_ids, run_ids, attention_mask):
        """
        Args:
            features: dict of modality tensors, each of shape [B, T, C_i]
            subject_ids: tensor or list of subject IDs of length B
            run_ids: tensor or list of run IDs of length B
            attention_mask: tensor of shape [B, T] for valid timepoints
        Returns:
            Averaged predictions tensor of shape [B, T, V]
        """
        preds = []
        for model in self.models:
            preds.append(model(features, subject_ids, run_ids, attention_mask))
        stacked = torch.stack(preds, dim=0)  # [N_models, B, T, V]
        return stacked.mean(dim=0)           # [B, T, V]


class ROIAdaptiveEnsemble(nn.Module):
    """
    For each ROI selects predictions from the checkpoint
    that had the best validation score on that ROI.
    """
    def __init__(self,
                 roi_labels:   list,       # [V] with ROI indices 0..R-1
                 roi_to_epoch: dict,                 # {roi_idx: epoch_int}
                 ckpt_dir:     Path,
                 device:       str = "cuda"):
        super().__init__()
        self.device = device

        # Preload one model per unique epoch in roi_to_epoch
        self.roi_to_epoch = roi_to_epoch
        self.roi_labels = np.array(roi_labels, dtype="<U20")
        self.epochs = sorted(set(roi_to_epoch.values()))
        self.models = {}
        for e in self.epochs:
            ckpt_path = ckpt_dir / f"epoch_{e}_final_model.pt"
            model, _ = load_model_from_ckpt(str(ckpt_path), ckpt_dir / "config.yaml")
            model.eval()
            self.models[e] = model


    @torch.no_grad()
    def forward(self, features, subject_ids, run_ids, attention_mask):
        """
        Run each required model, then stitch outputs voxel‐wise.
        Returns [B, T, V].
        """
        # 1) Run all needed models once:
        preds = {}  # maps epoch → [B,T,V] tensor
        for e, m in self.models.items():
            preds[e] = m(features, subject_ids, run_ids, attention_mask)  # [B,T,V]

        # 2) Build final output by picking per-voxel predictions
        # We'll create a [B, T, V] tensor by stacking and indexing
        B, T, V = preds[self.epochs[0]].shape
        out = torch.empty((B, T, V), device=self.device)

        # For each ROI index r, mask voxels and copy from preds[e_r]
        for r, e_r in self.roi_to_epoch.items():
            # Create a 1D voxel mask for this ROI
            voxel_mask = (self.roi_labels == r)  # shape [V]
            # Copy predictions for voxels in this ROI
            out[:, :, voxel_mask] = preds[e_r][:, :, voxel_mask]

        return out