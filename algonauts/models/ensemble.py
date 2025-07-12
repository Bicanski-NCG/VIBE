import random
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
    def __init__(self, models, device="cuda", normalize=False, norm_dim=(1, 2), eps=1e-8):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()
        self.normalize = normalize
        self.norm_dim = norm_dim
        self.eps = eps
        for model in models:
            # Load the model from checkpoint, move to device, set eval mode
            model.eval()
            self.models.append(model)
            # if len(self.models) > 20:
            #     # If we have many models, move them to CPU to save VRAM
            #     model.cpu()

    def _row_normalise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per‑batch normalisation over the specified `norm_dim` axes.
        x: [B, T, V]
        Returns a tensor with the same shape, zero‑mean and unit‑var
        along time+voxel dims for each batch row.
        """
        if not self.normalize:
            return x
        mu  = x.mean(dim=self.norm_dim, keepdim=True)
        std = x.std(dim=self.norm_dim, keepdim=True, unbiased=False).clamp(min=self.eps)
        return (x - mu) / std

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
            # Ensure model is on the correct device
            #model.to(self.device)
            device = next(model.parameters()).device
            features = {m: f.to(device) for m, f in features.items()}
            attention_mask = attention_mask.to(device)
            pred = model(features, subject_ids, run_ids, attention_mask)
            pred = self._row_normalise(pred).to(self.device)          # [B, T, V] normalised
            preds.append(pred)
            #if len(self.models) > 7:
                # Send models to CPU to same VRAM
                #model.cpu()
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
        self.device = torch.device(f"cuda:{random.randrange(torch.cuda.device_count())}") if torch.cuda.is_available() else torch.device("cpu")


        # Preload one model per unique epoch in roi_to_epoch
        self.roi_to_epoch = roi_to_epoch
        self.roi_labels = np.array(roi_labels, dtype="<U20")
        self.epochs = sorted(set(roi_to_epoch.values()))
        self.models = {}
        for e in self.epochs:
            ckpt_path = ckpt_dir / f"epoch_{e}_final_model.pt"
            model, _ = load_model_from_ckpt(str(ckpt_path), ckpt_dir / "config.yaml")
            model = model.to(self.device).eval()
            self.models[e] = model


    @torch.no_grad()
    def forward(self, features, subject_ids, run_ids, attention_mask):
        """
        Run each required model, then stitch outputs voxel‐wise.
        Returns [B, T, V].
        """
        # 1) Run all needed models once:
        preds = {}  # maps epoch → [B,T,V] tensor
        features = {m: f.to(self.device) for m, f in features.items()}
        attention_mask = attention_mask.to(self.device)
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