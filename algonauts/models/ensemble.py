import torch
import torch.nn as nn

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
