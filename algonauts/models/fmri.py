import numpy as np
import torch
import torch.nn as nn
from algonauts.models.rope import PredictionTransformerRoPE

# ────────────────────────────────────────────────────────────
# Stand‑alone projection block
#   – keeps projection logic separate from fusion logic
# ────────────────────────────────────────────────────────────
class MultiModalProjector(nn.Module):
    """
    Maps raw modality‑specific feature dims → shared hidden_dim.
    Kept as an independent module so we can re‑use or swap it out
    without touching ModalityFusionTransformer.
    """
    def __init__(self, input_dims: dict, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.projections = nn.ModuleDict({
            mod: self._build_projection(d_in, hidden_dim, num_layers)
            for mod, d_in in input_dims.items()
        })

    @staticmethod
    def _build_projection(input_dim: int, output_dim: int, num_layers: int):
        """Linear → GELU stack ending at output_dim."""
        layers = []
        dims = np.linspace(input_dim, output_dim, num_layers + 1, dtype=int)
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            inputs:  {modality: (B, T, D_in)}
        Returns:
            dict with the same keys, each value (B, T, hidden_dim)
        """
        return {m: self.projections[m](inputs[m]) for m in self.projections}


class ModalityFusionTransformer(nn.Module):
    def __init__(
        self,
        modalities,
        subject_count=4,
        hidden_dim=1024,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.3,
        subject_dropout_prob=0.0,
        fuse_mode: str = "concat",
        use_transformer: bool = True,
        use_run_embeddings: bool = False,
    ):
        super().__init__()
        self.fuse_mode = fuse_mode
        self.hidden_dim = hidden_dim
        self.modalities = modalities

        self.subject_dropout_prob = subject_dropout_prob
        self.subject_embeddings = nn.Embedding(subject_count + 1, hidden_dim)

        self.use_run_embeddings = use_run_embeddings
        if self.use_run_embeddings:
            self.run_embeddings = nn.Embedding(3, hidden_dim)
        self.null_subject_index = subject_count

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
                dropout=dropout_rate,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else: 
            self.transformer = nn.Identity()

    def forward(self, inputs: dict, subject_ids, run_ids):
        # inputs are already projected – just stack in the preset modality order
        keys = self.modalities
        projected = [inputs[k] for k in keys]
        x = torch.stack(projected, dim=2)  # (B, T, num_modalities, D)

        B, T, _, _ = x.shape

        subject_ids = torch.tensor(subject_ids, device=x.device, dtype=torch.long)
        run_ids = torch.tensor(run_ids, device=x.device, dtype=torch.long)
        if self.training and self.subject_dropout_prob > 0:
            drop_mask = (
                torch.rand(subject_ids.size(0), device=subject_ids.device)
                < self.subject_dropout_prob
            )
            subject_ids = subject_ids.clone()
            subject_ids[drop_mask] = self.null_subject_index

        subj_emb = self.subject_embeddings(subject_ids).unsqueeze(1).unsqueeze(2)
        subj_emb = subj_emb.expand(-1, T, 1, -1)

        if self.use_run_embeddings:
            run_emb = self.run_embeddings(run_ids).unsqueeze(1).unsqueeze(2)
            run_emb = run_emb.expand(-1, T, 1, -1)

        x = torch.cat([x, subj_emb, run_emb], dim=2) if self.use_run_embeddings else torch.cat([x, subj_emb], dim=2)
        x = x.view(B * T, x.shape[2], -1)

        fused = self.transformer(x)

        if self.fuse_mode == "concat":
            fused = fused.view(B * T, -1)
        elif self.fuse_mode == "mean":
            fused = fused.mean(dim=1)

        fused = fused.view(B, T, -1)
        return fused
    
    @property
    def output_dim(self):
        """
        Returns:
            Output dimension after fusion.
            For concat mode: hidden_dim * (num_modalities + 1 + use_run_embeddings)
            For mean mode: hidden_dim
        """
        if self.fuse_mode == "concat":
            return self.hidden_dim * (len(self.modalities) + 1 + int(self.use_run_embeddings))
        elif self.fuse_mode == "mean":
            return self.hidden_dim
        else:
            raise ValueError(f"Unknown fuse_mode: {self.fuse_mode}")


class FMRIModel(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        *,
        # fusion‑stage hyper‑params
        fusion_hidden_dim=256,
        fusion_layers=1,
        fusion_heads=4,
        fusion_dropout=0.3,
        subject_dropout_prob=0.0,
        use_fusion_transformer=True,
        use_run_embeddings=False,
        proj_layers=1,
        fuse_mode="concat",
        subject_count=4,
        # temporal predictor hyper‑params
        pred_layers=3,
        pred_heads=8,
        pred_dropout=0.3,
        rope_pct=1.0,
        # padding
        n_prepend_zeros=10,
        # training
        mask_prob=0.2,
    ):
        """
        FMRIModel combines modality fusion and temporal prediction.

        Args:
            input_dims (dict): Mapping modality names to input dimensions.
            output_dim (int): Output feature dimension.
            fusion_hidden_dim (int): Hidden dimension for fusion transformer.
            fusion_layers (int): Number of layers in fusion transformer.
            fusion_heads (int): Number of attention heads in fusion transformer.
            fusion_dropout (float): Dropout rate in fusion transformer.
            subject_dropout_prob (float): Probability to drop subject embedding during training.
            use_fusion_transformer (bool): Whether to use transformer for fusion stage.
            proj_layers (int): Number of layers in modality projection.
            fuse_mode (str): Fusion mode, e.g., "concat" or "mean".
            subject_count (int): Number of subjects.
            pred_layers (int): Number of layers in prediction transformer.
            pred_heads (int): Number of attention heads in prediction transformer.
            pred_dropout (float): Dropout rate in prediction transformer.
            rope_pct (float): Percentage parameter for RoPE positional encoding.
            mask_prob (float): Probability to mask input during training.
        """
        super().__init__()

        self.projector = MultiModalProjector(
            input_dims,
            hidden_dim=fusion_hidden_dim,
            num_layers=proj_layers,
        )

        self.encoder = ModalityFusionTransformer(
            modalities=list(input_dims.keys()),
            subject_count=subject_count,
            hidden_dim=fusion_hidden_dim,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            dropout_rate=fusion_dropout,
            subject_dropout_prob=subject_dropout_prob,
            fuse_mode=fuse_mode,
            use_transformer=use_fusion_transformer,
            use_run_embeddings=use_run_embeddings,
        )

        self.predictor = PredictionTransformerRoPE(
            input_dim=self.encoder.output_dim,
            output_dim=output_dim,
            num_layers=pred_layers,
            num_heads=pred_heads,
            dropout=pred_dropout,
            rope_pct=rope_pct,
        )

        self.n_prepend_zeros = n_prepend_zeros
        self.mask_prob = mask_prob

    def forward(self, features, subject_ids, run_ids, attention_mask):

        # attention masking logic
        if self.training and self.mask_prob > 0:
            mask = (
                torch.rand(attention_mask.shape, device=attention_mask.device)
                < self.mask_prob
            )
            attention_mask = attention_mask.clone()
            attention_mask[mask] = False

        # Project first, then fuse
        projected_features = self.projector(features)
        fused = self.encoder(projected_features, subject_ids, run_ids)

        # Prepend zeros to fused
        prepended_zeros = torch.zeros(
            fused.shape[:-2] + (self.n_prepend_zeros, fused.shape[-1]),
            device=fused.device,
            dtype=fused.dtype,
        )
        fused = torch.cat((prepended_zeros, fused), dim=-2)

        # Extend attention_mask for prepended zeros (set to False/masked)
        attention_mask_pre = torch.zeros(
            attention_mask.shape[:-1] + (self.n_prepend_zeros,),
            device=attention_mask.device,
            dtype=torch.bool,
        )
        attention_mask = torch.cat((attention_mask_pre, attention_mask), dim=-1)

        preds = self.predictor(fused, attention_mask)

        # Remove the appended zeros from preds
        preds = preds[..., self.n_prepend_zeros:preds.shape[-2], :] 

        return preds
