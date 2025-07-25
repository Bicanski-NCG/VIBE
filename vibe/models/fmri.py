import numpy as np
import torch
import torch.nn as nn
from vibe.models.rope import PredictionTransformerRoPE


class ModalityFusionTransformer(nn.Module):
    def __init__(
        self,
        input_dims,
        subject_count=4,
        hidden_dim=1024,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.3,
        subject_dropout_prob=0.0,
        fuse_mode: str = "concat",
        use_transformer: bool = True,
        use_run_embeddings: bool = False,
        num_layers_projection: int = 1,
    ):
        super().__init__()
        self.fuse_mode = fuse_mode
        self.projections = nn.ModuleDict({
            modality: self.build_projection(dim, hidden_dim, num_layers_projection)
            for modality, dim in input_dims.items()
        })

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

    def build_projection(self, input_dim, output_dim, num_layers):
        layers = []
        dims = np.linspace(input_dim, output_dim, num_layers + 1, dtype=int)

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU())

        return nn.Sequential(*layers)

    def forward(self, inputs: dict, subject_ids, run_ids):
        B, T, _ = next(iter(inputs.values())).shape

        projected_dict = {
            name: self.projections[name](inputs[name])
            for name in self.projections
        }

        projected = [projected_dict[name] for name in self.projections]
        x = torch.stack(projected, dim=2)

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
        num_pre_tokens: int = 0,
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
            use_hrf_conv (bool): Whether to use HRF convolution on outputs.
            learn_hrf (bool): Whether HRF convolution weights are learnable.
            hrf_size (int): Kernel size for HRF convolution.
            tr_sec (float): Repetition time in seconds for HRF.
            mask_prob (float): Probability to mask input during training.
        """
        super().__init__()
        self.encoder = ModalityFusionTransformer(
            input_dims,
            subject_count=subject_count,
            hidden_dim=fusion_hidden_dim,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            dropout_rate=fusion_dropout,
            subject_dropout_prob=subject_dropout_prob,
            fuse_mode=fuse_mode,
            use_transformer=use_fusion_transformer,
            use_run_embeddings=use_run_embeddings,
            num_layers_projection=proj_layers,
        )

        fused_dim = (
            fusion_hidden_dim * (len(input_dims) + 1 + int(use_run_embeddings))
            if fuse_mode == "concat"
            else fusion_hidden_dim
        )

        self.num_pre_tokens = int(num_pre_tokens)
        if self.num_pre_tokens > 0:
            # (T_p, D) where T_p == num_pre_tokens
            self.pre_tokens = nn.Parameter(
                torch.randn(self.num_pre_tokens, fused_dim) * 0.02
            )
        else:
            # register a placeholder so .to(device) works later
            self.register_buffer("pre_tokens", torch.empty(0, fused_dim))


        self.predictor = PredictionTransformerRoPE(
            input_dim=fused_dim,
            output_dim=output_dim,
            num_layers=pred_layers,
            num_heads=pred_heads,
            dropout=pred_dropout,
            rope_pct=rope_pct,
        )

        self.n_prepend_zeros = n_prepend_zeros
            
        self.mask_prob = mask_prob

    def forward(self, features, subject_ids, run_ids, attention_mask):
        num_pre_post_timepoints = self.n_prepend_zeros

        # Original attention_mask masking logic
        if self.training and self.mask_prob > 0:
            mask = (
                torch.rand(attention_mask.shape, device=attention_mask.device)
                < self.mask_prob
            )
            attention_mask = attention_mask.clone()
            attention_mask[mask] = False

        fused = self.encoder(features, subject_ids, run_ids)

        # Prepend zeros to fused
        # TODO: experiment what happens if we put torch.randn here instead of zeros
        zeros_pre_fused = torch.zeros(
            fused.shape[:-2] + (num_pre_post_timepoints, fused.shape[-1]),
            device=fused.device,
            dtype=fused.dtype,
        )
        fused = torch.cat((zeros_pre_fused, fused), dim=-2)

        attention_mask_pre = torch.zeros(
            attention_mask.shape[:-1] + (num_pre_post_timepoints,),
            device=attention_mask.device,
            dtype=torch.bool,
        )
        attention_mask = torch.cat((attention_mask_pre, attention_mask), dim=-1)

        if self.num_pre_tokens > 0:
            B = fused.size(0)
            prefix = self.pre_tokens.unsqueeze(0).expand(B, -1, -1)   # [B, T_p, D]
            fused = torch.cat([prefix, fused], dim=1)                 # [B, T_p+T, D]

            prefix_mask = torch.ones(
                B,
                self.num_pre_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, T_p+T]

        preds = self.predictor(fused, attention_mask)

        if self.num_pre_tokens > 0:
            preds = preds[:, self.num_pre_tokens:, :] 

        # Remove the appended zeros from preds
        preds = preds[..., num_pre_post_timepoints:preds.shape[-2], :] 

        return preds
