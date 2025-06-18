import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gamma
from Ropegat import PredictionTransformerRoPE
from torch_geometric.nn.models import GAT
#from graph_models import pad_to_fixed_length
#from graph_models import *

def spm_hrf(tr: float, size: int):
    length = tr * size
    t = np.arange(0, length, tr)

    peak1 = gamma.pdf(t, 6)
    peak2 = gamma.pdf(t, 16)
    hrf = peak1 - 0.5 * peak2
    return hrf / np.sum(hrf)


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
        num_layers_projection: int = 1
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

        projected = [self.projections[name](inputs[name]) for name in self.projections]
        x = torch.stack(projected, dim=2)  # (B, T, num_modalities, D)

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


class FixedPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=600):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class PredictionTransformer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        output_dim=1000,
        num_layers=2,
        num_heads=16,
        max_len=500,
        dropout=0.3,
    ):
        super().__init__()
        self.pos_encoder = FixedPositionalEncoding(input_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=input_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(input_dim, output_dim)

    def forward(self, x, attn_mask):
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        device = x.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=~attn_mask)
        return self.output_head(x)

class PredictionLSTM(nn.Module):
    def __init__(self, input_dim, output_dim=1000, num_layers=2, dropout=0.3):
        super().__init__()
        hidden_dim = input_dim*2
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, attn_mask):
        # Pack the sequence to handle variable-length sequences
        lengths = attn_mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=attn_mask.size(1)  # <<< Fix
        )
        return self.output_head(output)




class FMRIModel(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        *,
        adjacency_matrix = None, #TODO: make default none case to equal no graph layer
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
        num_pre_tokens: int = 5,
        #Graph concolver
        pad_length = 650, # this has to be larger than any time-series in the data
        # HRF-related
        use_hrf_conv=False,
        learn_hrf=False,
        n_prepend_zeros=10,
        hrf_size=8,
        tr_sec=1.49,
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
            adjacency_matrix =adjacency_matrix
        )

        #self.graph_convolver = FixedNetworkGraphAttention(dim = pad_length,adjacency_matrix=adjacency_matrix)

        self.n_prepend_zeros = n_prepend_zeros
        
        self.use_hrf_conv = use_hrf_conv
        self.learn_hrf = learn_hrf
        self.hrf_size = hrf_size

        if use_hrf_conv:
            self.hrf_conv = nn.Conv1d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=self.hrf_size,
                padding=0,
                groups=output_dim,
                bias=False,
            )
            with torch.no_grad():
                hrf = spm_hrf(tr=tr_sec, size=self.hrf_size)
                # reshape to (output_dim, 1, kernel_size) and broadcast
                hrf_weight = torch.tensor(hrf).view(1, 1, -1).repeat(output_dim, 1, 1)
                self.hrf_conv.weight.copy_(hrf_weight)
            self.hrf_conv.weight.requires_grad = learn_hrf
            self.register_buffer("hrf_prior", hrf_weight.clone().detach())

        else:
            self.hrf_conv = nn.Identity()
    
        self.mask_prob = mask_prob
        self.pad_length = pad_length

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

        # Extend attention_mask for prepended zeros (set to False/masked)
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

            # extend attention mask with valid (=True) entries
            prefix_mask = torch.ones(
                B,
                self.num_pre_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, T_p+T]


        preds = self.predictor(fused, attention_mask)

       # preds,padded = pad_to_fixed_length(preds,self.pad_length)

        #preds = preds+ self.graph_convolver(preds)
        #preds = preds[:,padded:]

        if self.num_pre_tokens > 0:
            preds = preds[:, self.num_pre_tokens :, :] 

        if self.use_hrf_conv:
            preds = preds.transpose(1, 2)
            preds = F.pad(preds, (self.hrf_size - 1, 0))
            preds = self.hrf_conv(preds)
            preds = preds.transpose(1, 2)

        # Remove the appended zeros from preds
        preds = preds[..., num_pre_post_timepoints : preds.shape[-2],:] 
        #preds = preds[..., num_pre_post_timepoints : preds.shape[-2] - num_pre_post_timepoints, :]


        return preds
