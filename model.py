import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityFusionTransformer(nn.Module):
    def __init__(
        self,
        input_dims,
        subject_count=4,
        hidden_dim=1024,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.3,
        subject_dropout_prob=0.2,
        fuse_mode: str = "concat",
        use_transformer: bool = True
    ):
        super().__init__()
        self.fuse_mode = fuse_mode
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, (hidden_dim + dim)  // 2),
                    nn.LeakyReLU(),
                    nn.Linear((hidden_dim + dim)  // 2, hidden_dim),
                )
                for modality, dim in input_dims.items()
            }
        )

        self.subject_dropout_prob = subject_dropout_prob
        self.subject_embeddings = nn.Embedding(subject_count + 1, hidden_dim)
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

    def forward(self, inputs: dict, subject_ids):
        B, T, _ = next(iter(inputs.values())).shape

        projected = [self.projections[name](inputs[name]) for name in self.projections]
        x = torch.stack(projected, dim=2)  # (B, T, num_modalities, D)

        subject_ids = torch.tensor(subject_ids, device=x.device, dtype=torch.long)
        if self.training and self.subject_dropout_prob > 0:
            drop_mask = (
                torch.rand(subject_ids.size(0), device=subject_ids.device)
                < self.subject_dropout_prob
            )
            subject_ids = subject_ids.clone()
            subject_ids[drop_mask] = self.null_subject_index

        subj_emb = self.subject_embeddings(subject_ids).unsqueeze(1).unsqueeze(2)
        subj_emb = subj_emb.expand(-1, T, 1, -1)

        x = torch.cat([x, subj_emb], dim=2)  # (B, T, num_modalities+1, D)
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
        input_dims,  # dict like {"audio": 128, "video_clip": 512, ...}
        output_dim,
        subject_count=4,
        hidden_dim=256,
        max_len=500,
        mask_prob=0.2,
        fuse_mode="concat",
        use_hrf_conv=False
    ):
        super().__init__()
        self.encoder = ModalityFusionTransformer(
            input_dims, subject_count, hidden_dim=hidden_dim, fuse_mode=fuse_mode
        )

        fused_dim = (
            hidden_dim * (len(input_dims) + 1)
            if fuse_mode == "concat"
            else hidden_dim
        )

        self.predictor = PredictionLSTM(
            input_dim=fused_dim,
            output_dim=output_dim,
        )

        self.use_hrf_conv = use_hrf_conv

        if use_hrf_conv:
            self.hrf_conv = nn.Conv1d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=5,
                padding=0,
                groups=output_dim,
            )
        else:
            self.hrf_conv = nn.Identity()

        self.mask_prob = mask_prob

    def forward(self, features, subject_ids, attention_mask):
        if self.training and self.mask_prob > 0:
            mask = (
                torch.rand(attention_mask.shape, device=attention_mask.device)
                < self.mask_prob
            )
            attention_mask = attention_mask.clone()
            attention_mask[mask] = False

        fused = self.encoder(features, subject_ids)
        preds = self.predictor(fused, attention_mask)

        if self.use_hrf_conv:
            preds = preds.transpose(1, 2)
            preds = F.pad(preds, (4, 0))
            preds = self.hrf_conv(preds)
            preds = preds.transpose(1, 2)

        return preds
