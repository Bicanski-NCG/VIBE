import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityFusionTransformer(nn.Module):
    def __init__(self, input_dims, subject_count=4, hidden_dim=1024, num_layers=1, num_heads=4, dropout_rate=0.3, subject_dropout_prob=0.4):
        super().__init__()
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_dim)
            for modality, dim in input_dims.items()
        })

        self.subject_dropout_prob = subject_dropout_prob

        self.subject_embeddings = nn.Embedding(subject_count+1, hidden_dim)
        self.null_subject_index = subject_count  # reserve the last index for "null"

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True, norm_first=True, activation='gelu', dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs: dict, subject_ids):
        """
        inputs: dict with keys matching input_dims, each of shape (B, T, D)
        subject_ids: tensor of shape (B,)
        """
        B, T, _ = next(iter(inputs.values())).shape

        projected = [
            proj(inputs[name])
            for name, proj in self.projections.items()
        ]
        x = torch.stack(projected, dim=2)  # (B, T, num_modalities, D)
        subject_ids = torch.tensor(subject_ids, device=x.device, dtype=torch.long)

        if self.training and self.subject_dropout_prob > 0:
            drop_mask = torch.rand(subject_ids.size(0), device=subject_ids.device) < self.subject_dropout_prob
            subject_ids = subject_ids.clone()
            subject_ids[drop_mask] = self.null_subject_index

        subj_emb = self.subject_embeddings(subject_ids).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, D)
        subj_emb = subj_emb.expand(-1, T, 1, -1)  # (B, T, 1, D)

        x = torch.cat([x, subj_emb], dim=2)  # (B, T, num_modalities+1, D)
        x = x.view(B * T, x.shape[2], -1)    # (B*T, num_modalities+1, D)

        fused = self.transformer(x)  # (B*T, num_modalities+1, D)
        fused = fused.mean(dim=1)   # (B*T, D)
        fused = fused.view(B, T, -1)  # (B, T, D)

        return fused


class FixedPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=600):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class PredictionTransformer(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=1000, num_layers=1, num_heads=8, max_len=600, dropout=0.3):
        super().__init__()
        self.pos_encoder = FixedPositionalEncoding(hidden_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_dim*4, batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x, attn_mask):
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        device = x.device
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=~attn_mask)
        return self.output_head(x)



class FMRIModel(nn.Module):
    def __init__(self, input_dims, output_dim, subject_count=4, hidden_dim=1024, max_len=500, mask_prob=0.2):
        super().__init__()
        self.encoder = ModalityFusionTransformer(input_dims, subject_count, hidden_dim=hidden_dim)
        self.predictor = PredictionTransformer(hidden_dim=hidden_dim, output_dim=output_dim, max_len=max_len)
        self.mask_prob = mask_prob

    def forward(self, audio, video, text, subject_ids, attention_mask):
        if self.training and self.mask_prob > 0:
            mask = torch.rand(attention_mask.shape, device=attention_mask.device) < self.mask_prob
            attention_mask[mask] = False
        inputs = {'audio': audio, 'video': video, 'text': text}
        fused = self.encoder(inputs, subject_ids)
        return self.predictor(fused, attention_mask)
