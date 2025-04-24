import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityFusionSimple(nn.Module):
    def __init__(self, input_dims, subject_count=4, hidden_dim=512, embedding_dim=128, dropout_rate=0.3):
        super().__init__()
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                # Maybe add activation/norm here if needed
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate) # Add dropout after projection
            )
            for modality, dim in input_dims.items()
        })
        # Total projected dim
        projected_dim = hidden_dim * len(input_dims)

        self.subject_embeddings = nn.Embedding(subject_count, embedding_dim) # Keep subject embedding size reasonable

        # Optional: A final linear layer to mix concatenated features and subject embedding
        # Adjust input dimension calculation if subject embedding is concatenated vs added
        self.combined_input_dim = projected_dim + embedding_dim # If concatenating subject embedding


    def forward(self, inputs: dict, subject_ids: torch.LongTensor):
        B, T, _ = next(iter(inputs.values())).shape

        projected = [
            self.projections[name](inputs[name]) # (B, T, hidden_dim)
            for name in self.projections.keys() # Ensure consistent order
        ]
        x = torch.cat(projected, dim=-1)  # (B, T, hidden_dim * num_modalities)

        # Subject embedding - Choose one way:
        # Option A: Concatenate
        subj_emb = self.subject_embeddings(torch.tensor(subject_ids, device=x.device)).unsqueeze(1) # (B, 1, hidden_dim)
        subj_emb = subj_emb.expand(-1, T, -1) # (B, T, hidden_dim)
        x = torch.cat([x, subj_emb], dim=-1) # (B, T, hidden_dim * num_modalities + hidden_dim)

        # Option B: Add (if using a mixer that brings back to hidden_dim, maybe add *after* mixer?)
        # subj_emb = self.subject_embeddings(subject_ids).unsqueeze(1) # (B, 1, hidden_dim)
        # subj_emb = subj_emb.expand(-1, T, -1) # (B, T, hidden_dim)
        # Ensure dimensions match if adding directly to some intermediate state
        # If adding subject embedding after mixer (adjust mixer output dim if needed)
        # fused = fused + subj_emb # If mixer outputs hidden_dim

        return x


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
    def __init__(self, hidden_dim=256, output_dim=1000, num_layers=3, num_heads=8, max_len=600, dropout=0.3):
        super().__init__()
        self.pos_encoder = FixedPositionalEncoding(hidden_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_dim*4, batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, attn_mask):
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=~attn_mask)
        return self.output_head(x)



class FMRIModel(nn.Module):
    def __init__(self, input_dims, output_dim, subject_count=4, hidden_dim=256, max_len=500):
        super().__init__()
        self.encoder = ModalityFusionSimple(input_dims, subject_count, hidden_dim=hidden_dim)
        self.predictor = PredictionTransformer(hidden_dim=self.encoder.combined_input_dim, output_dim=output_dim, max_len=max_len)

    def forward(self, audio, video, text, subject_ids, attention_mask):
        inputs = {'audio': audio, 'video': video, 'text': text}
        fused = self.encoder(inputs, subject_ids)
        return self.predictor(fused, attention_mask)
