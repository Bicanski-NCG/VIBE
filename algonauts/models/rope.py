import torch
from torch import nn
from labml_nn.transformers.rope import RotaryPositionalEmbeddings   # ← already in labml-nn
# If you don’t have labml-nn installed, just copy the implementation from the file you attached.

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 rope_pct: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RotaryPositionalEmbeddings(int(self.head_dim * rope_pct))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x                : [B, S, D]
        attn_mask        : [S, S]         (`True` = block) – causal mask
        key_padding_mask : [B, S]         (`True` = PAD)
        """
        B, S, _ = x.size()
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE expects [S, B, H, D_head]
        q = self.rope(q.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
        k = self.rope(k.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale   # [B,H,S,S]

        # ---- 1) MASK KEYS (columns) -------------------------------------------------
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]                  # [B,1,1,S]
            attn_scores = attn_scores.masked_fill(key_mask, -1e30)

        if attn_mask is not None:                                          # causal
            attn_scores = attn_scores.masked_fill(attn_mask, -1e30)

        # ---- 2) SOFT-MAX ------------------------------------------------------------
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)                 # rows all -1e30

        # ---- 3) CONTEXT -------------------------------------------------------------
        ctx = torch.einsum('bhij,bhjd->bhid', attn_probs, v)               # [B,H,S,Dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.embed_dim)

        # ---- 4) MASK QUERY OUTPUT (rows) -------------------------------------------
        if key_padding_mask is not None:
            query_mask = key_padding_mask[:, :, None]                      # [B,S,1]
            ctx = ctx.masked_fill(query_mask, 0.0)

        return self.out_proj(ctx)


class RoPEEncoderLayer(nn.Module):
    def __init__(self, in_dim, num_heads, dropout, rope_pct: float = 1.0):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(in_dim, num_heads, rope_pct=rope_pct, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * in_dim, in_dim)
        )
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self.dropout(self.self_attn(x, attn_mask, key_padding_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        return self.norm2(x)

class PredictionTransformerRoPE(nn.Module):
    def __init__(self,
                 input_dim=256,
                 output_dim=1000,
                 num_layers=1,
                 num_heads=8,
                 dropout=0.3,
                 rope_pct: float = 1.0):
        """
        x         : [batch, seq_len, input_dim]
        attn_mask : [batch, seq_len] – **True for real tokens, False for pads**

        Parameters:
        - input_dim: dimension of input embeddings
        - output_dim: dimension of output predictions
        - num_layers: number of transformer layers
        - num_heads: number of attention heads
        - dropout: dropout rate
        - rope_pct: percentage of head dimension to apply RoPE to
        """
        super().__init__()
        # ← no absolute/learned positional encodings needed
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(input_dim, num_heads, dropout, rope_pct=rope_pct)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(input_dim, output_dim)

    def forward(self, x, attn_mask):
        """
        x         : [batch, seq_len, input_dim]
        attn_mask : [batch, seq_len] – **True for real tokens, False for pads**
        """
        key_padding_mask = ~attn_mask          # True = PAD (matches our attention fn)
        # causal = torch.triu(
        #     torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1
        # )

        causal = None

        for layer in self.layers:
            x = layer(x, attn_mask=causal, key_padding_mask=key_padding_mask)

        return self.output_head(x)
