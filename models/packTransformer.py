"""
pack_transformer.py

Transformer-based Pack model tailored for 114-cell series pack.

Usage:
    - Prepare X_tensor (N,114,d_in) and y_arr (N, out_dim)
    - Fit StandardScalers externally (for X and y) or use sklearn inside main()
    - Instantiate model and train using the provided train_one_epoch / validate functions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias table.
    For clipped distance r = min(|i-j|, max_rel), returns bias per head: (num_heads, max_rel+1)
    """
    def __init__(self, num_heads: int, max_rel: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel = max_rel
        # initialize small
        self.bias_table = nn.Parameter(torch.zeros(num_heads, max_rel + 1) * 0.0)
        # optionally init so nearby have slightly larger magnitude
        nn.init.normal_(self.bias_table, mean=0.0, std=0.02)

    def forward(self, qlen: int, klen: int, device=None):
        """
        returns bias tensor of shape (num_heads, qlen, klen)
        """
        if device is None:
            device = self.bias_table.device
        # compute pairwise clipped distances
        q_idx = torch.arange(qlen, device=device).view(qlen, 1)
        k_idx = torch.arange(klen, device=device).view(1, klen)
        dist = (q_idx - k_idx).abs().clamp(max=self.max_rel).long()  # (qlen, klen)
        # bias_table: (num_heads, max_rel+1) -> index by dist to (num_heads, qlen, klen)
        bias = self.bias_table[:, dist]  # (num_heads, qlen, klen)
        return bias  # to be added to attention logits per head


class MultiHeadSelfAttentionWithRelBias(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, max_rel=16):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = RelativePositionBias(num_heads=num_heads, max_rel=max_rel)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """
        x: (B, L, dim)
        attn_mask: optional bool mask (B, L, L) with True for allowed positions OR None
        returns: (B, L, dim)
        """
        B, L, _ = x.shape
        qkv = self.qkv(x)  # (B, L, 3*dim)
        q, k, v = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q,k,v each: (B, num_heads, L, head_dim)
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        attn_logits = torch.einsum("b h i d, b h j d -> b h i j", q, k)  # (B, heads, L, L)
        attn_logits = attn_logits * self.scale

        # add relative position bias (shared across batch)
        bias = self.rel_bias(qlen=L, klen=L, device=x.device)  # (heads, L, L)
        attn_logits = attn_logits + bias.unsqueeze(0)  # (B, heads, L, L)

        # optional attn_mask (True = allowed)
        if attn_mask is not None:
            # attn_mask expected shape (B, L, L) or (L,L)
            if attn_mask.dim() == 2:
                am = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
                attn_logits = attn_logits.masked_fill(~am.bool(), float("-1e9"))
            elif attn_mask.dim() == 3:
                am = attn_mask.unsqueeze(1)  # (B,1,L,L)
                attn_logits = attn_logits.masked_fill(~am.bool(), float("-1e9"))

        attn = torch.softmax(attn_logits, dim=-1)  # (B, heads, L, L)
        attn = self.dropout(attn)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, self.dim)  # (B, L, dim)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PackTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, max_rel=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttentionWithRelBias(dim, num_heads=num_heads, dropout=dropout, max_rel=max_rel)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


class PackTransformer(nn.Module):
    """
    Transformer model for pack-level regression.

    Input: x (B, n_cells, in_dim)
    Output: y (B, out_dim)
    """
    def __init__(self,
                 in_dim,
                 model_dim=192,
                 num_layers=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 out_dim=1,
                 n_cells: int = 114,
                 max_rel: int = 16,
                 use_pack_token: bool = True):
        super().__init__()
        self.n_cells = n_cells
        self.use_pack_token = use_pack_token

        # input projection
        self.input_proj = nn.Linear(in_dim, model_dim)

        # positional embedding (absolute) optional + pack token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_cells + (1 if use_pack_token else 0), model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # pack token
        if use_pack_token:
            self.pack_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.trunc_normal_(self.pack_token, std=0.02)

        # transformer layers (use relative position bias inside attention)
        self.layers = nn.ModuleList([
            PackTransformerLayer(model_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, max_rel=max_rel)
            for _ in range(num_layers)
        ])

        # output head (from pack token if used, else pooled mean)
        if use_pack_token:
            self.head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, out_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, out_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """
        x: (B, n_cells, in_dim)
        attn_mask: optional (B, L, L) boolean mask True=allowed
        returns: (B, out_dim) or (B,) if out_dim==1
        """
        B, n, d_in = x.shape
        assert n == self.n_cells, f"Expected n={self.n_cells}, got {n}"

        # project
        h = self.input_proj(x)  # (B, n, model_dim)

        # insert pack token if used
        if self.use_pack_token:
            pack_tok = self.pack_token.expand(B, -1, -1)  # (B,1,model_dim)
            h = torch.cat([pack_tok, h], dim=1)  # (B, 1+n, D)
        L = h.shape[1]  # L = n + 1 if pack token

        # add pos emb (absolute for stability) - we still have relative bias in attention
        h = h + self.pos_embed[:, :L, :]

        # optionally build attn_mask to disallow pack token attending to nothing? we allow full attention
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        # readout from pack token or pooled mean
        if self.use_pack_token:
            pack_rep = h[:, 0, :]  # (B, D)
        else:
            pack_rep = h.mean(dim=1)  # (B, D)

        out = self.head(pack_rep)  # (B, out_dim)
        return out.squeeze(-1) if out.shape[1] == 1 else out
