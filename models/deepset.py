import torch
import torch.nn as nn

class DeepSetModel(nn.Module):
    def __init__(self, in_dim, emb_dim=64, hidden_dim=128, out_dim=1, agg='mean'):
        super().__init__()
        # phi: per-element embed
        self.phi = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        self.agg = agg
        # rho: after aggregation
        self.rho = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        # x: (B, n, in_dim)
        B, n, d = x.shape
        x_flat = x.view(B * n, d)
        emb = self.phi(x_flat)  # (B*n, emb_dim)
        emb = emb.view(B, n, -1)  # (B, n, emb_dim)
        if self.agg == 'mean':
            pooled = emb.mean(dim=1)  # (B, emb_dim)
        elif self.agg == 'sum':
            pooled = emb.sum(dim=1)
        else:
            pooled = emb.mean(dim=1)
        out = self.rho(pooled)  # (B, out_dim)
        return out.squeeze(-1)  # (B,)