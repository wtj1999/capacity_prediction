import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= Readout 模块 =========
class ReadoutMaxMean(nn.Module):
    def forward(self, h):
        mean_pool = h.mean(dim=1)          # (B, hidden)
        max_pool, _ = h.max(dim=1)         # (B, hidden)
        min_pool, _ = h.min(dim=1)
        return torch.cat([mean_pool, max_pool, min_pool], dim=-1)  # (B, 2*hidden)


class AttentionReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        attn_scores = torch.tanh(self.attn(h))      # (B, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, N, 1)
        pack_emb = (attn_weights * h).sum(dim=1)    # (B, hidden)
        return pack_emb


class ConvReadout(nn.Module):
    def __init__(self, hidden_dim, out_dim, n_cells):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, out_dim, kernel_size=n_cells, stride=1)

    def forward(self, h):
        # h: (B, N, hidden_dim)
        h = h.transpose(1, 2)               # (B, hidden_dim, N)
        conv_out = F.relu(self.conv(h))     # (B, out_dim, 1)
        return conv_out.squeeze(-1)         # (B, out_dim)


class TransformerReadout(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # learnable query

    def forward(self, h):
        B = h.size(0)
        query = self.query.expand(B, -1, -1)   # (B,1,hidden)
        out, _ = self.attn(query, h, h)       # (B,1,hidden)
        return out.squeeze(1)

class PackChainGraphModel(nn.Module):
    """
    - Input: x (B, n_cells, in_dim)
    - Output: (B, out_dim)
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, n_cells=114, gnn_type="gcn", heads=4, readout_type="mean"):
        super().__init__()
        self.n_cells = n_cells
        self.gnn_type = gnn_type.lower()
        self.out_dim = out_dim
        self.heads = heads
        self.readout_type = readout_type.lower()

        # ----- Layers -----
        if self.gnn_type == "gcn":
            self.gnn1 = GCNLayer(in_dim, hidden_dim)
            self.gnn2 = GCNLayer(hidden_dim, hidden_dim)
        elif self.gnn_type == "gat":
            self.gnn1 = GATLayer(in_dim, hidden_dim, heads=heads)
            self.gnn2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1)
        else:
            raise ValueError("gnn_type must be 'gcn' or 'gat'")

        # ----- Readout -----
        if self.readout_type == "mean":
            self.readout = lambda h: h.mean(dim=1)
            readout_dim = hidden_dim
        elif self.readout_type == "maxmean":
            self.readout = ReadoutMaxMean()
            readout_dim = hidden_dim * 3
        elif self.readout_type == "attn":
            self.readout = AttentionReadout(hidden_dim)
            readout_dim = hidden_dim
        elif self.readout_type == "conv":
            self.readout = ConvReadout(hidden_dim, hidden_dim, n_cells)
            readout_dim = hidden_dim
        elif self.readout_type == "transformer":
            self.readout = TransformerReadout(hidden_dim)
            readout_dim = hidden_dim
        else:
            raise ValueError("Invalid readout_type")

        # MLP for output
        self.mlp = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 8), out_dim)
        )

        # precompute adjacency matrix for chain
        self.register_buffer("adj", self._make_chain_adj(n_cells))

    def _make_chain_adj(self, n):
        """构造 (n,n) 的邻接矩阵，链式双向"""
        A = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        return A

    def forward(self, x):
        """
        x: (B, n_cells, in_dim)
        """
        if x.dim() != 3:
            raise ValueError("x must be (B, n_cells, in_dim)")

        B, N, d = x.shape
        if N != self.n_cells:
            raise ValueError(f"Expected n_cells={self.n_cells}, got {N}")

        if self.gnn_type == "gcn":
            h = self.gnn1(x, self.adj)
            h = self.gnn2(h, self.adj)
        elif self.gnn_type == "gat":
            h = self.gnn1(x)  # (B, N, hidden*heads)
            h = self.gnn2(h)  # (B, N, hidden)
        else:
            raise ValueError

        # readout
        pack_emb = self.readout(h)  # (B, hidden)

        out = self.mlp(pack_emb)  # (B, out_dim)
        return out.squeeze(-1) if out.shape[1] == 1 else out


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x: (B, N, d)
        adj: (N, N)
        """
        B, N, d = x.shape
        I = torch.eye(N, device=x.device)
        A_hat = adj.to(x.device) + I  # 加自环
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        D_hat_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
        norm_A = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt  # (N,N)

        h = self.linear(x)  # (B,N,out_dim)
        h = torch.matmul(norm_A, h)  # 邻接传播 (B,N,out_dim)
        return F.relu(h)


class GATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_dim))
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, x):
        """
        x: (B, N, d)
        return: (B, N, out_dim*heads)
        """
        B, N, d = x.shape
        h = self.W(x)  # (B,N,out_dim*heads)
        h = h.view(B, N, self.heads, -1)  # (B,N,H,d_out)

        # 注意力得分
        alpha_l = torch.einsum("bnhd,hd->bnh", h, self.attn_l)  # (B,N,H)
        alpha_r = torch.einsum("bnhd,hd->bnh", h, self.attn_r)  # (B,N,H)

        scores = alpha_l.unsqueeze(2) + alpha_r.unsqueeze(1)  # (B,N,N,H)
        scores = F.leaky_relu(scores, 0.2)
        attn = F.softmax(scores, dim=2)  # 对邻居 softmax (B,N,N,H)

        # 消息聚合
        out = torch.einsum("bnih,bnhd->bnhd", attn, h)  # (B,N,H,d_out)
        out = out.reshape(B, N, -1)  # (B,N,H*d_out)
        return F.elu(out)
