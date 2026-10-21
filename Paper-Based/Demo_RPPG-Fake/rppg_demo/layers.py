"""Building blocks shared by the three RPPG-Fake modules."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter, softmax


class PaperGATConv(MessagePassing):
    """Graph attention layer exactly as written in Eq. (2):

        u._k     = ReLU([W1 u_k ; W1 u_i])
        alpha_ik = exp(a^T u._k) / sum_{k in N(i)} exp(a^T u._k)
        u_i^L    = sum_{k in N(i)} alpha_ik W1 u_k          (the outer sigma is applied by GATEncoder)

    a = [a_src ; a_dst] splits over the concatenation, so the score is a_src.ReLU(W u_k) + a_dst.ReLU(W u_i).
    Self-loops are added so every node attends to itself as well.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, concat: bool = True, dropout: float = 0.0):
        super().__init__(aggr="add", node_dim=0)
        self.heads, self.out_dim, self.concat, self.dropout = heads, out_dim, concat, dropout
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index):
        h = self.lin(x).view(-1, self.heads, self.out_dim)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        r = F.relu(h)
        e_src = (r * self.a_src).sum(-1)
        e_dst = (r * self.a_dst).sum(-1)
        out = self.propagate(edge_index, h=h, e_src=e_src, e_dst=e_dst)
        out = out.reshape(-1, self.heads * self.out_dim) if self.concat else out.mean(1)
        return out + self.bias

    def message(self, h_j, e_src_j, e_dst_i, index, ptr, size_i):
        alpha = softmax(e_src_j + e_dst_i, index, ptr, size_i)
        alpha = F.dropout(alpha, self.dropout, self.training)
        return h_j * alpha.unsqueeze(-1)


class GATEncoder(nn.Module):
    """Two-layer graph attention network (Eq. 2): multi-head first layer, single-head output layer."""

    def __init__(self, in_dim: int = 128, hidden: int = 128, out_dim: int = 128, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = PaperGATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.conv2 = PaperGATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index)


class SubgraphAttentionPool(nn.Module):
    """Subgraph embedding Pi_t (Eq. 9).

    Each node is first refined with its neighbours inside the subgraph,
        u~_h = tanh(W0 [u_h ; sum_{u in N(u_h)} u]),
    then all nodes are aggregated with attention weights
        beta(u_h) = softmax(w . ELU(W1 u~_h)),   Pi = sum_h beta(u_h) u~_h.
    Works on a batch of graphs given as flat node features + ``batch`` vector.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.W0 = nn.Linear(2 * dim, dim)
        self.W1 = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch, num_graphs: int):
        n = x.size(0)
        if edge_index.numel():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)  # neighbourhood N(u_h) is undirected
            nb = scatter(x[edge_index[0]], edge_index[1], dim=0, dim_size=n, reduce="sum")
        else:
            nb = torch.zeros_like(x)
        u = torch.tanh(self.W0(torch.cat([x, nb], -1)))
        score = self.w(F.elu(self.W1(u))).squeeze(-1)
        score = score - scatter(score, batch, dim=0, dim_size=num_graphs, reduce="max")[batch]
        e = score.exp()
        beta = e / scatter(e, batch, dim=0, dim_size=num_graphs, reduce="sum")[batch].clamp_min(1e-12)
        return scatter(beta.unsqueeze(-1) * u, batch, dim=0, dim_size=num_graphs, reduce="sum")
