"""Module 3 - Temporal-aware fake news detection (Sec. III-D, Eqs. 20-22)."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter, to_dense_batch

from src.models.components.rppg.layers import GATEncoder


class TemporalAwareDetector(nn.Module):
    """Dynamic propagation graph encoder + temporal-structure fusion + content.

    * Each snapshot G_t of the (real early + generated) path goes through a GAT; mean-pool readout gives
      h_t, and a temporal position embedding p_t is added: H^ = {h_0 + p_0, ..., h_T + p_T} (Eq. 20).
    * The final graph G_T goes through a second GAT -> structure features U^_T.
    * Fusion (Eq. 21):  H' = LN(SelfAttn(H^) + H^)
                        U  = LN(CrossAttn(query=H', key/value=U^_T) + H')
                        H* = LN(FeedForward(U) + H')
    * h* = mean-pool(H*);  P(y | n, G) = sigmoid(W_P [h* ; n] + b), n = [n_T ; n_V]  (Sec. III-D3).
    Returns the logit; the loss is binary cross-entropy (Eq. 22), with y = 1 for real news.
    """

    def __init__(self, user_dim: int = 128, content_dim: int = 1024, dim: int = 128, heads: int = 4,
                 dropout: float = 0.2, max_steps: int = 64):
        super().__init__()
        self.user_in = nn.Linear(user_dim, dim)
        self.seed_in = nn.Linear(content_dim, dim)       # the news seed node carries the content
        self.temporal_gat = GATEncoder(dim, dim, dim, heads, dropout)
        self.structure_gat = GATEncoder(dim, dim, dim, heads, dropout)
        self.pos = nn.Embedding(max_steps, dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim + content_dim, 1)
        self.max_steps = max_steps

    def _nodes(self, g, content):
        h = self.user_in(g["x"])
        seed = g["seed_of"] >= 0
        if seed.any():
            h = h.clone()
            h[seed] = self.seed_in(content[g["seed_of"][seed]])
        return h

    def forward(self, batch):
        content = batch["content"]
        B = content.size(0)
        # ---- dynamic propagation graph encoder (Eq. 20)
        t = batch["temporal"]
        ht = self.temporal_gat(self._nodes(t, content), t["edge_index"])
        snap = scatter(ht, t["batch"], dim=0, dim_size=t["num_graphs"], reduce="mean")       # h_t per snapshot
        steps = t["step"].clamp(max=self.max_steps - 1)
        snap = snap + self.pos(steps)
        H, tmask = to_dense_batch(snap, t["news"], batch_size=B)                              # (B, L, d)
        # ---- structure features of the final graph
        s = batch["structure"]
        hs = self.structure_gat(self._nodes(s, content), s["edge_index"])
        S, smask = to_dense_batch(hs, s["batch"], batch_size=B)                               # (B, N, d)
        # ---- temporal-structure fusion (Eq. 21)
        a, _ = self.self_attn(H, H, H, key_padding_mask=~tmask)
        H1 = self.ln1(a + H)
        c, _ = self.cross_attn(H1, S, S, key_padding_mask=~smask)
        H2 = self.ln2(c + H1)
        H3 = self.ln3(self.ffn(H2) + H1)  # residual to H' as written in Eq. 21
        m = tmask.unsqueeze(-1).float()
        h_star = (H3 * m).sum(1) / m.sum(1).clamp_min(1.0)                                   # readout = mean pool
        return self.out(torch.cat([self.drop(h_star), content], -1)).squeeze(-1)
