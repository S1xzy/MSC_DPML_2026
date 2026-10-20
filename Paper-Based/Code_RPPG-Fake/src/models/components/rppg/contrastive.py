"""Module 1 - Heterogeneous graph contrastive learning (Sec. III-A, Eqs. 2-5)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.rppg.layers import GATEncoder


class HeteroGraphContrastive(nn.Module):
    """Dual-view self-supervised contrastive learning on the social graph H.

    * Node view: PGD-perturbed node features u~ = u + delta [18]; for every centre node the top-K1 most and
      least cosine-similar nodes form the positive / negative sets [19] (similarity of the current, detached
      GAT embeddings of the clean graph, searched in a random pool of nodes); a GAT encodes the perturbed
      graph and the loss is Eq. (3) with temperature tau1.
    * Meta-path view: one GAT per meta-path semantic graph H^l; the same node under two meta-paths is a
      positive pair, other nodes in the batch are negatives, Eq. (4) with temperature tau2.
    * L_CL = lambda1 * L_node + (1 - lambda1) * L_meta (Eq. 5).
    News nodes start from their (frozen) CLIP content feature, users from a learnable embedding table.
    """

    def __init__(self, num_users: int, num_news: int, content_dim: int, n_meta: int, dim: int = 128,
                 heads: int = 4, dropout: float = 0.2, tau1: float = 0.4, tau2: float = 0.3,
                 lambda1: float = 0.4, k1: int = 8, pgd_eps: float = 0.05, pgd_steps: int = 2,
                 pos_neg: str = "embedding"):
        super().__init__()
        # pos_neg="embedding": V^p / V^n by cosine of the model's own current embeddings (rounds 1-6). This is
        # self-reinforcing and collapsed the node view to ~1 dimension (found 2026-10-19).
        # pos_neg="structure": cosine of a fixed neighbourhood sketch (A + I) R of graph H (set_structure), so
        # nodes that share neighbours (same news / repost) are positives and unrelated nodes are negatives.
        self.pos_neg = pos_neg
        self.register_buffer("sketch", torch.zeros(0), persistent=False)
        self.num_users, self.num_news = num_users, num_news
        self.user_emb = nn.Embedding(num_users, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        self.content_proj = nn.Linear(content_dim, dim)
        self.node_gat = GATEncoder(dim, dim, dim, heads, dropout)
        self.meta_gats = nn.ModuleList([GATEncoder(dim, dim, dim, heads, dropout) for _ in range(n_meta)])
        self.tau1, self.tau2, self.lambda1, self.k1 = tau1, tau2, lambda1, k1
        self.pgd_eps, self.pgd_steps = pgd_eps, pgd_steps

    @torch.no_grad()
    def set_structure(self, edge_index, n_all: int, dim: int = 64, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        R = torch.randn(n_all, dim, generator=g).to(edge_index.device)
        S = R.clone()  # + I: a node's own id is part of its neighbourhood
        S.index_add_(0, edge_index[1], R[edge_index[0]])  # + A R (edge_index holds both directions)
        self.sketch = F.normalize(S, dim=-1)

    def features(self, content):
        return torch.cat([self.user_emb.weight, self.content_proj(content)], 0)

    # ------------------------------------------------------------------ node view (Eq. 2-3)
    @torch.no_grad()
    def _pos_neg(self, x, edge_index, centres, pool):
        """V_i^p / V_i^n: top-K1 most / least cosine-similar nodes to the centre (clean graph, detached)."""
        if self.pos_neg == "structure" and self.sketch.numel():
            z = self.sketch
        else:
            was = self.node_gat.training
            self.node_gat.eval()
            z = F.normalize(self.node_gat(x, edge_index), dim=-1)
            self.node_gat.train(was)
        xs = z[centres] @ z[pool].T
        same = pool.unsqueeze(0) == centres.unsqueeze(1)
        pos = torch.topk(xs.masked_fill(same, -2.0), self.k1, dim=1).indices
        neg = torch.topk(xs.masked_fill(same, 2.0), self.k1, dim=1, largest=False).indices
        return pos, neg

    def _node_loss(self, x, edge_index, centres, pool, pos, neg):
        z = F.normalize(self.node_gat(x, edge_index), dim=-1)
        sim = (z[centres] @ z[pool].T) / self.tau1
        pos_s = torch.logsumexp(sim.gather(1, pos), 1)
        neg_s = torch.logsumexp(sim.gather(1, neg), 1)
        return (neg_s - pos_s).mean()  # -log( sum_pos exp / sum_neg exp )

    def node_view_loss(self, content, edge_index, centres, pool):
        x = self.features(content)
        pos, neg = self._pos_neg(x.detach(), edge_index, centres, pool)
        if self.pgd_steps > 0 and self.training:  # PGD: ascend the contrastive loss inside an eps-ball
            delta = torch.zeros_like(x, requires_grad=True)
            step = self.pgd_eps / self.pgd_steps
            for _ in range(self.pgd_steps):
                loss = self._node_loss(x.detach() + delta, edge_index, centres, pool, pos, neg)
                g, = torch.autograd.grad(loss, delta)
                delta = (delta + step * g.sign()).clamp(-self.pgd_eps, self.pgd_eps).detach().requires_grad_(True)
            x = x + delta.detach()
        return self._node_loss(x, edge_index, centres, pool, pos, neg)

    # ------------------------------------------------------------------ meta-path view (Eq. 4)
    def meta_view_loss(self, content, meta_edges, batch):
        x = self.features(content)
        hs = [F.normalize(g(x, e)[batch], dim=-1) for g, e in zip(self.meta_gats, meta_edges)]
        loss, n = 0.0, 0
        eye = torch.eye(len(batch), dtype=torch.bool, device=x.device)
        for l in range(len(hs)):
            for m in range(len(hs)):
                if l == m:
                    continue
                sim = hs[l] @ hs[m].T / self.tau2
                pos = sim.diagonal()
                neg = torch.logsumexp(sim.masked_fill(eye, float("-inf")), 1)  # I[i != k]
                loss = loss + (neg - pos).mean()
                n += 1
        return loss / max(n, 1)

    def loss(self, content, edge_index, meta_edges, centres, pool, meta_batch):
        l_node = self.node_view_loss(content, edge_index, centres, pool)
        l_meta = self.meta_view_loss(content, meta_edges, meta_batch)
        return self.lambda1 * l_node + (1 - self.lambda1) * l_meta, l_node.detach(), l_meta.detach()

    @torch.no_grad()
    def embeddings(self, content, edge_index, meta_edges):
        """Node-view embedding u^ and sum_{l in Phi(u_i)} h^l (inputs of Eq. 7).

        Phi(u_i) = the meta-paths node i actually takes part in (it has at least one neighbour in H^l);
        e.g. users belong to UNU and UU but not to NUN.
        """
        x = self.features(content)
        node = self.node_gat(x, edge_index)
        meta = torch.zeros_like(node)
        for g, e in zip(self.meta_gats, meta_edges):
            member = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            if e.numel():
                member[e[0]] = True
            meta = meta + g(x, e) * member.unsqueeze(-1)
        return node, meta
