"""Module 2 - Reinforced propagation path generation (Sec. III-B and III-C, Eqs. 6-19)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.components.politifact import NEWS_ROOT, REL_ENGAGE, REL_REPOST
from src.models.components.rppg.layers import SubgraphAttentionPool

REL_COENGAGE, REL_CONTENT = 1, 2  # action relation types (repost = 0); index 3 = self-loop


def _csr(rows, cols, extra, n):
    order = np.argsort(rows, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, rows + 1, 1)
    return np.cumsum(ptr), cols[order], (extra[order] if extra is not None else None)


class PropagationEnv:
    """MDP environment on the training social graph H (Sec. III-B1).

    State s_t = (n, U_t, V_t, R_t); an action connects a user already in the path to a new user through an
    outgoing relation of H:
      * repost      (u_h -> u) : u reposted from u_h in some *other* training cascade (outgoing edge only)
      * co-engage   (u_h -> u) : u engaged a training news that u_h also engaged (U-N-U)
      * content     (n   -> u) : u is among the users whose embedding is closest to the news content; this
                                  lets generation start "even if no propagation paths exist" (Sec. III, B)
    Edges that come from the news being simulated are masked out, so the generator never sees that news'
    own real cascade through H. ``exclude[news]`` removes that news' real engagers from the content
    candidates: their embeddings were trained on H (which contains the training cascades), so for a
    training news they would be retrieved by content similarity although a test news never gets that
    advantage. Transitions are deterministic (Eq. 6).
    """

    def __init__(self, data: dict, content_topk: np.ndarray, max_sources: int = 64, max_repost: int = 8,
                 max_coengage: int = 6, max_actions: int = 256, exclude=None):
        U, N = data["num_users"], data["num_news"]
        ei = data["graph"]["edge_index"].numpy()
        et = data["graph"]["edge_type"].numpy()
        eo = data["graph"]["edge_origin"].numpy()
        fwd = data["graph"]["edge_forward"].numpy()
        rep = (et == REL_REPOST) & fwd
        u2n = (et == REL_ENGAGE) & (ei[0] < U)
        n2u = (et == REL_ENGAGE) & (ei[0] >= U)
        self.rep = _csr(ei[0][rep], ei[1][rep], eo[rep], U)
        self.u2n = _csr(ei[0][u2n], ei[1][u2n] - U, None, U)
        self.n2u = _csr(ei[0][n2u] - U, ei[1][n2u], None, N)
        self.content_topk = content_topk
        self.max_sources, self.max_repost = max_sources, max_repost
        self.max_coengage, self.max_actions = max_coengage, max_actions
        self.exclude = exclude or {}

    def candidates(self, news: int, path_users: np.ndarray, in_path: set, rng: np.random.Generator):
        """Return candidate (user, parent, relation) arrays for the current state."""
        users, parents, rels = [], [], []
        src = path_users if len(path_users) <= self.max_sources else rng.choice(path_users, self.max_sources, replace=False)
        ptr, col, org = self.rep
        uptr, ucol, _ = self.u2n
        nptr, ncol, _ = self.n2u
        for h in src:
            a, b = ptr[h], ptr[h + 1]
            if b > a:
                nb, o = col[a:b], org[a:b]
                nb = nb[o != news]
                if len(nb) > self.max_repost:
                    nb = rng.choice(nb, self.max_repost, replace=False)
                users.append(nb), parents.append(np.full(len(nb), h)), rels.append(np.full(len(nb), REL_REPOST))
            a, b = uptr[h], uptr[h + 1]
            if b > a:
                nws = ucol[a:b]
                nws = nws[nws != news]
                if len(nws):
                    m = nws[rng.integers(len(nws))]
                    na, nb_ = nptr[m], nptr[m + 1]
                    co = ncol[na:nb_]
                    if len(co) > self.max_coengage:
                        co = rng.choice(co, self.max_coengage, replace=False)
                    users.append(co), parents.append(np.full(len(co), h)), rels.append(np.full(len(co), REL_COENGAGE))
        ct = self.content_topk[news]
        if news in self.exclude:
            ct = ct[~np.isin(ct, self.exclude[news])]
        users.append(ct), parents.append(np.full(len(ct), -1)), rels.append(np.full(len(ct), REL_CONTENT))
        users, parents, rels = np.concatenate(users), np.concatenate(parents), np.concatenate(rels)
        keep = np.array([u not in in_path and u != NEWS_ROOT for u in users], dtype=bool)
        users, parents, rels = users[keep], parents[keep], rels[keep]
        users, first = np.unique(users, return_index=True)
        parents, rels = parents[first], rels[first]
        if len(users) > self.max_actions:
            sel = rng.choice(len(users), self.max_actions, replace=False)
            users, parents, rels = users[sel], parents[sel], rels[sel]
        return users, parents, rels


class PathGraph:
    """Growing propagation path G'_t: node 0 is the news seed, others are users (local indices)."""

    def __init__(self):
        self.users: List[int] = []
        self.index: Dict[int, int] = {}
        self.edges: List[tuple] = []  # local (parent, child)

    def add(self, user: int, parent: int) -> bool:
        if user in self.index:
            return False
        self.index[user] = len(self.users) + 1
        self.users.append(user)
        self.edges.append((self.index.get(parent, 0), self.index[user]))
        return True

    def tensors(self, device):
        ei = torch.tensor(self.edges, dtype=torch.long, device=device).T if self.edges else torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.tensor(self.users, dtype=torch.long, device=device), ei


class RPPGGenerator(nn.Module):
    """Policy network (Eqs. 7-11), critic (Eq. 15) and path discriminator D (Eq. 13)."""

    def __init__(self, base_dim: int = 128, content_dim: int = 1024, dim: int = 128, n_rel: int = 3,
                 standardize: bool = False):
        super().__init__()
        # standardize=True: z-score the Module 1 embeddings over users before Eq. 7. Without it the GAT outputs
        # share a large common component, tanh saturates and every user in H gets the same vector (found 2026-10-19).
        self.standardize = standardize
        # users that exist in graph H (train news); others share one "unknown" vector (set_known)
        self.register_buffer("known", torch.zeros(0, dtype=torch.bool), persistent=False)
        self.fuse = nn.Linear(2 * base_dim, dim)                 # Eq. 7: u_i = tanh(W_u [u^_i ; sum_l h^l_i])
        self.news_proj = nn.Linear(content_dim, dim)             # news content embedding n
        self.rel_emb = nn.Embedding(n_rel + 1, dim)              # r_i (last index = self-loop)
        self.self_loop = nn.Parameter(torch.zeros(dim))          # "no interaction happens" action
        self.state_pool = SubgraphAttentionPool(dim)             # Eq. 9
        self.actor = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))   # Eq. 11 (W3, W2)
        self.critic = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))  # Eq. 15 (W1Q, W0Q)
        self.disc_pool = SubgraphAttentionPool(dim)              # D shares the structure of Eq. 9
        self.disc_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.n_rel = n_rel

    def set_known(self, mask):
        self.known = mask

    def user_table(self, base_node, base_meta):
        k = self.known if self.known.numel() else None
        if self.standardize:  # z-score with statistics of the users in H only (isolated users are ~70% of rows)
            def z(x):
                ref = x[k] if k is not None else x
                return (x - ref.mean(0)) / (ref.std(0) + 1e-6)
            base_node, base_meta = z(base_node), z(base_meta)
        U = torch.tanh(self.fuse(torch.cat([base_node, base_meta], -1)))
        if k is not None:  # users not in H (val/test-only, unseen) -> one shared UNK vector = mean of known users
            U = torch.where(k.unsqueeze(-1), U, U[k].mean(0, keepdim=True))
        return U

    def graph_nodes(self, g: PathGraph, U, n_vec):
        users, ei = g.tensors(U.device)
        x = torch.cat([n_vec.unsqueeze(0), U[users]], 0) if len(users) else n_vec.unsqueeze(0)
        return x, ei

    def state(self, g: PathGraph, new_users: List[int], U, n_vec):
        """s_t = [Pi_t ; v_t ; n] (Eqs. 8-10)."""
        x, ei = self.graph_nodes(g, U, n_vec)
        pi = self.state_pool(x, ei, torch.zeros(x.size(0), dtype=torch.long, device=x.device), 1)[0]
        v = U[torch.tensor(new_users, device=U.device)].mean(0) if new_users else torch.zeros_like(n_vec)
        return torch.cat([pi, v, n_vec])

    def action_embeddings(self, users, rels, U):
        a = U[torch.as_tensor(users, device=U.device)] + self.rel_emb(torch.as_tensor(rels, device=U.device))
        return torch.cat([a, (self.self_loop + self.rel_emb.weight[self.n_rel]).unsqueeze(0)], 0)

    def scores(self, net, s, a):
        return net(torch.cat([s.unsqueeze(0).expand(a.size(0), -1), a], -1)).squeeze(-1)

    def discriminate(self, graphs: List[PathGraph], U, n_vecs):
        xs, eis, bs, off = [], [], [], 0
        for i, g in enumerate(graphs):
            x, ei = self.graph_nodes(g, U, n_vecs[i])
            xs.append(x), eis.append(ei + off), bs.append(torch.full((x.size(0),), i, device=x.device))
            off += x.size(0)
        h = self.disc_pool(torch.cat(xs), torch.cat(eis, 1), torch.cat(bs), len(graphs))
        return self.disc_head(h).squeeze(-1)  # logit of D


def generated_prefix(early: List[tuple], steps: List[List[tuple]], size: int) -> PathGraph:
    """Generated path G'_t restricted to its first ``size`` users (input to D, Eq. 13)."""
    g = PathGraph()
    for h, u, _ in early:
        if len(g.users) >= size:
            return g
        g.add(u, -1 if h == NEWS_ROOT else h)
    for step in steps:
        for p, c in step:
            if len(g.users) >= size:
                return g
            g.add(c, p)
    return g


def real_prefix(eng: List[tuple], size: int) -> PathGraph:
    """Real propagation path restricted to its first ``size`` engaged users (input to D)."""
    g = PathGraph()
    for h, u, _ in eng:
        if len(g.users) >= size:
            break
        g.add(u, -1 if h == NEWS_ROOT else h)
    return g


def rollout(model: RPPGGenerator, env: PropagationEnv, news: int, early: List[tuple], U, n_vec,
            rng: np.random.Generator, T: int, K2: int, max_steps: int, slots: Optional[List[List[int]]] = None,
            lambda2: float = 0.4, gamma: float = 0.9, train: bool = True, global_reward: str = "eq13",
            real_users: Optional[set] = None, hit_weight: float = 0.0, policy: str = "rl"):
    """Generate one propagation path; when ``train`` also return actor / critic losses.

    Rewards (Sec. III-B3):
      R_T(u_i, t) = cos(u_i, mean_{u_j in Lambda(t)} u_j)            temporal reward (Eq. 12)
      R_G        = -log(1 - D(G'_t))                                   global reward (Eq. 13). Eq. 13 is
                   -(E_real log D(G) + E_gen log(1 - D(G'))); the first term does not depend on the generator,
                   so the generator receives the second term for its own path (computed as softplus(logit D)).
                   global_reward="scaled" uses 2*D - 1 instead (bounded like the cosine temporal reward)
      R_C        = lambda2 * R_T + (1 - lambda2) * R_G               (Eq. 14)
      + hit_weight * 1[u_i really engaged this training news]          (optional, NOT in the paper: a direct
                   supervision signal used to test whether a more faithful G' helps detection)
    Actor-critic (Sec. III-C): q_t = R_C + gamma * E_a[Q(s_{t+1}, a)] (Eq. 16), L_critic = (Q - q_t)^2 (Eq. 17),
    actor gradient Q(s_t, a_t) * grad log pi(s_t, a_t) (Eq. 19). K2 actions are absorbed per step [24].
    ``policy`` (generation only, Table IV): "rl" = learned policy, "random" = "Random Sample" (K2 random
    candidates per step), "similarity" = "Node Similarity" (top-K2 candidates most similar to the news).
    """
    g = PathGraph()
    for h, u, _ in early:
        g.add(u, -1 if h == NEWS_ROOT else h)
    new_users = list(g.users)
    steps: List[List[tuple]] = []
    recs = []
    for t in range(max_steps):
        if len(g.users) >= T:
            break
        users, parents, rels = env.candidates(news, np.asarray(g.users, dtype=np.int64), set(g.users), rng)
        if policy != "rl":
            k = min(K2, T - len(g.users), len(users))
            if k <= 0:
                break
            if policy == "random":
                pick = rng.choice(len(users), k, replace=False)
            else:
                sims = F.normalize(U[torch.as_tensor(users, device=U.device)], dim=-1) @ F.normalize(n_vec, dim=0)
                pick = torch.topk(sims, k).indices.cpu().numpy()
            step = []
            for j in pick.tolist():
                if g.add(int(users[j]), int(parents[j])):
                    step.append((int(parents[j]), int(users[j])))
            steps.append(step)
            continue
        s = model.state(g, new_users, U, n_vec)
        a = model.action_embeddings(users, rels, U)
        logits = model.scores(model.actor, s, a)
        probs = logits.softmax(-1)
        k = min(K2, T - len(g.users), a.size(0))
        with torch.no_grad():
            pick = torch.multinomial(probs, k, replacement=False)
        added, step = [], []
        for j in pick.tolist():
            if j == len(users):  # self-loop
                continue
            if g.add(int(users[j]), int(parents[j])):
                added.append(int(users[j]))
                step.append((int(parents[j]), int(users[j])))
        steps.append(step)
        if train:
            q = model.scores(model.critic, s, a[pick])
            v_all = (probs.detach() * model.scores(model.critic, s, a).detach()).sum()
            recs.append({"logp": probs[pick].clamp_min(1e-9).log(), "q": q, "pick": pick, "n_cand": len(users),
                         "users": users, "v": v_all, "t": t})
        new_users = added
        if not added and len(users) == 0:
            break
    if not train:
        return steps, g, None
    # ---------------- rewards
    with torch.no_grad():
        Ud = U.detach()
        n_slots = len(slots)
        r_g_steps = []
        prefix = PathGraph()
        for h, u, _ in early:
            prefix.add(u, -1 if h == NEWS_ROOT else h)
        for step in steps:
            for p, c in step:
                prefix.add(c, p)
            logit = model.discriminate([prefix], Ud, n_vec.detach().unsqueeze(0))[0]
            r_g_steps.append(F.softplus(logit) if global_reward == "eq13" else 2 * torch.sigmoid(logit) - 1)
    actor_loss, critic_loss = 0.0, 0.0
    for i, r in enumerate(recs):
        t = min(r["t"], n_slots - 1)
        lam = slots[t] if slots[t] else [u for sl in slots for u in sl]
        target = F.normalize(Ud[torch.tensor(lam, device=U.device)].mean(0), dim=0)
        rt = torch.zeros(len(r["pick"]), device=U.device)
        is_user = r["pick"] < r["n_cand"]
        if is_user.any():
            sel = torch.as_tensor(r["users"], device=U.device)[r["pick"][is_user]]
            rt[is_user] = F.normalize(Ud[sel], dim=-1) @ target
        rc = lambda2 * rt + (1 - lambda2) * r_g_steps[i]
        if hit_weight and real_users and is_user.any():
            hit = torch.zeros(len(r["pick"]), device=U.device)
            picked = np.asarray(r["users"])[r["pick"][is_user].cpu().numpy()]
            hit[is_user] = torch.tensor([float(u in real_users) for u in picked], device=U.device)
            rc = rc + hit_weight * hit
        nxt = recs[i + 1]["v"] if i + 1 < len(recs) else torch.zeros((), device=U.device)
        q_target = (rc + gamma * nxt).detach()
        critic_loss = critic_loss + F.mse_loss(r["q"], q_target, reduction="sum")
        actor_loss = actor_loss - (r["q"].detach() * r["logp"]).sum()
    return steps, g, {"actor": actor_loss, "critic": critic_loss, "reward_g": torch.stack(r_g_steps).mean() if r_g_steps else torch.zeros(())}
