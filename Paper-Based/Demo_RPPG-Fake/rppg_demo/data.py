"""Stage-2 input building, copied from Code_RPPG-Fake/src/data/rppg_datamodule.py (make_sample, Collate).

A news item becomes cumulative snapshots G_0 ... G_T: the real early part (3-5 first engagements) split into
``real_slots`` steps, then one snapshot per step of the generated path G'.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .constants import NEWS_ROOT


def make_sample(eng, early, gen_steps, mode: str, real_slots: int, max_real_users: int, use_generated: bool,
                max_gen_users: Optional[int] = None, unk: Optional[set] = None):
    users: List[int] = []
    index: Dict[int, int] = {}
    edges: List[tuple] = []

    def add(u, parent):
        if u in index or u == NEWS_ROOT:
            return
        index[u] = len(users) + 1
        users.append(u)
        edges.append((index.get(parent, 0), index[u]))

    real = early if mode == "early" else eng
    for h, u, _ in real:
        if len(users) >= max_real_users:
            break
        add(u, -1 if h == NEWS_ROOT else h)
    n_real = len(users)
    bounds = []
    if n_real:
        for chunk in np.array_split(np.arange(1, n_real + 1), min(real_slots, n_real)):
            bounds.append(int(chunk[-1]) + 1)  # number of local nodes (seed included) in this snapshot
    else:
        bounds.append(1)
    if use_generated:
        cap = n_real + max_gen_users if max_gen_users is not None else None  # diagnostic, not in paper
        for step in gen_steps:
            before = len(users)
            for p, c in step:
                if cap is not None and len(users) >= cap:
                    break
                add(c, p)
            if len(users) > before:
                bounds.append(len(users) + 1)
    out = {"users": users, "edges": edges, "bounds": bounds}
    if unk:  # these users keep their node but get the UNK feature (row 0 of the user table, a user not in H)
        out["feat"] = [0 if u in unk else u for u in users]
    return out


class Collate:
    def __init__(self, user_emb: torch.Tensor, content: torch.Tensor):
        self.user_emb, self.content = user_emb, content

    def _graphs(self, items, temporal: bool):
        xs, seed_of, eis, gid, news_of, step_of = [], [], [], [], [], []
        off, g = 0, 0
        for b, (_, s, _) in enumerate(items):
            users = torch.tensor(s.get("feat", s["users"]), dtype=torch.long)
            e = torch.tensor(s["edges"], dtype=torch.long).reshape(-1, 2)
            bounds = s["bounds"] if temporal else [len(s["users"]) + 1]
            for t, nb in enumerate(bounds):
                feat = torch.zeros(nb, self.user_emb.size(1))
                if nb > 1:
                    feat[1:] = self.user_emb[users[: nb - 1]]
                so = torch.full((nb,), -1, dtype=torch.long)
                so[0] = b
                ee = e[e[:, 1] < nb] if len(e) else e
                xs.append(feat), seed_of.append(so), eis.append(ee.T + off)
                gid.append(torch.full((nb,), g, dtype=torch.long))
                news_of.append(b), step_of.append(t)
                off += nb
                g += 1
        return {"x": torch.cat(xs), "seed_of": torch.cat(seed_of), "edge_index": torch.cat(eis, 1),
                "batch": torch.cat(gid), "num_graphs": g, "news": torch.tensor(news_of), "step": torch.tensor(step_of)}

    def __call__(self, items):
        idx = torch.tensor([i for i, _, _ in items])
        return {"idx": idx, "content": self.content[idx], "y": torch.tensor([float(y) for _, _, y in items]),
                "temporal": self._graphs(items, True), "structure": self._graphs(items, False)}


