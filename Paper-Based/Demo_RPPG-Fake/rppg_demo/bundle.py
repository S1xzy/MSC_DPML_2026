"""Load the exported RPPG-Fake models and data (data/<dataset>/) and run each module for the demo.

A news item is addressed by a *key*: an int = index of a news item in the dataset (train / val / test), or a
string "ext<j>" = item j of the external set (news from events that are not in the training data).
"""
from __future__ import annotations

import csv
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .constants import NEWS_ROOT, REL_ENGAGE, REL_REPOST
from .data import Collate, make_sample
from .detector import TemporalAwareDetector
from .generator import REL_COENGAGE, REL_CONTENT, PathGraph, PropagationEnv, RPPGGenerator, rollout

REL_NAMES = {REL_REPOST: "repost (H)", REL_COENGAGE: "co-engage (H)", REL_CONTENT: "content"}
Key = Union[int, str]


class Bundle:
    """Everything one trained model needs: graph H, content features, Module 1-3 weights and the demo items."""

    def __init__(self, folder: str, device: str = "cpu"):
        self.folder, self.device = folder, torch.device(device)
        self.data = torch.load(os.path.join(folder, "graph.pt"), weights_only=False)
        self.content = torch.load(os.path.join(folder, "content.pt"), weights_only=False).float()
        m = torch.load(os.path.join(folder, "model.pt"), weights_only=False)
        self.settings = s = m["settings"]
        self.base_node, self.base_meta, self.known = m["base_node"], m["base_meta"], m["known"]
        self.topk = m["topk"].numpy().astype(np.int64)
        self.paths = m["paths"]                                   # G' generated at the end of Stage 1
        g, d = s["generator"], s["detector"]
        self.gen = RPPGGenerator(self.base_node.size(1), self.content.size(1), g["dim"], standardize=g["standardize"])
        self.gen.load_state_dict(m["generator"])
        self.gen.set_known(self.known)
        self.gen.eval()
        self.detector = TemporalAwareDetector(**d)
        self.detector.load_state_dict(m["detector"])
        self.detector.eval()
        with torch.no_grad():
            self.user_emb = self.gen.user_table(self.base_node, self.base_meta)   # user vectors used by Module 2 + 3
        self.env = PropagationEnv(self.data, self.topk, g["max_sources"], g["max_repost"], g["max_coengage"], g["max_actions"])
        self.collate = Collate(self.user_emb, self.content)
        self.demo = []
        demo_csv = os.path.join(folder, "demo.csv")
        if os.path.exists(demo_csv):
            with open(demo_csv, encoding="utf-8") as f:
                self.demo = [dict(r, index=int(r["index"])) for r in csv.DictReader(f)]
        self.U, self.N = self.data["num_users"], self.data["num_news"]
        self.labels = self.data["labels"].numpy() if torch.is_tensor(self.data["labels"]) else np.asarray(self.data["labels"])
        self._split_of = {i: k for k, v in self.data["split"].items() for i in v}
        self.ext = None                                           # news from events outside the training data
        ext = os.path.join(folder, "external.pt")
        if os.path.exists(ext):
            self.ext = torch.load(ext, weights_only=False)
            self.ext["topk"] = self.ext["topk"].numpy().astype(np.int64)
            self.ext_collate = Collate(self.user_emb, self.ext["content"])

    # ------------------------------------------------------------------ data
    @staticmethod
    def is_ext(key: Key) -> bool:
        return isinstance(key, str) and key.startswith("ext")

    def ext_keys(self) -> List[str]:
        return [f"ext{j}" for j in range(len(self.ext["items"]))] if self.ext else []

    def _j(self, key):
        return int(key[3:])

    def label(self, key: Key) -> int:
        return self.ext["items"][self._j(key)]["label"] if self.is_ext(key) else int(self.labels[key])

    def item(self, key: Key) -> dict:
        if self.is_ext(key):
            it = self.ext["items"][self._j(key)]
            img = os.path.join(self.folder, "external_images", f"{it['id']}.jpg")
            return {"index": key, "id": it["id"], "text": it["text"], "split": f"external · {it['event']}",
                    "label": "real" if it["label"] == 1 else "fake", "image": img if os.path.exists(img) else None,
                    "eng": it["eng"], "early": it["early"], "event": it["event"]}
        img = os.path.join(self.folder, "images", f"{self.data['ids'][key]}.jpg")
        return {"index": key, "id": self.data["ids"][key], "text": self.data["texts"][key], "split": self._split_of[key],
                "label": "real" if self.labels[key] == 1 else "fake", "image": img if os.path.exists(img) else None,
                "eng": self.data["eng"][key], "early": self.data["early"][key]}

    def stored_steps(self, key: Key):
        """G' used for the reported results (Stage 1 output, or the logged predict.py run for external news)."""
        return self.ext["items"][self._j(key)]["steps"] if self.is_ext(key) else self.paths[key]["steps"]

    def content_topk(self, key: Key):
        return self.ext["topk"][self._j(key)] if self.is_ext(key) else self.topk[key]

    def _content(self, key: Key):
        return self.ext["content"][self._j(key)] if self.is_ext(key) else self.content[key]

    def find(self, text: str, split: Optional[str] = "test") -> int:
        """Index of the first news whose text contains ``text`` (inside ``split`` when given; texts can repeat)."""
        pool = set(self.data["split"][split]) if split else None
        return next(i for i, t in enumerate(self.data["texts"]) if text.lower() in t.lower() and (pool is None or i in pool))

    def known_share(self, key: Key) -> float:
        early = self.item(key)["early"]
        return sum(bool(self.known[u]) for _, u, _ in early) / max(len(early), 1)

    # ------------------------------------------------------------------ Module 1: users in graph H
    def fake_share(self):
        """Share of fake news among the training news each user engaged (only for colouring plots)."""
        if not hasattr(self, "_share"):
            fake, tot = np.zeros(self.U), np.zeros(self.U)
            for i in self.data["split"]["train"]:
                for _, u, _ in self.data["eng"][i]:
                    tot[u] += 1
                    fake[u] += self.labels[i] == 0
            self._share = (fake / np.maximum(tot, 1), tot)
        return self._share

    def neighbours(self, user: int, k: int = 5):
        """Most similar known users to ``user`` (cosine of the Module 1 -> Eq. 7 user vectors)."""
        E = F.normalize(self.user_emb, dim=-1)
        sims = E @ E[user]
        sims[~self.known] = -2
        sims[user] = -2
        top = torch.topk(sims, k)
        return list(zip(top.indices.tolist(), top.values.tolist()))

    def ego(self, user: int, max_news: int = 6, max_users: int = 12):
        """Part of graph H around one user: news it engaged and users linked by repost edges."""
        ei, et = self.data["graph"]["edge_index"], self.data["graph"]["edge_type"]
        U = self.U
        all_news = ei[1][(ei[0] == user) & (et == REL_ENGAGE) & (ei[1] >= U)].unique().tolist()
        news = all_news[:max_news]
        reps = ei[1][(ei[0] == user) & (et == REL_REPOST)].unique().tolist()
        reps += ei[0][(ei[1] == user) & (et == REL_REPOST)].unique().tolist()
        reps = [u for u in dict.fromkeys(reps) if u != user]
        n_rep = len(reps)
        reps = reps[:max_users]
        co = {}
        for n in news:
            us = ei[1][(ei[0] == n) & (et == REL_ENGAGE) & (ei[1] < U)].unique().tolist()
            co[n] = [u for u in us if u != user][:4]
        return {"user": user, "news": [n - U for n in news], "repost": reps, "co": {n - U: v for n, v in co.items()},
                "n_news": len(all_news), "n_repost": n_rep}

    def first_known_user(self, key: Key):
        """First early user of the news that exists in graph H (None if all are new users)."""
        return next((u for _, u, _ in self.item(key)["early"] if self.known[u]), None)

    # ------------------------------------------------------------------ Module 2: generate G'
    def generate(self, key: Key, seed: int = 0):
        """Generate G' for a news item with the trained policy and record what happened at every step."""
        g = self.settings["generator"]
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        n_vec = self.gen.news_proj(self._content(key))
        news = -1 if self.is_ext(key) else key
        saved = self.env.content_topk
        if self.is_ext(key):   # news outside graph H: its content candidates come from the external file
            self.env.content_topk = self.ext["topk"][self._j(key): self._j(key) + 1]
        try:
            return self._rollout_trace(news, self.item(key)["early"], n_vec, rng, g)
        finally:
            self.env.content_topk = saved

    @torch.no_grad()
    def _rollout_trace(self, news, early, n_vec, rng, g):
        """Same steps as generator.rollout(train=False), plus a per-step record for plotting."""
        U = self.user_emb
        path = PathGraph()
        for h, u, _ in early:
            path.add(u, -1 if h == NEWS_ROOT else h)
        new_users = list(path.users)
        steps, records = [], []
        for t in range(g["max_steps"]):
            if len(path.users) >= g["T"]:
                break
            users, parents, rels = self.env.candidates(news, np.asarray(path.users, dtype=np.int64), set(path.users), rng)
            s = self.gen.state(path, new_users, U, n_vec)
            a = self.gen.action_embeddings(users, rels, U)
            probs = self.gen.scores(self.gen.actor, s, a).softmax(-1)
            k = min(g["K2"], g["T"] - len(path.users), a.size(0))
            pick = torch.multinomial(probs, k, replacement=False)
            added, step = [], []
            for j in pick.tolist():
                if j == len(users):
                    continue
                if path.add(int(users[j]), int(parents[j])):
                    added.append(int(users[j]))
                    step.append((int(parents[j]), int(users[j])))
            steps.append(step)
            p = probs.numpy()
            records.append({"step": t, "n_candidates": len(users),
                            "cand_rel": {r: int((rels == r).sum()) for r in REL_NAMES},
                            "pick_rel": {r: int(sum(rels[j] == r for j in pick.tolist() if j < len(users))) for r in REL_NAMES},
                            "p_picked": float(p[pick.numpy()].sum()), "p_stop": float(p[-1]),
                            "entropy": float(-(p * np.log(p + 1e-12)).sum()), "path_size": len(path.users)})
            new_users = added
            if not added and len(users) == 0:
                break
        return steps, records

    def hits(self, key: Key, steps) -> List[int]:
        it = self.item(key)
        real = {u for _, u, _ in it["eng"]} - {0}
        early = {u for _, u, _ in it["early"]}
        return [c for st in steps for _, c in st if c in real and c not in early]

    # ------------------------------------------------------------------ Module 3: detection
    def sample(self, key: Key, steps=None, n_steps: Optional[int] = None):
        d = self.settings["data"]
        it = self.item(key)
        steps = self.stored_steps(key) if steps is None else steps
        if n_steps is not None:
            steps = steps[:n_steps]
        return make_sample(it["eng"], it["early"], steps, d["eval_real"], d["real_slots"], d["max_real_users"], d["use_generated"])

    def _batch(self, keys, samples):
        if self.is_ext(keys[0]):
            return self.ext_collate([(self._j(k), s, float(self.label(k))) for k, s in zip(keys, samples)])
        return self.collate([(k, s, float(self.label(k))) for k, s in zip(keys, samples)])

    @torch.no_grad()
    def predict(self, key: Key, steps=None, n_steps: Optional[int] = None, explain: bool = False):
        """P(real) for one news item; with explain=True also the attention maps and the split of the final logit."""
        s = self.sample(key, steps, n_steps)
        batch = self._batch([key], [s])
        if not explain:
            return torch.sigmoid(self.detector(batch)).item()
        cap, hooks = {}, []
        det = self.detector
        hooks.append(det.self_attn.register_forward_hook(lambda m, a, o: cap.__setitem__("self", o[1][0])))
        hooks.append(det.cross_attn.register_forward_hook(lambda m, a, o: cap.__setitem__("cross", o[1][0])))
        hooks.append(det.out.register_forward_hook(lambda m, a, o: cap.__setitem__("z", a[0][0])))
        logit = det(batch).item()
        for h in hooks:
            h.remove()
        W, b = det.out.weight[0], det.out.bias.item()
        dg = W.numel() - self.content.size(1)
        half = self.content.size(1) // 2
        z = cap["z"]
        parts = {"path h*": float(W[:dg] @ z[:dg]), "text": float(W[dg:dg + half] @ z[dg:dg + half]),
                 "image": float(W[dg + half:] @ z[dg + half:]), "bias": b}
        return {"index": key, "p_real": float(torch.sigmoid(torch.tensor(logit))), "logit": logit, "parts": parts,
                "self_attn": cap["self"].numpy(), "cross_attn": cap["cross"].numpy(), "sample": s}

    @torch.no_grad()
    def evaluate(self, split: str = "test", batch_size: int = 32):
        """Metrics on a split of the dataset ('train' / 'val' / 'test') or on the external set ('external')."""
        keys = self.ext_keys() if split == "external" else list(self.data["split"][split])
        probs = []
        for b in range(0, len(keys), batch_size):
            ks = keys[b:b + batch_size]
            probs += torch.sigmoid(self.detector(self._batch(ks, [self.sample(k) for k in ks]))).tolist()
        p, y = np.array(probs), np.array([self.label(k) for k in keys])
        pred = p > 0.5
        tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        fp, fn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        pos, neg = p[y == 1], p[y == 0]
        auc = float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / max(len(pos) * len(neg), 1))
        return {"n": len(keys), "real": int((y == 1).sum()), "fake": int((y == 0).sum()), "acc": float((pred == y).mean()),
                "precision": pre, "recall": rec, "f1": 2 * pre * rec / max(pre + rec, 1e-9), "auc": auc,
                "acc_real": float(pred[y == 1].mean()) if len(pos) else float("nan"),
                "acc_fake": float((~pred[y == 0]).mean()) if len(neg) else float("nan"),
                "confusion": [[tp, fn], [fp, tn]], "index": keys, "p_real": p, "y": y}
