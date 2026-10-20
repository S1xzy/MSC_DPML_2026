"""Why does the generated path G' not help detection?  Diagnostics on existing Stage 1 outputs (no training).

    python scripts/diagnose_generated.py --datasets pheme gossipcop politifact --gen_name generated_paper

For every dataset / seed (test news only) it reports
  * source of generated users: content candidates (parent = news) vs graph actions (repost / co-engage)
  * hit-rate: share of generated users that really engaged the news, against two trivial baselines
    (random training users, the news' top content-similar users)
  * diversity: distinct generated users / all generated users, share of the 20 most frequent users
  * label signal: AUC of a user-credibility score (share of a user's *training* engagements that are fake news)
    computed on (a) the early real users, (b) the generated users, (c) the real users after the early stage.
    If (b) carries no more signal than (a) while (c) does, G' adds nothing the detector does not already see.
Results -> logs/diagnose_generated.json (+ printed table).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auc(scores, y):
    """ROC-AUC of predicting fake (y == 0) from a higher score; NaN scores are dropped."""
    s, y = np.asarray(scores, float), np.asarray(y)
    ok = ~np.isnan(s)
    s, pos = s[ok], (y[ok] == 0)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def credibility(data, train):
    """Smoothed share of fake news among each user's training engagements (Beta(1,1) prior); NaN = unseen."""
    fake, tot = Counter(), Counter()
    for i in train:
        y = int(data["labels"][i])
        for u in {u for _, u, _ in data["eng"][i]}:
            tot[u] += 1
            fake[u] += y == 0
    return {u: (fake[u] + 1) / (tot[u] + 2) for u in tot}


def mean_score(users, cred):
    v = [cred[u] for u in users if u in cred]
    return (float(np.mean(v)) if v else float("nan")), len(v) / max(len(users), 1)


def diagnose(ds, seed, gen_name, T):
    data = torch.load(os.path.join(ROOT, "data", ds, "processed", f"seed{seed}_time20_T{T}.pt"), weights_only=False)
    gen = torch.load(os.path.join(ROOT, "data", ds, gen_name, f"seed{seed}", "stage1.pt"), weights_only=False)
    rng = np.random.default_rng(seed)
    train, test = data["split"]["train"], data["split"]["test"]
    labels = [int(data["labels"][i]) for i in test]
    cred = credibility(data, train)
    train_users = np.array(sorted(cred))
    n_content = n_all = hit = hit_rand = 0
    allgen = Counter()
    sc = {"early": [], "generated": [], "future": []}
    cov = {"early": [], "generated": [], "future": []}
    for i in test:
        real = {u for _, u, _ in data["eng"][i]}
        early = [u for _, u, _ in data["early"][i]]
        steps = gen["paths"][i]["steps"]
        g = [c for st in steps for _, c in st]
        n_content += sum(p == -1 for st in steps for p, _ in st)
        n_all += len(g)
        hit += sum(u in real for u in g)
        rnd = rng.choice(train_users, min(len(g), len(train_users)), replace=False) if g else []
        hit_rand += sum(u in real for u in rnd)
        allgen.update(g)
        future = [u for _, u, _ in data["eng"][i] if u not in set(early)]
        for k, us in (("early", early), ("generated", g), ("future", future)):
            s, c = mean_score(us, cred)
            sc[k].append(s), cov[k].append(c)
    top20 = sum(c for _, c in allgen.most_common(20))
    return {
        "content_share": n_content / max(n_all, 1),
        "hit_rate": hit / max(n_all, 1),
        "hit_rate_random": hit_rand / max(n_all, 1),
        "distinct_share": len(allgen) / max(n_all, 1),
        "top20_share": top20 / max(n_all, 1),
        "auc": {k: auc(v, labels) for k, v in sc.items()},
        "coverage": {k: float(np.mean(v)) for k, v in cov.items()},
        "avg_generated": n_all / len(test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pheme", "gossipcop", "politifact"])
    ap.add_argument("--gen_name", default="generated_paper")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--out", default="logs/diagnose_generated.json")
    a = ap.parse_args()
    out = {}
    for ds in a.datasets:
        rows = [diagnose(ds, s, a.gen_name, a.T) for s in a.seeds
                if os.path.exists(os.path.join(ROOT, "data", ds, a.gen_name, f"seed{s}", "stage1.pt"))]
        flat = lambda f: float(np.nanmean([f(r) for r in rows]))  # noqa: E731
        out[ds] = {"n_seeds": len(rows),
                   "content_share": flat(lambda r: r["content_share"]),
                   "hit_rate": flat(lambda r: r["hit_rate"]), "hit_rate_random": flat(lambda r: r["hit_rate_random"]),
                   "distinct_share": flat(lambda r: r["distinct_share"]), "top20_share": flat(lambda r: r["top20_share"]),
                   "avg_generated": flat(lambda r: r["avg_generated"]),
                   "auc": {k: flat(lambda r, k=k: r["auc"][k]) for k in ("early", "generated", "future")},
                   "coverage": {k: flat(lambda r, k=k: r["coverage"][k]) for k in ("early", "generated", "future")},
                   "per_seed": rows}
        o = out[ds]
        print(f"\n== {ds} ({o['n_seeds']} seeds, test news)")
        print(f"  generated users / news      {o['avg_generated']:.1f}")
        print(f"  from content candidates     {o['content_share']:.1%}")
        print(f"  hit-rate  RL {o['hit_rate']:.2%}   random train users {o['hit_rate_random']:.2%}")
        print(f"  distinct users              {o['distinct_share']:.1%}   top-20 users = {o['top20_share']:.1%} of picks")
        print("  credibility AUC (fake)      " + "  ".join(f"{k} {o['auc'][k]:.3f} (cov {o['coverage'][k]:.0%})" for k in o["auc"]))
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
