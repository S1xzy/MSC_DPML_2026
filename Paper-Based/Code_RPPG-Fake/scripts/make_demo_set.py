"""Pick a small, presentation-ready demo set from the TEST split of the model that will be presented.

    python scripts/make_demo_set.py --dataset pheme --tag early_fix --n_fake 3 --n_real 3
    python scripts/make_demo_set.py --dataset gossipcop --tag early_fix --n_fake 2 --n_real 2

The seed is the one with the best VALIDATION accuracy for ``--tag`` (same rule as scripts/predict.py), so every
picked item is a news the presented model never trained on. Items are drawn at random (fixed seed) among test news
that have an image and >= 3 early engagements; short texts are preferred so they fit on a slide.
Output: ``data/demo_present/<dataset>/`` in the predict.py input format (CSV + images/), plus ``demo_items.json``
(id, label, short text, early users) for the slides. Then run predict.py on that folder.
"""
from __future__ import annotations

import argparse
import csv
import io
import contextlib
import json
import os
import random
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
with contextlib.redirect_stderr(io.StringIO()):
    from predict import find_runs  # noqa: E402

csv.field_size_limit(2**31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="early_fix")
    ap.add_argument("--n_fake", type=int, default=3)
    ap.add_argument("--n_real", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=40, help="prefer texts (first line) up to this many words")
    a = ap.parse_args()
    runs = find_runs(a.dataset, a.tag)
    seed = max(runs, key=lambda s: runs[s][0])
    proc = [p for p in os.listdir(os.path.join(ROOT, "data", a.dataset, "processed")) if p.startswith(f"seed{seed}_time20_T200")][0]
    d = torch.load(os.path.join(ROOT, "data", a.dataset, "processed", proc), weights_only=False)
    test = set(d["split"]["test"])
    src = os.path.join(ROOT, "data", a.dataset)
    with open(os.path.join(src, f"news_actions_{a.dataset}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["id"]: r for r in rows}
    imgs = {os.path.splitext(n)[0]: n for n in os.listdir(os.path.join(src, "images"))}
    cand = {0: [], 1: []}
    for i in sorted(test):
        nid = d["ids"][i]
        first = str(d["texts"][i]).strip().split("\n")[0]
        if nid in imgs and len(d["early"][i]) >= 3:
            cand[int(d["labels"][i])].append((len(first.split()) > a.max_words, i, nid, first))
    rng = random.Random(0)
    pick = []
    for lab, n in ((0, a.n_fake), (1, a.n_real)):
        pool = cand[lab]
        rng.shuffle(pool)
        pool.sort(key=lambda x: x[0])  # short texts first, random order among them
        pick += [(lab,) + x[1:] for x in pool[:n]]
    out = os.path.join(ROOT, "data", "demo_present", a.dataset)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"))
    with open(os.path.join(out, f"news_actions_demo_{a.dataset}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "actions_id", "event"])
        w.writeheader()
        for lab, i, nid, _ in pick:
            r = by_id[nid]
            w.writerow({"id": nid, "text": r["text"], "label": r["label"], "actions_id": r["actions_id"], "event": a.dataset})
            shutil.copy2(os.path.join(src, "images", imgs[nid]), os.path.join(out, "images", imgs[nid]))
    items = [{"id": nid, "label": "real" if lab else "fake", "text": first[:200], "image": f"images/{imgs[nid]}",
              "early": [(h, u) for h, u, _ in d["early"][i]], "n_engagements": len(d["eng"][i])}
             for lab, i, nid, first in pick]
    with open(os.path.join(out, "demo_items.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": a.dataset, "tag": a.tag, "seed": seed, "val_acc": runs[seed][0], "items": items}, f,
                  ensure_ascii=False, indent=2)
    print(f"{a.dataset}: seed {seed} (val {runs[seed][0]:.3f}) -> {len(pick)} test items in {out}")
    for it in items:
        print(f"  [{it['label']}] {it['id']}  {it['text'][:90]}")


if __name__ == "__main__":
    main()
