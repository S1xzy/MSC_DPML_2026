"""Compare Stage 2 configurations by VALIDATION accuracy (selection) next to their test results (report).

    python scripts/compare_tags.py --dataset gossipcop --tags early_final early_final_g20 early_final_g50

For each tag: mean best-epoch validation accuracy over seeds (from the CSV logs, sanity check excluded) and test
Acc / F1 / AUC mean ± std. The tag with the highest mean validation accuracy is marked; test is never used to choose.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs(dataset, tag, seeds):
    best = {}
    for p in glob.glob(os.path.join(ROOT, "logs", "stage2_detect", "runs", "*", "test_metrics.json")):
        with open(p) as f:
            m = json.load(f)
        if m.get("tag") != tag or m.get("dataset") != dataset or m["seed"] not in seeds:
            continue
        if m["seed"] in best and os.path.getmtime(p) <= best[m["seed"]][0]:
            continue
        with open(glob.glob(os.path.join(os.path.dirname(p), "csv", "*", "metrics.csv"))[0]) as fh:
            val = max(float(r["val/acc"]) for r in csv.DictReader(fh) if r.get("val/acc"))
        best[m["seed"]] = (os.path.getmtime(p), val, m)
    return {s: v[1:] for s, v in best.items()}


def ms(v):
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    rows = []
    for t in a.tags:
        r = runs(a.dataset, t, set(a.seeds))
        if not r:
            print(f"{t:28s} (no runs)")
            continue
        val = [v for v, _ in r.values()]
        rows.append((t, len(r), st.mean(val), [m["test/acc"] for _, m in r.values()],
                     [m["test/f1"] for _, m in r.values()], [m["test/auc"] for _, m in r.values()]))
    top = max(rows, key=lambda x: x[2])[0] if rows else None
    print(f"{a.dataset}: {'tag':28s} n  val Acc   test Acc        test F1         test AUC")
    for t, n, v, acc, f1, au in rows:
        print(f"{'*' if t == top else ' '} {t:28s} {n}  {v:.3f}   {ms(acc):15s} {ms(f1):15s} {ms(au)}")


if __name__ == "__main__":
    main()
