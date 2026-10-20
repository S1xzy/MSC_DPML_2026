"""Run the full RPPG-Fake pipeline for several seeds and report mean +/- std (paper: 5 runs, Sec. IV-B).

    python scripts/run_all.py --seeds 0 1 2 3 4                 # Stage 1 + Stage 2 (early + full)
    python scripts/run_all.py --seeds 0 1 2 3 4 --skip_stage1   # reuse existing Stage 1 outputs
    python scripts/run_all.py --aggregate_only                  # just rebuild the results table

Each seed uses its own stratified 7:1:2 split. Results go to logs/results_<setting>.csv and
logs/results_summary.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SETTINGS = {
    "early": ["data.train_real=early", "data.eval_real=early"],                                # Table III
    "full": ["data.train_real=full", "data.eval_real=full"],                                   # Table II
    "early_no_gen": ["data.train_real=early", "data.eval_real=early", "data.use_generated=false"],  # ablation
}
METRICS = ["test/acc", "test/pre", "test/rec", "test/f1", "test/macro_f1", "test/auc"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def latest_metrics(tag: str, dataset: str):
    rows = []
    for p in glob.glob(os.path.join(ROOT, "logs", "stage2_detect", "runs", "*", "test_metrics.json")):
        with open(p) as f:
            m = json.load(f)
        if m.get("tag") == tag and m.get("dataset", "politifact") == dataset:
            rows.append((os.path.getmtime(p), m))
    best = {}
    for _, m in sorted(rows, key=lambda r: r[0]):
        best[m["seed"]] = m  # keep the newest run per seed
    return [best[s] for s in sorted(best)]


def aggregate(settings, dataset):
    lines = ["| Setting | n | " + " | ".join(m.split("/")[1] for m in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for name in settings:
        rows = latest_metrics(name, dataset)
        if not rows:
            continue
        with open(os.path.join(ROOT, "logs", f"results_{dataset}_{name}.csv"), "w") as f:
            f.write("seed," + ",".join(METRICS) + "\n")
            for r in rows:
                f.write(f"{r['seed']}," + ",".join(f"{r.get(m, float('nan')):.4f}" for m in METRICS) + "\n")
        cells = []
        for m in METRICS:
            v = [r[m] for r in rows if m in r]
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            cells.append(f"{statistics.mean(v):.3f} ± {sd:.3f}")
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    with open(os.path.join(ROOT, "logs", f"results_summary_{dataset}.md"), "w", encoding="utf-8") as f:
        f.write(f"# RPPG-Fake reproduction on {dataset} (mean ± std over seeds)\n\n" + table + "\n")
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="politifact")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--skip_stage1", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for Stage 2")
    ap.add_argument("--stage1_extra", nargs="*", default=[], help="extra Hydra overrides for Stage 1")
    ap.add_argument("--tag_suffix", default="", help="appended to the setting names, e.g. _mm for text+image runs")
    ap.add_argument("--gen_name", default="generated", help="folder under data/<dataset>/ for Stage 1 output")
    a = ap.parse_args()
    gen_root = os.path.join(ROOT, "data", a.dataset, a.gen_name).replace("\\", "/")
    if not a.aggregate_only:
        for s in a.seeds:
            if not a.skip_stage1 or not os.path.exists(os.path.join(gen_root, f"seed{s}", "stage1.pt")):
                run([PY, "src/train_hete_RL.py", f"seed={s}", f"dataset={a.dataset}", "extras.print_config=False",
                     f"output_dir={gen_root}/seed{s}", *a.stage1_extra])
            for name in a.settings:
                run([PY, "src/train.py", f"seed={s}", f"dataset={a.dataset}", "extras.print_config=False",
                     "callbacks.rich_progress_bar=null", f"+tag={name}{a.tag_suffix}", f"data.generated_dir={gen_root}",
                     *SETTINGS[name], *a.extra])
    aggregate([n + a.tag_suffix for n in a.settings], a.dataset)


if __name__ == "__main__":
    main()
