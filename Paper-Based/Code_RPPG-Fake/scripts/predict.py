"""Run a trained RPPG-Fake model on news it has never seen (inference / demo, Algorithm 1 at test time).

    python scripts/predict.py --dataset pheme --input data/pheme_demo --tag early_final
    python scripts/predict.py --dataset pheme --input data/pheme_demo --tag early_final --seed 1

Input folder (same format as training): ``news_actions_<name>.csv`` (id, text, label?, actions_id[, event]),
``images/<id>.jpg`` and ``user_id_map.json`` built with the training map, so users seen in training keep their
id; users the model has never seen (id >= number of training users) are mapped to the unused id 0.

For every news item: early-stage engagements (3-5 earliest, Sec. IV-B) -> CLIP content n = [n_T ; n_V] ->
generated path G' from the trained generator (Module 2, on the training graph H) -> temporal-aware detector
(Module 3) -> P(real). Without ``--seed`` the seed with the best *validation* accuracy is used.
Output: ``logs/predict_<input>_<tag>_seed<k>.csv`` (+ ``.json`` with metrics when labels are given).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.components.encode_content import encode  # noqa: E402
from src.data.components.politifact import NEWS_ROOT, early_engagements, load_news  # noqa: E402
from src.data.rppg_datamodule import Collate, make_sample  # noqa: E402
from src.models.components.rppg.contrastive import HeteroGraphContrastive  # noqa: E402
from src.models.components.rppg.generator import PropagationEnv, RPPGGenerator, rollout  # noqa: E402
from src.models.rppg_detector_module import RPPGDetectorModule  # noqa: E402
from src.train_hete_RL import content_candidates  # noqa: E402

META_ORDER = ("UNU", "NUN", "UU")


def val_acc(run_dir):
    """Best epoch-level validation accuracy, read from the CSV log (excludes the sanity-check batches)."""
    f = glob.glob(os.path.join(run_dir, "csv", "*", "metrics.csv"))[0]
    with open(f) as fh:
        return max(float(r["val/acc"]) for r in csv.DictReader(fh) if r.get("val/acc"))


def find_runs(dataset, tag):
    """Stage 2 runs of this dataset / tag: {seed: (val_acc, run_dir, metrics)} (newest run per seed)."""
    best = {}
    for p in glob.glob(os.path.join(ROOT, "logs", "stage2_detect", "runs", "*", "test_metrics.json")):
        with open(p) as f:
            m = json.load(f)
        if m.get("tag") != tag or m.get("dataset") != dataset:
            continue
        d = os.path.dirname(p)
        if m["seed"] not in best or os.path.getmtime(p) > best[m["seed"]][3]:
            best[m["seed"]] = (val_acc(d), d, m, os.path.getmtime(p))
    return {s: v[:3] for s, v in best.items()}


def auc(prob, y):
    prob, y = np.asarray(prob), np.asarray(y)
    pos, neg = prob[y == 1], prob[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="training dataset of the model (folder under data/)")
    ap.add_argument("--input", required=True, help="folder with the unseen news")
    ap.add_argument("--tag", required=True, help="Stage 2 run tag of the detector, e.g. early_final")
    ap.add_argument("--seed", type=int, default=None, help="default: seed with the best validation accuracy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    runs = find_runs(a.dataset, a.tag)
    if not runs:
        raise SystemExit(f"no Stage 2 run with tag={a.tag} for dataset={a.dataset}")
    seed = a.seed if a.seed is not None else max(runs, key=lambda s: runs[s][0])
    val_acc, run_dir, m2 = runs[seed]
    cfg2 = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    gen_dir = os.path.join(cfg2.data.generated_dir, f"seed{seed}")
    cfg1 = OmegaConf.load(os.path.join(gen_dir, "stage1_config.yaml"))
    print(f"model: dataset={a.dataset} tag={a.tag} seed={seed} (val acc {val_acc:.3f}, test acc {m2['test/acc']:.3f})")
    print(f"  detector {m2['best_ckpt']}\n  generator {gen_dir}")

    # ---- training-side state: graph H, user embeddings, generator, contrastive encoder
    proc = os.path.join(ROOT, "data", a.dataset, "processed",
                        f"seed{seed}_{cfg1.data.slot_mode}{cfg1.data.n_slots}_T{cfg1.generator.T}.pt")
    data = torch.load(proc, weights_only=False)
    st1 = torch.load(os.path.join(gen_dir, "stage1.pt"), weights_only=False)
    U = data["num_users"]
    base_node, base_meta = st1["base_node"].to(dev), st1["base_meta"].to(dev)
    g = cfg1.generator
    gen = RPPGGenerator(base_node.size(1), 1024, g.dim, standardize=g.get("standardize", False)).to(dev)
    gen.load_state_dict(torch.load(os.path.join(gen_dir, "generator.ckpt"), map_location=dev))
    known = st1["known"].to(dev) if "known" in st1 else None
    if known is not None:
        gen.set_known(known)
    gen.eval()
    c = cfg1.contrastive
    cl = HeteroGraphContrastive(U, data["num_news"], 1024, len(META_ORDER), c.dim, c.heads, c.dropout).to(dev)
    cl.load_state_dict(torch.load(os.path.join(gen_dir, "contrastive.ckpt"), map_location=dev))
    cl.eval()
    det = RPPGDetectorModule.load_from_checkpoint(m2["best_ckpt"], map_location=dev).to(dev).eval()

    # ---- unseen news: content features and early engagements (unknown users -> id 0)
    news = load_news(a.input)
    to_id = lambda x: x if (x == NEWS_ROOT or x < U) else 0  # noqa: E731
    for n in news:
        n["eng"] = [(to_id(h), to_id(u), t) for h, u, t in n["eng"]]
    events = {}
    with open(glob.glob(os.path.join(a.input, "news_actions_*.csv"))[0], encoding="utf-8") as f:
        for r in csv.DictReader(f):
            events[r["id"]] = r.get("event", "")
    feats = encode([n["text"] for n in news], [n["id"] for n in news], os.path.join(a.input, "images"), a.device,
                   text_chunks=cfg2.data.get("text_chunks", 1))
    content = torch.cat([F.normalize(feats["text"], dim=-1), F.normalize(feats["image"], dim=-1)], -1).to(dev)
    with torch.no_grad():
        iso = cl.node_gat(cl.content_proj(content), torch.zeros(2, 0, dtype=torch.long, device=dev))
        Ut = gen.user_table(base_node, base_meta)
    topk = content_candidates(iso, base_node, g.content_topk, center=g.get("standardize", False), known=known)
    env = PropagationEnv(data, topk[:1], g.max_sources, g.max_repost, g.max_coengage, g.max_actions)
    rng, np_rng = random.Random(seed), np.random.default_rng(seed + 1000)
    samples, rows = {}, []
    with torch.no_grad():
        for j, n in enumerate(news):
            early = early_engagements(n["eng"], rng)
            env.content_topk = topk[j : j + 1]  # news index -1 = an item outside the training graph
            steps, _, _ = rollout(gen, env, -1, early, Ut, gen.news_proj(content[j]), np_rng, g.T, g.K2, g.max_steps,
                                  train=False, policy=g.get("mode", "rl"))
            samples[j] = make_sample(n["eng"], early, steps, cfg2.data.eval_real, cfg2.data.real_slots,
                                     cfg2.data.max_real_users, cfg2.data.use_generated)
            rows.append({"id": n["id"], "event": events.get(n["id"], ""), "label": n["label"],
                         "early_users": len(early), "early_known": sum(bool(known[u]) if known is not None else u != 0 for _, u, _ in early),
                         "generated_users": sum(len(s) for s in steps), "text": n["text"][:140].replace("\n", " ")})
        collate = Collate(st1["user_emb"].float(), content.cpu())
        probs = []
        for b in range(0, len(news), 16):
            items = [(j, samples[j], rows[j]["label"]) for j in range(b, min(b + 16, len(news)))]
            batch = collate(items)
            batch = {k: ({kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict)
                         else v.to(dev)) for k, v in batch.items()}
            probs += torch.sigmoid(det(batch)).cpu().tolist()
    for r, p in zip(rows, probs):
        r["p_real"] = round(p, 4)
        r["pred"] = "real" if p > 0.5 else "fake"
        r["truth"] = "real" if r["label"] == 1 else "fake"
        r["correct"] = int((p > 0.5) == (r["label"] == 1))

    name = os.path.basename(os.path.normpath(a.input))
    out = os.path.join(ROOT, "logs", f"predict_{name}_{a.tag}_seed{seed}")
    cols = ["id", "event", "truth", "pred", "p_real", "correct", "early_users", "early_known", "generated_users", "text"]
    with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    y = np.array([r["label"] for r in rows])
    pr = np.array(probs) > 0.5
    tp, tn = int(((pr == 1) & (y == 1)).sum()), int(((pr == 0) & (y == 0)).sum())
    fp, fn = int(((pr == 1) & (y == 0)).sum()), int(((pr == 0) & (y == 1)).sum())
    pre = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    summary = {"dataset": a.dataset, "input": name, "tag": a.tag, "seed": seed, "val_acc": val_acc, "n": len(rows),
               "fake": int((y == 0).sum()), "real": int((y == 1).sum()), "acc": float((pr == y).mean()), "pre": pre,
               "rec": rec, "f1": 2 * pre * rec / max(pre + rec, 1e-9), "auc": auc(probs, y),
               "confusion": {"real_as_real": tp, "fake_as_fake": tn, "fake_as_real": fp, "real_as_fake": fn},
               "by_event": {ev: {"n": sum(r["event"] == ev for r in rows),
                                 "acc": float(np.mean([r["correct"] for r in rows if r["event"] == ev]))}
                            for ev in sorted({r["event"] for r in rows})},
               "known_user_share_early": float(np.mean([r["early_known"] / max(r["early_users"], 1) for r in rows]))}
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_event"}, indent=1))
    print("by event:", summary["by_event"])
    print("->", out + ".csv")


if __name__ == "__main__":
    main()
