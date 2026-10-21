"""Add news from outside the training data (PHEME events never seen in training) to the demo bundle.

Same steps as Code_RPPG-Fake/scripts/predict.py (seed with the best validation accuracy): unknown users -> UNK,
early path of 3-5 engagements, CLIP content, G' from the trained generator. The result is saved so the notebook
does not need CLIP:  data/<dataset>/external.pt  +  data/<dataset>/external_images/

    cd Paper-Based/Code_RPPG-Fake
    .venv/Scripts/python ../Demo_RPPG-Fake/tools/export_external.py --dataset pheme --input data/pheme_demo
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from predict import META_ORDER, find_runs  # noqa: E402
from src.data.components.encode_content import encode  # noqa: E402
from src.data.components.politifact import NEWS_ROOT, early_engagements, load_news  # noqa: E402
from src.data.rppg_datamodule import Collate, make_sample  # noqa: E402
from src.models.components.rppg.contrastive import HeteroGraphContrastive  # noqa: E402
from src.models.components.rppg.generator import PropagationEnv, RPPGGenerator, rollout  # noqa: E402
from src.models.rppg_detector_module import RPPGDetectorModule  # noqa: E402
from src.train_hete_RL import content_candidates  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SOURCE = {"pheme_demo": "PHEME 9 events (figshare 6392078, CC BY 4.0): events ebola-essien, gurlitt, prince-toronto, "
                        "putinmissing, which are not in the training data; only threads with text + image + path"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pheme")
    ap.add_argument("--input", default="data/pheme_demo")
    ap.add_argument("--tag", default="early_fix")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_runs(a.dataset, a.tag)
    seed = max(runs, key=lambda s: runs[s][0])
    val_acc, run_dir, m2 = runs[seed]
    cfg2 = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    gen_dir = os.path.join(cfg2.data.generated_dir, f"seed{seed}")
    cfg1 = OmegaConf.load(os.path.join(gen_dir, "stage1_config.yaml"))
    proc = os.path.join(ROOT, "data", a.dataset, "processed", f"seed{seed}_{cfg1.data.slot_mode}{cfg1.data.n_slots}_T{cfg1.generator.T}.pt")
    data = torch.load(proc, weights_only=False)
    st1 = torch.load(os.path.join(gen_dir, "stage1.pt"), weights_only=False)
    U = data["num_users"]
    base_node, base_meta = st1["base_node"].to(dev), st1["base_meta"].to(dev)
    g = cfg1.generator
    gen = RPPGGenerator(base_node.size(1), 1024, g.dim, standardize=g.get("standardize", False)).to(dev)
    gen.load_state_dict(torch.load(os.path.join(gen_dir, "generator.ckpt"), map_location=dev))
    known = st1["known"].to(dev)
    gen.set_known(known)
    gen.eval()
    c = cfg1.contrastive
    cl = HeteroGraphContrastive(U, data["num_news"], 1024, len(META_ORDER), c.dim, c.heads, c.dropout).to(dev)
    cl.load_state_dict(torch.load(os.path.join(gen_dir, "contrastive.ckpt"), map_location=dev))
    cl.eval()
    det = RPPGDetectorModule.load_from_checkpoint(m2["best_ckpt"], map_location=dev).to(dev).eval()

    news = load_news(a.input)
    to_id = lambda x: x if (x == NEWS_ROOT or x < U) else 0  # noqa: E731
    raw_users = []
    for n in news:
        raw_users.append([u for _, u, _ in n["eng"]])
        n["eng"] = [(to_id(h), to_id(u), t) for h, u, t in n["eng"]]
    events = {}
    with open(glob.glob(os.path.join(a.input, "news_actions_*.csv"))[0], encoding="utf-8") as f:
        for r in csv.DictReader(f):
            events[r["id"]] = r.get("event", "")
    feats = encode([n["text"] for n in news], [n["id"] for n in news], os.path.join(a.input, "images"), a.device if hasattr(a, "device") else str(dev),
                   text_chunks=cfg2.data.get("text_chunks", 1))
    content = torch.cat([F.normalize(feats["text"], dim=-1), F.normalize(feats["image"], dim=-1)], -1).to(dev)
    with torch.no_grad():
        iso = cl.node_gat(cl.content_proj(content), torch.zeros(2, 0, dtype=torch.long, device=dev))
        Ut = gen.user_table(base_node, base_meta)
    topk = content_candidates(iso, base_node, g.content_topk, center=g.get("standardize", False), known=known)
    env = PropagationEnv(data, topk[:1], g.max_sources, g.max_repost, g.max_coengage, g.max_actions)
    rng, np_rng = random.Random(seed), np.random.default_rng(seed + 1000)
    items = []
    with torch.no_grad():
        for j, n in enumerate(news):
            early = early_engagements(n["eng"], rng)
            env.content_topk = topk[j : j + 1]
            steps, _, _ = rollout(gen, env, -1, early, Ut, gen.news_proj(content[j]), np_rng, g.T, g.K2, g.max_steps,
                                  train=False, policy=g.get("mode", "rl"))
            items.append({"id": n["id"], "text": n["text"], "label": int(n["label"]), "event": events.get(n["id"], ""),
                          "eng": n["eng"], "early": early, "steps": steps,
                          "users_unknown": sum(u >= U for u in raw_users[j]), "users_total": len(raw_users[j])})
        collate = Collate(st1["user_emb"].float(), content.cpu())
        batch_items = [(j, make_sample(it["eng"], it["early"], it["steps"], cfg2.data.eval_real, cfg2.data.real_slots,
                                       cfg2.data.max_real_users, cfg2.data.use_generated), it["label"]) for j, it in enumerate(items)]
        probs = []
        for b in range(0, len(batch_items), 16):
            batch = collate(batch_items[b:b + 16])
            batch = {k: ({kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev))
                     for k, v in batch.items()}
            probs += torch.sigmoid(det(batch)).cpu().tolist()
    for it, p in zip(items, probs):
        it["p_real_logged"] = p
    y = np.array([it["label"] for it in items])
    acc = float(((np.array(probs) > 0.5) == y).mean())
    logged = json.load(open(os.path.join(ROOT, "logs", f"predict_{os.path.basename(os.path.normpath(a.input))}_{a.tag}_seed{seed}.json")))
    print(f"external {len(items)} news · acc {acc:.4f} (logged {logged['acc']:.4f})")
    out = os.path.join(OUT, a.dataset)
    img_out = os.path.join(out, "external_images")
    os.makedirs(img_out, exist_ok=True)
    for it in items:
        src = os.path.join(a.input, "images", f"{it['id']}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, img_out)
    torch.save({"name": os.path.basename(os.path.normpath(a.input)), "source": SOURCE.get(os.path.basename(os.path.normpath(a.input)), a.input),
                "items": items, "content": content.cpu().float(), "topk": torch.as_tensor(topk, dtype=torch.int32),
                "logged": {k: logged[k] for k in ("n", "fake", "real", "acc", "f1", "auc", "confusion")}},
               os.path.join(out, "external.pt"))
    print("saved", os.path.join(out, "external.pt"))


if __name__ == "__main__":
    main()
