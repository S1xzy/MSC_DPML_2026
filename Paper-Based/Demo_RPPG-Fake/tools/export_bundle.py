"""Build the demo data bundle (Demo_RPPG-Fake/data/<dataset>/) from the trained round-8 models in Code_RPPG-Fake.

Run from Code_RPPG-Fake with its venv (needs Hydra / Lightning to read the training runs):

    cd Paper-Based/Code_RPPG-Fake
    .venv/Scripts/python ../Demo_RPPG-Fake/tools/export_bundle.py --datasets pheme gossipcop

Per dataset it writes:
  graph.pt     graph H, news text / label / split, real engagements, early path   (processed data)
  content.pt   CLIP content n = [n_T ; n_V] of every news item (1024-d, already normalised)
  model.pt     Module 1 embeddings (base_node, base_meta, known), content candidates, stored G' paths,
               generator (Module 2) and detector (Module 3) weights, settings and logged metrics
  demo.csv     the demo news items (ids from data/demo_present/<dataset>/demo_items.json)
  images/      images of the test news
The script checks that the exported detector reproduces the logged test accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from predict import find_runs  # noqa: E402
from src.models.components.rppg.contrastive import HeteroGraphContrastive  # noqa: E402
from src.models.rppg_detector_module import RPPGDetectorModule  # noqa: E402
from src.train_hete_RL import content_candidates  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PAPER = {"pheme": {"early": 0.829, "full": 0.844}, "gossipcop": {"early": 0.873, "full": 0.891},
         "politifact": {"early": 0.902, "full": 0.924}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pheme", "gossipcop"])
    ap.add_argument("--tag", default="early_fix")
    a = ap.parse_args()
    for ds in a.datasets:
        runs = find_runs(ds, a.tag)
        seed = max(runs, key=lambda s: runs[s][0])
        val_acc, run_dir, m2 = runs[seed]
        cfg2 = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
        gen_dir = os.path.join(cfg2.data.generated_dir, f"seed{seed}")
        cfg1 = OmegaConf.load(os.path.join(gen_dir, "stage1_config.yaml"))
        print(f"[{ds}] seed {seed}  val {val_acc:.3f}  test {m2['test/acc']:.3f}")
        out = os.path.join(OUT, ds)
        os.makedirs(os.path.join(out, "images"), exist_ok=True)

        dm = instantiate(cfg2.data)
        dm.setup()
        proc = os.path.join(ROOT, "data", ds, "processed", f"seed{seed}_{cfg1.data.slot_mode}{cfg1.data.n_slots}_T{cfg1.generator.T}.pt")
        data = torch.load(proc, weights_only=False)
        st1 = torch.load(os.path.join(gen_dir, "stage1.pt"), weights_only=False)
        keep = ("num_users", "num_news", "ids", "texts", "labels", "split", "graph", "eng", "early", "seed")
        torch.save({k: data[k] for k in keep}, os.path.join(out, "graph.pt"))
        torch.save(dm.content.float(), os.path.join(out, "content.pt"))

        # content candidates of every news item (Module 1 content view -> users closest to the news)
        c, g = cfg1.contrastive, cfg1.generator
        U, N = data["num_users"], data["num_news"]
        cl = HeteroGraphContrastive(U, N, dm.content.size(1), 3, c.dim, c.heads, c.dropout)
        cl.load_state_dict(torch.load(os.path.join(gen_dir, "contrastive.ckpt"), map_location="cpu"))
        cl.eval()
        with torch.no_grad():
            iso = cl.node_gat(cl.features(dm.content), torch.zeros(2, 0, dtype=torch.long))[U:]
        known = st1["known"]
        topk = content_candidates(iso, st1["base_node"], g.content_topk, center=g.get("standardize", False),
                                  known=known if g.get("standardize", False) else None)

        det = RPPGDetectorModule.load_from_checkpoint(m2["best_ckpt"], map_location="cpu").eval()
        hp = {k: det.hparams[k] for k in ("user_dim", "content_dim", "dim", "heads", "dropout", "max_steps")}
        settings = {
            "dataset": ds, "seed": seed, "tag": a.tag,
            "generator": {k: g[k] for k in ("dim", "T", "K2", "max_steps", "max_sources", "max_repost", "max_coengage",
                                            "max_actions", "content_topk", "standardize", "mode", "lambda2", "gamma")},
            "contrastive": {k: c[k] for k in ("dim", "heads", "tau1", "tau2", "lambda1", "k1", "steps")},
            "detector": hp,
            "data": {k: cfg2.data[k] for k in ("eval_real", "real_slots", "max_real_users", "use_generated", "text_chunks")},
            "metrics": {"val_acc": val_acc, **{k: v for k, v in m2.items() if k.startswith("test/")}},
            "paper": PAPER.get(ds, {}),
        }
        torch.save({"base_node": st1["base_node"].float(), "base_meta": st1["base_meta"].float(), "known": known,
                    "topk": torch.as_tensor(topk, dtype=torch.int32), "paths": st1["paths"],
                    "generator": torch.load(os.path.join(gen_dir, "generator.ckpt"), map_location="cpu"),
                    "detector": det.net.state_dict(), "settings": settings}, os.path.join(out, "model.pt"))
        with open(os.path.join(out, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=1, default=float)

        # check: the exported detector on the stored test inputs gives the logged accuracy
        test = dm.sets["test"]
        correct = 0
        with torch.no_grad():
            for b in range(0, len(test.idx), 32):
                items = [(i, test.samples[i], test.labels[i]) for i in test.idx[b:b + 32]]
                batch = dm.collate(items)
                correct += int(((det.net(batch) > 0).float() == batch["y"]).sum())
        acc = correct / len(test.idx)
        print(f"  check: test acc {acc:.4f} (logged {m2['test/acc']:.4f})")
        assert abs(acc - m2["test/acc"]) < 1e-3

        # demo items + images of the test news
        demo_json = os.path.join(ROOT, "data", "demo_present", ds, "demo_items.json")
        index = {nid: i for i, nid in enumerate(data["ids"])}
        rows = []
        if os.path.exists(demo_json):
            for it in json.load(open(demo_json, encoding="utf-8"))["items"]:
                rows.append({"index": index[it["id"]], "id": it["id"], "label": it["label"], "text": it["text"]})
        with open(os.path.join(out, "demo.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["index", "id", "label", "text"])
            w.writeheader()
            w.writerows(rows)
        img_dir = os.path.join(ROOT, "data", ds, "images")
        size = n_img = 0
        for i in data["split"]["test"]:
            src = os.path.join(img_dir, f"{data['ids'][i]}.jpg")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, "images"))
                size += os.path.getsize(src)
                n_img += 1
        print(f"  demo items {len(rows)} · test images {n_img} ({size / 1e6:.1f} MB)")
        for f in ("graph.pt", "content.pt", "model.pt"):
            print(f"  {f}: {os.path.getsize(os.path.join(out, f)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
