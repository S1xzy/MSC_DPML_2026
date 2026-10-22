"""Diagnostics for the PolitiFact gap: re-train Stage 2 (detector) and record val + test accuracy after every epoch.

Uses the data config of an existing run (same Stage 1 output, same split), so only Module 3 is re-trained.

    .venv/Scripts/python scripts/diagnose_protocol.py --dataset politifact --tag early_fix --seeds 0 1 2
    .venv/Scripts/python scripts/diagnose_protocol.py --dataset gossipcop --tag early_fix --seeds 0 1 2 --zero-image

Reports per run:
  ours       test acc at the epoch with the best val acc (our protocol, = Lightning ModelCheckpoint on val/acc)
  same-set   best test acc over epochs = selecting and reporting on the same set (what the official code's
             val/test swap leads to; see Code_RPPG-Fake/RUNNING.md)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from predict import find_runs  # noqa: E402
from src.models.rppg_detector_module import RPPGDetectorModule  # noqa: E402


@torch.no_grad()
def accuracy(model, loader, dev):
    model.eval()
    ok = n = 0
    for batch in loader:
        batch = to_dev(batch, dev)
        pred = (model(batch) > 0).float()
        ok += int((pred == batch["y"]).sum())
        n += len(pred)
    return ok / n


def to_dev(batch, dev):
    return {k: ({kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev))
            for k, v in batch.items()}


def run(cfg, seed, epochs, zero_image, dev, text_chunks=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if text_chunks is not None:  # detector content only; G' (Stage 1) is unchanged
        cfg.data.text_chunks = text_chunks
    dm = instantiate(cfg.data)
    dm.setup()
    if zero_image:  # content = [n_T ; n_V]: drop the image half (the Collate holds the same tensor)
        half = dm.content.size(1) // 2
        dm.content[:, half:] = 0
    model = RPPGDetectorModule(**{k: v for k, v in OmegaConf.to_container(cfg.model).items() if k != "_target_"}).to(dev)
    opt = model.configure_optimizers()
    tr, va, te = dm.train_dataloader(), dm.val_dataloader(), dm._loader("test", False)
    curve = []
    for ep in range(epochs):
        model.train()
        for batch in tr:
            batch = to_dev(batch, dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        curve.append((accuracy(model, va, dev), accuracy(model, te, dev)))
        print(f"  seed {seed} epoch {ep:2d}  val {curve[-1][0]:.3f}  test {curve[-1][1]:.3f}", flush=True)
    v = np.array([c[0] for c in curve]); t = np.array([c[1] for c in curve])
    best = int(np.argmax(v))  # first epoch with the best val acc (ModelCheckpoint keeps the first maximum)
    return {"seed": seed, "n_val": len(dm.sets["val"].idx), "n_test": len(dm.sets["test"].idx),
            "ours": float(t[best]), "best_epoch": best, "val_best": float(v[best]), "same_set": float(t.max()),
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="early_fix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--zero-image", action="store_true")
    ap.add_argument("--text-chunks", type=int, default=None, help="override data.text_chunks of the detector input")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_runs(a.dataset, a.tag)
    res = []
    for s in a.seeds:
        cfg = OmegaConf.load(os.path.join(runs[s][1], ".hydra", "config.yaml"))
        res.append(run(cfg, s, a.epochs, a.zero_image, dev, a.text_chunks))
        r = res[-1]
        print(f"[{a.dataset} {a.tag} seed {s}{' no-image' if a.zero_image else ''}] ours {r['ours']:.3f} (epoch {r['best_epoch']}) "
              f"· same-set {r['same_set']:.3f} · logged test {runs[s][2]['test/acc']:.3f}", flush=True)
    summ = {k: float(np.mean([r[k] for r in res])) for k in ("ours", "same_set", "val_best")}
    print("MEAN", json.dumps(summ))
    out = a.out or os.path.join(ROOT, "logs", f"diagnose_{a.dataset}_{a.tag}{'_noimg' if a.zero_image else ''}{f'_c{a.text_chunks}' if a.text_chunks else ''}.json")
    json.dump({"args": vars(a), "runs": res, "mean": summ}, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
