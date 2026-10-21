"""Run the trained PHEME detector (round 8, seed 2, early_fix) on two test news and record what Module 3 computes.

Run from Code_RPPG-Fake with the project venv:  .venv/Scripts/python <this> <out_dir>
"""
import json, os, sys
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from predict import find_runs  # noqa: E402
from src.data.rppg_datamodule import make_sample  # noqa: E402
from src.models.rppg_detector_module import RPPGDetectorModule  # noqa: E402

out = sys.argv[1]
runs = find_runs("pheme", "early_fix")
val, run_dir, m = runs[2]
cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
dm = instantiate(cfg.data); dm.setup()
det = RPPGDetectorModule.load_from_checkpoint(m["best_ckpt"], map_location="cpu").eval()
net = det.net if hasattr(det, "net") else det
core = [mod for mod in net.modules() if mod.__class__.__name__ == "TemporalAwareDetector"][0]
cap = {}
core.self_attn.register_forward_hook(lambda mod, i, o: cap.__setitem__("self", o[1].detach()))
core.cross_attn.register_forward_hook(lambda mod, i, o: cap.__setitem__("cross", o[1].detach()))
core.out.register_forward_hook(lambda mod, i, o: cap.__setitem__("z", i[0].detach()))

W, b = core.out.weight[0].detach(), core.out.bias.detach().item()
def logit_parts(z):
    d = z.numel() - 1024
    return {"graph": float(W[:d] @ z[:d]), "text": float(W[d:d + 512] @ z[d:d + 512]),
            "image": float(W[d + 512:] @ z[d + 512:]), "bias": b}
test = dm.sets["test"]
data = torch.load(os.path.join(cfg.data.processed_dir, f"seed2_{cfg.data.slot_mode}{cfg.data.n_slots}_T200.pt"), weights_only=False)
gen = torch.load(os.path.join(cfg.data.generated_dir, "seed2", "stage1.pt"), weights_only=False)
texts = data["texts"]
pick = {"real": [i for i, t in enumerate(texts) if "Love is Stronger than Hate" in t][0],
        "fake": [i for i, t in enumerate(texts) if "Two female hostages have run out" in t][0]}
res = {"val_acc": val, "test_acc": m["test/acc"], "items": {}}
with torch.no_grad():
    for key, i in pick.items():
        assert i in test.idx, (key, i)
        s = test.samples[i]
        batch = dm.collate([(i, s, test.labels[i])])
        p = torch.sigmoid(core(batch)).item()
        item = {"i": i, "text": texts[i], "label": int(test.labels[i]), "p_real": p,
                "users": s["users"], "edges": [list(e) for e in s["edges"]], "bounds": s["bounds"],
                "n_real": len(data["early"][i]), "self_attn": cap["self"][0].tolist(), "logit_parts": logit_parts(cap["z"][0]), "cross_attn": cap["cross"][0].tolist()}
        # what the detector would say if generation had stopped after k steps (same model, shorter G')
        curve = []
        steps = gen["paths"][i]["steps"]
        for k in range(0, len(steps) + 1):
            sk = make_sample(data["eng"][i], data["early"][i], steps[:k], cfg.data.eval_real, cfg.data.real_slots,
                             cfg.data.max_real_users, cfg.data.use_generated, cfg.data.get("max_gen_users"))
            if "feat" in s:
                sk["feat"] = s["feat"][: len(sk["users"])]
            curve.append(torch.sigmoid(core(dm.collate([(i, sk, test.labels[i])]))).item())
        item["curve"] = curve
        res["items"][key] = item
        print(key, i, "label", item["label"], "P(real) %.3f" % p, "snapshots", len(s["bounds"]), "nodes", len(s["users"]) + 1,
              "parts", {k: round(v, 2) for k, v in item["logit_parts"].items()}, "| early-only %.3f" % curve[0])
json.dump(res, open(os.path.join(out, "m3.json"), "w"))
