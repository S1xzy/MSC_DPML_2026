import os, sys, json, torch, numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from predict import find_runs
from src.models.rppg_detector_module import RPPGDetectorModule
rows = []
for ds in ["pheme", "gossipcop", "politifact"]:
    for tag in ["early_fix", "full_fix"]:
        runs = find_runs(ds, tag)
        for seed in sorted(runs):
            val, run_dir, m = runs[seed]
            cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
            dm = instantiate(cfg.data); dm.setup()
            det = RPPGDetectorModule.load_from_checkpoint(m["best_ckpt"], map_location="cpu").eval()
            core = [x for x in det.modules() if x.__class__.__name__ == "TemporalAwareDetector"][0]
            cap = {}
            core.out.register_forward_hook(lambda mod, i, o: cap.__setitem__("z", i[0].detach()))
            W = core.out.weight[0].detach(); d = W.numel() - 1024
            Z, ys, ps = [], [], []
            with torch.no_grad():
                for batch in dm._loader("test", False):
                    lg = core(batch); Z.append(cap["z"]); ys += batch["y"].tolist(); ps += torch.sigmoid(lg).tolist()
            Z = torch.cat(Z); ys = np.array(ys); ps = np.array(ps)
            g = (Z[:, :d] @ W[:d]).numpy()
            rest = np.log(ps / (1 - ps)) - g
            r = dict(ds=ds, tag=tag, seed=seed, acc=round(float(((ps > .5) == ys).mean()), 3),
                     h_std=round(float(Z[:, :d].std(0).mean()), 4), w_graph=round(float(W[:d].norm()), 3),
                     w_content=round(float(W[d:].norm()), 3), graph_std=round(float(g.std()), 4),
                     acc_graph_const=round(float(((rest + g.mean() > 0) == ys).mean()), 3))
            rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(sys.argv[1] + "/m3_seeds.json", "w"), indent=1)
