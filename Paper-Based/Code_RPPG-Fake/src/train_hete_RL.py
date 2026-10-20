"""Stage 1 of RPPG-Fake: user embeddings + reinforced propagation path generation (Algorithm 1, lines 1-14).

    python src/train_hete_RL.py seed=0

1. Heterogeneous graph contrastive learning on H (Module 1, Eqs. 2-5).
2. Actor-critic path generation with temporal + global rewards (Module 2, Eqs. 6-19), trained on the
   training news only.
3. Generate a propagation path G' for every news item (train / val / test) starting from its early-stage
   engagements, and save it together with the user embeddings for Stage 2 (``src/train.py``).
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import hydra
import numpy as np
import pyrootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils  # noqa: E402
from src.data.components.encode_content import cached as content_features  # noqa: E402
from src.data.components.politifact import NEWS_ROOT, build  # noqa: E402
from src.models.components.rppg.contrastive import HeteroGraphContrastive  # noqa: E402
from src.models.components.rppg.generator import (PropagationEnv, RPPGGenerator, generated_prefix,  # noqa: E402
                                                   real_prefix, rollout)

log = utils.get_pylogger(__name__)
META_ORDER = ("UNU", "NUN", "UU")


def set_seed(seed: int):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)


def load_inputs(cfg, device):
    path = build(cfg.data.raw_dir, cfg.data.processed_dir, cfg.seed, n_slots=cfg.data.n_slots, slot_mode=cfg.data.slot_mode,
                 slot_users=cfg.generator.T)
    data = torch.load(path, weights_only=False)
    feats = content_features(path, cfg.data.image_dir, device, cfg.data.get("text_chunks", 1))
    content = torch.cat([F.normalize(feats["text"], dim=-1), F.normalize(feats["image"], dim=-1)], -1)
    return data, content.to(device)


def train_contrastive(cfg, data, content, device):
    c = cfg.contrastive
    U, N = data["num_users"], data["num_news"]
    ei = data["graph"]["edge_index"].to(device)
    meta = [data["meta"][k].to(device) for k in META_ORDER]
    model = HeteroGraphContrastive(U, N, content.size(1), len(meta), c.dim, c.heads, c.dropout, c.tau1, c.tau2,
                                   c.lambda1, c.k1, c.pgd_eps, c.pgd_steps, c.get("pos_neg", "embedding")).to(device)
    active = torch.unique(ei.flatten())  # nodes that have at least one edge in H (train news)
    active_users = active[active < U]
    sample_active = c.get("pos_neg", "embedding") == "structure"
    if sample_active:
        model.set_structure(ei, U + N, seed=cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    n_all = U + N
    for step in range(1, c.steps + 1):
        model.train()
        if sample_active:  # only users / nodes that exist in H (isolated val/test-only users carry no structure)
            centres = active_users[torch.randint(0, len(active_users), (c.batch_centres,), device=device)]
            pool = active[torch.randint(0, len(active), (c.pool,), device=device)]
        else:
            centres = torch.randint(0, U, (c.batch_centres,), device=device)  # Algorithm 1, line 2: sampled user nodes
            pool = torch.randint(0, n_all, (c.pool,), device=device)
        mb = (active[torch.randint(0, len(active), (c.meta_batch,), device=device)] if sample_active
              else torch.randint(0, n_all, (c.meta_batch,), device=device))
        loss, l_node, l_meta = model.loss(content, ei, meta, centres, pool, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % c.log_every == 0 or step == 1:
            log.info(f"[CL] step {step:4d}  L_CL={loss.item():.4f}  L_node={l_node.item():.4f}  L_meta={l_meta.item():.4f}")
    model.eval()
    node, meta_sum = model.embeddings(content, ei, meta)
    with torch.no_grad():  # content-only view of every news node (no edges) -> no cascade information
        iso = model.node_gat(model.features(content), torch.zeros(2, 0, dtype=torch.long, device=device))[U:]
    return node[:U].detach(), meta_sum[:U].detach(), iso.detach(), model


def content_candidates(iso_news, user_node, k, center: bool = False, known=None):
    """Users closest to each news item's content embedding (fallback actions when no path exists yet).

    center=True removes the users' mean vector from both sides first; otherwise the common component of the GAT
    outputs dominates the cosine and every news item gets almost the same candidates."""
    if center:
        mu = (user_node[known] if known is not None else user_node).mean(0)
        iso_news, user_node = iso_news - mu, user_node - mu
    sims = F.normalize(iso_news, dim=-1) @ F.normalize(user_node, dim=-1).T
    sims[:, NEWS_ROOT] = -2
    sims[:, 0] = -2
    return torch.topk(sims, k, dim=1).indices.cpu().numpy()


def hit_rate(paths, data, idx):
    """Share of generated users that really engaged that news (diagnostic, not a training signal)."""
    hit = tot = 0
    for i in idx:
        real = {u for _, u, _ in data["eng"][i]}
        gen = [c for st in paths[i]["steps"] for _, c in st]
        hit += sum(u in real for u in gen)
        tot += len(gen)
    return hit / max(tot, 1)


def train_generator(cfg, data, content, base_node, base_meta, iso_news, device, out_dir):
    g = cfg.generator
    model = RPPGGenerator(base_node.size(1), content.size(1), g.dim, standardize=g.get("standardize", False)).to(device)
    disc_params = list(model.disc_pool.parameters()) + list(model.disc_head.parameters())
    disc_ids = {id(p) for p in disc_params}
    pol_params = [p for p in model.parameters() if id(p) not in disc_ids]
    opt = torch.optim.AdamW(pol_params, lr=g.lr)
    opt_d = torch.optim.AdamW(disc_params, lr=g.disc_lr)
    U_ = data["num_users"]
    gei = data["graph"]["edge_index"]
    known = torch.zeros(U_, dtype=torch.bool)
    known[gei[0][gei[0] < U_]] = True
    if g.get("standardize", False):
        model.set_known(known.to(device))
    topk = content_candidates(iso_news, base_node, g.content_topk, center=g.get("standardize", False),
                              known=known.to(device) if g.get("standardize", False) else None)
    own = {i: np.fromiter({u for _, u, _ in data["eng"][i]}, dtype=np.int64) for i in data["split"]["train"]}
    real_sets = {i: set(own[i].tolist()) for i in own}
    env = PropagationEnv(data, topk, g.max_sources, g.max_repost, g.max_coengage, g.max_actions, exclude=own)
    rng = np.random.default_rng(cfg.seed)
    train_idx = list(data["split"]["train"])
    history = []
    if g.mode != "rl":  # Table IV generation ablations: no RL training, optionally reuse a trained generator
        if g.init_from:
            model.load_state_dict(torch.load(g.init_from, map_location=device))
            log.info(f"[GEN] {g.mode} generation, generator weights from {g.init_from}")
    for epoch in range(1, (g.epochs if g.mode == "rl" else 0) + 1):
        model.train()
        random.shuffle(train_idx)
        t0, stats = time.time(), {"actor": 0.0, "critic": 0.0, "R_G": 0.0, "D_loss": 0.0, "users": 0.0}
        for b in range(0, len(train_idx), g.batch_size):
            batch = train_idx[b : b + g.batch_size]
            Ut = model.user_table(base_node, base_meta)
            a_loss, c_loss, fakes, reals, n_vecs = 0.0, 0.0, [], [], []
            for i in batch:
                n_vec = model.news_proj(content[i])
                steps, graph, losses = rollout(model, env, i, data["early"][i], Ut, n_vec, rng, g.T, g.K2, g.max_steps,
                                               data["slots"][i], g.lambda2, g.gamma, train=True, global_reward=g.global_reward,
                                               real_users=real_sets[i] if g.hit_weight else None, hit_weight=g.hit_weight)
                a_loss = a_loss + losses["actor"]
                c_loss = c_loss + losses["critic"]
                stats["R_G"] += float(losses["reward_g"])
                stats["users"] += len(graph.users)
                # D sees real and generated prefixes of the same random size, because R_G is computed on
                # every intermediate prefix G'_t (Eq. 13), not only on the finished path
                lo = len(data["early"][i]) + 1
                m = int(rng.integers(min(lo, len(graph.users)), len(graph.users) + 1)) if graph.users else 1
                if g.get("standardize", False):  # same size on both sides: real cascades are often < T users
                    m = max(1, min(m, len(real_sets[i])))
                fakes.append(generated_prefix(data["early"][i], steps, m))
                reals.append(real_prefix(data["eng"][i], m))
                n_vecs.append(n_vec.detach())
            loss = (a_loss + c_loss) / len(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol_params, 5.0)
            opt.step()
            # discriminator D: real paths vs generated paths (Eq. 13)
            Ud = Ut.detach()
            nv = torch.stack(n_vecs)
            logit_r = model.discriminate(reals, Ud, nv)
            logit_f = model.discriminate(fakes, Ud, nv)
            d_loss = F.binary_cross_entropy_with_logits(logit_r, torch.ones_like(logit_r)) + \
                F.binary_cross_entropy_with_logits(logit_f, torch.zeros_like(logit_f))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            stats["actor"] += float(a_loss.detach()) / len(batch)
            stats["critic"] += float(c_loss.detach()) / len(batch)
            stats["D_loss"] += float(d_loss)
        nb = math.ceil(len(train_idx) / g.batch_size)
        row = {"epoch": epoch, "actor": stats["actor"] / nb, "critic": stats["critic"] / nb, "D_loss": stats["D_loss"] / nb,
               "R_G": stats["R_G"] / len(train_idx), "avg_users": stats["users"] / len(train_idx), "sec": time.time() - t0}
        history.append(row)
        log.info("[RL] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    # ---------------- generate G' for every news item (Algorithm 1, line 14)
    model.eval()
    gen_rng = np.random.default_rng(cfg.seed + 1000)
    paths = []
    with torch.no_grad():
        Ut = model.user_table(base_node, base_meta)
        for i in range(data["num_news"]):
            steps, _, _ = rollout(model, env, i, data["early"][i], Ut, model.news_proj(content[i]), gen_rng, g.T, g.K2,
                                  g.max_steps, train=False, policy=g.mode)
            early = [(-1 if h == NEWS_ROOT else h, u) for h, u, _ in data["early"][i]]
            paths.append({"early": early, "steps": steps})
    torch.save({"user_emb": Ut.detach().cpu(), "paths": paths, "split": data["split"], "seed": cfg.seed,
                "base_node": base_node.cpu(), "base_meta": base_meta.cpu(), "known": known},
               os.path.join(out_dir, "stage1.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "generator.ckpt"))
    rates = {k: hit_rate(paths, data, data["split"][k]) for k in ("train", "val", "test")}
    sizes = float(np.mean([len(p["early"]) + sum(len(s) for s in p["steps"]) for p in paths]))
    summary = {"history": history, "hit_rate": rates, "avg_path_users": sizes}
    with open(os.path.join(out_dir, "stage1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[GEN] paths saved for {len(paths)} news  avg users={sizes:.1f}  hit-rate={rates}")
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_Hete_RL.yaml")
def main(cfg: DictConfig):
    utils.extras(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, "stage1_config.yaml"))
    data, content = load_inputs(cfg, device)
    log.info(f"news={data['num_news']} users={data['num_users']} train/val/test="
             f"{[len(data['split'][k]) for k in ('train', 'val', 'test')]}")
    t0 = time.time()
    base_node, base_meta, iso_news, cl_model = train_contrastive(cfg, data, content, device)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(cl_model.state_dict(), os.path.join(out_dir, "contrastive.ckpt"))  # needed by scripts/predict.py
    log.info(f"[CL] done in {time.time() - t0:.0f}s")
    train_generator(cfg, data, content, base_node, base_meta, iso_news, device, out_dir)


if __name__ == "__main__":
    main()
