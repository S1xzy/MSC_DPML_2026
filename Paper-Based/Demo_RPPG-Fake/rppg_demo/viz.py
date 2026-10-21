"""Plots for the demo notebook (matplotlib; labels in English so they render without Thai fonts, e.g. on Colab).

Functions take a Bundle and news *keys* (int = dataset index, "ext<j>" = external news). The *_pair functions
draw a real and a fake news item side by side.
"""
from __future__ import annotations

import math
import textwrap
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .bundle import REL_NAMES
from .generator import REL_COENGAGE, REL_CONTENT, REL_REPOST

INK, MUTED, GRID = "#14202C", "#4F5D6B", "#D8DFE6"
BLUE, ORANGE, PURPLE, RED, GREEN, GREY = "#2950B8", "#D9822B", "#7A4FC4", "#C0392B", "#2E8B57", "#8A97A6"
LABEL_COLOR = {"real": GREEN, "fake": RED}
REL_COLOR = {REL_REPOST: BLUE, REL_COENGAGE: "#5DA5DA", REL_CONTENT: PURPLE}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "axes.titleweight": "bold", "axes.titlesize": 12})


def _name(B, key):
    it = B.item(key)
    return f"{it['label'].upper()} news #{key}"


# ---------------------------------------------------------------- input
def _item_text(B, key, p_real=None, width=58):
    it = B.item(key)
    t0 = it["eng"][0][2]
    lines = [f"#{key}  ({it['split']})", f"label: {it['label'].upper()}" + (f"   ->   P(real) = {p_real:.3f}" if p_real is not None else ""),
             "", *textwrap.wrap(it["text"].replace("\n", " "), width)[:5], "",
             f"real engagements: {len(it['eng'])}", f"early path G (model input): {len(it['early'])}"]
    for h, u, t in it["early"]:
        lines.append(f"  user {u:>6} +{(t - t0) / 60:6.1f} min  {'in H' if B.known[u] else 'UNK (new user)'}")
    return "\n".join(lines)


def show_item(B, key, p_real=None):
    show_pair(B, [key], [p_real])


def show_pair(B, keys, probs=None):
    """News items side by side: image on top, text + early path below."""
    probs = probs or [None] * len(keys)
    fig, axes = plt.subplots(2, len(keys), figsize=(7.2 * len(keys), 6.0), gridspec_kw=dict(height_ratios=[1, 0.75]), squeeze=False)
    for c, (key, p) in enumerate(zip(keys, probs)):
        it = B.item(key)
        ax = axes[0][c]
        if it["image"]:
            ax.imshow(plt.imread(it["image"]))
        else:
            ax.text(0.5, 0.5, "no image", ha="center", va="center", color=MUTED)
        ax.axis("off")
        ax.set_title(f"{it['label'].upper()} news", color=LABEL_COLOR[it["label"]], fontsize=14)
        ax = axes[1][c]
        ax.axis("off")
        ax.text(0, 1, _item_text(B, key, p), va="top", family="monospace", fontsize=9, color=INK)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- Module 1
def _ego_ax(B, ego, ax, title):
    u0 = ego["user"]
    pos = {("u", u0): (0, 0)}
    news, reps = ego["news"], ego["repost"]
    for k, n in enumerate(news):
        a = 2 * math.pi * k / max(len(news), 1)
        pos[("n", n)] = (1.6 * math.cos(a), 1.6 * math.sin(a))
        for j, u in enumerate(ego["co"].get(n, [])):
            b = a + (j - 1.5) * 0.18
            pos.setdefault(("u", u), (2.6 * math.cos(b), 2.6 * math.sin(b)))
    for k, u in enumerate(reps):
        a = 2 * math.pi * (k + 0.5) / max(len(reps), 1)
        pos.setdefault(("u", u), (0.9 * math.cos(a), 0.9 * math.sin(a)))
    for n in news:
        ax.plot(*zip(pos[("u", u0)], pos[("n", n)]), color=GREY, lw=1)
        for u in ego["co"].get(n, []):
            ax.plot(*zip(pos[("n", n)], pos[("u", u)]), color=GREY, lw=0.6, ls=":")
    for u in reps:
        ax.plot(*zip(pos[("u", u0)], pos[("u", u)]), color=BLUE, lw=1.4)
    for (kind, v), (x, y) in pos.items():
        if kind == "n":
            ax.scatter(x, y, s=170, marker="s", color=ORANGE if B.labels[v] == 0 else GREEN, zorder=3)
            ax.text(x, y - 0.25, f"news {v}", ha="center", fontsize=8, color=MUTED)
        else:
            ax.scatter(x, y, s=260 if v == u0 else 60, color=INK if v == u0 else BLUE, zorder=3)
            if v == u0 or v in reps:
                ax.text(x, y + 0.15, f"u{v}", ha="center", fontsize=8, color=INK)
    ax.set_title(f"{title}\nuser {u0}: {ego['n_news']} train news (showing {len(news)}), "
                 f"{ego['n_repost']} repost links (showing {len(reps)})", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_ego(B, ego, title=""):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _ego_ax(B, ego, ax, title)
    _ego_legend(ax)
    plt.show()


def _ego_legend(ax):
    h = [plt.Line2D([], [], color=BLUE, lw=1.4, label="U-U repost"), plt.Line2D([], [], color=GREY, label="U-N engage"),
         plt.Line2D([], [], marker="s", ls="", color=GREEN, label="real news (train)"),
         plt.Line2D([], [], marker="s", ls="", color=ORANGE, label="fake news (train)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)


def plot_ego_pair(B, keys):
    """Graph H around the first early user (that exists in H) of each news item."""
    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 * len(keys), 6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        u = B.first_known_user(key)
        if u is None:
            ax.text(0.5, 0.5, f"{_name(B, key)}:\nnone of the early users is in graph H\n(all new users -> UNK vector)",
                    ha="center", va="center", fontsize=11, color=MUTED)
            ax.axis("off")
        else:
            _ego_ax(B, B.ego(u), ax, f"{_name(B, key)} · first early user in H")
    _ego_legend(axes[0][-1])
    plt.show()


def plot_user_tsne(B, n=4000, highlight=None, seed=0):
    """t-SNE of user embeddings; ``highlight`` = list of (user, label, colour) to mark with their 5 neighbours."""
    from sklearn.manifold import TSNE
    share, tot = B.fake_share()
    cand = np.where(B.known.numpy() & (tot >= 1))[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand, min(n, len(cand)), replace=False)
    highlight = [h for h in (highlight or []) if h[0] is not None]
    groups = [[u] + [v for v, _ in B.neighbours(u, 5)] for u, _, _ in highlight]
    if groups:
        pick = np.unique(np.concatenate([pick, *groups]))
    Y = TSNE(2, perplexity=40, init="pca", random_state=seed).fit_transform(B.user_emb[pick].numpy())
    cmap = LinearSegmentedColormap.from_list("rf", [BLUE, "#C9D3E3", ORANGE])
    c = share[pick]
    order = np.argsort(np.abs(c - 0.5))
    fig, ax = plt.subplots(figsize=(9, 6.8))
    sc = ax.scatter(Y[order, 0], Y[order, 1], c=c[order], cmap=cmap, vmin=0, vmax=1, s=6, alpha=0.8, linewidths=0)
    loc = {u: k for k, u in enumerate(pick)}
    for (u, lab, col), grp in zip(highlight, groups):
        for v in grp[1:]:
            ax.plot(*zip(Y[loc[u]], Y[loc[v]]), color=col, lw=0.9)
        ax.scatter(*Y[loc[u]], s=220, marker="*", color=col, edgecolors="white", zorder=4)
        ax.annotate(f"{lab}: user {u} + 5 nearest", Y[loc[u]], xytext=(15, 15), textcoords="offset points", color=col,
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Module 1 user embeddings (t-SNE of {len(pick):,} users in H)")
    fig.colorbar(sc, ax=ax, fraction=0.04).set_label("share of fake news the user engaged (train)")
    plt.show()


def _cc_ax(B, key, ax, k):
    share, tot = B.fake_share()
    users = B.content_topk(key)[:k]
    y = np.arange(k)[::-1]
    ax.hlines(y, 0, 1, color=GRID, lw=0.8)
    ax.scatter(share[users], y, s=40 + 25 * np.minimum(tot[users], 10), c=[ORANGE if v > 0.5 else BLUE for v in share[users]], zorder=3)
    for yy, u in zip(y, users):
        ax.text(1.03, yy, f"{int(tot[u])} news", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([f"u{u}" for u in users], fontsize=8)
    ax.set_xlim(-0.05, 1.25); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0 (only real)", "0.5", "1 (only fake)"])
    ax.set_xlabel("fake share of the user (dot size = number of train news)")
    ax.set_title(f"{_name(B, key)}: mean fake share of top-{k} = {share[users].mean():.2f}", color=LABEL_COLOR[B.item(key)["label"]], fontsize=11)


def plot_content_candidates(B, key, k=10):
    fig, ax = plt.subplots(figsize=(8.5, 0.35 * k + 1.2))
    _cc_ax(B, key, ax, k)
    plt.show()


def plot_content_candidates_pair(B, keys, k=10):
    """Users whose embedding is closest to the news content (the 'content' actions of Module 2)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 * len(keys), 0.35 * k + 1.4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        _cc_ax(B, key, ax, k)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- Module 2
def _radial(early, steps):
    parent, ring, via = {}, {}, {}
    for h, u, _ in early:
        parent[u], ring[u], via[u] = "N", 0, "early"
    for k, st in enumerate(steps):
        for p, u in st:
            if u in parent:
                continue
            parent[u] = "N" if p == -1 or p not in parent else p
            ring[u], via[u] = k + 1, ("content" if p == -1 else "graph")
    children = defaultdict(list)
    for u, p in parent.items():
        children[p].append(u)
    leaves, ang = {}, {}

    def count(v):
        leaves[v] = 1 if not children[v] else sum(count(c) for c in children[v])
        return leaves[v]

    def place(v, a0, a1):
        ang[v] = (a0 + a1) / 2
        a = a0
        for c in sorted(children[v], key=lambda x: (via[x] != "early", ring[x])):
            span = (a1 - a0) * leaves[c] / leaves[v]
            place(c, a, a + span)
            a += span

    count("N")
    place("N", 0, 2 * math.pi)
    xy = {u: ((1 + 0.42 * ring[u]) * math.cos(ang[u]), (1 + 0.42 * ring[u]) * math.sin(ang[u])) for u in parent}
    xy["N"] = (0.0, 0.0)
    return parent, ring, via, xy


def plot_generation(B, key, steps, records=None):
    it = B.item(key)
    real = {u for _, u, _ in it["eng"]} - {0}
    early_users = {u for _, u, _ in it["early"]}
    parent, ring, via, xy = _radial(it["early"], steps)
    fig = plt.figure(figsize=(15, 6.2))
    fig.suptitle(_name(B, key), color=LABEL_COLOR[it["label"]], fontsize=15, fontweight="bold", x=0.02, ha="left")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    t0 = it["eng"][0][2]
    seen = []
    for h, u, t in it["eng"]:
        if u == 0 or u not in [s[0] for s in seen]:
            seen.append((u, (t - t0) / 3600))
    for k, (u, hr) in enumerate(seen):
        col = ORANGE if k < len(it["early"]) else (RED if u in parent and u not in early_users and u != 0 else GREY)
        ax.plot([0, hr], [0, k + 1], color=col, lw=0.8, alpha=0.6)
        ax.scatter(hr, k + 1, color=col, s=40 if k < len(it["early"]) else 18, zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("hours after posting"); ax.set_ylabel("order of real engagement")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(f"Real path: {len(seen)} engagements (orange = early G)", fontsize=11)
    ax = fig.add_subplot(gs[1])
    for r in range(0, len(steps) + 1, 5):
        ax.add_patch(plt.Circle((0, 0), 1 + 0.42 * r, fill=False, color=GRID, lw=0.7))
    for u, p in parent.items():
        ls = (0, (3, 2)) if via[u] == "content" else "-"
        col = PURPLE if via[u] == "content" else (ORANGE if via[u] == "early" else BLUE)
        ax.plot(*zip(xy[p], xy[u]), color=col, lw=0.8, ls=ls, alpha=0.6)
    for kind, col, s in [("graph", BLUE, 10), ("content", PURPLE, 18), ("early", ORANGE, 70)]:
        us = [u for u in parent if via[u] == kind]
        ax.scatter([xy[u][0] for u in us], [xy[u][1] for u in us], s=s, color=col, zorder=3)
    hits = [u for u in parent if u in real and u not in early_users]
    for u in hits:
        ax.scatter(*xy[u], s=220, marker="*", color=RED, zorder=5)
        ax.annotate(f"u{u}", xy[u], xytext=(6, 6), textcoords="offset points", color=RED, fontsize=9)
    ax.scatter(0, 0, s=150, marker="s", color=INK, zorder=6)
    lim = 1 + 0.42 * len(steps) + 0.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    n_g = sum(v == "graph" for v in via.values()); n_c = sum(v == "content" for v in via.values())
    ax.set_title(f"Generated G': {n_g + n_c} users in {len(steps)} steps\n{len(hits)} of them really engaged later (red stars)", fontsize=11)
    h = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="early (real)"),
         plt.Line2D([], [], color=BLUE, label=f"via graph H ({n_g})"),
         plt.Line2D([], [], color=PURPLE, ls="--", label=f"via content ({n_c})")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=9)
    ax = fig.add_subplot(gs[2])
    if records:
        x = np.arange(len(records))
        bottom = np.zeros(len(records))
        for r, name in REL_NAMES.items():
            v = np.array([rec["pick_rel"][r] for rec in records])
            ax.bar(x, v, bottom=bottom, color=REL_COLOR[r], label=name, width=0.8)
            bottom += v
        ax2 = ax.twinx()
        ax2.plot(x, [rec["n_candidates"] for rec in records], color=INK, lw=1.2, marker=".")
        ax2.set_ylabel("candidate actions (line)", color=INK)
        ax2.spines["right"].set_visible(True)
        ax.set_xlabel("step"); ax.set_ylabel("users added (by relation)")
        ax.set_ylim(0, max(bottom.max(), 1) * 1.45)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
        ax.set_title("What the policy picked at each step", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- Module 3
def plot_snapshots(B, out, show=(0, 2, 7, 14, -1)):
    key = out["index"]
    it = B.item(key)
    s = out["sample"]
    bounds, edges = s["bounds"], s["edges"]
    n = len(s["users"]) + 1
    early_n = len({u for _, u, _ in it["early"]})
    parent = {c: p for p, c in edges}
    snap = {0: 0}
    for t, b in enumerate(bounds):
        for v in range(1, b):
            snap.setdefault(v, t)
    children = defaultdict(list)
    for c, p in parent.items():
        children[p].append(c)
    leaves, ang = {}, {}

    def count(v):
        leaves[v] = 1 if not children[v] else sum(count(c) for c in children[v])
        return leaves[v]

    def place(v, a0, a1):
        ang[v] = (a0 + a1) / 2
        a = a0
        for c in sorted(children[v]):
            span = (a1 - a0) * leaves[c] / leaves[v]
            place(c, a, a + span)
            a += span

    count(0)
    place(0, 0, 2 * math.pi)
    rad = {v: 0 if v == 0 else 1 + 0.25 * snap[v] for v in range(n)}
    xy = {v: (rad[v] * math.cos(ang[v]), rad[v] * math.sin(ang[v])) for v in range(n)}
    shows = sorted({t if t >= 0 else len(bounds) + t for t in show if -len(bounds) <= t < len(bounds)})
    fig = plt.figure(figsize=(15, 7.4))
    fig.suptitle(f"{_name(B, key)} · P(real) = {out['p_real']:.3f} · {len(bounds)} snapshots",
                 color=LABEL_COLOR[it["label"]], fontsize=14, fontweight="bold", x=0.02, ha="left")
    top = fig.add_gridspec(1, len(shows), left=0.02, right=0.98, top=0.9, bottom=0.55, wspace=0.05)
    lim = 1 + 0.25 * (len(bounds) - 1) + 0.2
    for k, t in enumerate(shows):
        ax = fig.add_subplot(top[k])
        nodes = range(bounds[t])
        for v in nodes:
            if v:
                ax.plot(*zip(xy[parent[v]], xy[v]), color=ORANGE if v <= early_n else BLUE, lw=0.5, alpha=0.6)
        for v in nodes:
            col, sz, mk = (INK, 60, "s") if v == 0 else ((ORANGE, 35, "o") if v <= early_n else (BLUE, 6, "o"))
            ax.scatter(*xy[v], color=col, s=sz, marker=mk, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"snapshot {t} · {bounds[t]} nodes", fontsize=10)
    bot = fig.add_gridspec(1, 2, left=0.05, right=0.97, top=0.44, bottom=0.05, wspace=0.45, width_ratios=[1, 1.3])
    ax = fig.add_subplot(bot[0])
    im = ax.imshow(out["self_attn"], cmap="Blues", aspect="auto")
    ax.set_xlabel("key snapshot"); ax.set_ylabel("query snapshot")
    ax.set_title("Self-attention between snapshots", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(bot[1])
    C = out["cross_attn"].mean(0)
    groups = [("news node", C[0], INK), (f"early real ({early_n})", C[1:early_n + 1].sum(), ORANGE),
              (f"generated ({n - 1 - early_n})", C[early_n + 1:].sum(), BLUE)]
    y = np.arange(3)[::-1]
    ax.barh(y, [g[1] for g in groups], color=[g[2] for g in groups])
    for yy, g in zip(y, groups):
        ax.text(g[1] + 0.01, yy, f"{g[1] * 100:.1f}%", va="center")
    ax.set_yticks(y); ax.set_yticklabels([g[0] for g in groups]); ax.set_xlim(0, 1.15); ax.set_xticks([])
    k90 = int((np.cumsum(np.sort(C)[::-1]) < 0.9).sum() + 1)
    ax.set_title(f"Cross-attention to final graph (90% of weight on {k90} nodes)", fontsize=11)
    plt.show()


def plot_decision(B, outs, curves=None):
    """outs: {name: predict(..., explain=True)}; curves: {name: [P(real) with 0..T steps of G']}"""
    fig, axes = plt.subplots(1, 2 if curves else 1, figsize=(14 if curves else 7.5, 4.6), squeeze=False)
    ax = axes[0][0]
    names = ["path h*", "text", "image", "bias"]
    cols = [BLUE, GREEN, PURPLE, GREY]
    w = 0.8 / len(outs)
    for j, (lab, o) in enumerate(outs.items()):
        v = [o["parts"][k] for k in names]
        x = np.arange(4) + (j - (len(outs) - 1) / 2) * w
        ax.bar(x, v, width=w, color=cols, alpha=1 if j == 0 else 0.5, hatch=None if j == 0 else "//", edgecolor=INK if j else "none")
        for xi, vi in zip(x, v):
            ax.text(xi, vi, f"{vi:+.2f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_ylabel("contribution to logit (+ = real)")
    ax.set_title("Split of the final logit (solid = first, hatched = second)\n" +
                 " | ".join(f"{k}: P={o['p_real']:.3f}" for k, o in outs.items()), fontsize=10)
    if curves:
        ax = axes[0][1]
        for (lab, c), col in zip(curves.items(), [GREEN, RED, BLUE, ORANGE]):
            ax.plot(range(len(c)), c, marker="o", ms=3, color=col, label=lab)
        ax.axhline(0.5, ls="--", color=MUTED, lw=0.8)
        ax.set_ylim(0, 1.02); ax.set_xlabel("steps of G' given to the detector (0 = early only)"); ax.set_ylabel("P(real)")
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.7))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.set_title("Does a longer generated path change the decision?")
    plt.show()


def plot_evaluation(res, title="test set"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cm = np.array(res["confusion"])
    ax[0].imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax[0].text(c, r, str(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else INK)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["pred real", "pred fake"])
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels([f"true real ({res['real']})", f"true fake ({res['fake']})"])
    ax[0].set_title(f"{title}: Acc {res['acc']:.3f} · F1 {res['f1']:.3f} · AUC {res['auc']:.3f}", fontsize=10)
    p, y = res["p_real"], res["y"]
    ax[1].hist(p[y == 1], bins=20, range=(0, 1), alpha=0.7, color=GREEN, label="real news")
    ax[1].hist(p[y == 0], bins=20, range=(0, 1), alpha=0.7, color=RED, label="fake news")
    ax[1].axvline(0.5, ls="--", color=MUTED)
    ax[1].set_xlabel("P(real)"); ax[1].set_ylabel("news"); ax[1].legend(frameon=False)
    ax[1].set_title("Predicted probability by true label")
    plt.show()


def plot_compare_sets(results):
    """results: {set name: evaluate(...)} -> accuracy on real / fake news and AUC side by side."""
    names = list(results)
    metrics = [("acc_real", "acc on REAL news", GREEN), ("acc_fake", "acc on FAKE news", RED), ("acc", "overall acc", BLUE), ("auc", "AUC", GREY)]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (m, lab, col) in enumerate(metrics):
        v = [results[n][m] for n in names]
        ax.bar(x + (k - 1.5) * w, v, width=w, color=col, label=lab)
        for xi, vi in zip(x + (k - 1.5) * w, v):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({results[n]['real']} real / {results[n]['fake']} fake)" for n in names])
    ax.set_ylim(0, 1.12); ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center")
    ax.set_title("Real vs fake news: how well does the model do on each set?")
    plt.show()
