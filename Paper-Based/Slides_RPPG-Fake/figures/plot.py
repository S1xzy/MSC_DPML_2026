"""Draw user-embedding and generated-path figures for the slides (system Python + matplotlib + sklearn)."""
import json, math, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

D = sys.argv[1]
plt.rcParams["font.family"] = ["Leelawadee UI", "Segoe UI Symbol", "Segoe UI"]
INK, MUTED, GRID = "#14202C", "#4F5D6B", "#D8DFE6"
BLUE, ORANGE, PURPLE, RED = "#2950B8", "#D9822B", "#7A4FC4", "#C0392B"

# ---------------------------------------------------------------- user embedding
known = np.load(f"{D}/known.npy"); fake = np.load(f"{D}/fake.npy"); tot = np.load(f"{D}/tot.npy")
emb_fix = np.load(f"{D}/emb_fix.npy"); emb_old = np.load(f"{D}/emb_old.npy")
cand = np.where(known & (tot >= 1))[0]
rng = np.random.default_rng(0)
focus = [258, 4110, 5777, 3783, 412, 4097]
sample = np.unique(np.concatenate([rng.choice(cand, 6000, replace=False), focus]))
share = fake[sample] / np.maximum(tot[sample], 1)
cmap = LinearSegmentedColormap.from_list("rf", [BLUE, "#C9D3E3", ORANGE])

def tsne(X):
    return TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(X)

Xo = emb_old[sample] - emb_old[sample].mean(0)
Y_old = Xo @ np.linalg.svd(Xo, full_matrices=False)[2][:2].T if np.abs(Xo).max() > 1e-6 else np.zeros((len(sample), 2))
Y_fix = tsne(emb_fix[sample])  # t-SNE (perplexity 40)
n_unique_old = len(np.unique(emb_old[sample].round(5), axis=0))

fig, axes = plt.subplots(1, 2, figsize=(16.6, 7.0), dpi=150)
for ax, Y, title in [(axes[0], Y_old, f"ก่อนแก้ bug (รอบ 6): embedding ยุบ"),
                     (axes[1], Y_fix, "หลังแก้ bug (รอบ 8): แต่ละ user มีตำแหน่งของตัวเอง")]:
    order = np.argsort(np.abs(share - 0.5))
    sc = ax.scatter(Y[order, 0], Y[order, 1], c=share[order], cmap=cmap, vmin=0, vmax=1, s=(260 if Y is Y_old else 7), alpha=0.75, linewidths=0)
    ax.set_title(title, fontsize=17, color=INK, loc="left", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    if Y is Y_old:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.annotate(f"{len(sample):,} users → 1 จุด\n(ทุกคนมีเวกเตอร์เดียวกัน)", (0, 0), xytext=(0, -60), textcoords="offset points", ha="center", fontsize=15, color=INK)
    for s in ax.spines.values(): s.set_color(GRID)
pos = {u: k for k, u in enumerate(sample)}
ax = axes[1]
for u in focus[1:]:
    ax.plot(*zip(Y_fix[pos[258]], Y_fix[pos[u]]), color=INK, lw=0.8, alpha=0.6)
ax.scatter(*Y_fix[[pos[u] for u in focus[1:]]].T, s=60, facecolors="none", edgecolors=INK, linewidths=1.4)
ax.scatter(*Y_fix[pos[258]], s=150, marker="*", color=INK)
ax.annotate("u258 และเพื่อนบ้านใกล้สุด 5 คน", Y_fix[pos[258]], xytext=(18, 18), textcoords="offset points",
            fontsize=13, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015)
cb.set_label("สัดส่วนข่าว fake ที่ user เคยแชร์ (train)", fontsize=13, color=MUTED)
cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0 (real)", "0.5", "1 (fake)"])
fig.savefig(f"{D}/user_embedding.png", bbox_inches="tight", facecolor="white")
print("user_embedding.png", len(sample), "users; old unique", n_unique_old)

# ---------------------------------------------------------------- generated path
P = json.load(open(f"{D}/path.json"))
eng = P["eng"]; t0 = eng[0][2]
early = [u for _, u in P["path"]["early"]]
steps = P["path"]["steps"]
real_users = []
for _, u, t in eng:
    if u not in [r[0] for r in real_users]:
        real_users.append((u, (t - t0) / 3600))
real_set = {u for u, _ in real_users}

# tree: node -> parent, ring (0 = early, k+1 = step k)
parent, ring, via = {}, {}, {}
for u in early:
    parent[u], ring[u], via[u] = "N", 0, "early"
for k, st in enumerate(steps):
    for p, u in st:
        if u in parent:
            continue
        parent[u] = "N" if p == -1 or p not in parent else p
        ring[u] = k + 1
        via[u] = "content" if p == -1 else "graph"
children = defaultdict(list)
for u, p in parent.items():
    children[p].append(u)
leaves = {}
def count(n):
    c = children.get(n, [])
    leaves[n] = 1 if not c else sum(count(x) for x in c)
    return leaves[n]
count("N")
ang = {}
def place(n, a0, a1):
    ang[n] = (a0 + a1) / 2
    a = a0
    for c in sorted(children.get(n, []), key=lambda x: (via[x] != "early", ring[x])):
        span = (a1 - a0) * leaves[c] / leaves[n]
        place(c, a, a + span)
        a += span
place("N", 0, 2 * math.pi)
R = lambda u: 1.0 + ring[u] * 0.42
xy = {u: (R(u) * math.cos(ang[u]), R(u) * math.sin(ang[u])) for u in parent}
xy["N"] = (0.0, 0.0)

fig = plt.figure(figsize=(16.6, 7.6), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.08)
# left: real path as a timeline
ax = fig.add_subplot(gs[0])
hours = [h for _, h in real_users]
for k, (u, h) in enumerate(real_users):
    col = ORANGE if u in early else (RED if u == 585 else "#8A97A6")
    ax.plot([0, h], [0, k + 1], color=col, lw=1.0, alpha=0.6)
    ax.scatter(h, k + 1, s=90 if u in early or u == 585 else 45, color=col, zorder=3,
               marker="*" if u == 585 else "o")
ax.scatter(0, 0, s=220, color=INK, marker="s", zorder=4)
ax.annotate("ข่าว", (0, 0), xytext=(10, -4), textcoords="offset points", fontsize=13, color=INK)
ax.annotate("3 คนแรก = early path (G)", (hours[2], 3), xytext=(30, -6), textcoords="offset points", fontsize=13, color=ORANGE)
ax.annotate("u585 (ชั่วโมงที่ %.0f)" % hours[-1], (hours[-1], len(real_users)), xytext=(-150, -2), textcoords="offset points",
            fontsize=13, color=RED)
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("ชั่วโมงหลังโพสต์", fontsize=13, color=MUTED)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.set_ylabel("ลำดับคนที่แชร์จริง", fontsize=13, color=MUTED)
ax.set_title(f"เส้นทางจริง: {len(real_users)} คน (ตอบกลับต้นโพสต์)", fontsize=17, color=INK, loc="left", pad=10)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)

# right: generated G' as a radial tree (ring = generation step)
ax = fig.add_subplot(gs[1])
for k in range(0, len(steps) + 1, 5):
    ax.add_patch(plt.Circle((0, 0), 1.0 + k * 0.42, fill=False, color=GRID, lw=0.8, zorder=0))
for u, p in parent.items():
    (x0, y0), (x1, y1) = xy[p], xy[u]
    if via[u] == "content":
        ax.plot([x0, x1], [y0, y1], color=PURPLE, lw=0.9, ls=(0, (3, 2)), alpha=0.8, zorder=1)
    else:
        ax.plot([x0, x1], [y0, y1], color=ORANGE if via[u] == "early" else BLUE, lw=0.9, alpha=0.55, zorder=1)
for kind, col, size in [("graph", BLUE, 16), ("content", PURPLE, 26), ("early", ORANGE, 90)]:
    us = [u for u in parent if via[u] == kind]
    ax.scatter([xy[u][0] for u in us], [xy[u][1] for u in us], s=size, color=col, zorder=3, linewidths=0)
hits = [u for u in parent if u in real_set and u not in early]
for u in hits:
    ax.scatter(*xy[u], s=320, marker="*", color=RED, zorder=5)
    ax.annotate(f"u{u}: ตรงกับคนที่แชร์จริง (step {ring[u]})", xy[u], xytext=(0, -40), ha="center", textcoords="offset points",
                fontsize=13, color=RED, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
ax.scatter(0, 0, s=260, color=INK, marker="s", zorder=6)
lim = 1.0 + len(steps) * 0.42 + 0.3
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
n_graph = sum(v == "graph" for v in via.values()); n_content = sum(v == "content" for v in via.values())
ax.set_title(f"เส้นทางที่ generate (G′): {n_graph + n_content} คน ใน {len(steps)} step", fontsize=17, color=INK, loc="left", pad=10)
handles = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=9, label="early (จริง)"),
           plt.Line2D([], [], marker="o", ls="-", color=BLUE, ms=5, label=f"เลือกจากกราฟ H ({n_graph})"),
           plt.Line2D([], [], marker="o", ls="--", color=PURPLE, ms=6, label=f"เลือกจากเนื้อหาข่าว ({n_content})"),
           plt.Line2D([], [], marker="*", ls="", color=RED, ms=14, label="ตรงกับคนแชร์จริง")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.94, 1.0), frameon=False, fontsize=13, labelcolor=INK)
ax.annotate("วงใน → วงนอก = step 0 → 20", (0, -(1.0 + 20 * 0.42)), xytext=(0, -22), textcoords="offset points",
            ha="center", fontsize=12, color=MUTED)
fig.savefig(f"{D}/generated_path.png", bbox_inches="tight", facecolor="white")
print("generated_path.png graph", n_graph, "content", n_content, "hits", hits, "real", len(real_users))
