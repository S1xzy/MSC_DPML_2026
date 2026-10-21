"""Module 3 figures from m3.json (system Python + matplotlib)."""
import json, math, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

D = sys.argv[1]
plt.rcParams["font.family"] = ["Leelawadee UI", "Segoe UI Symbol", "Segoe UI"]
INK, MUTED, GRID = "#14202C", "#4F5D6B", "#D8DFE6"
BLUE, ORANGE, PURPLE, RED, GREEN = "#2950B8", "#D9822B", "#7A4FC4", "#C0392B", "#2E8B57"
M = json.load(open(f"{D}/m3.json"))
A = M["items"]["real"]

# ---------------------------------------------------------------- layout of the final graph (radial tree)
n = len(A["users"]) + 1                      # local node 0 = news
parent = {c: p for p, c in A["edges"]}
children = defaultdict(list)
for c, p in parent.items():
    children[p].append(c)
bounds = A["bounds"]
snap_of = {0: 0}
for t, b in enumerate(bounds):
    for v in range(1, b):
        snap_of.setdefault(v, t)
leaves = {}
def count(v):
    leaves[v] = 1 if not children[v] else sum(count(c) for c in children[v])
    return leaves[v]
count(0)
ang = {}
def place(v, a0, a1):
    ang[v] = (a0 + a1) / 2
    a = a0
    for c in sorted(children[v]):
        span = (a1 - a0) * leaves[c] / leaves[v]
        place(c, a, a + span); a += span
place(0, 0, 2 * math.pi)
rad = {v: (0 if v == 0 else 1 + 0.25 * snap_of[v]) for v in range(n)}
xy = {v: (rad[v] * math.cos(ang[v]), rad[v] * math.sin(ang[v])) for v in range(n)}
n_real = A["n_real"]
kind = lambda v: "news" if v == 0 else ("early" if v <= n_real else "gen")

# ---------------------------------------------------------------- figure 1: snapshots + attention
fig = plt.figure(figsize=(16.6, 7.4), dpi=150)
top = fig.add_gridspec(1, 5, left=0.0, right=1.0, top=0.97, bottom=0.53, wspace=0.05)
show = [0, 2, 7, 14, len(bounds) - 1]
lim = 1 + 0.25 * (len(bounds) - 1) + 0.2
for k, t in enumerate(show):
    ax = fig.add_subplot(top[k])
    nodes = range(bounds[t])
    for v in nodes:
        if v:
            p = parent[v]
            ax.plot(*zip(xy[p], xy[v]), color=ORANGE if kind(v) == "early" else BLUE, lw=0.6, alpha=0.6)
    for kd, col, s in [("gen", BLUE, 7), ("early", ORANGE, 40), ("news", INK, 70)]:
        vs = [v for v in nodes if kind(v) == kd]
        if vs:
            ax.scatter([xy[v][0] for v in vs], [xy[v][1] for v in vs], s=s, color=col, marker="s" if kd == "news" else "o", zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    lab = "early (จริง)" if t < n_real else "+ G′"
    ax.set_title(f"snapshot {t}  ·  {bounds[t]} node\n{lab}", fontsize=14, color=INK)
fig.text(0.0, 0.50, "ซ้าย → ขวา: กราฟโตขึ้นทีละ snapshot (แสดง 5 จาก %d) · ส้ม = early จริง · น้ำเงิน = user ที่ generate" % len(bounds),
         fontsize=13, color=MUTED)

bot = fig.add_gridspec(1, 2, left=0.04, right=0.98, top=0.42, bottom=0.04, wspace=0.55, width_ratios=[1, 1.4])
ax = fig.add_subplot(bot[0])
S = np.array(A["self_attn"])
im = ax.imshow(S, cmap="Blues", aspect="auto")
ax.set_xlabel("snapshot ที่ถูกมอง (key)", fontsize=12, color=MUTED); ax.set_ylabel("snapshot (query)", fontsize=12, color=MUTED)
ax.set_title("Self-attention ระหว่าง snapshot", fontsize=14, color=INK, loc="left")
ax.tick_params(labelsize=10, colors=MUTED)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=10)
ax = fig.add_subplot(bot[1])
C = np.array(A["cross_attn"]).mean(0)              # average over the snapshot queries -> weight per node
groups = [("ข่าว (seed)", C[0], INK), (f"early จริง ({n_real} คน)", C[1:n_real + 1].sum(), ORANGE),
          (f"user ที่ generate ({n - 1 - n_real} คน)", C[n_real + 1:].sum(), BLUE)]
ys = np.arange(len(groups))[::-1]
ax.barh(ys, [g[1] for g in groups], color=[g[2] for g in groups], height=0.6)
for y, g in zip(ys, groups):
    ax.text(g[1] + 0.01, y, f"{g[1] * 100:.0f}%", va="center", fontsize=13, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([g[0] for g in groups], fontsize=13, color=INK)
ax.set_xlim(0, 1.1); ax.set_xticks([])
for s in ["top", "right", "bottom"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID)
srt = np.sort(C)[::-1]; k90 = int((np.cumsum(srt) < 0.9).sum() + 1)
ax.set_title("Cross-attention: snapshot มองไปที่ node ไหนของกราฟสุดท้าย", fontsize=14, color=INK, loc="left")
ax.text(0.98, 0.5, f"90% ของน้ำหนักอยู่ที่\nuser ที่ generate แค่ {k90} คน", transform=ax.transAxes, ha="right", va="center", fontsize=13, color=BLUE)
fig.savefig(f"{D}/m3_snapshots.png", bbox_inches="tight", facecolor="white")
print("m3_snapshots.png", [round(g[1], 3) for g in groups], "self diag mean", round(float(np.diag(S).mean()), 3))

# ---------------------------------------------------------------- figure 2: what decides P(real)
fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.2), dpi=150, gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25))
ax = axes[0]
names = [("graph", "เส้นทาง (h*)", BLUE), ("text", "ข้อความ", GREEN), ("image", "ภาพ", PURPLE), ("bias", "ค่าคงที่", "#8A97A6")]
items = [("real", "ข่าว real: Love is Stronger than Hate"), ("fake", "ข่าว fake: Two female hostages have run out")]
w = 0.36
for j, (key, lab) in enumerate(items):
    parts = M["items"][key]["logit_parts"]
    x = np.arange(len(names)) + (j - 0.5) * w
    vals = [parts[k] for k, _, _ in names]
    bars = ax.bar(x, vals, width=w, color=[c for _, _, c in names], alpha=1.0 if j == 0 else 0.5,
                  edgecolor=INK if j == 1 else "none", hatch="//" if j == 1 else None, linewidth=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.12 if v >= 0 else -0.12), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=11, color=INK)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(np.arange(len(names))); ax.set_xticklabels([l for _, l, _ in names], fontsize=13, color=INK)
ax.set_ylabel("ส่วนของ logit (บวก = ไปทาง real)", fontsize=12, color=MUTED)
pr, pf = M["items"]["real"]["p_real"], M["items"]["fake"]["p_real"]
ax.set_title(f"แยก logit สุดท้าย: แท่งทึบ = ข่าว real (P {pr:.3f}) · แท่งลาย = ข่าว fake (P {pf:.3f})", fontsize=14, color=INK, loc="left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED)
ax = axes[1]
for key, col, lab in [("real", BLUE, "ข่าว real"), ("fake", RED, "ข่าว fake")]:
    c = M["items"][key]["curve"]
    ax.plot(range(len(c)), c, color=col, lw=2.2, marker="o", ms=4, label=lab)
ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
ax.text(len(c) - 1, 0.45, "threshold 0.5", ha="right", fontsize=11, color=MUTED)
ax.set_ylim(0, 1.02); ax.set_xlabel("จำนวน step ของ G′ ที่ใส่ให้ detector (0 = early อย่างเดียว)", fontsize=12, color=MUTED)
ax.set_ylabel("P(real)", fontsize=12, color=MUTED)
ax.set_title("ถ้าหยุด generate เร็วขึ้น ผลเปลี่ยนไหม", fontsize=14, color=INK, loc="left")
ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.88))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED)
fig.savefig(f"{D}/m3_decision.png", bbox_inches="tight", facecolor="white")
print("m3_decision.png")
