"""Make a copy of a dataset whose real/fake ratio matches the paper's Table I (e.g. GossipCop: 3,591 real of 5,526 = 65%).

    python scripts/make_ratio_subset.py --src gossipcop --dst gossipcop_r65 --real_share 0.65

Keeps every news item of the larger-needed class and randomly drops items of the other class (fixed seed), so the
class prior is the same as in the paper. Images of the kept items are copied; nothing in ``--src`` is changed.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil

csv.field_size_limit(2**31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gossipcop")
    ap.add_argument("--dst", default="gossipcop_r65")
    ap.add_argument("--real_share", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, dst = os.path.join("data", a.src), os.path.join("data", a.dst)
    with open(os.path.join(src, f"news_actions_{a.src}.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    real = [r for r in rows if r["label"] == "1"]
    fake = [r for r in rows if r["label"] == "0"]
    n_fake = round(len(real) * (1 - a.real_share) / a.real_share)
    if n_fake > len(fake):  # not enough fake items: keep all fake, drop real instead
        n_fake, real = len(fake), random.Random(a.seed).sample(real, round(len(fake) * a.real_share / (1 - a.real_share)))
    fake = random.Random(a.seed).sample(fake, n_fake)
    keep = real + fake
    keep.sort(key=lambda r: rows.index(r))
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    with open(os.path.join(dst, f"news_actions_{a.dst}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    imgs = {os.path.splitext(n)[0]: n for n in os.listdir(os.path.join(src, "images"))}
    for r in keep:
        if r["id"] in imgs:
            shutil.copy2(os.path.join(src, "images", imgs[r["id"]]), os.path.join(dst, "images", imgs[r["id"]]))
    print(f"{dst}: {len(keep)} news  real {len(real)} ({len(real) / len(keep):.1%})  fake {len(fake)}  "
          f"images {len(os.listdir(os.path.join(dst, 'images')))}")


if __name__ == "__main__":
    main()
