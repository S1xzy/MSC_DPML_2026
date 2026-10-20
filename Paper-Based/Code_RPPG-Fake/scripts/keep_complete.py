"""Keep only news items that have all three paper components: news text, image and propagation path.

For each dataset folder the full version is moved to ``data/_archive/<name>_full/`` (nothing is deleted),
and ``data/<name>/`` is rebuilt with
  * ``news_actions_<name>.csv``  - only complete news items
  * ``images/<news_id>.jpg``     - their images (copied from data/external)
A news item is complete when it has an image file, real text (PHEME: the tweet; GossipCop / PolitiFact-UPFD:
title + recovered article body) and at least two engagements in its propagation path.
Datasets with no complete item (PolitiFact from the authors' data.zip, the title-only PolitiFact-UPFD copy)
and the raw downloads in data/external are moved to data/_archive/ as well.

    python scripts/keep_complete.py
"""
from __future__ import annotations

import ast
import csv
import os
import shutil

csv.field_size_limit(2**31 - 1)
DATA = "data"
ARCHIVE = os.path.join(DATA, "_archive")
SETS = {
    "pheme": {"images": "external/pheme/images", "needs_body": False},
    "gossipcop": {"images": "external/fnn_content/gossipcop/images", "needs_body": True},
    "politifact_upfd": {"images": "external/fnn_content/politifact/images", "needs_body": True},
}
NO_COMPLETE = ["politifact", "politifact_upfd_title"]


def move(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            raise FileExistsError(dst)
        shutil.move(src, dst)
        print(f"archived {src} -> {dst}")


def main():
    os.makedirs(ARCHIVE, exist_ok=True)
    for name, cfg in SETS.items():
        src_dir = os.path.join(DATA, name)
        csv_path = os.path.join(src_dir, f"news_actions_{name}.csv")
        img_src = os.path.join(DATA, cfg["images"])
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        keep = []
        for r in rows:
            has_img = os.path.exists(os.path.join(img_src, f"{r['id']}.jpg"))
            has_text = ("\n" in r["text"]) if cfg["needs_body"] else bool(r["text"].strip())
            has_prop = len(ast.literal_eval(r["actions_id"])) >= 2
            if has_img and has_text and has_prop:
                keep.append(r)
        tmp = os.path.join(DATA, f"_{name}_complete")
        os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
        with open(os.path.join(tmp, f"news_actions_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)
        for r in keep:
            shutil.copy2(os.path.join(img_src, f"{r['id']}.jpg"), os.path.join(tmp, "images", f"{r['id']}.jpg"))
        if os.path.exists(os.path.join(src_dir, "user_id_map.json")):
            shutil.copy2(os.path.join(src_dir, "user_id_map.json"), tmp)
        move(src_dir, os.path.join(ARCHIVE, f"{name}_full"))
        os.rename(tmp, src_dir)
        fake = sum(r["label"] == "0" for r in keep)
        print(f"{name}: kept {len(keep)} / {len(rows)} news (fake {fake}, real {len(keep) - fake})")
    for name in NO_COMPLETE:
        move(os.path.join(DATA, name), os.path.join(ARCHIVE, name))
    move(os.path.join(DATA, "external"), os.path.join(ARCHIVE, "external"))


if __name__ == "__main__":
    main()
