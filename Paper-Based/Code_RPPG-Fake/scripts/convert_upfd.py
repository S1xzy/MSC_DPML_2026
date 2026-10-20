"""Convert UPFD propagation graphs + FakeNewsNet titles into the RPPG-Fake input format.

GossipCop / PolitiFact tweets can no longer be crawled (paid X API), so the propagation comes from UPFD
(GNN-FakeNews, built from FakeNewsNet): retweet trees, node -> Twitter user id and node -> timestamp
mappings. The news text is the FakeNewsNet title (article bodies and images are not available).

Inputs (``data/external``):
  upfd/<name>/raw/{A.txt,node_graph_id.npy,graph_labels.npy}   (downloaded by torch_geometric UPFD)
  upfd/mapping/{gos|pol}_id_twitter_mapping.pkl, {gos|pol}_id_time_mapping.pkl   (GNN-FakeNews repo)
  fakenewsnet/<name>_{fake,real}.csv
Output: ``data/<name>/news_actions_<name>.csv`` (id, text, label, actions_id), label 0 = fake, 1 = real.

    python scripts/convert_upfd.py --name gossipcop
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict

import numpy as np

csv.field_size_limit(2**31 - 1)


class SafeUnpickler(pickle.Unpickler):
    """The mapping files are plain dicts of int -> str / int; refuse anything else."""

    ALLOWED = {("builtins", "dict"), ("builtins", "int"), ("builtins", "str")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gossipcop", choices=["gossipcop", "politifact"])
    ap.add_argument("--ext", default="data/external")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--content", action="store_true",
                    help="use article text from data/external/fnn_content/<name>/articles.jsonl (title + body)")
    a = ap.parse_args()
    pre = {"gossipcop": "gos", "politifact": "pol"}[a.name]
    raw = os.path.join(a.ext, "upfd", a.name, "raw")
    with open(os.path.join(a.ext, "upfd", "mapping", f"{pre}_id_twitter_mapping.pkl"), "rb") as f:
        tw = SafeUnpickler(f).load()
    with open(os.path.join(a.ext, "upfd", "mapping", f"{pre}_id_time_mapping.pkl"), "rb") as f:
        tm = SafeUnpickler(f).load()
    gid = np.load(os.path.join(raw, "node_graph_id.npy"))
    edges = np.loadtxt(os.path.join(raw, "A.txt"), delimiter=",", dtype=np.int64)
    titles, labels = {}, {}
    for lab in ("fake", "real"):
        with open(os.path.join(a.ext, "fakenewsnet", f"{a.name}_{lab}.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                titles[r["id"]], labels[r["id"]] = r["title"], 0 if lab == "fake" else 1
    bodies = {}
    if a.content:
        with open(os.path.join(a.ext, "fnn_content", a.name, "articles.jsonl"), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("text"):
                    bodies[rec["id"]] = rec["text"]
    roots = {n for n, v in tw.items() if str(v).startswith(a.name)}
    uid = {}

    def u(node):
        x = str(tw[node])
        if x not in uid:
            uid[x] = len(uid) + 2
        return uid[x]

    per_graph = defaultdict(list)
    for s, d in edges:
        head = 1 if s in roots else u(s)
        per_graph[int(gid[d])].append({"head_user": str(head), "user": str(u(d)), "time": str(int(tm[d]))})
    rows = []
    for r in sorted(roots):
        nid = tw[r]
        eng = sorted(per_graph[int(gid[r])], key=lambda e: int(e["time"]))
        text = titles[nid] + ("\n" + bodies[nid] if nid in bodies else "")
        rows.append({"id": nid, "text": text, "label": labels[nid], "actions_id": repr(eng)})
    out_dir = os.path.join(a.out_root, a.name if a.name == "gossipcop" else "politifact_upfd")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"news_actions_{os.path.basename(out_dir)}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "actions_id"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "user_id_map.json"), "w") as f:
        json.dump(uid, f)
    fake = sum(r["label"] == 0 for r in rows)
    root_edges = sum(1 for s, _ in edges if s in roots)
    print(f"with article body: {sum(r['id'] in bodies for r in rows)}/{len(rows)}")
    print(f"{out}: news={len(rows)} fake={fake} real={len(rows) - fake} engagements={len(edges)} "
          f"(direct from news={root_edges}) users={len(uid)}")


if __name__ == "__main__":
    main()
