"""Build an unseen demo / test set from the 4 PHEME events that are not in the training data.

Source: PHEME dataset for Rumour Detection and Veracity Classification (figshare 6392078,
PHEME_veracity.tar.bz2, CC BY 4.0), extracted to ``data/external_new/pheme9/all-rnr-annotated-threads``.
The training PHEME (figshare 4010619) has 5 events; this release adds ebola-essien, gurlitt,
prince-toronto and putinmissing. Output ``data/pheme_demo/``:

* ``news_actions_pheme_demo.csv`` - same format as ``data/pheme`` (id, text, label, actions_id, event)
* ``images/<tweet_id>.jpg``       - first photo of the source tweet (downloaded, see fetch_pheme_images.py)
* ``user_id_map.json``            - training ids are kept (``data/pheme/user_id_map.json``); users never seen
                                     in training get new ids above the training range (unknown to the model)

Only *complete* news items are kept: text + image + propagation path (>= 2 engagements), as in training.

    python scripts/make_pheme_demo.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_pheme import ts  # noqa: E402
from fetch_pheme_images import fetch, photo_url  # noqa: E402

NEW_EVENTS = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external_new/pheme9/all-rnr-annotated-threads")
    ap.add_argument("--train_map", default="data/pheme/user_id_map.json")
    ap.add_argument("--train_csv", default="data/pheme/news_actions_pheme.csv")
    ap.add_argument("--out", default="data/pheme_demo")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    with open(a.train_map) as f:
        uid = json.load(f)
    n_train_ids = max(uid.values()) + 1
    csv.field_size_limit(1 << 30)
    with open(a.train_csv, encoding="utf-8") as f:
        train_ids = {r["id"] for r in csv.DictReader(f)}

    def u(x):
        x = str(x)
        if x not in uid:
            uid[x] = len(uid) + 2
        return uid[x]

    rows, photos = [], []
    for ev in NEW_EVENTS:
        for cls in ("rumours", "non-rumours"):
            for thread in sorted(glob.glob(os.path.join(a.src, f"{ev}-all-rnr-threads", cls, "*"))):
                if os.path.basename(thread).startswith("._"):
                    continue
                srcs = [p for p in glob.glob(os.path.join(thread, "source-tweets", "*.json"))
                        if not os.path.basename(p).startswith("._")]
                if not srcs:
                    continue
                with open(srcs[0], encoding="utf-8") as f:
                    s = json.load(f)
                if s["id_str"] in train_ids:
                    continue
                tweets, reacts = {s["id_str"]: s}, []
                for p in glob.glob(os.path.join(thread, "reactions", "*.json")):
                    if os.path.basename(p).startswith("._"):
                        continue
                    with open(p, encoding="utf-8") as f:
                        r = json.load(f)
                    tweets[r["id_str"]] = r
                    reacts.append(r)
                eng = [{"head_user": "1", "user": str(u(s["user"]["id_str"])), "time": str(ts(s["created_at"]))}]
                for r in sorted(reacts, key=lambda x: ts(x["created_at"])):
                    parent = r.get("in_reply_to_status_id_str")
                    head = 1 if parent in (None, s["id_str"]) or parent not in tweets else u(tweets[parent]["user"]["id_str"])
                    eng.append({"head_user": str(head), "user": str(u(r["user"]["id_str"])), "time": str(ts(r["created_at"]))})
                rows.append({"id": s["id_str"], "text": s["text"], "label": 0 if cls == "rumours" else 1,
                             "actions_id": repr(eng), "event": ev})
                url = photo_url(s)
                if url:
                    photos.append((s["id_str"], url))
    print(f"threads in new events: {len(rows)}  with a photo link: {len(photos)}", flush=True)
    img_dir = os.path.join(a.out, "images")
    stats = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(lambda j: fetch(j, img_dir, a.delay), photos):
            stats[res] = stats.get(res, 0) + 1
    print(f"image download: {stats}", flush=True)
    have = {os.path.splitext(n)[0] for n in os.listdir(img_dir)}
    keep = [r for r in rows if r["id"] in have and len(eval(r["actions_id"])) >= 2]  # noqa: S307 - own repr output
    with open(os.path.join(a.out, "news_actions_pheme_demo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "actions_id", "event"])
        w.writeheader()
        w.writerows(keep)
    with open(os.path.join(a.out, "user_id_map.json"), "w") as f:
        json.dump(uid, f)
    seen = sum(1 for r in keep for e in eval(r["actions_id"]) if int(e["user"]) < n_train_ids)  # noqa: S307
    tot = sum(len(eval(r["actions_id"])) for r in keep)  # noqa: S307
    by_ev = {ev: [sum(1 for r in keep if r["event"] == ev and r["label"] == y) for y in (0, 1)] for ev in NEW_EVENTS}
    print(f"complete news kept: {len(keep)}  fake/real by event: {by_ev}")
    print(f"engagements: {tot}  by users seen in training: {seen} ({seen / max(tot, 1):.1%})")


if __name__ == "__main__":
    main()
