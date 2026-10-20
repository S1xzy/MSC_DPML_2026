"""Convert the PHEME rumour dataset into the RPPG-Fake input format.

Source: figshare 10.6084/m9.figshare.4010619 (pheme-rnr-dataset.tar.bz2, CC BY 4.0), extracted to
``data/external/pheme/pheme-rnr-dataset``. Output: ``data/pheme/news_actions_pheme.csv`` with the same
columns as the PolitiFact file shipped with the code (id, text, label, actions_id).

* news item  = source tweet of a thread; text = its tweet text
* label      = 0 for rumour (treated as fake), 1 for non-rumour (real) - paper: y = 0 fake, y = 1 real
* engagement = (head_user, user, time): the source author engages first (head = news root, id 1); a reply
  to the source tweet has head = news root; a reply to another reply has head = that reply's author.
User ids are mapped to integers starting at 2 (0 is unused, 1 is the news root, as in the PolitiFact file).

    python scripts/convert_pheme.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone

FMT = "%a %b %d %H:%M:%S +0000 %Y"


def ts(s: str) -> int:
    return int(datetime.strptime(s, FMT).replace(tzinfo=timezone.utc).timestamp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/pheme/pheme-rnr-dataset")
    ap.add_argument("--out", default="data/pheme")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    uid = {}

    def u(x):
        x = str(x)
        if x not in uid:
            uid[x] = len(uid) + 2
        return uid[x]

    rows, n_rep = [], 0
    for thread in sorted(glob.glob(os.path.join(a.src, "*", "*", "*"))):
        cls = os.path.basename(os.path.dirname(thread))
        if cls not in ("rumours", "non-rumours"):
            continue
        srcs = glob.glob(os.path.join(thread, "source-tweet", "*.json"))
        if not srcs:
            continue
        with open(srcs[0], encoding="utf-8") as f:
            s = json.load(f)
        tweets = {s["id_str"]: s}
        reacts = []
        for p in glob.glob(os.path.join(thread, "reactions", "*.json")):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            tweets[r["id_str"]] = r
            reacts.append(r)
        eng = [{"head_user": "1", "user": str(u(s["user"]["id_str"])), "time": str(ts(s["created_at"]))}]
        for r in sorted(reacts, key=lambda x: ts(x["created_at"])):
            parent = r.get("in_reply_to_status_id_str")
            head = 1 if parent in (None, s["id_str"]) or parent not in tweets else u(tweets[parent]["user"]["id_str"])
            eng.append({"head_user": str(head), "user": str(u(r["user"]["id_str"])), "time": str(ts(r["created_at"]))})
            n_rep += 1
        rows.append({"id": s["id_str"], "text": s["text"], "label": 0 if cls == "rumours" else 1, "actions_id": repr(eng)})
    out = os.path.join(a.out, "news_actions_pheme.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "actions_id"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "user_id_map.json"), "w") as f:
        json.dump(uid, f)
    fake = sum(r["label"] == 0 for r in rows)
    print(f"{out}: news={len(rows)} fake(rumour)={fake} real={len(rows) - fake} reactions={n_rep} users={len(uid)}")


if __name__ == "__main__":
    main()
