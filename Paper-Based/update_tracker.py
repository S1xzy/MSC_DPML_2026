"""Refresh the live block of Project_Tracker_RPPG-Fake.html from the Round 4 (paper protocol) logs.

    Code_RPPG-Fake/.venv/Scripts/python update_tracker.py

Reads Code_RPPG-Fake/data/<ds>/generated_paper/seed<k>/ and logs/stage2_detect/runs/*/test_metrics.json
(tags *_paper) and rewrites the text between /*LIVE*/ and /*END LIVE*/ in the HTML.
"""
import datetime
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "Code_RPPG-Fake")
HTML = os.path.join(HERE, "Project_Tracker_RPPG-Fake.html")
DATASETS = ["pheme", "gossipcop", "politifact"]
SETTINGS = ["early", "full", "early_no_gen"]
SEEDS = range(5)
METRICS = ["test/acc", "test/pre", "test/rec", "test/f1", "test/macro_f1", "test/auc"]

runs = {}
for p in glob.glob(os.path.join(CODE, "logs", "stage2_detect", "runs", "*", "test_metrics.json")):
    with open(p) as f:
        m = json.load(f)
    tag = m.get("tag") or ""
    if not tag.endswith("_paper"):
        continue
    key = (m.get("dataset"), tag[: -len("_paper")], m["seed"])
    if key not in runs or os.path.getmtime(p) > runs[key][0]:
        runs[key] = (os.path.getmtime(p), m)

live = {"updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "status": {}, "results": {}}
for ds in DATASETS:
    st = []
    for s in SEEDS:
        gen = os.path.join(CODE, "data", ds, "generated_paper", f"seed{s}")
        n2 = sum((ds, k, s) in runs for k in SETTINGS)
        if n2 == len(SETTINGS):
            st.append("done")
        elif os.path.exists(os.path.join(gen, "stage1.pt")):
            st.append(f"s2:{n2}")
        elif os.path.isdir(gen):
            st.append("s1")
        else:
            st.append("wait")
    live["status"][ds] = st
    res = {}
    for k in SETTINGS:
        rows = [runs[(ds, k, s)][1] for s in SEEDS if (ds, k, s) in runs]
        if not rows:
            continue
        mean = [round(statistics.mean(r[m] for r in rows), 4) for m in METRICS]
        sd = [round(statistics.stdev([r[m] for r in rows]), 4) if len(rows) > 1 else 0 for m in METRICS]
        res[k] = {"n": len(rows), "mean": mean, "sd": sd}
    if res:
        live["results"][ds] = res

with open(HTML, encoding="utf-8") as f:
    html = f.read()
new = re.sub(r"/\*LIVE\*/.*?/\*END LIVE\*/", lambda _: "/*LIVE*/" + json.dumps(live, ensure_ascii=False) + "/*END LIVE*/",
             html, count=1, flags=re.S)
with open(HTML, "w", encoding="utf-8") as f:
    f.write(new)
print(json.dumps(live, ensure_ascii=False, indent=1))
