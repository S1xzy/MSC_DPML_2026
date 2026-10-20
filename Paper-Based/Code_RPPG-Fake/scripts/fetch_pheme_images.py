"""Download the images attached to PHEME source tweets (n_V for the multimodal content, Sec. III-D3).

The PHEME release keeps the tweet JSON, including ``media_url_https`` of attached photos, but not the image
files. This fetches the first photo of every source tweet into ``data/external/pheme/images/<tweet_id>.jpg``
(resumable, rate-limited, every file is checked to be a real image). Twitter keeps the rights to the images;
they are used for research only and are not committed to git.

    python scripts/fetch_pheme_images.py
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 (research; RPPG-Fake reproduction)"}


def photo_url(src: dict):
    for key in ("extended_entities", "entities"):
        for m in (src.get(key) or {}).get("media") or []:
            if m.get("type") == "photo":
                return m.get("media_url_https") or m.get("media_url")
    return None


def fetch(job, out_dir, delay):
    tid, url = job
    path = os.path.join(out_dir, f"{tid}.jpg")
    if os.path.exists(path):
        return "cached"
    time.sleep(delay)
    try:
        r = requests.get(url + "?name=medium", headers=UA, timeout=20)
        if r.status_code != 200:
            r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200:
            return f"http{r.status_code}"
        img = Image.open(io.BytesIO(r.content))
        img.verify()
        Image.open(io.BytesIO(r.content)).convert("RGB").save(path, "JPEG", quality=90)
        return "ok"
    except Exception as e:  # network error or not an image
        return type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/pheme/pheme-rnr-dataset")
    ap.add_argument("--out", default="data/external/pheme/images")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.25, help="seconds each worker waits before a request")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for p in glob.glob(os.path.join(a.src, "*", "*", "*", "source-tweet", "*.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        u = photo_url(s)
        if u:
            jobs.append((s["id_str"], u))
    print(f"source tweets with a photo: {len(jobs)}", flush=True)
    stats = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda j: fetch(j, a.out, a.delay), jobs), 1):
            stats[res] = stats.get(res, 0) + 1
            if i % 250 == 0:
                print(f"{i}/{len(jobs)} {stats}", flush=True)
    print(f"done {stats}  images in {a.out}: {len(os.listdir(a.out))}", flush=True)


if __name__ == "__main__":
    main()
