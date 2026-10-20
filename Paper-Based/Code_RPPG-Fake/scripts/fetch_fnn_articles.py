"""Fetch FakeNewsNet article text and lead image for the news that have UPFD propagation graphs.

FakeNewsNet only ships ids, URLs and titles; its official crawler downloads the article from the publisher
and falls back to the Wayback Machine. This does the same for the GossipCop / PolitiFact news in UPFD:
  1. fetch the article URL; if that fails or yields no body text, use the closest Wayback snapshot (2018);
  2. extract the main text with trafilatura and the lead image from og:image / twitter:image;
  3. download that image (live, then Wayback), check it decodes, save as ``images/<news_id>.jpg``.
Output: ``data/external/fnn_content/<name>/articles.jsonl`` + ``images/``. Resumable and rate-limited.
Articles and images belong to their publishers; they are used for research only and are not committed.

    python scripts/fetch_fnn_articles.py --name gossipcop
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import trafilatura
from PIL import Image

csv.field_size_limit(2**31 - 1)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-crawler (RPPG-Fake reproduction)"}
_wb_lock = threading.Lock()
_wb_last = [0.0]
META_IMG = re.compile(r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)(?::src)?["\'][^>]*content=["\']([^"\']+)', re.I)
META_IMG2 = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\'](?:og:image|twitter:image)', re.I)


def wb_get(url, gap):
    """Wayback requests go through one lock so archive.org sees at most one request per ``gap`` seconds."""
    with _wb_lock:
        wait = _wb_last[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _wb_last[0] = time.time()
    return requests.get(url, headers=UA, timeout=30)


WB_URL = re.compile(r"^https?://web\.archive\.org/web/(\d+)[a-z_]*/(.+)$")


def snapshot(url, gap):
    """Closest Wayback snapshot: first near 2018 (when FakeNewsNet was collected), then any time."""
    m = WB_URL.match(url)
    if m:
        return m.group(1)
    for stamp in ("&timestamp=20180601", ""):
        try:
            r = wb_get(f"https://archive.org/wayback/available?url={requests.utils.quote(url, safe='')}{stamp}", gap)
            s = r.json().get("archived_snapshots", {}).get("closest")
            if s and s.get("available"):
                return s["timestamp"]
        except Exception:
            pass
    return None


def original(url):
    m = WB_URL.match(url)
    return m.group(2) if m else url


def extract(html):
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    m = META_IMG.search(html) or META_IMG2.search(html)
    return text.strip(), (m.group(1).strip() if m else None)


def save_image(content, path):
    Image.open(io.BytesIO(content)).verify()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if min(img.size) < 50:
        raise ValueError("too small")
    img.thumbnail((1024, 1024))
    img.save(path, "JPEG", quality=90)


def process(item, img_dir, gap):
    nid, url = item
    url = url if url.startswith("http") else "https://" + url
    url_raw = url
    url = original(url)
    rec = {"id": nid, "url": url, "source": None, "text": "", "image_url": None, "image": False}
    html = None
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
            html, rec["source"] = r.text, "live"
    except Exception:
        pass
    ts = None
    if html is not None:
        rec["text"], rec["image_url"] = extract(html)
    if html is None or len(rec["text"]) < 200:
        ts = snapshot(url_raw, gap)
        if ts:
            try:
                r = wb_get(f"https://web.archive.org/web/{ts}id_/{url}", gap)
                if r.status_code == 200:
                    t, im = extract(r.text)
                    if len(t) > len(rec["text"]):
                        rec["text"], rec["source"] = t, "wayback"
                    rec["image_url"] = rec["image_url"] or im
            except Exception:
                pass
    if rec["image_url"]:
        img_url = requests.compat.urljoin(url, rec["image_url"])
        path = os.path.join(img_dir, f"{nid}.jpg")
        for attempt in ("live", "wayback"):
            try:
                if attempt == "live":
                    r = requests.get(img_url, headers=UA, timeout=20)
                else:
                    ts = ts or snapshot(url, gap)
                    if not ts:
                        break
                    r = wb_get(f"https://web.archive.org/web/{ts}im_/{img_url}", gap)
                if r.status_code == 200:
                    save_image(r.content, path)
                    rec["image"] = True
                    break
            except Exception:
                continue
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gossipcop", choices=["gossipcop", "politifact"])
    ap.add_argument("--ext", default="data/external")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wayback_gap", type=float, default=1.0, help="min seconds between archive.org requests")
    ap.add_argument("--retry_failed", action="store_true", help="re-fetch records that got no text")
    a = ap.parse_args()
    out = os.path.join(a.ext, "fnn_content", a.name)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl = os.path.join(out, "articles.jsonl")
    done = set()
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if a.retry_failed:  # keep the successful records, fetch the rest again
            recs = [r for r in recs if r.get("text")]
            with open(jsonl, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        done = {r["id"] for r in recs}
    # only the news that have a UPFD propagation graph
    import pickle

    class Safe(pickle.Unpickler):
        def find_class(self, m, n):
            if (m, n) in {("builtins", "dict"), ("builtins", "int"), ("builtins", "str")}:
                return super().find_class(m, n)
            raise pickle.UnpicklingError(f"blocked {m}.{n}")

    pre = {"gossipcop": "gos", "politifact": "pol"}[a.name]
    with open(os.path.join(a.ext, "upfd", "mapping", f"{pre}_id_twitter_mapping.pkl"), "rb") as f:
        upfd_ids = {v for v in Safe(f).load().values() if str(v).startswith(a.name)}
    items = []
    for lab in ("fake", "real"):
        with open(os.path.join(a.ext, "fakenewsnet", f"{a.name}_{lab}.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["id"] in upfd_ids and r["id"] not in done and r["news_url"]:
                    items.append((r["id"], r["news_url"]))
    print(f"{a.name}: {len(upfd_ids)} UPFD news, {len(done)} already done, {len(items)} to fetch", flush=True)
    lock = threading.Lock()
    stats = {"live": 0, "wayback": 0, "none": 0, "image": 0}
    with open(jsonl, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(lambda it: process(it, img_dir, a.wayback_gap), items), 1):
            with lock:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
                stats[rec["source"] or "none"] += 1
                stats["image"] += rec["image"]
            if i % 100 == 0:
                print(f"{i}/{len(items)} {stats}", flush=True)
    print(f"done {stats}", flush=True)


if __name__ == "__main__":
    main()
