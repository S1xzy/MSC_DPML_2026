"""Insert (or refresh) the "code ที่ใช้รันจริง" cells inside each section of the existing notebooks without touching their saved outputs.

    python tools/add_source_cells.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from source_cells import insert_source  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("RPPG-Fake_Demo_Colab.ipynb", "RPPG-Fake_Demo_Local.ipynb"):
    p = os.path.join(HERE, name)
    nb = json.load(open(p, encoding="utf-8"))
    before = len(nb["cells"])
    nb["cells"] = insert_source(nb["cells"])
    with open(p, "w", encoding="utf-8", newline="") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(name, before, "->", len(nb["cells"]), "cells")
