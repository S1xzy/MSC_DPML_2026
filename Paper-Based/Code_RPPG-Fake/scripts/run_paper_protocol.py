"""Paper protocol (Sec. IV): 5 runs per dataset with the paper's settings (configs as shipped:
T=200, K2=10, 20 time steps, RL 10 epochs), reporting early (Table III), full (Table II) and the
no-generated-path ablation. Datasets run one after another; the PC is kept awake while this runs.

    python scripts/run_paper_protocol.py --datasets pheme politifact
    python scripts/run_paper_protocol.py --datasets gossipcop

Results: logs/results_summary_<dataset>.md, rows tagged *_paper. Re-running resumes: Stage 1 outputs
already in data/<dataset>/generated_paper/seed<k>/ are reused.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def keep_awake():
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: block idle sleep while this process lives (no settings change)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    keep_awake()
    for ds in a.datasets:
        cmd = [sys.executable, "-u", "scripts/run_all.py", "--dataset", ds, "--seeds", *map(str, a.seeds),
               "--tag_suffix", "_paper", "--gen_name", "generated_paper", "--skip_stage1"]
        print(">> [paper protocol]", ds, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(">> [paper protocol] done", ds, flush=True)


if __name__ == "__main__":
    main()
