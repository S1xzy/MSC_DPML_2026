# 📄 Paper Based

This section collects all work related to the paper assigned by the professor for in-depth study (research project basis).

> ⏳ **Paper Based**: Early Detection of Multimodal Fake News via Reinforced Propagation Path Generation \
> 🔎 **Author**: L. Zhang et al. \
> 💻 **Official Code (RPPG-Fake)**: [github.com/LitianD/RPPG-Fake](https://github.com/LitianD/RPPG-Fake)

## Contents

| Item | Description |
|---|---|
| `2P Early_Detection_of_...pdf` | The paper |
| [`Code_RPPG-Fake/`](Code_RPPG-Fake/) | Runnable re-implementation of the three modules (the official code does not run). How to run, data and deviations from the paper: [`RUNNING.md`](Code_RPPG-Fake/RUNNING.md) |
| [`HANDOFF.md`](HANDOFF.md) | Full experiment log (rounds 1–8), results, bugs found and fixed, leakage checks, demo set (Thai) |
| `Paper_Summary_RPPG-Fake.html` | Interactive paper summary (open in a browser) |
| `RPPG-Fake_Methodology_Interactive.html` | Interactive walkthrough of the method |
| `RPPG-Fake_Results.html` | Reproduction results |
| `Project_Tracker_RPPG-Fake.html`, `update_tracker.py` | Progress tracker |
| [`Slides_RPPG-Fake/`](Slides_RPPG-Fake/) | Source of the presentation deck (52 slides, Thai): background, problem formulation, methodology, experiments, ablation, research critique, conclusion |

## Reproduction status (round 8, 3 seeds, early-stage accuracy)

| Dataset | Ours | Paper (Table III) |
|---|---|---|
| PHEME | 0.837 ± 0.012 | 0.829 |
| GossipCop | 0.878 ± 0.087 | 0.873 |
| PolitiFact (authors' data, no images) | 0.826 ± 0.043 | 0.902 |

The generated propagation paths (G′) did not improve accuracy in our runs; see `HANDOFF.md` for the analysis and limitations.

Datasets, trained models and logs (`Code_RPPG-Fake/data/`, `logs/`, `.venv/`) are not in the repository. The authors' PolitiFact data (`data/data.zip`) is in the official repository; the other datasets are rebuilt with the scripts in `Code_RPPG-Fake/scripts/`.
