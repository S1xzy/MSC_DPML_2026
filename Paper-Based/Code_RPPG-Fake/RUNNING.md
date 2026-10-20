# Running RPPG-Fake (fixed version)

The code published with the paper does not run: config files and source modules are missing, several
names are undefined and the generation reward / discriminator are not implemented. This branch
(`runnable`) re-implements the three modules of the paper so the full pipeline runs on the PolitiFact
data that ships with the repository, following the paper's equations as closely as the data allows.

## 1. Environment (Windows, NVIDIA GPU)

```bash
python -m pip install --user uv
python -m uv venv .venv --python 3.11
python -m uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv/Scripts/python.exe "lightning>=2.2" torchmetrics "hydra-core==1.3.2" hydra-colorlog pyrootutils rich torch_geometric pandas ftfy regex tqdm tensorboard "git+https://github.com/openai/CLIP.git"
```

Tested with Python 3.11, PyTorch 2.11 + CUDA 12.8 on an RTX 5050 Laptop GPU (8 GB).

## 2. Data

The model needs, for every news item, **news text + image + propagation path** (plus the label).
The working datasets in `data/` keep only news items that have all three:

| Folder | News | How it was built |
|---|---|---|
| `data/gossipcop/` | 2,827 | UPFD propagation + FakeNewsNet title + crawled article body and lead image (`convert_upfd.py --content`, `fetch_fnn_articles.py`) |
| `data/pheme/` | 2,004 | PHEME threads + source-tweet images (`convert_pheme.py`, `fetch_pheme_images.py`) |
| `data/politifact_upfd/` | 79 | as GossipCop (too small for reliable results) |

Each folder holds `news_actions_<name>.csv` and `images/<news_id>.jpg`; `scripts/keep_complete.py` builds
them from the full conversions and moves everything else to `data/_archive/` (full sets, the authors'
PolitiFact without images, raw downloads). The authors' PolitiFact can still be used with
`unzip data/data.zip -d data/politifact` (text + propagation only).

Preprocessing and CLIP content features are built automatically on first use (cached in
`data/<name>/processed/`).

## 3. Run

```bash
.venv/Scripts/python src/train_hete_RL.py seed=0          # Stage 1: Module 1 + Module 2, writes data/politifact/generated/seed0/
.venv/Scripts/python src/train.py seed=0                  # Stage 2: Module 3, early stage (Table III)
.venv/Scripts/python src/train.py seed=0 data.train_real=full data.eval_real=full   # Stage 2, full paths (Table II)
.venv/Scripts/python scripts/run_all.py --seeds 0 1 2 3 4      # 5 seeds -> logs/results_summary.md
```

| Stage | Time on RTX 5050 (seed 0) |
|---|---|
| Stage 1 (300 contrastive steps + 10 RL epochs + generation) | ~45 min |
| Stage 2 (30 epochs, early stopping) | ~3–5 min |

### Predicting unseen news (demo)

```bash
.venv/Scripts/python scripts/make_pheme_demo.py            # 4 PHEME events not in training -> data/pheme_demo/ (text + image + path)
.venv/Scripts/python scripts/predict.py --dataset pheme --input data/pheme_demo --tag early_final
```
`predict.py` picks the seed with the best validation accuracy, builds the early path (3-5 engagements), encodes
text + image with CLIP, generates G' with the trained generator and outputs P(real) per news item
(`logs/predict_<input>_<tag>_seed<k>.csv` + `.json` with metrics). Users never seen in training are mapped to id 0.
Stage 1 must have been run with this version (it saves `contrastive.ckpt` and the base embeddings).

## 4. What maps to what

| Paper | File |
|---|---|
| Data, split 7:1:2, graph H, meta-path RWR, Λ(t), early stage (3–5 engagements) | `src/data/components/politifact.py` |
| Content features n = [n_T ; n_V] (CLIP ViT-B/32) | `src/data/components/encode_content.py` |
| Module 1 – node view + PGD (Eq. 2–3), meta-path view (Eq. 4), L_CL (Eq. 5) | `src/models/components/rppg/contrastive.py` |
| Module 2 – MDP (Eq. 6), policy (Eq. 7–11), rewards (Eq. 12–14), actor-critic (Eq. 15–19) | `src/models/components/rppg/generator.py` |
| Module 3 – dynamic GAT (Eq. 20), temporal-structure fusion (Eq. 21), prediction + BCE (Eq. 22) | `src/models/components/rppg/detector.py` |
| Algorithm 1 lines 1–14 | `src/train_hete_RL.py`, `configs/train_Hete_RL.yaml` |
| Algorithm 1 lines 15–17 | `src/train.py`, `src/models/rppg_detector_module.py`, `src/data/rppg_datamodule.py`, `configs/train_detection_graph.yaml` |

## 5. Where this differs from the paper (and why)

### Implemented as written in the paper
GAT attention of Eq. 2 (custom layer, ReLU before the attention vector) · PGD-perturbed node view with
top-K1 / bottom-K1 sets and Eq. 3 · meta-path semantic graphs from RWR and Eq. 4 · Eq. 5 · Eq. 7 (summed
over the meta-paths a node belongs to) · state Eq. 8-10 · policy Eq. 11 with K2 actions, self-loop and stop
at T · temporal reward Eq. 12 with Λ(t) from evenly divided propagation time · global reward from a
discriminator trained real-vs-generated (Eq. 13), R_G = −log(1 − D(G′_t)) · Eq. 14 · actor-critic Eq. 15-19
(TD target with E_π[Q(s_{t+1}, ·)], actor = Q · ∇log π, no baseline) · dynamic GAT + position embedding Eq. 20 ·
fusion Eq. 21 (query = temporal sequence, key/value = structure features, residuals as written) ·
σ(W_P[h*; n] + b) with binary cross-entropy Eq. 22 · AdamW, lr 5e-3, batch 16, hidden 128 · 7:1:2 split ·
5 runs · early stage = 3-5 earliest engagements.

### Deviations forced by the data
| # | Paper | Here | Reason |
|---|---|---|---|
| 1 | Publisher nodes; meta-paths PNU, NUN, UNU | No publishers; meta-paths UNU, NUN, **UU** (repost) | Every cascade in the released data starts from the same placeholder user, so publishers are unknown |
| 2 | Multimodal content (text + image) | CLIP ViT-B/32 text; image feature = zeros (`data.image_dir` can point to images) | No images in the released data; the paper's own encoder [28] is not released, CLIP is what the official code loads |
| 3 | 3 datasets | PolitiFact only | Only PolitiFact is included in the repository |

### Interpretation choices (the paper is silent or ambiguous)
| # | Topic | Choice |
|---|---|---|
| 4 | Λ(t) | Built from the first T real engagements (the part of the cascade the agent simulates), divided into T/K2 equal time intervals; an empty interval reuses the previous one's users (`data.slot_mode=rank` = equal engagements per step) |
| 5 | Initial state | s0 = news seed + its 3-5 early engagements (generation is conditioned on the early path G, Eq. 1) |
| 6 | Action candidates | Outgoing repost edges of path users in H, co-engagement (U-N-U) neighbours, and users closest to the news content (so a path can start without any real engagement, Sec. III-B) |
| 7 | Graph H | Built from training news only; when a news item is generated its own edges are masked and its real engagers are removed from the content candidates (no self-leakage) |
| 8 | D training | Real and generated prefixes of the same random size, because R_G is used on every prefix G′_t |
| 9 | V_i^p / V_i^n | Cosine similarity of the current (detached) GAT embeddings of the clean graph |
| 10 | Detection input | Train and test use the same kind of real part (early for Table III, full for Table II) plus G′, so snapshot positions mean the same thing |
| 11 | Split | Random 7:1:2, stratified by label; a different split per seed |
| 12 | log π of K2 picks | Sum of log π of the picked actions (without-replacement renormalisation ignored) |
| 13 | Hyper-parameters τ1, τ2, λ1, λ2, K1, K2, T | Peaks of Fig. 4 (exact values not reported) |

### Compute caps (not in the paper)
| Setting | Value |
|---|---|
| Path users expanded per step (`max_sources`) | 64 |
| Repost / co-engagement neighbours per source | 8 / 6 |
| Candidate actions per step (`max_actions`) | 256 (+ 64 content candidates) |
| Pool searched for V_i^p / V_i^n | 4,096 random nodes per step |
| Real users per news in Stage 2 (`max_real_users`) | 500 |

Everything else that is not stated in the paper (epochs, dropout, GAT heads, PGD ε, γ, …) is marked
"not in paper" in the configs.

## 6. Original files

The authors' original components are left untouched for reference
(`src/models/components/hete_reinforcement.py`, `detect_temporal_graph.py`, `src/models/*_module.py`,
`src/data/*politifiact*`, `configs/model/detection_graph.yaml`, `configs/model/hete_reinforce.yaml`, …).
They are no longer used by the entry points.
