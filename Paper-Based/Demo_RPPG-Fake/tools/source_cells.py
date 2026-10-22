"""Attach the exact code used to train and run the model to the demo notebooks, inside the section it belongs to.

Each file of Code_RPPG-Fake is embedded in a `%%writefile code_used/<path>` cell at the end of the matching section
(e.g. contrastive.py in "2. Module 1"): the code is visible in Colab / GitHub, and running the cell writes the file to
code_used/. Section 8 keeps only the run commands and a map of where each file is. All these cells have ids "src..",
so insert_source() can remove and re-insert them without touching the other cells or their outputs.
"""
import os

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Code_RPPG-Fake"))

# (heading prefix of the notebook section, title, [(file, what it does)])
GROUPS = [
    ("## เลือก dataset", "ตั้งค่าที่ใช้ train (Hydra configs)", [
        ("configs/train_Hete_RL.yaml", "Stage 1: Module 1 + 2 (τ1, τ2, λ1, K1, T, K2, λ2, epoch, การแก้รอบ 8)"),
        ("configs/train_detection_graph.yaml", "Stage 2: Module 3"),
        ("configs/data/rppg_politifact.yaml", "data ของ Stage 2 (early / full, real_slots, text_chunks)"),
        ("configs/model/rppg_detector.yaml", "detector: hidden 128, heads 4, AdamW lr 5e-3"),
    ]),
    ("## 1. Input", "เตรียมข้อมูล input", [
        ("src/data/components/politifact.py", "อ่านข่าว + engagement · split 7:1:2 · กราฟ H (train เท่านั้น) · meta-path RWR · Λ(t) · early 3-5"),
        ("src/data/components/encode_content.py", "CLIP ViT-B/32: n = [n_T ; n_V]"),
    ]),
    ("## 2. Module 1", "Heterogeneous Graph Contrastive Learning", [
        ("src/models/components/rppg/layers.py", "GAT ตาม Eq. 2 และ attention pooling (ใช้ใน Eq. 9)"),
        ("src/models/components/rppg/contrastive.py", "node view + PGD (Eq. 2-3), meta-path view (Eq. 4), L_CL (Eq. 5)"),
    ]),
    ("## 3. Module 2", "Reinforced Propagation Path Generation", [
        ("src/models/components/rppg/generator.py", "MDP (Eq. 6), policy (Eq. 7-11), reward (Eq. 12-14), actor-critic (Eq. 15-19)"),
        ("src/train_hete_RL.py", "Stage 1 = Algorithm 1 บรรทัด 1-14: train Module 1 แล้ว Module 2 แล้วสร้าง G′ ทุกข่าว"),
    ]),
    ("## 4. Module 3", "Temporal-aware Fake News Detection", [
        ("src/data/rppg_datamodule.py", "ตัดเส้นทาง (early จริง + G′) เป็น snapshot ให้ detector"),
        ("src/models/components/rppg/detector.py", "dynamic GAT + position (Eq. 20), fusion (Eq. 21), P(real) (Eq. 22)"),
        ("src/models/rppg_detector_module.py", "loss BCE (Eq. 22) + metric (positive class = real)"),
        ("src/train.py", "Stage 2 = Algorithm 1 บรรทัด 15-17: train detector, เลือก checkpoint ด้วย val acc, วัด test"),
    ]),
    ("## 5. ผลบน test set", "รันทุก seed และสรุปผล", [
        ("scripts/run_all.py", "รัน Stage 1 + 2 ทุก seed / setting แล้วสรุป mean ± std"),
        ("logs/run_fix.sh", "คำสั่งที่ใช้จริงในรอบ 8 (ผลที่นำเสนอ)"),
    ]),
    ("## 6. ข้อมูลจากแหล่งอื่น", "ทำนายข่าวที่ model ไม่เคยเห็น", [
        ("scripts/predict.py", "สิ่งที่ใช้สร้างผลส่วนนี้ (และ Demo A / B / C)"),
    ]),
]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def group_cells(title, files, first=False):
    cells = [md(f"#### 🔧 Code ที่ใช้รันจริง · {title}",
                "",
                "ไฟล์จริงจาก `Paper-Based/Code_RPPG-Fake/` ที่ใช้ train model รอบ 8 · cell ด้านล่างทำหน้าที่แสดง code "
                "(รัน cell = เขียนไฟล์ลง `code_used/` ไม่กระทบ demo ด้านบน ซึ่งใช้ `rppg_demo/` ที่คัดลอกจาก code ชุดนี้)")]
    if first:
        dirs = sorted({os.path.dirname(f"code_used/{p}") for _, _, fs in GROUPS for p, _ in fs})
        cells.append(code("# สร้างโฟลเดอร์สำหรับ cell %%writefile ทั้ง notebook (%%writefile ไม่สร้างโฟลเดอร์ให้)\nimport os\n"
                          f"for d in {dirs!r}:\n    os.makedirs(d, exist_ok=True)\nprint('code_used/ ready')"))
    for path, desc in files:
        body = open(os.path.join(CODE, path), encoding="utf-8").read().rstrip("\n")
        cells.append(md(f"**`{path}`** · {desc}"))
        cells.append(code(f"%%writefile code_used/{path}\n{body}\n"))
    return cells


def commands_cells():
    return [md(
        "## 8. Code ที่ใช้รันจริง: คำสั่งและตำแหน่งไฟล์",
        "",
        "code ของแต่ละ module แนบไว้ท้ายส่วนนั้น (หัวข้อ 🔧) · ทั้งหมดมาจาก `Paper-Based/Code_RPPG-Fake/` (GitHub: `S1xzy/MSC_DPML_2026`)",
        "",
        "| ส่วนใน notebook | ไฟล์ที่แนบ |",
        "|---|---|",
        *[f"| {h.lstrip('# ')} · {t} | " + " · ".join(f"`{p}`" for p, _ in fs) + " |" for h, t, fs in GROUPS],
        "",
        "**ลำดับการรันจริง (ต่อ 1 dataset, 1 seed):**",
        "```bash",
        "cd Paper-Based/Code_RPPG-Fake",
        "# Stage 1: Module 1 + 2 -> data/<ds>/generated_fix/seed<k>/stage1.pt (user embedding + G′ ของทุกข่าว)",
        ".venv/Scripts/python src/train_hete_RL.py seed=0 dataset=pheme output_dir=data/pheme/generated_fix/seed0 data.text_chunks=1 generator.epochs=5",
        "# Stage 2: Module 3 (Table III = early, Table II = full, ablation = ไม่ใช้ G′)",
        ".venv/Scripts/python src/train.py seed=0 dataset=pheme +tag=early_fix data.generated_dir=data/pheme/generated_fix data.train_real=early data.eval_real=early data.text_chunks=1",
        ".venv/Scripts/python src/train.py seed=0 dataset=pheme +tag=full_fix data.generated_dir=data/pheme/generated_fix data.train_real=full data.eval_real=full data.text_chunks=1",
        ".venv/Scripts/python src/train.py seed=0 dataset=pheme +tag=early_no_gen_fix data.generated_dir=data/pheme/generated_fix data.train_real=early data.eval_real=early data.use_generated=false data.text_chunks=1",
        "```",
        "ทั้งหมดนี้คือสิ่งที่ `logs/run_fix.sh` สั่งผ่าน `scripts/run_all.py` (3 dataset × seed 0-2 · PolitiFact ใช้ `text_chunks=8`)",
        "",
        "ข้อจำกัด: การ train ต้องใช้ข้อมูลเต็มใน `Code_RPPG-Fake/data/` ซึ่งไม่อยู่ใน git (สร้างใหม่ได้ด้วย `scripts/convert_*.py`, `fetch_*.py`) ·",
        "บนเครื่องเรา Stage 1 ใช้ ~20-60 นาที / seed และ Stage 2 ~3-5 นาที / setting (RTX 5050 8 GB) · ขั้นตอนติดตั้ง: `Code_RPPG-Fake/RUNNING.md`",
    )]


def insert_source(cells):
    """Remove old source cells, then put each group at the end of its section and section 8 at the end."""
    cells = [c for c in cells if not str(c.get("id", "")).startswith("src")]
    heads = [i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith("## ")]
    out, k = [], 0
    pending = {}
    for gi, (prefix, title, files) in enumerate(GROUPS):
        start = next(i for i in heads if _src(cells[i]).startswith(prefix))
        end = next((i for i in heads if i > start), len(cells))  # first cell of the next section
        pending.setdefault(end, []).extend(group_cells(title, files, first=gi == 0))
    for i, c in enumerate(cells):
        if i in pending:
            out += pending.pop(i)
        out.append(c)
    out += pending.pop(len(cells), [])
    out += commands_cells()
    for c in out:
        if not c.get("id"):
            c["id"] = f"src{k:02d}"
            k += 1
    return out
