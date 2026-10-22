# HANDOFF — Paper-Based: RPPG-Fake reproduction

> อัปเดตล่าสุด: 2026-10-21 (⏸ หาสาเหตุ PolitiFact ค้างอยู่ — ดูหัวข้อ ⏸ ในส่วน Presentation) · เดิม: 2026-10-20 (รอบที่ 8 ครบ · Slide version 18 + หัวข้อวิพากษ์การวิจัย · ถัดไป: หน้า Results / tracker, rerun Table IV หลังแก้ bug ถ้าผู้ใช้ต้องการ)
> Paper ที่ได้รับมอบหมาย: **Early Detection of Multimodal Fake News via Reinforced Propagation Path Generation (RPPG-Fake)** — L. Zhang et al., IEEE TKDE 37(2), 2025 · Code ทางการ: https://github.com/LitianD/RPPG-Fake
> หน้าที่ของเรา: **ตรวจ / ทำให้ Code รันได้ และทดลอง reproduce ผล**

⚠️ อย่าสับสนกับ paper "Deepfake face detection via multi-level DWT + ViT" — เคยทำผิด paper ไปแล้วรอบหนึ่งและลบทิ้งหมดแล้ว ให้ยึด paper ใน README ของโฟลเดอร์นี้เสมอ

---

## 1. ตอนนี้กำลังทำอะไรอยู่

**รอบที่ 4 — "Paper protocol"** ✅ **เสร็จครบ 15/15 runs** (2026-10-14 15:57 → 2026-10-15 ~06:40) · ผลอยู่ในข้อ 3 · รอผู้ใช้เลือกขั้นต่อไป (ข้อ 7)
ตั้งค่าตาม paper / config ตั้งต้น: **5 seeds (0–4)**, T=200, K2=10, 20 time steps, RL 10 epoch, slot_users 200 · รายงาน early (Table III), full (Table II), early_no_gen (ablation) · tag `_paper`, Stage 1 output `data/<ds>/generated_paper/seed<k>/`

| Chain | Dataset | log |
|---|---|---|
| A | `pheme` (ครบ 3 ส่วน, 2,004) → `politifact` (ของผู้เขียน, 483, ไม่มีรูป — แตกใหม่จาก `data/data.zip` ไป `data/politifact/`) | `logs/run_paper_A.log` |
| B | `gossipcop` (ครบ 3 ส่วน, 2,827) — รันขนานกับ A บน GPU เดียว (ใช้ VRAM ~7.3 / 8 GB) | `logs/run_paper_B.log` |

- ตัวรัน: `scripts/run_paper_protocol.py` (เรียก `run_all.py` ต่อ dataset และกันเครื่อง sleep ด้วย `SetThreadExecutionState` ตลอดอายุ process)
- **ถ้า session หลุด / process ตาย:** รันคำสั่งเดิมซ้ำได้เลย — Stage 1 ที่มี `stage1.pt` แล้วจะถูกข้าม (Stage 2 ของ seed นั้นจะรันใหม่)
```bash
cd Paper-Based/Code_RPPG-Fake
nohup .venv/Scripts/python -u scripts/run_paper_protocol.py --datasets pheme politifact > logs/run_paper_A.log 2>&1 &
nohup .venv/Scripts/python -u scripts/run_paper_protocol.py --datasets gossipcop > logs/run_paper_B.log 2>&1 &
```
- **อัปเดต tracker:** `Code_RPPG-Fake/.venv/Scripts/python update_tracker.py` (รันจาก `Paper-Based/`) → เขียนสถานะ/ผลลง `Project_Tracker_RPPG-Fake.html` (บล็อก `/*LIVE*/`) แล้ว publish artifact ซ้ำที่ URL เดิม
- **ผลระหว่างทาง** PHEME seed 0 (2026-10-14 17:46): early 0.871 / F1 0.911 / AUC 0.927 · full 0.866 / 0.908 · no G′ 0.868 / 0.909 → สูงกว่า paper (early 0.829, full 0.844) แต่ G′ ช่วยแค่ +0.2 pp · hit-rate ของ user ที่ generate ใน path จริง ≈ 0.06–0.09% (รอบ 3 ก็ ≈ 0.08% สำหรับ PHEME; GossipCop ≈ 2.3%)
- **2026-10-15 03:45 — RAM เต็ม (16 GB):** PolitiFact Stage 2 (11 GB) + GossipCop Stage 1 (9.6 GB) พร้อมกัน → swap หนัก, RL epoch ช้าลง 3 เท่า, chain A มี `WinError 1450` → **หยุด chain A แล้ว** และตั้งคิวให้ `run_paper_protocol.py --datasets politifact` เริ่มเองหลัง GossipCop เสร็จ (log: `logs/run_paper_A2.log`; Stage 1 seed 0 ของ PolitiFact เก็บไว้แล้ว) · **อย่ารัน PolitiFact ขนานกับ dataset อื่นอีก**
- **PHEME ครบ 5 seeds (✅):** early 0.833 ± 0.027 / F1 0.882 ± 0.021 / AUC 0.907 · full 0.842 ± 0.017 / 0.890 ± 0.013 · no G′ 0.834 ± 0.020 / 0.883 ± 0.015 → **ตรงกับ paper** (early 0.829 / 0.867, full 0.844 / 0.879) แต่ G′ ไม่ช่วย (−0.1 pp) · hit-rate 0.04–0.09%
- ผลสรุป: `logs/results_summary_<dataset>.md` (แถว `*_paper`) — ⚠️ `run_all.py` เขียนไฟล์ summary ทับด้วยแถวของ tag ที่รันล่าสุดเท่านั้น (แถว `_complete` ของรอบ 3 ยังอยู่ใน `logs/results_<ds>_*_complete.csv`)

### รอบที่ 5 — หาสาเหตุที่ G′ ไม่ช่วย (เริ่ม 2026-10-15 12:00, ผู้ใช้สั่ง)

**Diagnostics (ไม่ต้อง train)** — `scripts/diagnose_generated.py` → `logs/diagnose_generated.json` (test news, เฉลี่ย 5 seeds ของรอบ 4)

| | PHEME | GossipCop | PolitiFact |
|---|---|---|---|
| user ที่ generate / ข่าว | 196 | 196 | 196 |
| มาจาก content candidates | 18% | 26% | 29% |
| hit-rate RL vs สุ่ม user จาก train | 0.07% vs 0.02% | 3.46% vs 0.10% | 0.76% vs 0.37% |
| AUC ของ "user credibility" (สัดส่วนข่าว fake ใน train ที่ user เคยแชร์): early / generated / future | 0.73 / 0.72 / 0.68 | **0.98** / 0.75 / 0.99 | 0.51 / **0.74** / 0.51 |

ตีความ: (1) G′ แทบไม่ตรงกับ path จริง (hit-rate ≤ 3.5%) · (2) **G′ ท่วม user จริง**: 3–5 user จริง + ~196 user ที่ generate → บน GossipCop user จริงมีสัญญาณ label แรงมาก (AUC 0.98) แต่ G′ อ่อนกว่า (0.75) จึงเจือจางสัญญาณ → G′ ทำให้แย่ลง −1.7 pp · (3) PolitiFact กลับกัน: G′ มีสัญญาณมากกว่า user จริง (0.74 vs 0.51) เพราะ generator ดึง user จาก train graph (คล้าย kNN) แต่ detector ไม่ได้ใช้ประโยชน์ (ผล −1.4 ± 5 pp) · (4) เพดาน: ต่อให้ G′ = path จริงทั้งหมด (= setting full) ได้แค่ +0.9 / +2.0 / +3.1 pp · (5) ข้างเคียง: บน GossipCop แค่ credibility ของ user ช่วง early ก็ได้ AUC 0.98 ขณะที่ model ได้ 0.84 → user embedding จาก contrastive (ไม่ใช้ label) ไม่เก็บ credibility

**การทดลอง (รันทีละ job เพราะ RAM):**
| # | การทดลอง | ตั้งค่า | tag | log | สถานะ |
|---|---|---|---|---|---|
| A | จำกัด user ที่ generate (20 / 50 คน) — ทดสอบ "เจือจาง" | Stage 2 อย่างเดียว, ใช้ Stage 1 ของรอบ 4, 5 seeds, early | `_g20`, `_g50` | `logs/run_capG.log` | ✅ 14:42 |
| B | Table IV ของ paper: "Random Sample" / "Node Similarity" แทน RL | ไม่ train RL, ใช้ generator.ckpt ของรอบ 4 seed เดียวกัน, seeds 0–2, early + full | `_random`, `_similarity` | `logs/run_ablation.log` | ⚠️ หยุดกลางทาง 15 ต.ค. 18:03 (Windows fork ไม่ได้ / resource หมด) — เสร็จแค่ random ของ PolitiFact (3 seeds) + PHEME (early 3, full 2) · ย้ายไปทำใหม่ในรอบ 6 |
| C | เพิ่ม hit reward (`generator.hit_weight=1`) — ให้ RL เลือก user ที่แชร์จริง | paper settings, seeds 0–2, early | `_hit` | `logs/run_ablation.log` | ❌ ไม่ได้รัน (ผู้ใช้เปลี่ยนเป้าหมายเป็นรอบ 6) |

**ผล A (จำกัด user ที่ generate, early Acc, 5 seeds, ต่างแบบจับคู่ seed):**
| Dataset | ทั้งหมด ~196 | cap 50 | cap 20 | ไม่ใช้ G′ | cap 20 − ทั้งหมด | cap 20 − ไม่ใช้ G′ |
|---|---|---|---|---|---|---|
| GossipCop | 0.769 | 0.771 | 0.776 | 0.786 | +0.7 ± 0.6 (ดีขึ้น 4/5 seeds) | −1.0 ± 1.8 |
| PHEME | 0.833 | 0.836 | 0.826 | 0.834 | −0.7 ± 1.5 | −0.8 ± 2.2 |
| PolitiFact | 0.790 | 0.804 | 0.802 | 0.804 | +1.2 ± 3.7 | −0.2 ± 6.5 |

→ ลด G′ ให้น้อยลงช่วยเล็กน้อยบน GossipCop (ยืนยันว่ามีการเจือจางสัญญาณ) แต่ **ไม่มีจุดไหนดีกว่าไม่ใช้ G′** → ปัญหาหลักคือ G′ เองไม่มีข้อมูลที่ detector ใช้ประโยชน์ได้ ไม่ใช่แค่จำนวนเยอะเกิน

code ที่เพิ่ม (ทั้งหมด default = ปิด, ไม่กระทบรอบ 4): `data.max_gen_users`, `generator.hit_weight`, `generator.mode` (rl | random | similarity), `generator.init_from` · สำรอง summary รอบ 4: `logs/results_summary_<ds>_paper.md` (เพราะ `run_all.py` เขียน summary ทับ)

**ผล B บางส่วน (Random Sample vs RL seed เดียวกัน):** PHEME early Acc **เท่ากันทุก seed** (0.871 / 0.843 / 0.821) แม้ path ต่างกัน (ซ้ำกันแค่ 20/196 user) → บน PHEME detector ตัดสินจาก content เกือบทั้งหมด · PolitiFact early random −6.6 ± 4.3 pp (ทิศเดียวกับ paper), full +1.7 ± 5.9 (noise)

### รอบที่ 6 — Final model สำหรับนำเสนออาจารย์ (เริ่ม 2026-10-17 15:00)
คำสั่งผู้ใช้: train/ปรับให้ใกล้ paper ที่สุด (algorithm + ผลลักษณะเดียวกับ paper), **3 seeds** (ผู้ใช้เลือกแบบเร็ว), ดาวน์โหลด dataset ใหม่ที่ **ครบ 3 ส่วน (ข้อความ + รูป + เส้นทาง)** มาเป็น demo test
หลักที่ใช้: เลือกทุกอย่างด้วย **validation** เท่านั้น ห้าม tune บน test / ห้ามเลือก seed จาก test · ถ้าไม่ตรง paper ให้รายงานตามจริง

1. **Demo data** ✅ — PHEME 9 events (figshare 6392078, `PHEME_veracity.tar.bz2`, 46.5 MB, CC BY 4.0) → `data/external_new/pheme9/` · ใช้เฉพาะ 4 event ที่ไม่อยู่ใน training (ebola-essien, gurlitt, prince-toronto, putinmissing: 623 threads, มีลิงก์รูป 165, โหลดรูปได้ 107) → `scripts/make_pheme_demo.py` → **`data/pheme_demo/` 64 ข่าวครบ 3 ส่วน (fake 47 / real 17)**; engagement จาก user ที่เคยเห็นใน training 21.7% (user ใหม่ → id 0)
2. **Code ใหม่** — `data.text_chunks` / `data.text_chunks` ของ Stage 1 (CLIP อ่านทีละ 50 คำแล้วเฉลี่ย แก้ deviation 77 token; default 1 = เหมือนเดิม) · Stage 1 เก็บ `contrastive.ckpt` + `base_node/base_meta` ใน `stage1.pt` (ต้องใช้ตอน predict) · `test_metrics.json` มี `val/acc_best` · `scripts/predict.py` (inference ข่าวใหม่: early 3–5 → CLIP → G′ จาก generator → detector → P(real); เลือก seed จาก val) — ทดสอบ end-to-end แล้ว
3. **Screening** ✅ (`logs/run_screen.log`, val acc เฉลี่ย seeds 0–2): PolitiFact c8 **0.857** vs c1 0.823 → ใช้ **8** · GossipCop c8 0.793 vs c1 **0.794** (เสมอ) → คง **1** · PHEME tweet สั้น → 1
4. **Final run** ✅ เสร็จ 2026-10-18 01:42 (เริ่ม 2026-10-17 ~15:45) — `logs/run_final.sh` (ห่อด้วย python ที่กันเครื่อง sleep) → `logs/run_final.log` · 3 datasets × seeds 0–2: RL (early, full, early_no_gen; tag `_final`, gen `generated_final`) แล้ว Table IV (random / similarity: early, full; tag `_random_f` / `_similarity_f`) · คาด ~12–13 ชม. · ถ้าหลุด: รัน `bash logs/run_final.sh` ซ้ำ (Stage 1 ที่มี stage1.pt จะข้าม)
5. **Demo** ✅ (PHEME, model เลือกจาก val = **seed 2**, val 0.890 / test 0.821)
   - **PHEME final (3 seeds):** early 0.838 ± 0.016 / F1 0.889 (paper 0.829 / 0.867) · full 0.828 ± 0.010 / 0.880 (paper 0.844 / 0.879) · no G′ 0.839 ± 0.014
   - **Demo A — test split ที่ model ไม่เคยเห็น (402 ข่าว, `data/pheme_test_seed2/`):** Acc 0.821 / F1 0.876 / AUC 0.906 = ตรงกับ test ตอน train → ยืนยันว่า `predict.py` ถูกต้อง · `logs/predict_pheme_test_seed2_early_final_seed2.csv`
   - **Demo B — 4 event ใหม่ (64 ข่าว, `data/pheme_demo/`):** Acc **0.25** / AUC **0.75** — ทายเกือบทุกข่าวเป็น real (p_real 0.55–0.95) · สาเหตุ: user 77% ไม่เคยเห็น (→ id 0), สัดส่วน fake 73% vs train 30%, event ใหม่ (prince-toronto 33 ข่าวเป็น rumour ทั้งหมด) → ปัญหา cross-event generalization ที่รู้กันใน PHEME · **ไม่ได้ปรับ threshold / mapping หลังเห็นผล** (จะเป็นการ tune บน test) · `logs/predict_pheme_demo_early_final_seed2.csv`
   - **Bug ที่แก้:** `val/acc_best` นับ sanity-check (2 batch = 32 ข่าว) → ได้ 0.969 ปลอม · checkpoint ไม่กระทบ (monitor `val/acc`) · แก้ `on_train_start` reset ใน `rppg_detector_module.py` และ `predict.py` อ่าน val จาก `metrics.csv`

**ผลรอบ 6 (final, 3 seeds, mean ± std, Acc / F1)** — ไฟล์ `logs/results_<ds>_<tag>.csv`, tag `*_final`, `*_random_f`, `*_similarity_f`
| Dataset | early (III) | Paper | full (II) | Paper | no G′ (early) | Random early / full | Similarity early / full | Paper Random / Sim (full) |
|---|---|---|---|---|---|---|---|---|
| PolitiFact | 0.865 ± 0.045 / 0.806 | 0.902 / 0.886 | 0.792 ± 0.052 / 0.758 | 0.924 / 0.914 | 0.844 ± 0.031 | 0.840 / 0.847 | 0.844 / 0.830 | 0.891 / 0.902 |
| GossipCop | 0.768 ± 0.031 / 0.781 | 0.873 / 0.913 | 0.793 ± 0.059 / 0.805 | 0.891 / 0.925 | 0.760 ± 0.035 | 0.764 / 0.778 | 0.763 / 0.774 | 0.872 / 0.873 |
| PHEME | 0.838 ± 0.016 / 0.889 | 0.829 / 0.867 | 0.828 ± 0.010 / 0.880 | 0.844 / 0.879 | 0.839 ± 0.014 | 0.834 / 0.846 | 0.823 / 0.828 | 0.824 / 0.826 |

สรุป: PHEME ตรง paper (early +0.9 pp) · PolitiFact early −3.7 pp (ดีขึ้นจากรอบ 4 −11.2 เพราะ text_chunks=8) · GossipCop ~−10 pp · Table IV: **early** → RL ดีที่สุดทุก dataset (เล็ก, อยู่ใน std); **full** → RL ชนะเฉพาะ GossipCop (+1.5 / +1.9 pp ≈ paper +1.9 / +1.8) · full ไม่ได้ดีกว่า early เสมอ (PolitiFact) ต่างจาก paper · hit-rate final: PHEME 0.06%, PolitiFact 0.75%, GossipCop 3.6% (`logs/diagnose_generated_final.json`)
**Model ที่ใช้นำเสนอ (เลือก seed จาก val):** PolitiFact seed 0 (val 0.898 / test 0.896) · GossipCop seed 1 (0.813 / 0.779) · PHEME seed 2 (0.890 / 0.821) — checkpoint ดูในหน้า Results แท็บภาพรวม
6. **หน้า Results สำหรับนำเสนอ** ✅ `Paper-Based/RPPG-Fake_Results.html` → https://claude.ai/artifact/ToP6dbVMKvXVWhbhAj68x7 (5 แท็บ: ภาพรวม, Table II/III, Table IV, Demo, ข้อจำกัด) · ข้อมูลฝังใน `/*DATA*/…/*END DATA*/` สร้างจาก CSV/JSON ใน logs · tracker อัปเดตเป็นเวอร์ชัน 7


### รอบที่ 7 — ปรับ model หลังวิเคราะห์สาเหตุ (เริ่ม 2026-10-18, ผู้ใช้สั่ง "ลองปรับ Model แล้วทดสอบใหม่")
สาเหตุที่วิเคราะห์ (ตอบผู้ใช้แล้ว): (1) dataset ต่าง — GossipCop เราสมดุล 50/50 vs paper real 65% (majority baseline 0.513 vs 0.650; lift เหนือ majority: เรา +25.5 vs paper +22.3 pp), PolitiFact ไม่มีรูป + test 96 ข่าว; (2) รายละเอียดที่ paper ไม่เปิดเผย (code รันไม่ได้, encoder, hyper-param); (3) G′ ตรงคนแชร์จริงน้อย + กลบสัญญาณ user จริง; (4) 3 seeds, std ±1.6–5.9 pp
`logs/run_tune.sh` → `logs/run_tune.log` (ห่อ keep-awake, ทีละ job, seeds 0–2, เลือกด้วย val — `scripts/compare_tags.py`):
- A: จำกัด G′ 20 / 50 users (Stage 2 อย่างเดียว ใช้ Stage 1 ของรอบ 6) · tag `early_final_g20/_g50`
- C: GossipCop สัดส่วนตาม paper (real 65%) — `scripts/make_ratio_subset.py` → `data/gossipcop_r65/` (2,229 ข่าว: real 1,449 / fake 780) · tag `*_final`
- B: hit reward (`generator.hit_weight=1`) Stage 1 ใหม่ · gen `generated_hit_f` · tag `early_hit_f/full_hit_f`
ถ้าหลุด: `bash logs/run_tune.sh` ซ้ำได้ (Stage 1 ที่มีแล้วข้าม)
**ผู้ใช้ขอให้เร็วขึ้น (2026-10-18 11:10) → เลือก "A + C แบบเร็ว":** ตัด B (hit reward) และ C แบบ train ใหม่ · chain จะถูกหยุดอัตโนมัติหลัง "PART A DONE" (`scratchpad/stop_tune.ps1`) · `data/gossipcop_r65/` สร้างไว้แล้วแต่ไม่ได้ train
**C แบบเร็ว (re-weight test ของ model รอบ 6 ให้เป็น real 65% เท่า paper):** early Acc 0.768 → **0.778 ± 0.017** (F1 0.781 → 0.825) · full 0.793 → 0.804 (F1 0.846) · no G′ 0.760 → 0.764 · model ทาย real/fake ถูกพอ ๆ กัน (TPR ≈ TNR ≈ 0.78) → **สัดส่วน class อธิบายช่องว่าง GossipCop ได้แค่ ~1 pp** (ที่เคยบอกผู้ใช้ว่า "ส่วนใหญ่มาจากสัดส่วน" เกินจริง — ต้องแก้) · ข้อจำกัด: model ยัง train บน 50/50 (การ train บน 65% อาจขยับ threshold ได้อีกเล็กน้อย)
**ผล A (2026-10-18 11:41, val เลือก):** PHEME — เดิม val 0.847 / test 0.838 ดีสุด (g20 0.842/0.837, g50 0.810/0.822) · PolitiFact — เดิม 0.871/0.865 ดีสุด (g20 0.857/0.819, g50 0.857/0.833) · GossipCop — g20 0.795/0.766, g50 0.796/0.768 ≈ เดิม 0.796/0.768; val สูงสุดคือ no G′ (0.812) แต่ test 0.760 และไม่ตรง paper → **คง model รอบ 6** · ผลรอบ 5 (cap 20 +0.7 pp GossipCop) ไม่ซ้ำกับ model ชุดใหม่ · chain หยุดแล้ว ไม่มี process ค้าง
**สรุปรอบ 7:** ไม่มีการปรับที่ดีขึ้นตาม val → model นำเสนอ = รอบ 6 · ช่องว่างที่เหลือน่าจะมาจากข้อมูลเนื้อหา (ดึงเอง/ครึ่งเดียว) + implementation ที่ paper ไม่เปิดเผย

### รอบที่ 8 — แก้ bug ร้ายแรงแล้ว train ใหม่ (เริ่ม 2026-10-19 18:48)
**Bug ที่พบ (ตอนทำสไลด์ตัวอย่างจริง):** `user_emb` ที่ส่งให้ Module 2/3 — user ทุกคนใน H ได้ vector **เดียวกันทุกตัวเลข** (unique rows = 1 ทั้ง 3 dataset) เพราะ base_node มี common component ใหญ่ (norm ~49) → tanh ใน Eq. 7 ตัน · และ node view ของ Module 1 ยุบเหลือ ~1 มิติ (effective rank 1.1) เพราะเลือก V^p/V^n จาก embedding ของตัวเอง (self-reinforcing) → **ผลรอบ 1–7 ทั้งหมด model มองไม่เห็นตัวตนของ user** (อธิบาย G′ ไม่ช่วย, random == RL บน PHEME, detector ใช้แค่ content)
**Audit เพิ่ม (subagent, read-only):** พบเพิ่ม (1) train/test shift: early user ของข่าว train อยู่ใน H 100% แต่ test 37% (user ที่ไม่อยู่ใน H ได้ embedding สุ่ม) (2) discriminator เทียบ prefix คนละขนาด (real ~12 vs fake ~105 users) (3) standardize ใช้ stat ของ user ทั้งหมด (70% isolated) (5) meta-view batch ส่วนใหญ่เป็น node isolated · ไม่พบ data leak / index error
**แก้แล้ว (ทุกตัวมี config flag, ค่า default ใหม่ = เปิด; รอบเก่าใช้ค่าเดิมได้):**
- `contrastive.pos_neg: structure` — V^p/V^n จาก cosine ของ neighbourhood sketch (A+I)R ของ H (คงที่) + sample centres/pool/meta-batch จาก node ที่อยู่ใน H
- `generator.standardize: true` — z-score base_node/base_meta ด้วย stat ของ user ใน H ก่อน Eq. 7 · user ที่ไม่อยู่ใน H → UNK vector เดียว (mean ของ known) · content candidates center ด้วย mean ของ known · D เทียบ prefix ขนาดเท่ากัน (≤ จำนวน user จริง) · `stage1.pt` เก็บ `known`
- `data.unk_users: true` — Stage 2: user ของข่าว train ที่แชร์แค่ข่าวนั้น → UNK feature (leave-one-out ให้ train เหมือน test)
- `predict.py` ใช้ known / UNK เหมือน train (+ แก้ชื่อตัวแปรชน)
- ผลทดสอบ PHEME seed 2 (Module 1 อย่างเดียว): base_node eff. rank 1.1 → 9.0 · user_emb unique 1 → 14,550 / 14,583 · eff. rank 30.3 · top-5 neighbours เคยแชร์ข่าวเดียวกัน 0.00 → 0.97
**Train ใหม่:** `logs/run_fix.sh` → `logs/run_fix.log` · ค่าตาม paper ยกเว้น **RL 5 epoch** (paper ไม่ระบุ; ผู้ใช้เลือกเพื่อประหยัดเวลา) · PHEME → GossipCop → PolitiFact · seeds 0–2 · early / full / early_no_gen · tag `_fix`, gen `generated_fix` · คาด ~5.5–6 ชม. · ถ้าหลุด: `bash logs/run_fix.sh` ซ้ำ
**ผลระหว่างทาง:** PHEME (3 seeds) early 0.837 ± 0.012 · full **0.844 ± 0.021 (= paper)** · no G′ 0.844 ± 0.028 · RL loss ยังลดที่ epoch 5 (ยังไม่อิ่ม) · GossipCop seed 0: early 0.791 → **0.933**, full 0.788 → **0.973** (สูงกว่า paper 0.873 / 0.891), no G′ 0.933
**ตรวจ data leakage (2026-10-19 21:30, GossipCop s0 + PHEME s2):** ✅ H ไม่มีเส้น/ node จากข่าว val/test · ✅ early = 3–5 แรกตามเวลา · ✅ test sample มีแค่ early + generated · ⚠️ ข้อความซ้ำ train↔test 5% / 4%, ภาพซ้ำ 10% / 17% (ลักษณะ dataset) · กฎ "label ส่วนใหญ่ของข่าว train ที่ early user เคยแชร์" ได้ 0.922 (GossipCop) / 0.688 (PHEME) → ผล 0.93 มาจากสัญญาณ user ที่มีจริงใน dataset ไม่ใช่ code รั่ว · **control สลับ label ของ train** (`data.shuffle_train_labels`) ตั้งคิวรันต่อหลัง FIX ALL DONE: `logs/run_controls.sh` → `logs/run_controls.log` (tag `early_fix_shuffled`; ถ้าไม่มี leakage test acc ต้อง ≈ 0.5)
**Demo set สำหรับนำเสนอ (ผู้ใช้ขอ 5–10 record):** `scripts/make_demo_set.py` เลือกจาก **test split ของ seed ที่ val ดีสุด** (model ไม่เคยเห็น) · มีรูป + early ≥ 3 + ข้อความสั้น · PHEME 6 (3 fake / 3 real) + GossipCop 4 (2 / 2) → `data/demo_present/<ds>/` (CSV + images + `demo_items.json`) แล้วรัน predict · ตั้งคิวไว้ `logs/run_demo_set.sh` → `logs/run_demo_set.log` (รอ CONTROLS DONE) · ผลจะอยู่ `logs/predict_<ds>_early_fix_seed<k>.csv`
**✅ Train รอบ 8 เสร็จ 2026-10-20 01:11 (3 datasets × 3 seeds, Acc test mean ± std):**
| Dataset | early | full | no G′ | paper early / full |
|---|---|---|---|---|
| PHEME | 0.837 ± 0.012 | 0.844 ± 0.021 | 0.844 ± 0.028 | 0.829 / 0.844 |
| GossipCop | 0.878 ± 0.087 | 0.908 ± 0.077 | 0.900 ± 0.044 | 0.873 / 0.891 |
| PolitiFact | 0.826 ± 0.043 | 0.868 ± 0.006 | 0.868 ± 0.022 | 0.902 / 0.924 |
(GossipCop seed 1 ต่ำ: early 0.777 vs seed 0/2 ≈ 0.93 · G′ ยังไม่ช่วย: no G′ ≥ early ทุก dataset)
**⏸ หยุดตามคำสั่งผู้ใช้ 2026-10-20 01:12** (ผู้ใช้จะปิดเครื่อง) — ยังไม่ได้ทำ: leakage control (สลับ label) + demo set 10 record · **รันต่อด้วย:** `cd Paper-Based/Code_RPPG-Fake && nohup .venv/Scripts/python -c "import ctypes,subprocess; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); subprocess.run(['bash','logs/run_resume.sh'])" > logs/run_resume.log 2>&1 &` (~20 นาที) · หลังจากนั้น: อัปเดต Slide / หน้า Results / tracker ด้วยผลรอบ 8 + สไลด์ตัวอย่างจริง (ผู้ใช้ขอ "step-by-step real example")
**✅ รันต่อเสร็จ 2026-10-20 12:44 (`logs/run_resume.log`):**
- **Leakage control (สลับ label ของ train, tag `early_fix_shuffled`):** GossipCop acc 0.506 (chance 0.51) · PHEME 0.701 (= ทายคลาสส่วนใหญ่ 0.701) · PolitiFact 0.604 (= คลาสส่วนใหญ่) → **ไม่มี data leakage** · AUC ห่าง 0.5 ได้ทั้งสองทาง (GossipCop 3 การสุ่ม: 0.345 / 0.385 / 0.527, tag `_k1`, `_k2`) เพราะ input มีข้อมูลเกี่ยวกับ label อยู่แล้ว model ที่เรียนจาก label สุ่มจึงให้ทิศคะแนนสุ่ม · เพิ่ม `data.shuffle_seed`
- **Demo set 10 record** (`data/demo_present/{pheme,gossipcop}/`, model seed 2 ทั้งคู่ เลือกจาก val): PHEME ถูก 5/6 (ผิด: rumour “Hostages seen fleeing from cafe… #SydneySiege” P(real) 0.74), GossipCop ถูก 4/4 · ผล `logs/predict_{pheme,gossipcop}_early_fix_seed2.csv` · แก้ `predict.py` ให้นับ early_known จาก `known` จริง (PHEME early user ที่ model รู้จัก 32%, GossipCop 92%)
- **Detector ใช้เส้นทางหรือไม่ (2026-10-20, วิเคราะห์หลังเลือก model):** แยก logit ชั้นสุดท้าย σ(W·[h*; n_T; n_V] + b) ทุก run (`Slides_RPPG-Fake/figures/m3_seeds.py` → `m3_seeds.json`) · ถ้าแทนส่วนเส้นทางด้วยค่าเฉลี่ย: **PHEME 0/6 run เปลี่ยน** (h* ยุบ, std ส่วนเส้นทาง ≤ 0.025 → ผลมาจาก CLIP ข้อความ+ภาพ), **GossipCop 5/6 run acc ลด 6–26 pp** (seed 1 early ไม่ใช้เส้นทาง = seed ที่แย่สุด 0.777), PolitiFact 1/6 (full seed 2) · ตัวอย่างข่าว 115: P(real) ไม่เปลี่ยนเมื่อให้ G′ 0–20 step · ไอเดียต่อ: ทำไม h* ยุบใน PHEME (LayerNorm/attention กระจุก, w_graph เล็ก)
**ยังไม่แก้ (บันทึกไว้):** สเกล content vs user ใน Module 1 (#4), T=200 สำหรับ PHEME, actor/critic loss scale, early user ซ้ำ, detector GAT ทิศเดียว

### Final presentation (2026-10-22)
- `MS_DPMLA_Final Presentation.pdf` = สไลด์ที่ผู้ใช้ทำเอง (42 หน้า, อัปเดต 22 ต.ค. 14:31: เพิ่มหน้า 8 "Data · Model training") · `Supawit_MS_DPMLA_Final Presentation.pdf` = เวอร์ชันก่อนหน้า 41 หน้า (ผู้ใช้เพิ่มเอง)
- `MS_DPMLA_Final Presentation_Script.pdf` = script ภาษาไทยรายหน้า (ตรงกับฉบับ 42 หน้า) · เวลารวม ~33 นาที · มี Q&A 6 ข้อ + จุดที่ควรแก้บนสไลด์ · สร้างจาก scratchpad `final_pres/script_data.py` + `build_script_pdf.py` (Chrome headless)
- Demo notebook: code ที่ใช้รันจริงแนบท้ายแต่ละส่วน (หัวข้อ 🔧) + ส่วน 8 คำสั่งรัน (`Demo_RPPG-Fake/tools/source_cells.py`, `add_source_cells.py`)

### ⏸ หาสาเหตุ PolitiFact ไม่ตรง paper (2026-10-21 · หยุดตามคำสั่งผู้ใช้ · ทำต่อพรุ่งนี้)
**พบแล้ว:**
1. **Paper วัด PolitiFact บนชุดเดียว 92 ข่าว ไม่ได้เฉลี่ย 5 รอบ:** Acc ทุกค่าใน Table II/III/IV + Fig. 4 = k/92 (0.880…0.924 = 81…85/92), Recall = k/41 → confusion matrix ลงตัว (Table II: TP 37 FP 3 TN 48; Table III: TP 35 FP 3 TN 48) · code ทางการ (`.git_upstream`: `configs/data/rl_pol.yaml` split [338, 48, 92], `src/data/rl_politifiact_datamodule.py` shuffle `random_state=665` แล้วตัด 478, `data_test`=48 / `data_val`=92 สลับกัน, checkpoint monitor `val/acc`) → ชุด 92 ข่าว = ชุดที่ใช้เลือก checkpoint = ชุดที่รายงาน (น่าจะ) · ไฟล์ข้อมูลของผู้เขียนตอน train (`phase_3/politifact.csv`, 478 แถว) ≠ `data.zip` ใน repo (484 แถว) → สร้าง split เดียวกันไม่ได้
2. **Content-only baseline:** logistic regression บน CLIP text (77 token) = **0.892** (seed 0–2: 0.927 / 0.896 / 0.854) > RPPG-Fake ของเรา 0.826 · ใช้ text_chunks=8 (ที่เราเลือกจาก val 48 ข่าว) LR ได้แค่ 0.858 → การเลือก text_chunks=8 ทำให้แย่ลง
3. **Selection บนชุดเดียวกับที่รายงาน** (`scripts/diagnose_protocol.py`, train Stage 2 ใหม่ 30 epoch, G′ เดิม): early ours 0.826 → same-set **0.878** (+5.2 pp) · full ours 0.882 → same-set 0.896 (+1.4) · ผล → `Code_RPPG-Fake/logs/diagnose_politifact_{early,full}_fix.json`
4. ไม่ใช่สาเหตุ: กราฟ H (early user อยู่ใน H 66% > PHEME 37%), ขนาด/สัดส่วนข้อมูล (484: fake 297 / real 187 vs paper 478: 294 / 184) · ข้อความของเรา mean 1,149 คำ (paper 1,546)
**ยังไม่ได้ทำ (ต่อพรุ่งนี้):**
- GossipCop early ปกติ vs ตัดภาพ (`--zero-image`) เพื่อประเมินผลของการไม่มีภาพ: `cd Code_RPPG-Fake && .venv/Scripts/python -u scripts/diagnose_protocol.py --dataset gossipcop --tag early_fix` และ `... --zero-image`
- PolitiFact detector ด้วยข้อความ 77 token: `... --dataset politifact --tag early_fix --text-chunks 1` (และ full_fix)
- สรุปลง slide (ผู้ใช้ขอ): หน้าใหม่ "ทำไม PolitiFact ไม่ตรง paper" แยกส่วนที่อธิบายได้จากแต่ละสาเหตุ

### หน้า Results / Tracker เป็นรอบ 8 (2026-10-21)
- **Results** (`RPPG-Fake_Results.html` → https://claude.ai/artifact/ToP6dbVMKvXVWhbhAj68x7, version 2): Table II/III ใช้ tag `*_fix` (รอบ 8) + แถว "ไม่ใช้ G′" · model ที่นำเสนอ = PolitiFact seed 1 / GossipCop seed 2 / PHEME seed 2 · panel ใหม่ "Detector ใช้เส้นทางหรือไม่" (จาก `Slides_RPPG-Fake/figures/m3_seeds.json`) · แท็บ Ablation ยังเป็นรอบ 6 (ติดป้าย) + การ์ดรอบ 8 ใช้ G′ vs ไม่ใช้ · Demo เพิ่ม C = 10 ข่าว · ข้อจำกัด / ข้อสังเกตเขียนใหม่ · ข้อมูลฝังใน `/*DATA*/` (script: scratchpad `build_results_data.py`)
- **Tracker** (`Project_Tracker_RPPG-Fake.html` → https://claude.ai/artifact/PaoukVWPyHnvmy9JCxfEfq, version 8): แท็บผลการรันเพิ่ม Round 8 (ค่าเริ่มต้น) · checklist เพิ่มงานรอบ 7-8 / slide / demo / push (31/32) · timeline ถึง 10-21 · hyper-param และคำสั่งรันเป็นของรอบ 8 · ประเด็นเปิดเขียนใหม่ · บล็อก `/*LIVE*/` ยังเป็นรอบ 4 (`update_tracker.py` อ่าน tag `*_paper`)

### Demo notebooks (2026-10-20)
`Paper-Based/Demo_RPPG-Fake/` — model + data พร้อมรัน · อัปเดต 2026-10-21: ทุกส่วนแสดงข่าว REAL / FAKE คู่กัน + ส่วนที่ 6 ข้อมูลจากแหล่งอื่น (PHEME 4 เหตุการณ์ใหม่ 64 ข่าว, `tools/export_external.py`, ได้ acc 0.25 ตรง log) + ส่วนที่ 7 สรุปจำนวนข้อมูล · zip 222 MB (ไม่ต้อง train / GPU) · `RPPG-Fake_Demo_Local.ipynb` (รันแล้ว บันทึก output 9 ภาพ) + `RPPG-Fake_Demo_Colab.ipynb` (cell แรก mount Drive → unzip `Demo_RPPG-Fake.zip` 214 MB → pip `torch_geometric`) · package `rppg_demo/` (คัดลอก Module 2-3 + make_sample/Collate, ไม่ต้องใช้ Hydra/Lightning; `bundle.py` = load/generate/predict/evaluate, `viz.py` = ภาพ, label อังกฤษเพื่อให้ Colab แสดงได้) · data จาก `tools/export_bundle.py` (PHEME seed 2, GossipCop seed 2; ตรวจแล้วได้ test acc ตรง log 0.8234 / 0.9239) · `data/` และ `*.zip` อยู่ใน `.gitignore` ของโฟลเดอร์ · ติดตั้ง matplotlib, scikit-learn, ipykernel, nbclient เพิ่มใน `Code_RPPG-Fake/.venv` แล้ว (ผู้ใช้อนุญาต) · **ยังไม่ได้ทดสอบบน Colab จริง** (ทดสอบ notebook จาก zip ที่แตกใหม่ในเครื่องแล้ว)

### Presentation (2026-10-18)
**อัปเดต 2026-10-20 (version 18): 52 สไลด์ · 7 หัวข้อ** — เพิ่มหัวข้อ **06 การวิพากษ์การวิจัย** (ผู้ใช้ขอ): `k-paper` (ทำซ้ำไม่ได้: code ทางการรันไม่ได้ / ค่าพารามิเตอร์ไม่ระบุ / ข้อมูลไม่ครบ), `k-method` (ไม่มีแถวไม่ใช้ G′ ใน Table IV, ต้องรู้จัก user ล่วงหน้า, benchmark ให้รางวัลการจำ user, ส่วนต่าง ablation < std), `k-ours` (Limitation ของเรา: เครื่อง 8 GB / RAM 16 GB, ข้อมูลไม่เหมือน paper, Table IV ยังไม่รันใหม่หลังแก้ bug, bug ของเราเอง) · สรุปผลเลื่อนเป็น 07 · อัปเดตผลรอบ 8 ทุกสไลด์: `e-rounds`, `e-bug` (ใหม่), `e-model`, `e-early`, `e-full`, `e-leak` (ใหม่), `e-output` (Demo A 0.823 / 0.879 / 0.906), `e-demo10` (ใหม่, 9/10), `e-newevents`, `e-why`, `c-summary` · ตัวอย่างจริงทีละ module: `w-m1`, `w-m2`, `w-m3` · `a-ours` / `e-tune` ติดป้ายว่าเป็นผลก่อนแก้ bug (รอบ 6–7)
**อัปเดต 2026-10-18 (version 12): 35 สไลด์ จัดใหม่ทุกหัวข้อเป็น "จาก Paper" (ป้ายเทา) → "สิ่งที่เราทำ" (ป้ายส้ม) · เพิ่ม bg-contrib, p-data, p-setup, p-results, p-figs (ช่องเส้นประให้วาง Fig. 3–5 จาก PDF), e-why (สาเหตุไม่ตรง paper), e-tune (รอบ 7), c-paper · footer ของสไลด์ paper บอก Section / Fig / หน้า PDF ที่ควรใช้ประกอบ** · Slide deck แก้ไข / comment ได้: **https://claude.ai/artifact/FctQ2dy4bDc1nihBzfR4ap** ("RPPG-Fake Paper Presentation", 27 สไลด์, ภาษาไทย, เน้นภาพ) · 6 หัวข้อตามที่ผู้ใช้กำหนด (ที่มา, นิยามปัญหา, ระเบียบวิธี, การทดลอง (เน้น 11 สไลด์), ablation, สรุป) · ทุกหัวข้อมีสไลด์ป้ายส้ม "สิ่งที่เราทำ" จากผลใน HANDOFF · speaker notes ทุกสไลด์ · ไฟล์ต้นฉบับใน scratchpad (`rppg_deck/project/`) — แก้ต่อผ่าน Artifact `read` + publish ไปที่ url เดิม · ชื่อผู้นำเสนอใช้ "Supawit Chaingam" (จาก git) · ใช้คำว่า Fake News (ผู้ใช้พิมพ์ "Deepfake" ในคำขอ แต่ paper เป็น fake news)

## 2. ทำอะไรไปแล้วบ้าง (ตามลำดับเวลา)

1. **Lectures** — เปลี่ยนชื่อไฟล์ Lecture 06–12 ตาม convention และ push ขึ้น GitHub แล้ว (commit `458dd9e`)
2. **Clone code ทางการ** → `Paper-Based/Code_RPPG-Fake/` (มี `.git` ของต้นทาง, ทำงานบน branch `runnable`)
3. **Interactive HTML summary** → `Paper-Based/Paper_Summary_RPPG-Fake.html` (ภาษาอังกฤษ, 8 แท็บ, Edit + Comment + Save HTML ได้) — **ยังไม่ push** รอผู้ใช้ตรวจ
4. **ตรวจ code ต้นฉบับ** → รันไม่ได้ (config/ไฟล์ source หาย, ตัวแปรไม่ได้ประกาศ, val/test สลับกัน, reward ไม่ตรง paper ฯลฯ)
5. **เขียน code ใหม่ให้รันได้ตาม paper** (Module 1 contrastive, Module 2 RL generator, Module 3 temporal detector) — รายละเอียดใน `Code_RPPG-Fake/RUNNING.md`
6. **Audit 2 รอบ** (ตัวเอง + subagent เทียบกับสมการ Eq. 1–22) → แก้ 9 จุดให้ตรงตามตัวอักษร (เช่น GAT ตาม Eq. 2, Λ(t), R_G ตาม Eq. 13, D train บน prefix, ปิดการรั่วผ่าน content candidates)
7. **Environment**: `.venv` (Python 3.11, PyTorch 2.11 + CUDA 12.8, RTX 5050 8 GB) สร้างด้วย `uv`
8. **รอบที่ 1 (text-only) seed 0** — PolitiFact (data ของผู้เขียน), PHEME, GossipCop (UPFD + หัวข้อข่าว)
9. **หา dataset ให้ครบองค์ประกอบ** → เติมรูป PHEME + เนื้อข่าว/รูป GossipCop & PolitiFact (UPFD) จากเว็บต้นฉบับ/Wayback
10. **รอบที่ 2 (multimodal) seed 0** — เสร็จ (2026-10-10)
11. **คัดเฉพาะข่าวครบ 3 ส่วน + รอบที่ 3 seed 0** — เสร็จ (2026-10-12)
12. **Dashboard** `Project_Tracker_RPPG-Fake.html` (2026-10-14)
13. **รอบที่ 4 Paper protocol (5 seeds, T=200, RL 10 epoch)** — เริ่ม 2026-10-14 (ดูข้อ 1)

---

## 3. ผลที่ได้แล้ว (seed 0, positive class = real)

### รอบที่ 1 — text-only
| Dataset | early Acc / F1 | full Acc / F1 | early_no_gen Acc / F1 | Paper early (Table III) | Paper full (Table II) |
|---|---|---|---|---|---|
| PolitiFact (ของผู้เขียน, T=200, RL 10 ep) | 0.844 / 0.769 | 0.865 / 0.806 | 0.833 / 0.750 | 0.902 / 0.886 | 0.924 / 0.914 |
| PHEME (T=100, RL 3 ep) | 0.835 / 0.874 | 0.835 / 0.874 | 0.838 / 0.875 | 0.829 / 0.867 | 0.844 / 0.879 |
| GossipCop (หัวข้อข่าว, T=100, RL 3 ep) | 0.762 / 0.767 | 0.782 / 0.794 | 0.765 / 0.776 | 0.873 / 0.913 | 0.891 / 0.925 |

### รอบที่ 2 — multimodal
| Dataset | early Acc / F1 | full Acc / F1 | early_no_gen Acc / F1 |
|---|---|---|---|
| PolitiFact-UPFD หัวข้อข่าวอย่างเดียว (baseline) | 0.855 / 0.852 | 0.855 / 0.870 | 0.839 / 0.839 |
| PolitiFact-UPFD เนื้อข่าว + รูป | 0.645 / 0.732 ⚠️ | 0.839 / 0.848 | 0.871 / 0.886 |
| PHEME + รูป | 0.833 / 0.870 | 0.829 / 0.869 | 0.829 / 0.869 |
| GossipCop เนื้อข่าว + รูป | 0.779 / 0.778 | 0.802 / 0.817 | 0.783 / 0.792 |

### รอบที่ 3 — เฉพาะข่าวที่ครบ 3 ส่วน (2026-10-12, tag `_complete`, `data/<ds>/generated/`)
| Dataset (train/val/test) | early Acc / F1 / AUC | full Acc / F1 / AUC | early_no_gen Acc / F1 |
|---|---|---|---|
| GossipCop 2,827 (1,979/283/565) | 0.788 / 0.791 / 0.854 | 0.793 / 0.799 / 0.856 | 0.782 / 0.798 |
| PHEME 2,004 (1,402/200/402) | 0.848 / 0.894 / 0.911 | 0.866 / 0.907 / 0.928 | 0.838 / 0.887 |
| PolitiFact-UPFD 79 (55/8/16) | 0.750 / 0.600 | 0.813 / 0.824 | 1.000 / 1.000 (16 ข่าว — ใช้ไม่ได้) |

ข้อสังเกต: PHEME สูงกว่า paper (+1.9 / +2.2 pp) แต่ test set ต่างกัน (เฉพาะ tweet ที่มีรูป) · G′ ช่วย +0.6 (GossipCop) / +1.0 pp (PHEME) — ยังต่ำกว่าที่ paper อ้าง และเป็น seed เดียว

### รอบที่ 4 — Paper protocol: 5 seeds, T=200, K2=10, 20 time steps, RL 10 epoch (2026-10-15, tag `_paper`)
mean ± std ของ 5 seeds · positive class = real

| Dataset | early Acc / F1 / AUC | full Acc / F1 / AUC | early no G′ Acc / F1 | Paper early / full (Acc / F1) | G′ (early − no G′, Acc) |
|---|---|---|---|---|---|
| PHEME (ครบ 3 ส่วน, 2,004) | 0.833 ± 0.027 / 0.882 ± 0.021 / 0.907 | 0.842 ± 0.017 / 0.890 ± 0.013 / 0.911 | 0.834 ± 0.020 / 0.883 ± 0.015 | 0.829 / 0.867 · 0.844 / 0.879 | −0.1 pp |
| GossipCop (ครบ 3 ส่วน, 2,827) | 0.769 ± 0.024 / 0.776 ± 0.018 / 0.843 | 0.789 ± 0.015 / 0.797 ± 0.018 / 0.863 | 0.786 ± 0.026 / 0.796 ± 0.020 | 0.873 / 0.913 · 0.891 / 0.925 | −1.7 pp |
| PolitiFact (ผู้เขียน, 483, ไม่มีรูป) | 0.790 ± 0.043 / 0.734 ± 0.034 / 0.883 | 0.821 ± 0.043 / 0.754 ± 0.054 / 0.892 | 0.804 ± 0.052 / 0.749 ± 0.063 | 0.902 / 0.886 · 0.924 / 0.914 | −1.4 pp |

สรุปรอบ 4: **PHEME reproduce ตัวเลขหลักได้** (ต่างจาก paper ≤ 0.4 pp Acc, อยู่ใน std) · GossipCop ต่ำกว่า ~10 pp, PolitiFact ~11 pp · **G′ ไม่ช่วยในทุก dataset** — user ที่ generate ตรงกับ path จริงน้อยมาก (hit-rate PHEME 0.04–0.09%, PolitiFact ~0.7%, GossipCop ~3.5%) · seed 0 ดูดีกว่าค่าเฉลี่ยเสมอ (PHEME 0.871, PolitiFact รอบ 1 0.844) → ผล seed เดียวของรอบ 1–3 มองในแง่ดีเกินไป
⚠️ PolitiFact Stage 1 seed 0 รันตอน RAM เต็ม (ช้า 3 เท่า) แต่ผลไม่ได้เพี้ยน — seed 1–4 รันเดี่ยว


⚠️ PolitiFact-UPFD early_mm ล่ม (ทายเกือบทุกข่าวเป็น real) — ชุดนี้เล็กมาก (train 220 / val 32 / test 62), 1 ข่าว ≈ 1.6 pp ต้องดูหลาย seed ก่อนสรุป

**ข้อสังเกตสำคัญ (รอบที่ 1):** path ที่ generate ช่วยน้อยมากหรือไม่ช่วย (ablation ต่างกัน −0.3 ถึง +1.1 pp)  · PHEME ใกล้ paper ที่สุด

---

> ⚠️ **แก้ความเข้าใจผิด (2026-10-15):** ตัวเลข "+2.7 / +4.0 / +4.1 %" ใน paper (Sec. IV-B) คือ **RPPG-Fake ดีกว่า baseline ที่ดีที่สุดที่ early stage** ไม่ใช่ "ผลของ G′" อย่างที่เคยเขียนไว้ใน HANDOFF / tracker / Paper Summary (แท็บ Reproduction บรรทัด ~710, 735, 801 ยังผิดอยู่ — ต้องแก้ตอนอัปเดตหน้า) · ablation ของ G′ ใน paper คือ **Table IV**: เปลี่ยน RL เป็น "Random Sample" (สุ่ม 10 user ต่อ step) → ลด 3.3 / 1.9 / 2.0 pp และ "Node Similarity" (top-10 ตาม semantic similarity) → ลด 2.2 / 1.8 / 1.8 pp (PolitiFact / GossipCop / PHEME, ตัวเลขชุดเดียวกับ Table II = full) · paper **ไม่มี** การทดลองแบบไม่ใช้ G′ เลย → "early_no_gen" ของเราเป็นการทดลองเพิ่มเอง

## 4. Dataset ที่ใช้งานอยู่ตอนนี้ — เฉพาะข่าวที่ครบ 3 ส่วน (ข้อความ + รูป + เส้นทางข่าว)

ตั้งแต่ 2026-10-10 (`scripts/keep_complete.py`): `data/` เหลือเฉพาะข่าวที่มีครบ ส่วนที่เหลือถูก **ย้าย** (ไม่ได้ลบ) ไป `data/_archive/`

| โฟลเดอร์ | ข่าว | fake / real | เนื้อหา |
|---|---|---|---|
| `data/gossipcop/` | 2,827 | 1,378 / 1,449 | `news_actions_gossipcop.csv` + `images/` (หัวข้อ + เนื้อข่าว, retweet tree) |
| `data/pheme/` | 2,004 | 599 / 1,405 | `news_actions_pheme.csv` + `images/` (tweet, reply thread) |
| `data/politifact_upfd/` | 79 | 43 / 36 | น้อยมาก — ใช้ทดสอบเท่านั้น |

- config ตั้ง default `dataset: gossipcop` และ `image_dir: data/${dataset}/images` แล้ว
- ต้องรัน Stage 1 + Stage 2 ใหม่ทั้งหมดสำหรับ subset นี้ (processed / generated ของชุดเต็มอยู่ใน `_archive/*_full/`)
- `data/_archive/` (ชุดเต็มเดิม, PolitiFact ของผู้เขียน, ไฟล์ดิบ `external`) **ถูกย้ายไป Windows Recycle Bin แล้ว** ตามคำสั่งผู้ใช้ (2026-10-10) — กู้คืนได้จาก Recycle Bin จนกว่าผู้ใช้จะ Empty; ถ้าหายแล้วต้องโหลดใหม่ด้วย `scripts/convert_*.py` / `fetch_*.py` (`data.zip` ของ repo ยังอยู่)

### (เดิม) Dataset ก่อนคัด — ตอนนี้อยู่ใน `data/_archive/`, path ด้านล่างคือที่เดิม

| โฟลเดอร์ใน `Code_RPPG-Fake/data/` | ที่มา | ข้อความ | รูป | เส้นทางข่าว | จำนวน |
|---|---|---|---|---|---|
| `politifact/` | `data.zip` ของผู้เขียน | เนื้อข่าวเต็ม | ❌ | ✅ | 483 |
| `pheme/` | figshare 4010619 (CC BY 4.0) | tweet | ✅ 2,086 (36%) ใน `external/pheme/images/` | ✅ reply thread | 5,802 |
| `gossipcop/` | UPFD + FakeNewsNet + ดึงเนื้อข่าวเอง | หัวข้อ + เนื้อข่าว 67% | ✅ 2,832 (52%) ใน `external/fnn_content/gossipcop/images/` | ✅ retweet tree | 5,464 |
| `politifact_upfd/` | UPFD + FakeNewsNet + ดึงเนื้อข่าวเอง | หัวข้อ + เนื้อข่าว 55% | ⚠️ 79 (25%) | ✅ | 314 |
| `politifact_upfd_title/` | เหมือนข้างบนแต่หัวข้ออย่างเดียว | หัวข้อ | ❌ | ✅ | 314 |
| `external/` | ไฟล์ดิบที่โหลดมา (PHEME, UPFD + mapping .pkl, FakeNewsNet CSV, fnn_content) | | | | |

หมายเหตุ: `gossipcop/processed_titleonly/` = processed ของรอบที่ 1 (หัวข้อข่าว) · `*/generated/` = path ของรอบที่ 1 · `*/generated_mm/` = path ของรอบที่ 2
ไฟล์ `.pkl` ของ UPFD ต้องเปิดด้วย restricted unpickler เท่านั้น (มีใน `scripts/convert_upfd.py`)

---

## 5. โครงสร้าง code ที่สำคัญ (`Code_RPPG-Fake/`)

| ไฟล์ | หน้าที่ |
|---|---|
| `RUNNING.md` | วิธีติดตั้ง/รัน + ตาราง mapping paper ↔ code + รายการ deviation ทั้งหมด |
| `src/data/components/politifact.py` | โหลด `news_actions_*.csv`, split 7:1:2, graph H (train only), meta-path RWR, Λ(t), early 3–5 |
| `src/data/components/encode_content.py` | CLIP ViT-B/32 text + image → cache `content_clip_vitb32[_img].pt` |
| `src/models/components/rppg/{layers,contrastive,generator,detector}.py` | Module 1–3 |
| `src/train_hete_RL.py` + `configs/train_Hete_RL.yaml` | Stage 1 (contrastive + RL + generate paths) |
| `src/train.py` + `configs/train_detection_graph.yaml` | Stage 2 (detection, Lightning) |
| `scripts/run_all.py` | รัน Stage 1 + Stage 2 (early / full / early_no_gen) แล้วสรุป mean ± std |
| `scripts/convert_pheme.py`, `convert_upfd.py` | แปลง PHEME / UPFD+FakeNewsNet เป็น format เดียวกับ PolitiFact |
| `scripts/fetch_pheme_images.py`, `fetch_fnn_articles.py` | ดึงรูป PHEME / เนื้อข่าว+รูป FakeNewsNet (resumable, rate-limited) |

ไฟล์ต้นฉบับของผู้เขียนยังอยู่ (ไม่ได้ใช้แล้ว) เพื่อเทียบ — ดู `git diff` บน branch `runnable`

---

## 6. Deviation จาก paper ที่ต้องบอกเสมอเวลารายงาน

- ไม่มี publisher (PNU → ใช้ UU แทน) · CLIP อ่านข้อความได้ 77 token · รูปมีแค่บางส่วน (ไม่มีรูป = ศูนย์)
- PHEME ใช้ 5,802 ข่าว (paper 3,506) · GossipCop/PolitiFact-UPFD ถูกคัดให้ 50/50 (paper ไม่สมดุล)
- PHEME/GossipCop ใช้ T=100, RL 3 epoch (paper ไม่ระบุ epoch; PolitiFact ใช้ T=200, 10 epoch) เพื่อประหยัดเวลา
- ยังรันแค่ seed 0 (paper เฉลี่ย 5 รอบ)

---

## 7. สิ่งที่ยังค้าง / ไอเดียถัดไป

- [x] รอบ multimodal เสร็จ → สรุปเทียบรอบก่อน + paper แล้ว (GossipCop ดีขึ้น +1.7/+2.0 pp; PHEME เท่าเดิม; path ที่ generate ยังไม่ช่วย)
- [x] รอบที่ 3: รัน subset ครบ 3 ส่วน (seed 0) แล้ว — ดูข้อ 3
- [x] รอบที่ 4: 5 seeds ตาม paper บน PHEME + GossipCop (ครบ 3 ส่วน) + PolitiFact ของผู้เขียน — เสร็จ 2026-10-15 (ข้อ 3) · tracker เวอร์ชัน 5 publish แล้ว
- [ ] เติมผลรอบ 4 ในแท็บ Reproduction ของ `Paper_Summary_RPPG-Fake.html`
- [ ] (ไอเดีย) หาสาเหตุที่ G′ ไม่ช่วย: hit-rate ต่ำมาก → ลอง reward / candidate set แบบอื่น หรือดู Fig. 5 ของ paper
- [x] เพิ่มแท็บ **Reproduction** (แท็บ 8) ใน `Paper_Summary_RPPG-Fake.html`: dataset ที่ใช้ + coverage, ผลรอบ 1/2 เทียบ paper, ablation G′, ตัวอย่าง path จริง vs generate (anonymised) — ข้อมูลฝังเป็น `<script id="repro-data">` สร้างจาก script ใน scratchpad (`extract_repro.py`, `insert_rep.py`)
- [x] เพิ่มแท็บ **Progress** (แท็บ 9, 2026-10-14): สถานะปัจจุบัน, 14 ขั้นตอน (คลิกดูรายละเอียด), timeline, ผลทุกรอบ, next steps, open issues · แท็บ Reproduction เพิ่มผลรอบ 3 แล้ว — แก้รายการขั้นตอนได้ที่ `const STEPS=[...]` ใน HTML
- [x] Dashboard ติดตามงาน `Paper-Based/Project_Tracker_RPPG-Fake.html` (2026-10-14, ภาษาไทย, ข้อความน้อย): checklist 19 งานติ๊กได้ + progress bar แยกเฟส, Dataset, ผลรอบ 1–3 (กราฟ + ตาราง), pipeline, ข้อจำกัด · artifact https://claude.ai/artifact/PaoukVWPyHnvmy9JCxfEfq · แก้งานได้ที่ `const PHASES=[...]`, ผลที่ `const ROUNDS=[...]`
- [ ] อัปเดตแท็บ Code ของ HTML ให้สะท้อน code ที่แก้แล้ว (ตอนนี้ยังอธิบาย code ต้นฉบับของผู้เขียน)
- [x] Push ขึ้น GitHub แล้ว (2026-10-20, ผู้ใช้สั่ง): HTML, HANDOFF, code, `Slides_RPPG-Fake/` (สำเนา source ของ deck) · `.git` ของต้นทางใน `Code_RPPG-Fake` ถูกเปลี่ยนชื่อเป็น `.git_upstream/` (ignore ไว้; เปลี่ยนกลับเป็น `.git` ได้ถ้าต้องการ diff กับต้นทาง) · ไม่ push `data/`, `.venv/`, `logs/`, `*.log`

---

## 8. ข้อควรระวังในการทำงาน

- **ห้าม commit/push จนกว่าผู้ใช้สั่ง** · commit ต้องลงท้าย `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- การโหลด dataset / ติดตั้ง package / ดึงข้อมูลจากเว็บ ต้องขออนุญาตก่อน (ระบุแหล่ง + ขนาด)
- รันยาวต้องกัน sleep (ครั้งแรก process ถูก kill ระหว่าง sleep)
- Stage 2 ของ dataset ใหญ่ช้าที่ CPU (collate สร้าง snapshot graph) ไม่ใช่ GPU
- เขียน Python ผ่าน bash heredoc บน Windows ระวัง `\n`/`±` เพี้ยน — ใช้ Edit tool สำหรับแก้ string ที่มี escape
