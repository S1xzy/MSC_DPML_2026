# Demo_RPPG-Fake: model + data พร้อมรัน สำหรับนำเสนอ

ชุด demo ของการ reproduce **RPPG-Fake** (Zhang et al., IEEE TKDE 37(2), 2025) ใช้ model ที่เรา train แล้ว (รอบ 8, seed ที่ validation ดีที่สุด) กับข้อมูลจริง
แล้วแสดงภาพทีละ module ของ Algorithm ใน Jupyter notebook ไม่ต้อง train ใหม่ และไม่ต้องใช้ GPU · ทุกขั้นแสดง **ข่าว REAL และ FAKE คู่กัน**

| Notebook | ใช้ที่ไหน |
|---|---|
| `RPPG-Fake_Demo_Local.ipynb` | คอมของเรา (บันทึกผลลัพธ์และภาพไว้แล้ว เปิดดูได้ทันทีโดยไม่ต้องรัน) |
| `RPPG-Fake_Demo_Colab.ipynb` | Google Colab (cell แรก mount Drive แตก zip และติดตั้ง `torch_geometric`) · บันทึกผลที่รันบน Colab แล้ว · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S1xzy/MSC_DPML_2026/blob/main/Paper-Based/Demo_RPPG-Fake/RPPG-Fake_Demo_Colab.ipynb) |

ทั้งสองไฟล์มีเนื้อหาเหมือนกัน ต่างกันแค่ cell setup

## เนื้อหาใน notebook

| ส่วน | สิ่งที่แสดง |
|---|---|
| 1. Input | ข่าว REAL / FAKE (ข้อความ + ภาพ) + early path G (คนแชร์ 3-5 คนแรก) · ตารางข่าว demo พร้อม P(real) |
| 2. Module 1 | กราฟ H รอบ user ของแต่ละข่าว · user embedding (t-SNE) · เพื่อนบ้านใกล้สุด · user ที่ใกล้เนื้อหาข่าว |
| 3. Module 2 | สร้าง G′ สด ๆ ให้ทั้งสองข่าว · เส้นทางจริง vs G′ · action แต่ละ step · ตารางเทียบ REAL / FAKE · hit rate แยก label |
| 4. Module 3 | snapshot · self / cross attention · แยก logit (เส้นทาง / ข้อความ / ภาพ) · P(real) เมื่อให้ G′ 0-20 step |
| 5. ผล | Accuracy / F1 / AUC บน test · ทายข่าว real / fake ถูกกี่ % · confusion matrix |
| 6. ข้อมูลจากแหล่งอื่น | PHEME 4 เหตุการณ์ที่ไม่อยู่ใน training (64 ข่าว: real 17 / fake 47) ผ่านทุก module · เทียบกับ test (เฉพาะ PHEME) |
| 7. สรุป | ตารางว่าแต่ละส่วนใช้ข้อมูลกี่ข่าว (real / fake) และผลของแต่ละชุด |

เปลี่ยน dataset ได้ที่ cell `DATASET = 'pheme'` → `'gossipcop'`

## โครงสร้างโฟลเดอร์

```
Demo_RPPG-Fake/
├ RPPG-Fake_Demo_Local.ipynb, RPPG-Fake_Demo_Colab.ipynb
├ rppg_demo/            code ที่รันได้ (ไม่ต้องใช้ Hydra / Lightning)
│  ├ layers.py, generator.py, detector.py   Module 2-3 (คัดลอกจาก Code_RPPG-Fake/src/models/components/rppg/)
│  ├ data.py            สร้าง snapshot ให้ detector (make_sample, Collate จาก rppg_datamodule.py)
│  ├ bundle.py          โหลด model + data และรันแต่ละ module (generate, predict, evaluate, ...)
│  └ viz.py             ฟังก์ชันวาดภาพ
├ data/<dataset>/       ข้อมูลและ weights ที่ export แล้ว (ไม่อยู่ใน git)
│  ├ graph.pt           กราฟ H, ข้อความ, label, split, engagement จริง, early path
│  ├ content.pt         CLIP feature ของทุกข่าว n = [n_T ; n_V] (1024 มิติ)
│  ├ model.pt           Module 1 embedding, content candidates, G′ ที่ Stage 1 สร้าง, weights ของ generator + detector
│  ├ settings.json      ค่าที่ใช้ train + ผลที่ log ไว้
│  ├ demo.csv           ข่าว demo (PHEME 6, GossipCop 4)
│  ├ images/            ภาพของข่าว test
│  ├ external.pt        (PHEME) ข่าวจากเหตุการณ์ใหม่ 64 ข่าว: CLIP feature + early path + G′ ที่คำนวณไว้
│  └ external_images/   ภาพของข่าวชุดนี้
├ tools/export_bundle.py   สร้าง data/ จาก model ใน Code_RPPG-Fake (ตรวจว่าได้ test acc ตรงกับ log)
├ tools/export_external.py เพิ่มข่าวจากเหตุการณ์ใหม่ (ทำแบบเดียวกับ scripts/predict.py ตรวจว่า acc ตรงกับ log)
├ tools/make_notebooks.py  สร้าง notebook ทั้งสองไฟล์
└ Demo_RPPG-Fake.zip    ทั้งโฟลเดอร์ (รวม data) สำหรับอัปโหลดไป Colab (ไม่อยู่ใน git เพราะมี data · สร้างจากเครื่องที่มี data/)
```

| Dataset | seed | ข่าว | test | test acc (เรา) | paper (Table III, early) |
|---|---|---|---|---|---|
| PHEME | 2 | 2,004 | 402 | 0.823 | 0.829 |
| GossipCop | 2 | 2,827 | 565 | 0.924 | 0.873 |
| PHEME เหตุการณ์ใหม่ (figshare 6392078) | 2 | 64 | 64 | 0.250 (AUC 0.75) | — |

## รันในคอมของเรา

ใช้ environment เดิมของ `Code_RPPG-Fake` (มี torch + torch_geometric และติดตั้ง matplotlib, scikit-learn, ipykernel เพิ่มแล้ว)

**VS Code:** เปิด `RPPG-Fake_Demo_Local.ipynb` → Select Kernel → เลือก `Code_RPPG-Fake\.venv\Scripts\python.exe` → Run All

**Jupyter ในเบราว์เซอร์** (ต้องติดตั้ง jupyter ก่อนครั้งเดียว):

```bash
../Code_RPPG-Fake/.venv/Scripts/python -m pip install jupyter
```

```bash
../Code_RPPG-Fake/.venv/Scripts/python -m jupyter notebook RPPG-Fake_Demo_Local.ipynb
```

เครื่องอื่น: `pip install -r requirements.txt` แล้วเปิด notebook จากโฟลเดอร์นี้ · ใช้ CPU ได้ ทั้ง notebook ใช้เวลา ~30-60 วินาที

## รันใน Google Colab

1. อัปโหลด `Demo_RPPG-Fake.zip` (~222 MB) ไปที่ Google Drive → `MyDrive/Demo_RPPG-Fake.zip`
2. กดปุ่ม **Open In Colab** ด้านบน (เปิด notebook จาก GitHub) หรือ File → Upload notebook → เลือก `RPPG-Fake_Demo_Colab.ipynb`
3. Runtime → Run all → อนุญาตให้เข้าถึง Drive เมื่อถูกถาม

ถ้าเก็บ zip ไว้ที่อื่น แก้ `ZIP_PATH` ใน cell แรก · ไม่อยากใช้ Drive ตั้ง `USE_DRIVE = False` แล้วเลือกไฟล์ zip ตอนรัน

## สร้าง data ใหม่ (ถ้า train model ใหม่)

```bash
cd ../Code_RPPG-Fake && .venv/Scripts/python ../Demo_RPPG-Fake/tools/export_bundle.py --datasets pheme gossipcop
```

```bash
cd ../Code_RPPG-Fake && .venv/Scripts/python ../Demo_RPPG-Fake/tools/export_external.py --dataset pheme --input data/pheme_demo
```

## ข้อควรรู้

- ทุกข่าว demo อยู่ใน test split ของ seed ที่ใช้ model ไม่เคยเห็นตอน train
- G′ ที่สร้างสดในส่วนที่ 3 ต่างจาก G′ ที่ใช้วัดผลเล็กน้อย เพราะ policy สุ่มตาม π (เปลี่ยน `seed` ได้)
- PHEME: detector แทบไม่ใช้ส่วนของเส้นทาง ผลมาจากข้อความ + ภาพ (ส่วนที่ 4 แสดงให้เห็น) · GossipCop ใช้เส้นทาง
- model ต้องรู้จัก user ล่วงหน้า (user เป็น node ใน H) ข่าวจากเหตุการณ์ใหม่ที่มีแต่ user ใหม่ทายได้ไม่ดี
- ไม่ได้ใส่ CLIP encoder มาด้วย จึงใช้ได้กับข่าวที่อยู่ในข้อมูลแล้วเท่านั้น (ข่าวใหม่ทั้งหมดใช้ `Code_RPPG-Fake/scripts/predict.py`)
- รายละเอียดการทดลองทั้งหมด: `../HANDOFF.md` · วิธี train: `../Code_RPPG-Fake/RUNNING.md`
