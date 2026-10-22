"""Write RPPG-Fake_Demo_Local.ipynb and RPPG-Fake_Demo_Colab.ipynb (same cells, different setup cell).

    python tools/make_notebooks.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from source_cells import insert_source  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "\n".join(lines)}


TITLE = md(
    "# RPPG-Fake Demo: ตรวจจับข่าวปลอมช่วงแรกด้วยการสร้างเส้นทางการแพร่",
    "",
    "Notebook นี้ใช้ model ที่เรา train แล้ว (รอบ 8, seed ที่ validation ดีที่สุด) กับข้อมูลจริง แล้วแสดงภาพทีละ module ของ Algorithm",
    "**ทุกขั้นแสดงข่าวจริง (REAL) และข่าวปลอม (FAKE) คู่กัน** เพื่อเทียบว่า model เห็นอะไรต่างกัน",
    "",
    "| ส่วน | สิ่งที่แสดง |",
    "|---|---|",
    "| 1. Input | ข่าว REAL / FAKE (ข้อความ + ภาพ) และ early path G (คนแชร์ 3-5 คนแรก) |",
    "| 2. Module 1 · Heterogeneous Graph Contrastive Learning | กราฟ H รอบ user ของแต่ละข่าว · user embedding (t-SNE) · user ที่ใกล้เนื้อหาข่าว |",
    "| 3. Module 2 · Reinforced Propagation Path Generation | สร้าง G′ สด ๆ ให้ทั้งสองข่าว · เทียบเส้นทางจริง · action ที่เลือกแต่ละ step |",
    "| 4. Module 3 · Temporal-aware Detection | snapshot · self / cross attention · แยก logit ว่ามาจากเส้นทาง ข้อความ หรือภาพ |",
    "| 5. ผลบน test set | Accuracy / F1 / AUC แยกข่าว real และ fake |",
    "| 6. ข้อมูลจากแหล่งอื่น | PHEME 4 เหตุการณ์ที่ model ไม่เคยเห็น (figshare) · REAL / FAKE คู่กัน |",
    "| 7. สรุป | notebook นี้ใช้ข้อมูลกี่ข่าวในแต่ละส่วน |",
    "| 8. Code ที่ใช้รันจริง | คำสั่งรัน + ตำแหน่งไฟล์ · code ของแต่ละ module แนบท้ายส่วนนั้น (หัวข้อ 🔧) |",
    "",
    "Paper: L. Zhang et al., *Early Detection of Multimodal Fake News via Reinforced Propagation Path Generation*, IEEE TKDE 37(2), 2025",
)

SETUP_LOCAL = [
    md("## 0. Setup (คอมของเรา)",
       "",
       "ต้องมี Python ที่ติดตั้ง `torch`, `torch_geometric`, `matplotlib`, `scikit-learn`, `pandas` แล้ว (ดู `requirements.txt` / `README.md`).",
       "เปิด notebook นี้จากโฟลเดอร์ `Demo_RPPG-Fake` เพื่อให้ import `rppg_demo` ได้"),
    code("import os, sys",
         "ROOT = os.path.abspath('.')",
         "assert os.path.isdir(os.path.join(ROOT, 'rppg_demo')), 'เปิด notebook จากโฟลเดอร์ Demo_RPPG-Fake'",
         "sys.path.insert(0, ROOT)"),
]

COLAB_BADGE = md('<a href="https://colab.research.google.com/github/S1xzy/MSC_DPML_2026/blob/main/Paper-Based/Demo_RPPG-Fake/RPPG-Fake_Demo_Colab.ipynb" '
                 'target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')

SETUP_COLAB = [
    md("## 0. Setup (Google Colab)",
       "",
       "1. อัปโหลดไฟล์ `Demo_RPPG-Fake.zip` ไปที่ Google Drive (เช่น `MyDrive/Demo_RPPG-Fake.zip`)",
       "2. รัน cell ด้านล่าง: mount Drive → แตกไฟล์ → ติดตั้ง `torch_geometric` (Colab มี torch, matplotlib, scikit-learn อยู่แล้ว)",
       "",
       "ถ้าไม่อยากใช้ Drive ให้ตั้ง `USE_DRIVE = False` แล้วเลือกไฟล์ zip จากเครื่องตอนรัน (อัปโหลดช้ากว่า)",
       "",
       "Runtime แบบ CPU ก็พอ (ทุกขั้นใช้เวลาไม่กี่วินาที ยกเว้น t-SNE ~1 นาที)"),
    code("USE_DRIVE = True",
         "ZIP_PATH = '/content/drive/MyDrive/Demo_RPPG-Fake.zip'   # แก้ path ถ้าเก็บไว้ที่อื่น",
         "",
         "import os, sys, subprocess",
         "if USE_DRIVE:",
         "    from google.colab import drive",
         "    drive.mount('/content/drive')",
         "else:",
         "    from google.colab import files",
         "    up = files.upload()",
         "    ZIP_PATH = '/content/' + next(iter(up))",
         "if not os.path.isdir('/content/Demo_RPPG-Fake'):",
         "    subprocess.run(['unzip', '-q', '-o', ZIP_PATH, '-d', '/content'], check=True)",
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric'], check=True)",
         "ROOT = '/content/Demo_RPPG-Fake'",
         "os.chdir(ROOT)",
         "sys.path.insert(0, ROOT)",
         "print('ready:', sorted(os.listdir(ROOT)))"),
]

BODY = [
    md("## เลือก dataset และโหลด model",
       "",
       "`DATASET = 'pheme'` (Twitter, ข่าวเหตุการณ์) หรือ `'gossipcop'` (ข่าวบันเทิง) · เปลี่ยนแล้ว `Run all` ใหม่ได้เลย",
       "(ส่วนที่ 6 ข้อมูลจากแหล่งอื่นมีเฉพาะ PHEME)"),
    code("DATASET = 'pheme'   # 'pheme' หรือ 'gossipcop'",
         "",
         "import json, numpy as np, pandas as pd, torch",
         "from IPython.display import display",
         "from rppg_demo.bundle import Bundle",
         "from rppg_demo import viz",
         "pd.set_option('display.max_colwidth', 80)",
         "",
         "B = Bundle(os.path.join(ROOT, 'data', DATASET), 'cpu')",
         "s = B.settings",
         "print(f\"dataset: {DATASET} · seed {s['seed']} · news {B.N:,} · users {B.U:,} (in graph H: {int(B.known.sum()):,})\")",
         "print(f\"logged: val acc {s['metrics']['val_acc']:.3f} · test acc {s['metrics']['test/acc']:.3f} · paper (Table III, early) {s['paper'].get('early')}\")",
         "split_counts = pd.DataFrame([{'split': k, 'news': len(v), 'real': int((B.labels[v] == 1).sum()), 'fake': int((B.labels[v] == 0).sum())}",
         "                             for k, v in B.data['split'].items()])",
         "split_counts"),

    md("## 1. Input: ข่าว REAL และ FAKE + early path G",
       "",
       "ตอน test model ได้ **ข้อความ + ภาพของข่าว** และ **คนแชร์ 3-5 คนแรก** (early stage, Sec. IV-B) เท่านั้น",
       "user ที่อยู่ในกราฟ H (มาจากข่าว train) มี embedding ของตัวเอง ส่วนคนใหม่ใช้เวกเตอร์ UNK ร่วมกัน",
       "",
       "ตาราง: ข่าว demo ที่เลือกไว้ (ทั้งหมดอยู่ใน test split · model ไม่เคยเห็นตอน train)"),
    code("demo = pd.DataFrame(B.demo)",
         "demo['P(real)'] = [round(B.predict(i), 3) for i in demo['index']]",
         "demo['pred'] = np.where(demo['P(real)'] > 0.5, 'real', 'fake')",
         "demo['correct'] = demo['pred'] == demo['label']",
         "demo.sort_values('label', ascending=False)[['index', 'label', 'pred', 'P(real)', 'correct', 'text']]"),
    code("# คู่ข่าวที่ใช้ตลอด notebook: REAL 1 ข่าว + FAKE 1 ข่าว (จาก test split)",
         "if DATASET == 'pheme':",
         "    REAL = B.find('Love is Stronger than Hate #CharlieHebdo cover. We will not')   # news #115 = ตัวอย่างในสไลด์",
         "    FAKE = B.find('Two female hostages have run out')",
         "else:",
         "    REAL = next(d['index'] for d in B.demo if d['label'] == 'real')",
         "    FAKE = next(d['index'] for d in B.demo if d['label'] == 'fake')",
         "PAIR = [REAL, FAKE]",
         "viz.show_pair(B, PAIR, [B.predict(k) for k in PAIR])"),

    md("## 2. Module 1 · Heterogeneous Graph Contrastive Learning",
       "",
       "กราฟ H สร้างจากข่าว **train เท่านั้น**: node = user และข่าว · edge = U-N (user แชร์ข่าว) และ U-U (repost ต่อจากกัน)",
       "Module 1 เรียน embedding ของ user จาก 2 มุมมอง (node view + meta-path view) แบบ contrastive โดยไม่ใช้ label",
       "",
       "**2.1** กราฟ H รอบ user คนแรก (ที่อยู่ใน H) ของข่าว REAL และ FAKE · สีของข่าว = label ของข่าว train ที่ user เคยแชร์"),
    code("viz.plot_ego_pair(B, PAIR)"),
    md("**2.2** user embedding ทั้งหมด (ลดเหลือ 2 มิติด้วย t-SNE) · สี = สัดส่วนข่าว fake ที่ user เคยแชร์ใน train",
       "ดาวเขียว = user ของข่าว REAL · ดาวแดง = user ของข่าว FAKE (พร้อมเพื่อนบ้านใกล้สุด 5 คน) · ใช้เวลา ~1 นาที"),
    code("U_REAL, U_FAKE = B.first_known_user(REAL), B.first_known_user(FAKE)",
         "viz.plot_user_tsne(B, n=4000, highlight=[(U_REAL, 'REAL news', viz.GREEN), (U_FAKE, 'FAKE news', viz.RED)])"),
    code("# เพื่อนบ้านใกล้สุดของ user ทั้งสองคน (cosine ของ user vector ที่ Module 2 / 3 ใช้)",
         "share, tot = B.fake_share()",
         "rows = []",
         "for lab, u in [('REAL news', U_REAL), ('FAKE news', U_FAKE)]:",
         "    if u is None:",
         "        continue",
         "    rows.append({'news': lab, 'user': u, 'role': 'early user', 'cosine': 1.0, 'train news': int(tot[u]), 'fake share': round(share[u], 2)})",
         "    rows += [{'news': lab, 'user': v, 'role': 'neighbour', 'cosine': round(c, 3), 'train news': int(tot[v]), 'fake share': round(share[v], 2)}",
         "             for v, c in B.neighbours(u, 5)]",
         "pd.DataFrame(rows)"),
    md("**2.3** user ที่ embedding ใกล้ **เนื้อหาข่าว** ที่สุด: เป็น action แบบ content ให้ Module 2 เริ่มสร้างเส้นทางได้แม้ยังไม่มีคนแชร์"),
    code("viz.plot_content_candidates_pair(B, PAIR, k=12)"),

    md("## 3. Module 2 · Reinforced Propagation Path Generation",
       "",
       "MDP: state = กราฟเส้นทางปัจจุบัน + user ล่าสุด + เนื้อหาข่าว · action = เพิ่ม user ใหม่ผ่านความสัมพันธ์ใน H (repost / co-engage) หรือผ่านความใกล้เนื้อหา",
       "policy (actor) เลือก K2 = 10 คนต่อ step จนครบ T = 200 คน · ตอน train ใช้ reward = λ2·R_T (ตรงกับคนที่แชร์ในช่วงเวลานั้น) + (1-λ2)·R_G (discriminator)",
       "",
       "cell ด้านล่าง **สร้าง G′ ใหม่สด ๆ** ให้ทั้งข่าว REAL และ FAKE ด้วย policy ที่ train แล้ว (เปลี่ยน `seed` แล้วได้เส้นทางต่างกัน เพราะ policy สุ่มตาม π)"),
    code("GEN = {k: B.generate(k, seed=0) for k in PAIR}",
         "for k in PAIR:",
         "    viz.plot_generation(B, k, *GEN[k])"),
    code("# เทียบ REAL กับ FAKE: สร้าง user มาจากไหน ตรงกับคนแชร์จริงกี่คน และ policy มั่นใจแค่ไหน",
         "def gen_summary(k, steps, rec):",
         "    via_content = sum(p == -1 for st in steps for p, _ in st)",
         "    n = sum(len(st) for st in steps)",
         "    return {'news': f\"#{k} {B.item(k)['label'].upper()}\", 'early users in H': f\"{B.known_share(k):.0%}\",",
         "            'generated users': n, 'via graph H': n - via_content, 'via content': via_content,",
         "            'match real later users': len(B.hits(k, steps)), 'mean candidates / step': round(np.mean([r['n_candidates'] for r in rec]), 1),",
         "            'mean entropy of π': round(np.mean([r['entropy'] for r in rec]), 2)}",
         "pd.DataFrame([gen_summary(k, *GEN[k]) for k in PAIR]).set_index('news').T"),
    code("# ทั้ง test set (G′ ที่ Stage 1 สร้างไว้และใช้วัดผลจริง): ตรงกับคนแชร์จริงแค่ไหน แยกข่าว real / fake",
         "test = list(B.data['split']['test'])",
         "rows = []",
         "for lab, v in [('real', 1), ('fake', 0)]:",
         "    ks = [i for i in test if B.labels[i] == v]",
         "    gen = sum(len(st) for i in ks for st in B.paths[i]['steps'])",
         "    hit = sum(len(B.hits(i, B.paths[i]['steps'])) for i in ks)",
         "    rows.append({'test news': lab, 'news': len(ks), 'generated users': gen, 'match real later users': hit, 'hit rate': f'{hit / max(gen, 1):.2%}'})",
         "pd.DataFrame(rows)"),

    md("## 4. Module 3 · Temporal-aware Fake News Detection",
       "",
       "เส้นทาง (early จริง + G′) ถูกตัดเป็น snapshot ตามเวลา → GAT อ่านทีละ snapshot + position embedding (Eq. 20)",
       "→ self-attention ระหว่าง snapshot, cross-attention ไปยังโครงสร้างกราฟสุดท้าย, FFN (Eq. 21) → mean pool = h*",
       "→ P(real) = σ(W·[h* ; n_T ; n_V] + b) (Eq. 22)"),
    code("OUT = {k: B.predict(k, explain=True) for k in PAIR}",
         "for k in PAIR:",
         "    viz.plot_snapshots(B, OUT[k])"),
    md("**อะไรตัดสินผล?** ชั้นสุดท้ายเป็นสมการเส้นตรง จึงแยก logit ได้พอดีว่ามาจากเส้นทาง (h*), ข้อความ, ภาพ หรือค่าคงที่",
       "และลองให้ detector ตัวเดิมเห็น G′ แค่ 0 ถึง 20 step ว่าผลของข่าว REAL / FAKE เปลี่ยนไหม"),
    code("name = {k: f\"#{k} ({B.item(k)['label']})\" for k in PAIR}",
         "curves = {name[k]: [B.predict(k, n_steps=n) for n in range(len(B.stored_steps(k)) + 1)] for k in PAIR}",
         "viz.plot_decision(B, {name[k]: OUT[k] for k in PAIR}, curves)"),
    code("# ทั้ง test set: ถ้าแทนส่วนของเส้นทางด้วยค่าเฉลี่ย accuracy เปลี่ยนเท่าไร (= model ใช้เส้นทางมากแค่ไหน)",
         "parts = [B.predict(i, explain=True)['parts'] for i in test]",
         "g = np.array([p['path h*'] for p in parts]); rest = np.array([p['text'] + p['image'] + p['bias'] for p in parts])",
         "y = B.labels[test]",
         "print(f'std of path term {g.std():.4f} · std of text+image {rest.std():.3f}')",
         "print(f'acc {((g + rest > 0) == y).mean():.3f} -> with path term = constant {((g.mean() + rest > 0) == y).mean():.3f}')"),

    md("## 5. ผลบน test set (แยกข่าว real / fake)"),
    code("RES = {'test': B.evaluate('test')}",
         "r = RES['test']",
         "print(f\"test {r['n']} news ({r['real']} real / {r['fake']} fake) · acc {r['acc']:.3f} · F1 {r['f1']:.3f} · AUC {r['auc']:.3f}\")",
         "print(f\"   real news predicted correctly {r['acc_real']:.1%} · fake news predicted correctly {r['acc_fake']:.1%}\")",
         "viz.plot_evaluation(r, f'{DATASET} test')"),

    md("## 6. ข้อมูลจากแหล่งอื่น: เหตุการณ์ที่ model ไม่เคยเห็น",
       "",
       "PHEME รุ่น 9 เหตุการณ์ (figshare 6392078, CC BY 4.0) มี 4 เหตุการณ์ที่ไม่อยู่ใน training: ebola-essien, gurlitt, prince-toronto, putinmissing",
       "เราเก็บเฉพาะข่าวที่มีครบ ข้อความ + ภาพ + เส้นทาง ได้ 64 ข่าว · user ที่ model ไม่รู้จักใช้เวกเตอร์ UNK",
       "ทุกอย่างเหมือนส่วนที่ 1-5 (CLIP feature และ G′ คำนวณไว้แล้วด้วย model ตัวเดียวกัน) · มีเฉพาะ DATASET = 'pheme'"),
    code("EXT_PAIR = []",
         "if B.ext is None:",
         "    print('ไม่มีข้อมูลจากแหล่งอื่นสำหรับ', DATASET, '-> ข้ามส่วนนี้ (ใช้ DATASET = \"pheme\")')",
         "else:",
         "    print(B.ext['source'])",
         "    ext = pd.DataFrame([{'key': k, 'event': B.item(k)['event'], 'label': B.item(k)['label'], 'P(real)': round(B.predict(k), 3),",
         "                         'early users in H': B.known_share(k)} for k in B.ext_keys()])",
         "    ext['pred'] = np.where(ext['P(real)'] > 0.5, 'real', 'fake')",
         "    ext['correct'] = ext['pred'] == ext['label']",
         "    display(ext.groupby(['event', 'label']).agg(news=('key', 'size'), correct=('correct', 'sum'),",
         "                                                early_users_in_H=('early users in H', 'mean')).round(2))",
         "    RES['external (new events)'] = B.evaluate('external')",
         "    viz.plot_evaluation(RES['external (new events)'], 'PHEME new events')",
         "    viz.plot_compare_sets(RES)"),
    code("# คู่ REAL / FAKE จากเหตุการณ์ใหม่: ผ่านทุก module เหมือนข้างบน",
         "if B.ext is not None:",
         "    # ข่าว real ส่วนใหญ่ทายถูก (16 / 17) จึงเลือกข่าว real ที่ทายถูก · ข่าว fake ทุกข่าวถูกทายผิดเป็น real",
         "    EXT_REAL = next(k for k in B.ext_keys() if B.label(k) == 1 and B.predict(k) > 0.5)",
         "    EXT_FAKE = next(k for k in B.ext_keys() if B.label(k) == 0)",
         "    EXT_PAIR = [EXT_REAL, EXT_FAKE]",
         "    viz.show_pair(B, EXT_PAIR, [B.predict(k) for k in EXT_PAIR])",
         "    viz.plot_ego_pair(B, EXT_PAIR)",
         "    viz.plot_content_candidates_pair(B, EXT_PAIR, k=12)"),
    code("if B.ext is not None:",
         "    for k in EXT_PAIR:",
         "        GEN[k] = B.generate(k, seed=0)",
         "        viz.plot_generation(B, k, *GEN[k])",
         "    display(pd.DataFrame([gen_summary(k, *GEN[k]) for k in EXT_PAIR]).set_index('news').T)",
         "    name.update({k: f\"{k} ({B.item(k)['label']})\" for k in EXT_PAIR})",
         "    for k in EXT_PAIR:",
         "        OUT[k] = B.predict(k, explain=True)",
         "        viz.plot_snapshots(B, OUT[k])",
         "    viz.plot_decision(B, {name[k]: OUT[k] for k in EXT_PAIR},",
         "                      {name[k]: [B.predict(k, n_steps=n) for n in range(len(B.stored_steps(k)) + 1)] for k in EXT_PAIR})"),
    md("**ทำไมเหตุการณ์ใหม่ทายไม่ได้:** user ช่วงแรกส่วนใหญ่ไม่อยู่ในกราฟ H (ได้เวกเตอร์ UNK เหมือนกันหมด) · เนื้อหาเป็นเรื่องใหม่ ·",
       "สัดส่วน fake สูงกว่าตอน train มาก · AUC ยังสูงกว่า 0.5 แปลว่าคะแนนข่าว fake ต่ำกว่าข่าว real โดยเฉลี่ย แต่ threshold 0.5 ใช้ไม่ได้",
       "(เราไม่ปรับ threshold หลังเห็นผล เพราะจะเป็นการปรับบนข้อมูลทดสอบ)"),

    md("## 7. สรุป: notebook นี้ใช้ข้อมูลกี่ข่าว"),
    code("def rl(keys):",
         "    ys = [B.label(k) for k in keys]",
         "    return {'news': len(keys), 'real': sum(y == 1 for y in ys), 'fake': sum(y == 0 for y in ys)}",
         "summary = [",
         "    {'ส่วน': 'dataset ทั้งหมด (train / val / test)', **rl(list(range(B.N)))},",
         "    {'ส่วน': '  - train (สร้างกราฟ H + train model)', **rl(list(B.data['split']['train']))},",
         "    {'ส่วน': '1. ข่าว demo (ตาราง)', **rl(list(demo['index']))},",
         "    {'ส่วน': '1-4. คู่ตัวอย่าง REAL / FAKE (แสดงทุก module)', **rl(PAIR)},",
         "    {'ส่วน': '3-5. test set (ทาย + hit rate + แยก logit)', **rl(test)},",
         "]",
         "if B.ext is not None:",
         "    summary += [{'ส่วน': '6. ข้อมูลจากแหล่งอื่น (4 เหตุการณ์ใหม่)', **rl(B.ext_keys())},",
         "                {'ส่วน': '6. คู่ตัวอย่าง REAL / FAKE จากเหตุการณ์ใหม่', **rl(EXT_PAIR)}]",
         "summary.append({'ส่วน': '3, 6. ข่าวที่สร้าง G′ สดใน notebook', **rl(list(GEN))})",
         "SUMMARY = pd.DataFrame(summary).set_index('ส่วน')",
         "SUMMARY"),
    code("shown = list(dict.fromkeys(list(demo['index']) + PAIR + EXT_PAIR))",
         "evaluated = len(test) + (len(B.ext_keys()) if B.ext is not None else 0)",
         "print(f'dataset: {DATASET} · model seed {s[\"seed\"]}')",
         "print(f'ข่าวที่แสดงเป็นรายข่าว: {len(shown)} ข่าว (real {sum(B.label(k) == 1 for k in shown)} / fake {sum(B.label(k) == 0 for k in shown)})')",
         "print(f'ข่าวที่สร้าง G′ สดและแสดงครบทุก module: {len(GEN)} ข่าว · ข่าวที่ model ทายเพื่อวัดผล: {evaluated} ข่าว')",
         "for name_, r in RES.items():",
         "    print(f\"  {name_:<24} {r['n']:>4} ข่าว ({r['real']} real / {r['fake']} fake) · acc {r['acc']:.3f} · real ถูก {r['acc_real']:.0%} · fake ถูก {r['acc_fake']:.0%} · AUC {r['auc']:.3f}\")"),
    md("### ข้อสรุป",
       "",
       "- **Module 1** วาง user ที่มีพฤติกรรมคล้ายกันไว้ใกล้กัน (t-SNE มีกลุ่มที่แชร์ข่าว fake เป็นส่วนใหญ่) โดยไม่ใช้ label",
       "- **Module 2** สร้าง G′ ได้ราว 200 คนต่อข่าว ส่วนใหญ่มาจากกราฟ H แต่ตรงกับคนที่แชร์จริงน้อยมาก (ดูตาราง hit rate)",
       "- **Module 3** ใน PHEME ส่วนของเส้นทางแทบเป็นค่าคงที่ ผลมาจากข้อความ + ภาพ · ใน GossipCop ส่วนของเส้นทางมีผลจริง",
       "- ข่าวใน test ของ dataset เดิม model ทายได้ใกล้เคียง paper แต่ข่าวจาก **เหตุการณ์ใหม่** ที่มีแต่ user ใหม่ ทายเกือบทั้งหมดเป็น real",
       "- G′ ที่สร้างสดต่างจาก G′ ที่ใช้วัดผลเล็กน้อยเพราะ policy สุ่ม แต่ P(real) แทบไม่เปลี่ยน",
       "- รายละเอียดทั้งหมด: `Paper-Based/HANDOFF.md` และ `Paper-Based/Code_RPPG-Fake/RUNNING.md`"),
]


def notebook(cells, colab):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if colab:
        meta["colab"] = {"provenance": []}
        meta["accelerator"] = "None"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


for name, setup, colab in [("RPPG-Fake_Demo_Local.ipynb", SETUP_LOCAL, False), ("RPPG-Fake_Demo_Colab.ipynb", SETUP_COLAB, True)]:
    nb = notebook(([COLAB_BADGE] if colab else []) + [TITLE, *setup, *BODY], colab)
    for k, c in enumerate(nb["cells"]):
        c["id"] = f"c{k:02d}"
    nb["cells"] = insert_source(nb["cells"])   # code that was actually run, inside each section + section 8
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", name, len(nb["cells"]), "cells")
