# Rubric — HW14: Cross-Validation & Bootstrap
**วิชา** 1145 201 | **CLO4**

---

## ข้อ 1 — k-Fold CV สำหรับ Polynomial Regression (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| 10-Fold CV + SE + plot | CV ถูก degree 1–8, คำนวณ SE ถูก, plot MSE±SE bar ถูก | CV ถูกแต่ไม่มี SE bar | CV ถูกบางส่วน | ทำไม่ได้ | 1 |
| Best CV + One-SE Rule | ระบุ best degree และ one-SE degree ถูก พร้อมเหตุผล | ระบุ best ถูกแต่ One-SE ผิด | ระบุ best ได้ | ผิดหรือไม่ระบุ | 1 |
| Polynomial curves + edu | plot polynomial curve 2 versions ถูก, เพิ่ม edu และ compare CV MSE | plot ถูกแต่ไม่มี edu | plot ได้ 1 version | ไม่ plot | 1 |

---

## ข้อ 2 — Bootstrap (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Bootstrap Median: SE + CI + plot | Bootstrap loop ถูก (replace=True), SE ถูก, CI ถูก, histogram ถูก | SE ถูกแต่ CI ผิดหรือไม่ plot | loop ถูกแต่ SE ผิด | ทำไม่ได้ | 1 |
| Bootstrap Spearman: SE + CI | Spearman corr ถูก, bootstrap loop ถูก, SE + CI ถูก | bootstrap ถูกแต่ใช้ Pearson แทน | bootstrap ถูกบางส่วน | ทำไม่ได้ | 1 |
| เปรียบเทียบ Normal approx | คำนวณ Normal approx ถูก, เปรียบเทียบกับ Bootstrap CI ชัดเจน | คำนวณ Normal ได้แต่ไม่เปรียบเทียบ | คำนวณ Normal ได้ | ไม่ทำ | 1 |

---

## ข้อ 3 — Full CV Model Selection Pipeline (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| CV Grid Search: KNN + Logistic | CV ถูกทั้ง KNN (8 K values) + Logistic (4 C values), stratified, AUC scoring | CV ถูก 1/2 method | CV ถูกบางส่วน | ทำไม่ได้ | 1 |
| Plot AUC vs K / vs C | plot ถูกทั้ง 2, แสดง mean±SE, mark optimal | plot ถูก 1/2 | plot ได้บางส่วน | ไม่ plot | 1 |
| Comparison table + best method | table ครบ ≥ 4 methods + LDA/NB, เลือก best ถูก, เหตุผลชัด | table 3/4 methods | table 2/4 | ไม่มีตาราง | 1 |
| Final test evaluation + CV vs Test AUC | test evaluation ครบ 5 metrics + ROC, อธิบาย CV vs test gap ถูก | evaluation ครบแต่ไม่อธิบาย gap | evaluation บางส่วน | ไม่ evaluate | 1 |
