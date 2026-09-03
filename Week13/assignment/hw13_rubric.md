# Rubric — HW13: Full Classification Project
**วิชา** 1145 201 | **CLO3**

---

## ข้อ 1 — Poisson GLM (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Fit GLM + summary | smf.glm ถูก, family=Poisson ถูก, แสดง summary ครบ, อ่าน coef/p-value ได้ | Fit ถูกแต่อ่าน summary ไม่ครบ | Fit ได้แต่ family ผิด | ทำไม่ได้ | 1 |
| IRR table + ตีความ | คำนวณ IRR = e^β̂ ถูกทุกตัว, ตีความ rain และ speed ถูก พร้อมหน่วย | IRR ถูกแต่ตีความไม่ครบ | คำนวณ IRR ได้บางตัว | ผิดหมด | 1 |
| Poisson vs Linear comparison | predict ทั้ง 2 model ถูก แสดงว่า Poisson ≥ 0 ตลอด Linear อาจลบ | predict ถูกแต่ไม่เปรียบเทียบ | predict ได้ 1 model | ไม่ทำ | 1 |

---

## ข้อ 2 — ROC + AUC (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Fit 5 classifiers + comparison table | ทุก method fit ถูก, scaler ถูก, table 5×5 ครบ Acc/P/R/F1/AUC | fit ถูก 4/5, table บางส่วน | fit ถูก 3/5 | fit ผิดหมด | 1 |
| ROC Curve plot | ROC ครบ 5 เส้น, AUC ใน legend, random diagonal, xlabel/ylabel ถูก | ROC ครบ 4/5 หรือขาด legend | ROC ได้ 2-3 เส้น | ไม่ plot | 1 |
| Optimal threshold + recommendation | Youden's J ถูก, threshold ถูก, อธิบาย diabetes screening + Recall สำคัญ ชัดเจน | threshold ถูกแต่ไม่อธิบาย | หา threshold ได้ | ผิดหรือไม่ตอบ | 1 |

---

## ข้อ 3 — Full Pipeline (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| EDA 3 plots | 3 plots ครบ มี title/label, distribution + boxplot + heatmap ถูก | 2/3 plots | 1/3 plots | ไม่มี EDA | 1 |
| Fit 6 methods + comparison table | 6 methods ถูก (incl. LR-balanced), table ครบ | 5/6 methods, table บางส่วน | 4/6 methods | ทำไม่ได้ | 1 |
| Threshold analysis 4 levels | ตาราง threshold 4 ค่าถูก, อธิบาย trade-off Precision/Recall ชัด | ตารางถูกแต่ไม่อธิบาย | ลอง 2 threshold | ไม่ทำ | 1 |
| Business recommendation ครบ 3 ประเด็น | เลือก classifier + เหตุผล, threshold พร้อม FN/FP cost reasoning, trade-off ชัดเจน | 2/3 ประเด็น | 1/3 ประเด็น | ไม่ตอบ | 1 |
