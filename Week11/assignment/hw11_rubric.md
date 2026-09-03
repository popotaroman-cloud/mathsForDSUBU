# Rubric — HW11: Logistic Regression Analysis
**วิชา** 1145 201 | **CLO3**

---

## ข้อ 1 — Simple Logistic Regression (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| fit + summary | fit ถูก แสดง summary ครบ อ่าน β̂ SE z p CI ถูก | อ่านได้ 4/5 | อ่านได้ 3/5 | fit ไม่ได้ | 1 |
| ตีความ β̂₁ + OR | คำนวณ OR = e^(1000β̂₁) ถูก อธิบาย log-odds ชัดเจน | OR ถูกแต่อธิบายไม่ครบ | คำนวณได้ | ผิดหมด | 1 |
| Sigmoid plot + decision boundary | sigmoid plot ถูก หา X* = 1936 ถูกต้อง | plot ถูกแต่ไม่หา X* | plot ได้ | ไม่ plot | 1 |

---

## ข้อ 2 — Multiple Logistic + Confounding (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Fit 2 models | M1 M2 fit ถูกทั้งคู่ แสดง summary | fit ได้ 1/2 | พยายาม | ทำไม่ได้ | 1 |
| ตาราง + OR | ตารางครบ OR ถูก ระบุ significant/not | ตารางครบแต่ OR ผิด | ตารางบางส่วน | ไม่ทำ | 1 |
| อธิบาย confounding | อธิบาย sign change ถูก อ้าง correlation balance-student | อธิบายได้บางส่วน | พยายาม | ไม่อธิบาย | 1 |
| Prediction table | ทุก prediction ถูกต้อง | 2/3 ถูก | 1/3 ถูก | ผิดหมด | 1 |

---

## ข้อ 3 — Confusion Matrix + Metrics (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Confusion matrix + Accuracy/P/R/F1 | CM ถูก metrics ถูกครบ visualize | metrics ถูก 3/4 | metrics ถูก 2/4 | ผิดหมด | 1 |
| Threshold analysis | ตาราง 3 threshold ถูก อธิบาย trade-off ชัด | ตารางถูกแต่ไม่อธิบาย | ลอง 2 threshold | ไม่ทำ | 1 |
| Business answer | แนะนำ threshold ต่ำ (0.2–0.3) อธิบาย FN cost สูง ชัดเจน | แนะนำถูกแต่ไม่อธิบาย | พยายาม | ผิดหรือไม่ตอบ | 1 |
