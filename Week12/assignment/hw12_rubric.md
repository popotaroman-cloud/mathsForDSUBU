# Rubric — HW12: Classification Methods Comparison
**วิชา** 1145 201 | **CLO3**

---

## ข้อ 1 — LDA บน Wine Quality Dataset (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Fit + StandardScaler ถูกต้อง | Fit LDA ถูก, scaler fit บน train เท่านั้น, แสดง class priors + class means | Fit ถูกแต่ scaler leak หรือแสดงผลไม่ครบ | Fit ได้แต่ไม่ standardize | ทำไม่ได้ | 1 |
| Confusion matrix + metrics ครบ | CM ถูก, แสดง precision/recall/F1 ครบ 3 class, accuracy ≈ 0.98 | metrics ถูก 2/3 class | metrics ผิดบางส่วน | ไม่มี | 1 |
| LDA projection scatter plot | transform ด้วย lda.transform() ถูก, scatter แยก 3 class ด้วย color ชัดเจน | plot ได้แต่ label ไม่ครบ | plot ได้ | ไม่ plot | 1 |

---

## ข้อ 2 — Assumption Testing (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| สร้าง 2 datasets + split + scale | ทั้ง 2 dataset ถูก, split ถูก, scale ถูก | 1/2 dataset ถูก | พยายามแต่ผิด | ไม่ทำ | 1 |
| Comparison table ครบ | ตาราง 4×4 ครบทุกช่อง, ค่า Accuracy/F1/AUC ถูก | ตารางครบแต่ค่าผิด 1–2 | ตารางบางส่วน | ไม่มีตาราง | 1 |
| Decision boundary plot | 4 subplots ครบ, boundary ถูกต้องตาม method, มี colormap + legend | plot ได้ 2/4 | plot ได้ 1/4 | ไม่ plot | 1 |
| อธิบาย assumption + ผล | อธิบาย LDA linear vs non-linear ถูก, อธิบาย QDA flexibility ถูก, กล่าวถึง NB independence | อธิบาย 2/3 ประเด็น | อธิบายได้บางส่วน | ไม่อธิบาย | 1 |

---

## ข้อ 3 — KNN Pipeline (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| F1 vs K plot + optimal K selection | plot F1 vs K ถูก (K=1,3,5,7,10,15), ระบุ optimal K ที่ peak F1 พร้อมเหตุผล | plot ถูกแต่ไม่ระบุ optimal K | plot ได้บางส่วน | ไม่ plot | 1 |
| Full comparison table + ROC curve | ตาราง 5 methods ครบ Acc/P/R/F1/AUC, ROC plot 5 เส้น + AUC labels | ตารางครบ ROC ขาด 1–2 method | ตารางบางส่วน | ไม่ทำ | 1 |
| Business recommendation | แนะนำ method ที่เน้น Recall, ระบุ threshold < 0.5, อธิบาย FN cost ชัดเจน | แนะนำถูกแต่ไม่อธิบาย FN cost | แนะนำได้ | ผิดหรือไม่ตอบ | 1 |
