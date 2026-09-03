# Rubric — Final Project (20 คะแนน)
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | **CLO1–4**

---

## รายงาน Jupyter Notebook (17 คะแนน)

### CLO1: Linear Algebra (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Covariance matrix | คำนวณ Σ ถูก, interpret pairs ที่ correlate สูง/ต่ำ ชัดเจน | คำนวณถูกแต่ interpret น้อย | คำนวณได้ | ผิดหรือไม่ทำ | 1 |
| Eigenvalue + variance explained | λ₁,...,λₚ ถูก, % variance ถูก, ระบุ PC กี่ตัวได้ 90% | ค่าถูกแต่ไม่ interpret | คำนวณได้บางส่วน | ผิดหรือไม่ทำ | 1 |
| PCA projection + visualization | transform ถูก, scatter PC1 vs PC2 ด้วย color by target, มี label ครบ | plot ถูกแต่ label ไม่ครบ | plot ได้แต่ไม่ใช่ PCA ที่แท้จริง | ไม่ทำ | 1 |
| Interpretation | อธิบาย PCA ใน context ของ dataset ได้ถูกต้อง มีเนื้อหาที่ relate กับ problem | อธิบายได้บางส่วน | พยายามอธิบาย | ไม่อธิบาย | 1 |

---

### CLO2: Statistical Analysis (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Descriptive + Distribution | stats table ครบ, histogram + QQ plot ถูก, กล่าวถึง skewness/normality | stats ครบแต่ plot ขาด | stats บางส่วน | ขาดมาก | 1 |
| Hypothesis testing | H₀/H₁ ชัดเจน, test เหมาะสม, interpret p-value ถูก พร้อม business conclusion | test ถูกแต่ interpret ไม่ครบ | test ได้แต่ H₀ ไม่ชัด | ผิดหรือไม่ทำ | 1 |
| Correlation + feature insight | heatmap ถูก, ระบุ top features ที่ relate กับ target พร้อมค่า r, มี direction | heatmap ถูกแต่ไม่ identify top features | heatmap ได้ | ไม่ทำ | 1 |

---

### CLO3: Machine Learning Models (5 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| ≥ 3 models fit ถูกต้อง | 3 models fit ถูก, pipeline ถูก (scaler → model), code clean | 2/3 models ถูก | 1/3 ถูก | ทำไม่ได้ | 2 |
| Evaluation metrics เหมาะสม | metrics เลือกถูก (classification: F1/AUC, regression: RMSE/R²), CM หรือ residual plot | metrics ถูกแต่ขาดบาง | metrics มีแต่ไม่ visualize | ผิดหรือขาดมาก | 1 |
| Comparison table + interpretation | ตารางครบ 3 models, เลือก best พร้อมเหตุผลที่ใช้ metric ที่เหมาะสม | ตารางครบแต่ไม่ justify | ตารางบางส่วน | ไม่มี | 1 |
| Business/Domain interpretation | อธิบาย model result ใน context ของ dataset — ไม่ใช่แค่ตัวเลข | อธิบายได้บางส่วน | พยายาม | ไม่อธิบาย | 1 |

---

### CLO4: Model Selection with CV (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| k-Fold CV ถูกต้อง | CV ถูก (k ≥ 5), stratified ถ้า classification, scaler ใน pipeline, report mean ± SE | CV ถูกแต่ขาด SE หรือ scaler ไม่ถูก | CV ได้บางส่วน | ทำไม่ได้ | 1 |
| One-SE Rule + selection | ระบุ best model + one-SE model ถูก พร้อมอธิบายเหตุผลการเลือก | ระบุ best ถูกแต่ One-SE ขาด | ระบุ best ได้ | ผิดหรือไม่ทำ | 1 |
| Final test evaluation | evaluate test **ครั้งเดียว** หลัง selection เสร็จ, report ครบ, เปรียบเทียบ CV vs test | evaluate ถูกแต่ไม่เปรียบเทียบ | evaluate ได้ | evaluate ระหว่าง selection (data leakage) | 1 |

---

### Code Quality + Documentation (2 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Code comments ครบ | ทุก block มี `# ─── ทำอะไร ───` + `# วัตถุประสงค์:` ตาม standard | 75% blocks มี comment | 50% blocks | ไม่มี comment | 1 |
| Markdown + structure | Notebook มี markdown cells ก่อน section, section ชัดเจน, plot มี label | structure ถูกแต่ markdown น้อย | structure พอรู้เรื่อง | ไม่มี structure | 1 |

---

## Presentation + Q&A (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Content + storyline | นำเสนอ CLO1–4 ครบ มี storyline ชัดเจนตั้งแต่ problem จนถึง conclusion, เนื้อหาถูกต้อง | ครบ 3/4 CLO | ครบ 2/4 CLO | ไม่ครบหรือ content ผิดมาก | 1 |
| Presentation skills | พูดชัดเจน บริหารเวลาถูก (10 นาที ±1), ใช้ visual ประกอบดี, ตอบคำถามได้ | บริหารเวลาเกิน/ขาด 2 นาที, visual พอใช้ | พูดได้แต่ visual น้อย | พูดไม่ได้หรือขาด presentation มาก | 1 |
| Q&A | ตอบคำถามถูก อธิบาย reasoning ได้ ไม่ใช่แค่อ่าน code | ตอบได้บางส่วน | พยายามตอบ | ตอบไม่ได้เลย | 1 |

---

## Summary Score

| Component | คะแนน |
|-----------|-------|
| CLO1: Linear Algebra | /4 |
| CLO2: Statistics | /3 |
| CLO3: Models | /5 |
| CLO4: CV + Model Selection | /3 |
| Code Quality | /2 |
| Presentation + Q&A | /3 |
| **รวม** | **/20** |

**เกรด:**
- 18–20: A
- 16–17: B+
- 14–15: B
- 12–13: C+
- 10–11: C
- 8–9: D+
- 6–7: D
- < 6: F
