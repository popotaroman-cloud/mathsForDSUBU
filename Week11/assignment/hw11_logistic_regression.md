# Homework 11: Logistic Regression Analysis
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO3 | **LLo**: สร้าง Logistic Regression, ตีความ coefficient และประเมิน model ด้วย classification metrics ได้  
**กำหนดส่ง**: สัปดาห์ที่ 12  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราเปลี่ยนจาก regression ที่มี quantitative Y มาสู่ **classification** ที่ Y เป็น categorical เช่น Default (Yes/No) หรือ Spam (Yes/No) Logistic Regression เป็น classifier พื้นฐานที่ interpretable ที่สุด เพราะ coefficient ตีความผ่าน log-odds และ Odds Ratio ได้ชัดเจน ต่างจาก Neural Network หรือ Random Forest ที่เป็น black box ในการบ้านนี้คุณจะสร้าง Logistic Regression, ตีความ z-statistic, p-value, CI, Odds Ratio, ทำ confusion matrix, คำนวณ precision/recall/F1 และเขียนรายงานสรุปผล

---

## ข้อ 1 — Simple Logistic Regression: Default ~ Balance (3 คะแนน)

ในข้อนี้คุณจะสร้าง Logistic Regression ที่ทำนาย probability ของการ default จาก credit card balance เพื่อให้เข้าใจ sigmoid function และการตีความ coefficient

**Dataset**: Default.csv (หรือ synthetic ที่ generate ไว้ใน lab)

**สิ่งที่ต้องทำ**:
1. Fit `default ~ balance` ด้วย statsmodels `logit()`
2. อ่านค่าจาก summary: β̂₀, β̂₁, SE, z-statistic, p-value, 95% CI
3. คำนวณ verify: z = β̂₁ / SE
4. ตีความ β̂₁: บอกว่า balance เพิ่ม $1,000 → Odds Ratio = e^(1000×β̂₁) = ?
5. Plot: sigmoid curve P(default|balance) พร้อม actual data points (alpha=0.1)
6. หา decision boundary (X* ที่ P = 0.5)

**Expected**: β̂₀ ≈ -10.65, β̂₁ ≈ 0.0055, X* ≈ $1,936

**Deliverable**: Code + summary output + sigmoid plot + คำอธิบาย 2–3 ประโยค

---

## ข้อ 2 — Multiple Logistic Regression + Confounding (4 คะแนน)

ในข้อนี้คุณจะเพิ่ม predictors เข้า model และสังเกต confounding effect ที่โด่งดังของ Default dataset

**สิ่งที่ต้องทำ**:
1. Fit 3 models:
   - M1: `default ~ student`
   - M2: `default ~ balance + income + student`
2. สร้างตาราง:

| Model | β̂_student | OR_student | p-value | ความหมาย |
|-------|-----------|-----------|---------|---------|
| M1 (SLR) | | | | |
| M2 (MLR) | | | | |

3. อธิบาย: ทำไม β̂_student เปลี่ยน sign? (ต้องการ 3–5 ประโยค พร้อม correlation matrix)
4. M2: ระบุ predictor ไหน significant ที่ α = 0.05 และ predictor ไหนไม่ significant
5. Predict P(default) สำหรับ customer 3 คน:

| Customer | Balance | Income | Student | P(default) | Predict |
|----------|---------|--------|---------|-----------|---------|
| A | $1,200 | $35,000 | No | ? | ? |
| B | $2,500 | $80,000 | Yes | ? | ? |
| C | $500 | $20,000 | No | ? | ? |

**Deliverable**: Code + ตาราง + คำอธิบาย confounding + prediction table

---

## ข้อ 3 — Model Evaluation: Confusion Matrix + Metrics (3 คะแนน)

ในข้อนี้คุณจะประเมิน classifier ด้วย metrics หลายตัวและปรับ threshold ตาม business objective

**ใช้ Multiple Logistic model จากข้อ 2 (M2)**

**สิ่งที่ต้องทำ**:
1. Split data: train 80% / test 20% (random_state=42)
2. Fit บน train set, predict บน test set (threshold = 0.5)
3. สร้าง confusion matrix (visualize ด้วย heatmap)
4. คำนวณ: Accuracy, Precision, Recall, F1 (โดยใช้ class "Yes" เป็น positive)
5. **Threshold analysis**:
   - ลอง threshold = 0.2 และ 0.3
   - เปรียบ Precision, Recall, F1 ของ 3 threshold (0.2, 0.3, 0.5) ในตาราง
   - แนะนำ threshold ที่เหมาะสมสำหรับ "ธนาคารต้องการลด default" พร้อมเหตุผล

**Business Question**: ถ้าธนาคารเสียหายจาก FN (miss default) มากกว่า FP 10 เท่า → ควรเลือก threshold เท่าไร?

**Deliverable**: Code + confusion matrix plot + metrics table + คำตอบ business question 2–3 ประโยค

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว
- ตั้งชื่อ: `hw11_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | Simple Logistic + Sigmoid | 3 |
| 2 | Multiple Logistic + Confounding | 4 |
| 3 | Confusion Matrix + Metrics | 3 |
| **รวม** | | **10** |
