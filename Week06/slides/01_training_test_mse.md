# Slide Deck: Training MSE vs Test MSE
> Week 6 | CLO2 | ISLP Ch.2.2.1 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 6**  
Measuring Model Accuracy: Training MSE vs Test MSE  
CLO2: คำนวณ MSE และอธิบายความแตกต่างระหว่าง Training Error กับ Test Error ได้

---

## Slide 2 — Week Overview
**Key Message**: Week 6 คือจุดเปลี่ยนสำคัญ — เราหยุดถามว่า "model ดีไหม?" แล้วเริ่มถามว่า "model generalizes ได้ไหม?"

สัปดาห์นี้เราก้าวจาก "วิธีสร้าง model" ไปสู่ "วิธีประเมิน model" ซึ่งเป็นทักษะที่แยก Data Scientist มืออาชีพออกจากมือใหม่ แนวคิดหลักคือ model ที่ดูดีบน training data อาจ "จำ" data แทนที่จะ "เรียนรู้" ซึ่งทำให้ predict ผิดบน data ใหม่ เป้าหมายของสัปดาห์นี้คือให้นักศึกษาเข้าใจว่าทำไม Test MSE ถึงสำคัญกว่า Train MSE และ U-curve ของ Test MSE เกิดขึ้นได้อย่างไรจากการ decompose เป็น Bias² + Variance + Irreducible Error ทักษะนี้ใช้ทุกวันในการเลือก model และ hyperparameter สำหรับปัญหาจริง

**4 หัวข้อสัปดาห์นี้:**
- Deck 1: Training MSE vs Test MSE + Overfitting/Underfitting
- Deck 2: Bias-Variance Decomposition + U-curve
- Deck 3: Classification Setting + Bayes Classifier + KNN
- Deck 4: Probability Foundations สำหรับ Statistical Learning

---

## Slide 3 — Mean Squared Error: นิยาม
**Key Message**: MSE วัด "ความห่าง" เฉลี่ยระหว่าง prediction กับ ค่าจริง — ยิ่งต่ำยิ่งดี

**Training MSE:**
```
MSE_train = (1/n) Σᵢ₌₁ⁿ (yᵢ − f̂(xᵢ))²

n  = จำนวน training observations
yᵢ = ค่าจริงของ observation i
f̂(xᵢ) = prediction ของ model ที่ xᵢ
```

**การตีความ:**
- MSE = 0: model predict ถูกต้องสมบูรณ์
- MSE = 4: prediction ผิดเฉลี่ย √4 = 2 หน่วย (RMSE)
- Units ของ MSE = units² ของ Y → ใช้ RMSE (√MSE) เพื่อ interpret ได้ง่ายกว่า

**ตัวอย่าง — House Price:**
```
Y (ราคาจริง): [200, 350, 150, 400] K บาท
Ŷ (prediction): [210, 330, 160, 380] K บาท
Errors: [-10, 20, -10, 20]
MSE = (100 + 400 + 100 + 400) / 4 = 250
RMSE = √250 ≈ 15.8 K บาท
→ ผิดเฉลี่ย 15,800 บาท
```

---

## Slide 4 — Training MSE ≠ Test MSE
**Key Message**: เราสนใจ Test MSE ไม่ใช่ Training MSE — model ต้องทำงานได้กับ data ที่ไม่เคยเห็น

```
Training MSE  = วัดบน data ที่ model ใช้ fit (คุ้นเคย)
Test MSE      = วัดบน data ใหม่ที่ model ไม่เคยเห็น ← สิ่งที่เราสนใจ
```

**ทำไม Test MSE ถึงสำคัญกว่า:**

| สถานการณ์จริง | Training Data | Test Data |
|-------------|--------------|-----------|
| Spam filter | Email ที่ผู้ใช้ label แล้ว | Email ใหม่ที่รับทุกวัน |
| Medical diagnosis | Patient cases ที่มี diagnosis แล้ว | Patient ใหม่ที่มาโรงพยาบาล |
| Stock prediction | ราคาหุ้นในอดีต | ราคาหุ้นวันพรุ่งนี้ |

**Problem:** Training MSE สามารถลดได้เรื่อยๆ โดยทำให้ model ซับซ้อนขึ้น  
→ แต่ Test MSE ไม่ได้ลดตามเสมอ

---

## Slide 5 — Overfitting คืออะไร?
**Key Message**: Overfitting เกิดเมื่อ model "จำ" training data รวมถึง noise — ทำให้ Train MSE ต่ำแต่ Test MSE สูง

```
Overfitting:
Training MSE ← ต่ำมาก (model จำทุก data point)
Test MSE     ← สูงมาก (model ไม่ generalize)

ตัวอย่าง KNN k=1:
Training MSE = 0 (interpolate ทุกจุดสมบูรณ์)
Test MSE = สูงมาก (ทำงานผิดบน data ใหม่)
```

**[FIGURE: Training data scatter + overfitted curve ที่ผ่านทุกจุดอย่าง wiggly + true f เป็น dashed line]**

**สัญญาณของ Overfitting:**
- Training MSE << Test MSE (gap ใหญ่)
- Model มี parameters มากกว่า n (observations)
- Polynomial degree สูงมาก, KNN k=1, Neural Network โตมากบน data น้อย

---

## Slide 6 — Underfitting คืออะไร?
**Key Message**: Underfitting เกิดเมื่อ model ง่ายเกินไปจน capture pattern จริงไม่ได้ — ทั้ง Train และ Test MSE สูง

```
Underfitting:
Training MSE ← สูง (model ไม่ fit data แม้แต่ training set)
Test MSE     ← สูง (ยิ่งไปกันใหญ่)

ตัวอย่าง: fit linear model กับ Y = sin(X)
→ เส้นตรงจะ miss curve ของ sin เสมอ
→ Error สูงทั้ง train และ test
```

**[FIGURE: Sine curve data + linear fit ที่ miss pattern อย่างชัดเจน]**

**สัญญาณของ Underfitting:**
- Training MSE สูงมาก (เทียบกับ irreducible error)
- Train MSE ≈ Test MSE (ทั้งคู่แย่)
- Model form ผิดประเภท (linear กับ nonlinear data)

---

## Slide 7 — The Gold Standard: Train/Test Split
**Key Message**: วิธีพื้นฐานสุดในการประเมิน model คือแบ่ง data ออกเป็น training set และ test set แยกกันก่อนเริ่ม

**Procedure:**
```
1. แบ่ง data: 70-80% สำหรับ Training, 20-30% สำหรับ Test
2. Fit model บน Training set เท่านั้น
3. ประเมิน Test MSE บน Test set
4. ไม่แตะ Test set จนกว่าจะ report final performance
```

**Python:**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model.fit(X_train, y_train)

# ─── ประเมินบน TEST set ─────────────────────────────────
# วัตถุประสงค์: Test MSE คือ estimate ของ performance จริง
test_mse = mean_squared_error(y_test, model.predict(X_test))
```

**ข้อควรระวัง:**
- ถ้า dataset เล็ก → test set เล็ก → estimate ไม่น่าเชื่อถือ
- แก้ได้ด้วย Cross-Validation (Week 14)

---

## Slide 8 — U-Curve: Complexity vs Test MSE
**Key Message**: Test MSE มักมี U-shape เป็น function ของ model complexity — อยู่ทางซ้ายมาก = underfit, ทางขวามาก = overfit

**[FIGURE: 2 panels ─ ซ้าย: data + 3 fitted curves (linear/moderate/overfit), ขวา: U-curve plot แสดง Train MSE (ลดเรื่อยๆ) และ Test MSE (U-shape) พร้อม horizontal line = Irreducible Error]**

```
← Underfitting                          Overfitting →
High Bias          Optimal              High Variance
Low Variance         ↓                  Low Bias
      ─────────────[★]────────────────────────────
                  Best model
```

**สิ่งที่สังเกตได้:**
- Training MSE: ลดเรื่อยๆ ไปถึง 0 ที่ high complexity
- Test MSE: มี minimum แล้วเพิ่มขึ้น → U-shape
- Irreducible Error: lower bound ที่ทำไม่ได้ดีกว่า

---

## Slide 9 — ตัวอย่างจริง: KNN Regression
**Key Message**: KNN แสดง U-curve อย่างชัดเจน — K เล็ก = overfit, K ใหญ่ = underfit, K เหมาะสม = optimal

**ทดสอบบน Advertising Dataset:**

| K | Train MSE | Test MSE | ลักษณะ |
|---|----------|---------|--------|
| 1 | 0.00 | 17.2 | Overfit ร้ายแรง |
| 5 | 3.2 | 9.5 | Better |
| 10 | 4.8 | 10.3 | Slightly underfit |
| 50 | 8.1 | 14.7 | Underfit |
| n | MSE_train | MSE_train | Worst (predict mean เสมอ) |

**Key Insight:**
- K=1: Train MSE=0 เพราะ model predict ค่าตัวเองเสมอ
- Optimal K ≈ 5 สำหรับ dataset นี้
- Linear Regression ชนะ KNN ทั้งหมด (เพราะ TV→Sales ใกล้ linear)

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **MSE**: วัด average squared prediction error — ยิ่งต่ำยิ่งดี
- **Training MSE ≠ Test MSE**: เราสนใจ Test MSE เพราะวัด generalization
- **Overfitting**: Train MSE ต่ำ, Test MSE สูง — model จำ noise
- **Underfitting**: ทั้ง Train และ Test MSE สูง — model ง่ายเกินไป
- **U-curve**: Test MSE มี minimum ที่ optimal complexity

| ปัญหา | Training MSE | Test MSE | วิธีแก้ |
|------|------------|---------|--------|
| Overfitting | ต่ำ | สูง | Simplify model, เพิ่ม data, Regularization |
| Underfitting | สูง | สูง | เพิ่ม complexity, เพิ่ม features |
| Good fit | ต่ำพอสมควร | ต่ำ | Keep it! |

**สัปดาห์ถัดไป (Slide Deck 2):** ทำไม U-curve ถึงเกิด? — Bias² + Variance + Irreducible Error decomposition ทางคณิตศาสตร์
