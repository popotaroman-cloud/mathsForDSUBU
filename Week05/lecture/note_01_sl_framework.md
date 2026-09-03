# Note 1: กรอบแนวคิด Statistical Learning — Y = f(X) + ε
> Week 5 | CLO2 | ISLP Reference: Ch.1 Introduction + Ch.2.1 What Is Statistical Learning?

## บทนำ

สัปดาห์นี้เราจะเข้าสู่หัวใจของวิชา — **Statistical Learning** คือกรอบคณิตศาสตร์ที่ใช้เรียนรู้ความสัมพันธ์ระหว่างข้อมูล input และ output จากข้อมูลตัวอย่าง เป้าหมายของสัปดาห์นี้คือให้นักศึกษาเข้าใจและอธิบายกรอบ Y = f(X) + ε ได้ รวมถึงแยกแยะว่าเราต้องการ **prediction** หรือ **inference** จากโจทย์ที่กำหนด ซึ่งตรงกับ LLo: *อธิบายกรอบแนวคิด Statistical Learning แยกแยะ Supervised/Unsupervised และ Regression/Classification ได้* ในโลกจริงของ Data Science เราใช้กรอบนี้ทุกครั้งที่สร้าง model — ไม่ว่าจะเป็นการพยากรณ์ยอดขาย, วินิจฉัยโรค, หรือแนะนำสินค้า ทุก model ล้วนพยายาม "ประมาณ" ฟังก์ชัน f ที่แท้จริงในธรรมชาติ เมื่อเข้าใจกรอบนี้แล้ว การเรียนรู้ algorithm ต่าง ๆ ในสัปดาห์ต่อไปจะกลายเป็น "วิธีต่าง ๆ ในการประมาณ f" แทนที่จะเป็นสูตรที่ต้องจำแบบไม่เข้าใจ

---

## 5.1 What Is Statistical Learning?  *(ISLP 2.1)*

ในส่วนนี้เราจะนิยาม Statistical Learning อย่างเป็นทางการ เพื่อสร้างฐานความเข้าใจสำหรับทุก algorithm ที่จะเรียนตลอดวิชา

### นิยาม

สมมติว่าเรามีตัวแปร input (เรียกว่า **predictors**, **features**, หรือ **independent variables**) จำนวน p ตัว คือ $X_1, X_2, \ldots, X_p$ และตัวแปร output (เรียกว่า **response** หรือ **dependent variable**) คือ $Y$

เราสมมติว่ามีความสัมพันธ์ทั่วไปในรูป:

$$Y = f(X) + \varepsilon$$

โดย:
- $f$: ฟังก์ชัน **ที่ไม่รู้จัก** (fixed but unknown function) แสดงถึงข้อมูลเชิงระบบ (systematic information) ที่ X ให้เกี่ยวกับ Y
- $\varepsilon$: **random error term** ที่เป็น independent จาก X และมีค่าเฉลี่ย = 0 คือ $E[\varepsilon] = 0$
- Statistical Learning คือชุดวิธีการสำหรับ **ประมาณ f** จากข้อมูล

### ตัวอย่างเชิงตัวเลข — Advertising Dataset

สมมติเรามีข้อมูลค่าใช้จ่ายโฆษณา (หน่วย: พันดอลลาร์) และยอดขาย (หน่วย: พันหน่วย):

| TV ($X_1$) | Radio ($X_2$) | Newspaper ($X_3$) | Sales ($Y$) |
|-----------|--------------|-----------------|------------|
| 230.1 | 37.8 | 69.2 | 22.1 |
| 44.5 | 39.3 | 45.1 | 10.4 |
| 17.2 | 45.9 | 69.3 | 9.3 |
| 151.5 | 41.3 | 58.5 | 18.5 |

Statistical Learning ถามว่า: มีฟังก์ชัน f ใดที่ทำให้ $\text{Sales} \approx f(\text{TV}, \text{Radio}, \text{Newspaper})$?

ตัวอย่างเบื้องต้น (linear approximation):
$$\text{Sales} \approx f(X) = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \beta_3 \cdot \text{Newspaper}$$

**DS Connection**: ทุก model ใน Data Science ไม่ว่าจะเป็น Linear Regression, Random Forest, หรือ Neural Network ล้วนพยายาม "ประมาณ f" ในรูปแบบต่าง ๆ กัน

---

## 5.2 Why Estimate f? — Prediction vs Inference  *(ISLP 2.1.1)*

ในส่วนนี้เราจะเรียนรู้ว่ามีเหตุผลสองประการในการประมาณ f เพื่อให้นักศึกษาเลือกวิธีการที่เหมาะสมกับโจทย์

### 2.1 Prediction (การพยากรณ์)

เมื่อ input X พร้อมใช้งานแต่ output Y ยังไม่ทราบ เราต้องการประมาณ:

$$\hat{Y} = \hat{f}(X)$$

ใน setting นี้ เราสนใจว่า $\hat{Y}$ ใกล้เคียง $Y$ แค่ไหน แต่ **ไม่สนใจ** ว่า $\hat{f}$ มีรูปร่างอย่างไรภายใน

**ตัวอย่าง**: ธนาคารต้องการพยากรณ์ว่าลูกค้ารายนี้จะผิดนัดชำระหรือไม่ สนใจแค่คำตอบ yes/no ไม่สนใจว่า feature ไหนสำคัญที่สุด

### 2.2 Inference (การอนุมาน)

เราต้องการ **เข้าใจ** ความสัมพันธ์ระหว่าง X และ Y โดยเฉพาะ:
- Feature ไหนสัมพันธ์กับ Y?
- ทิศทางและขนาดของความสัมพันธ์เป็นอย่างไร?
- Model เชิงเส้นเพียงพอหรือต้องการ non-linear?

**ตัวอย่าง**: นักระบาดวิทยาต้องการรู้ว่าพฤติกรรมใดของผู้ป่วย (สูบบุหรี่, ออกกำลังกาย, อาหาร) สัมพันธ์กับโรคหัวใจ — **ต้องอธิบายได้**

### 2.3 Reducible vs Irreducible Error

Expected prediction error แบ่งได้เป็นสองส่วน:

$$E\left[(Y - \hat{f}(X))^2\right] = \underbrace{\left[f(X) - \hat{f}(X)\right]^2}_{\text{Reducible Error}} + \underbrace{\text{Var}(\varepsilon)}_{\text{Irreducible Error}}$$

**Reducible Error**: เกิดจาก $\hat{f} \neq f$ — ลดได้ด้วยการเลือก algorithm และ training ที่ดีขึ้น

**Irreducible Error** ($\text{Var}(\varepsilon)$): เกิดจาก noise ที่แท้จริงในข้อมูล — **ลดไม่ได้เลย** ไม่ว่า model จะดีแค่ไหน

**ตัวอย่างตัวเลข**: ถ้าบ้านราคา 5 ล้านบาท แต่ $\hat{f}(X) = 4.8$ ล้านบาท ส่วนต่าง 0.2 ล้านบาทอาจมาจาก:
- Reducible: model ยังเรียนรู้ feature ไม่ครบ (ลดได้)
- Irreducible: ราคาบ้านขึ้นอยู่กับ negotiation ที่วัดไม่ได้ (ลดไม่ได้)

**DS Connection**: Irreducible error อธิบายว่าทำไม model ที่ดีที่สุดในโลกก็ยังมี error — นี่คือขีดจำกัดทางทฤษฎีที่ต้องเข้าใจก่อนตัดสินว่า model "แย่"

---

## Case Study: Spam Email Classifier — Supervised Learning Framework

**Scenario**: บริษัทอีเมลต้องการสร้างระบบกรอง spam อัตโนมัติ ปัจจุบันมี email ที่มนุษย์ label แล้ว 4,601 ฉบับ (2,788 Ham, 1,813 Spam)

**Data**:
- Input $X$: 57 features — ความถี่ของคำ (free, money, click), ความยาว email, สัดส่วน capital letters
- Output $Y$: 0 = Ham, 1 = Spam (categorical)

**Method**: ตั้งกรอบ Statistical Learning:
$$Y_{\text{spam}} = f(X_1, X_2, \ldots, X_{57}) + \varepsilon$$

ระบุว่าเป็น **Supervised + Classification** problem เพราะมี labeled Y และ Y เป็น categorical

**Result**: ด้วย Logistic Regression อย่างง่าย accuracy = 92.5%

**Insight**: แม้ accuracy จะดี แต่ Irreducible Error ยังคงอยู่ — มี email ที่เป็น spam แต่เขียนแบบ ham จริง ๆ ไม่มี model ใดแยกออกได้ 100%

```python
# ─── Statistical Learning Framework Demo ────────────────────────────
# วัตถุประสงค์: แสดงว่า spam classification คือการประมาณ f(X) ใน SL framework
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# สร้าง mock data แทน Spambase dataset (57 features)
np.random.seed(42)
n = 200
# features: ความถี่คำ 'free', 'money', ความยาว
X = np.random.randn(n, 3)
# สมมติ f(X): spam ถ้า 'free' + 'money' สูง
true_f = 1 / (1 + np.exp(-(2*X[:, 0] + 1.5*X[:, 1] - 0.5*X[:, 2])))
# Y มีทั้ง systematic part (true_f) และ irreducible error (noise)
Y = (true_f + np.random.normal(0, 0.1, n) > 0.5).astype(int)

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# ประมาณ f ด้วย Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# ประเมิน
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"จำนวนตัวอย่าง Train: {len(X_train)}, Test: {len(X_test)}")
# Accuracy ไม่ถึง 100% เพราะ irreducible error ε
```

---

## สรุป (Summary)

| แนวคิด | สูตร/Method | Python |
|--------|------------|--------|
| Statistical Learning | $Y = f(X) + \varepsilon$ | `sklearn` estimator API |
| Prediction | $\hat{Y} = \hat{f}(X)$, minimize error | `model.predict(X)` |
| Inference | เข้าใจ X → Y | `model.coef_` |
| Reducible Error | $[f(X) - \hat{f}(X)]^2$ | ลดด้วย better model |
| Irreducible Error | $\text{Var}(\varepsilon)$ | ลดไม่ได้ |

---

## เชื่อมกับสัปดาห์อื่น

- ← Week 4: Matrix factorization (PCA/SVD) — เป็นวิธีหนึ่งในการประมาณ f แบบ unsupervised
- → Week 6: เมื่อเรามี $\hat{f}$ แล้ว จะวัดความแม่นยำ (MSE, Bias-Variance) ได้อย่างไร?
- → Week 9: Linear Regression เป็น parametric method สำหรับประมาณ f ในรูปเส้นตรง
