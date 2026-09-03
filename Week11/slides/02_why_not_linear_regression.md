# Slide Deck: ทำไม Linear Regression ไม่เหมาะสำหรับ Classification
> Week 11 | CLO3 | ISLP Ch.4.2 | 8 slides

---
## Slide 1 — Title
**Why Not Linear Regression? ข้อจำกัดสำหรับ Classification**  
Week 11 | CLO3 | ISLP 4.2  
LLo: อธิบายว่าทำไม Linear Regression ไม่เหมาะสำหรับ Classification ได้

---
## Slide 2 — ลองใช้ Linear Regression สำหรับ Binary Y
**Key Message**: ถ้า encode Y=0/1 แล้วใช้ OLS — ดูเหมือนจะได้ แต่มีปัญหาสำคัญ 2 ข้อ

**Encoding**: Default = Yes → 1, Default = No → 0

```python
# ─── ลองใช้ Linear Regression สำหรับ classification ─────────
# วัตถุประสงค์: แสดงว่า OLS ให้ probability นอกช่วง [0,1]
default['default_num'] = (default['default'] == 'Yes').astype(int)
model_ols = smf.ols('default_num ~ balance', data=default).fit()

# บาง prediction จะออกนอก [0,1]!
preds = model_ols.fittedvalues
print(f'Predictions outside [0,1]: {((preds < 0) | (preds > 1)).sum()}')
# อาจได้ p̂ = -0.02 หรือ 1.3 ← ไม่ใช่ probability!
```

---
## Slide 3 — ปัญหาที่ 1: Predictions นอก [0,1]
**Key Message**: OLS ไม่รับประกันว่า ŷ ∈ [0,1] — ค่าลบหรือค่า > 1 ไม่มีความหมายในฐานะ probability

[FIGURE: scatter plot default (0/1) vs balance + OLS fitted line:
- บางส่วนของเส้นอยู่ต่ำกว่า 0 (balance ต่ำ)
- บางส่วนอาจสูงกว่า 1 (balance สูงมาก)
- Sigmoid curve (Logistic Regression) อยู่ใน [0,1] ตลอด]

**ปัญหา**:
- ŷ = -0.02 แปลว่าอะไร? probability ติดลบ?
- ŷ = 1.15 แปลว่าอะไร? probability 115%?
- ต้องการ model ที่ output อยู่ใน [0,1] เสมอ → Logistic Regression

---
## Slide 4 — ปัญหาที่ 2: Multi-class ไม่มี Natural Ordering
**Key Message**: ถ้า K > 2 → การ encode ตัวเลขเป็น arbitrary และให้ผลต่างกันตาม encoding

**ตัวอย่าง**: Medical condition: Mild (Y=1), Moderate (Y=2), Severe (Y=3)

```
OLS กับ Y = {1, 2, 3}:
  - สมมติว่า moderate = 2 × mild ในแง่ "ระยะ" ← ไม่มีเหตุผล
  - ถ้าเปลี่ยน encoding: Mild=1, Severe=2, Moderate=3 → ผลลัพธ์ต่าง!
```

**K = 2**: encoding เป็น 0/1 unique ไม่มีปัญหา multi-class

**ใน practice**: ถ้า K = 2 และ assumptions OK — OLS และ LDA ให้ผลใกล้เคียงกัน  
แต่ถ้า p̂ นอก [0,1] บ่อย → Logistic Regression ดีกว่าชัดเจน

---
## Slide 5 — Linear vs Logistic: Visual Comparison
**Key Message**: OLS เส้นตรงผ่านนอก [0,1]; Logistic sigmoid อยู่ใน [0,1] เสมอ

[FIGURE: สองกราฟ side-by-side:
ซ้าย — OLS: เส้นตรง ŷ = β₀ + β₁×balance, ส่วน balance ต่ำ → ŷ < 0
ขวา — Logistic: S-shaped sigmoid curve อยู่ใน [0,1] ตลอด, asymptote ที่ 0 และ 1]

```python
# ─── เปรียบเทียบ OLS กับ Logistic ──────────────────────────
# วัตถุประสงค์: แสดงภาพว่า sigmoid ดีกว่า OLS สำหรับ binary classification
import matplotlib.pyplot as plt
import numpy as np

bal_range = np.linspace(default['balance'].min(), default['balance'].max(), 100)

# OLS prediction (อาจออกนอก [0,1])
ols_pred = model_ols.params['Intercept'] + model_ols.params['balance'] * bal_range

# Logistic sigmoid
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(default[['balance']], default['default_num'])
log_pred = lr.predict_proba(bal_range.reshape(-1,1))[:,1]

fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, pred, title in [(axes[0], ols_pred, 'OLS (Linear)'), (axes[1], log_pred, 'Logistic')]:
    ax.scatter(default['balance'], default['default_num'], alpha=0.05, c='steelblue', s=5)
    ax.plot(bal_range, pred, 'r-', lw=2)
    ax.axhline(0, color='gray', lw=0.5); ax.axhline(1, color='gray', lw=0.5)
    ax.set_xlabel('Balance'); ax.set_ylabel('P(default=Yes)')
    ax.set_title(title)
plt.tight_layout(); plt.show()
```

---
## Slide 6 — เมื่อ Linear Regression พอใช้ได้?
**Key Message**: ถ้า class probabilities ไม่ชน 0 หรือ 1 — OLS และ Logistic ให้ผลใกล้เคียงกัน

**กรณีที่ OLS พอใช้ได้**:
- Binary outcome ที่ไม่ imbalanced มาก (p ≈ 0.3–0.7)
- ช่วงของ X ไม่กว้างมากจนทำให้ ŷ ออกนอก [0,1]
- ต้องการความเร็วและ interpretability มากกว่า accuracy

**LPM (Linear Probability Model)**:
- ใช้กันใน Econometrics — โดยยอมรับว่า ŷ อาจออกนอก [0,1]
- ยังนิยมใน social science เพราะ interpret coefficient ได้ง่าย

**Data Science standard**: ใช้ Logistic Regression เป็น default สำหรับ binary classification

---
## Slide 7 — Link Functions
**Key Message**: Logistic Regression = Linear Regression กับ logit link function — ส่วนหนึ่งของ GLM

**Generalized Linear Models (GLM)**:

| Distribution | Link | Model | ใช้เมื่อ |
|-------------|------|-------|---------|
| Normal | Identity | OLS | continuous Y |
| Binomial | Logit | **Logistic** | binary Y |
| Poisson | Log | Poisson Regression | count Y |

$$\text{Logit link}: \log\frac{p}{1-p} = \beta_0 + \beta_1 X$$

→ ทำให้ linear predictor map ไปเป็น probability ผ่าน sigmoid function

---
## Slide 8 — Summary
**Key Message**: Linear Regression fails สำหรับ classification เพราะ predictions ออกนอก [0,1] — ต้องใช้ Logistic

**สิ่งที่เรียนรู้วันนี้**:
- **ปัญหา 1**: OLS ให้ ŷ นอก [0,1] → ไม่ใช่ probability
- **ปัญหา 2**: Multi-class encoding เป็น arbitrary → ผลต่างกันตาม encoding
- **Logistic Regression**: sigmoid function รับประกัน p̂(X) ∈ [0,1]
- **LPM**: OLS ยังใช้ได้บ้างใน econometrics แต่ไม่ใช่ DS standard

**สัปดาห์ต่อไป — Slide 3**: Logistic Regression model — sigmoid, log-odds, MLE
