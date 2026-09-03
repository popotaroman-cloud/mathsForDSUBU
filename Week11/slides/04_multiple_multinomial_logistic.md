# Slide Deck: Multiple Logistic Regression และ Multinomial
> Week 11 | CLO3 | ISLP Ch.4.3.4–4.3.5 | 9 slides

---
## Slide 1 — Title
**Multiple Logistic Regression และ Multinomial Logistic**  
Week 11 | CLO3 | ISLP 4.3.4–4.3.5  
LLo: สร้าง Multiple Logistic Regression, ตีความ confounding และ Softmax model ได้

---
## Slide 2 — Multiple Logistic Regression Model
**Key Message**: Multiple Logistic = เพิ่ม predictors ใน log-odds — ตีความเหมือน MLR โดยยึดตัวอื่นคงที่

**Model**:
$$\log\frac{p(X)}{1-p(X)} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$$

$$p(X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}$$

**ตัวอย่าง Default**:
$$\log\frac{p(\text{default})}{1-p(\text{default})} = \beta_0 + \beta_1\text{balance} + \beta_2\text{income} + \beta_3\text{student}$$

**Interpretation**: β̂ⱼ = ผลของ Xⱼ ต่อ log-odds **โดยยึด predictors อื่นคงที่** — เหมือน MLR

---
## Slide 3 — Python: Multiple Logistic Regression
**Key Message**: formula ใน statsmodels เหมือน OLS ทุกประการ แค่เปลี่ยน `ols` → `logit`

```python
import statsmodels.formula.api as smf
import pandas as pd

default = pd.read_csv('Default.csv')
default['default_num'] = (default['default'] == 'Yes').astype(int)
default['student_num'] = (default['student'] == 'Yes').astype(int)

# ─── Multiple Logistic Regression ────────────────────────
# วัตถุประสงค์: fit model ที่ control ทั้ง balance, income, student พร้อมกัน
m_multi = smf.logit('default_num ~ balance + income + student_num', data=default).fit()
print(m_multi.summary())
```

```
              coef     std err    z       P>|z|
-----------------------------------------------
Intercept  -10.869      0.492  -22.08   0.000
balance      0.006      0.000   24.74   0.000
income       0.000      0.000    0.37   0.712
student_num -0.647      0.236   -2.74   0.006
```

**สังเกต**: income p = 0.712 → NOT significant ใน MLR แม้ SLR อาจ significant

---
## Slide 4 — Confounding: Student Effect
**Key Message**: Student = Yes → lower default rate ใน SLR แต่ higher ใน MLR — confounding จาก balance

**SLR: Default ~ Student**:
```
student_num   coef = 0.405 (p < 0.001)  ← student default MORE?
```
→ Student = Yes → default probability สูงกว่า

**MLR: Default ~ Balance + Income + Student**:
```
student_num   coef = -0.647 (p = 0.006)  ← student default LESS!
```
→ เมื่อ control balance → Student = Yes → default probability ต่ำกว่า

**ทำไม?**:
- นักศึกษามี balance สูงกว่า non-student เฉลี่ย
- Balance สูง → default probability สูง → SLR absorbs balance effect เข้า student
- MLR แยกออก: holding balance constant → student มี default probability ต่ำกว่า

[FIGURE: boxplot balance by student status — นักศึกษามี balance สูงกว่า]

---
## Slide 5 — Multinomial Logistic Regression (K > 2)
**Key Message**: Softmax function ขยาย sigmoid ให้รองรับ K class — เลือก reference class แล้ว fit K-1 models

**Problem**: K > 2 classes — เช่น ผลการวินิจฉัย: Low Risk / Medium Risk / High Risk

**Multinomial Model** (Softmax):
$$\Pr(Y=k|X) = \frac{e^{\beta_{0k} + \boldsymbol{\beta}_k^T\mathbf{X}}}{\sum_{l=1}^{K}e^{\beta_{0l} + \boldsymbol{\beta}_l^T\mathbf{X}}}$$

- Reference class K: log-odds vs class K = β₀ₖ + βₖᵀX
- K−1 sets of coefficients (เหมือน dummy variable)
- Σ probabilities = 1 (valid distribution)

---
## Slide 6 — Python: Multinomial Logistic
**Key Message**: sklearn `multi_class='multinomial'` และ statsmodels `MNLogit` ใช้ได้ทั้งคู่

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# ─── ตัวอย่าง: classify credit risk level ──────────────────
# วัตถุประสงค์: multinomial logistic สำหรับ 3-class problem
# สร้าง synthetic 3-class data จาก Default
default['risk'] = pd.cut(default['balance'],
                         bins=[0, 1000, 2000, 5000],
                         labels=['Low', 'Medium', 'High'])
default_clean = default.dropna(subset=['risk'])

# sklearn multinomial
lr_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_multi.fit(default_clean[['balance', 'income']], default_clean['risk'])

# predict probabilities สำหรับ 3 classes
proba = lr_multi.predict_proba([[1500, 50000]])
print('P(Low):', round(proba[0,0],4))
print('P(Medium):', round(proba[0,1],4))
print('P(High):', round(proba[0,2],4))
```

---
## Slide 7 — Odds Ratios: ตีความง่ายขึ้น
**Key Message**: e^β̂ = Odds Ratio — ตีความได้ชัดเจนกว่า coefficient ดิบ

**Odds Ratio**:
$$OR = e^{\hat{\beta}_1} = \frac{\text{Odds}(X=x+1)}{\text{Odds}(X=x)}$$

**ตัวอย่าง Default ~ Balance + Student**:
- β̂_balance = 0.0055 → OR = e^0.0055 = 1.0055
  - Balance เพิ่ม $1 → odds of default คูณ 1.0055 (+0.55%)
  - Balance เพิ่ม $100 → OR = e^0.55 = 1.73 (+73%)
- β̂_student = -0.647 → OR = e^{-0.647} = 0.52
  - Student มี odds of default เพียง 52% ของ non-student (เมื่อ control balance)

```python
# ─── แสดง Odds Ratios ──────────────────────────────────────
# วัตถุประสงค์: ตีความ coefficient ผ่าน OR ที่เข้าใจง่ายกว่า
import numpy as np
OR = np.exp(m_multi.params)
CI_OR = np.exp(m_multi.conf_int())
pd.concat([OR, CI_OR], axis=1).round(3)
```

---
## Slide 8 — Predict Probability บน New Data
**Key Message**: ขั้นตอน predict: สร้าง new DataFrame → predict → บอก class + probability

```python
# ─── Predict สำหรับ customer ใหม่ ──────────────────────────
# วัตถุประสงค์: วัตถุประสงค์จริง — score customer ว่าควร approve หรือไม่
new_customers = pd.DataFrame({
    'balance': [1000, 2000, 1500],
    'income': [30000, 50000, 40000],
    'student_num': [0, 1, 0]
})

probs = m_multi.predict(new_customers)
preds = (probs > 0.5).astype(int)

result = new_customers.copy()
result['P(default)'] = probs.round(4)
result['Prediction'] = ['Default' if p else 'No Default' for p in preds]
print(result)
```

---
## Slide 9 — Summary
**Key Message**: Multiple Logistic = Multiple Linear แต่ predict probability; Multinomial = Softmax

**สิ่งที่เรียนรู้วันนี้**:
- **Multiple Logistic**: log-odds linear ใน X₁,…,Xₚ — ตีความ "holding others fixed"
- **Confounding**: student example — ผลเปลี่ยนทิศทางเมื่อ control balance
- **Odds Ratio**: e^β̂ — ตีความง่ายกว่า coefficient ดิบ
- **Multinomial**: Softmax สำหรับ K > 2 classes

| Model | สูตร | Python |
|-------|------|--------|
| Binary Logistic | logit = β₀+β₁X | `smf.logit('y~x').fit()` |
| Multiple Logistic | logit = Σβⱼxⱼ | `smf.logit('y~x1+x2').fit()` |
| Multinomial | Softmax | `LogisticRegression(multi_class='multinomial')` |
| Odds Ratio | e^β̂ | `np.exp(model.params)` |

**สัปดาห์ต่อไป — Slide 5**: Classification Metrics — Confusion Matrix, Precision, Recall, F1
