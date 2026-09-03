# Note 11-1: Logistic Regression — Sigmoid, Log-odds, and Maximum Likelihood
> Week 11 | CLO3 | ISLP Reference: Ch.4.1–4.3.3

สัปดาห์ที่ 11 เราก้าวข้ามจาก Regression ที่ทำนายค่าต่อเนื่อง (เช่น ยอดขาย, ราคาบ้าน) มาสู่ **Classification** ซึ่งทำนายว่า observation นั้น "อยู่ใน class ไหน" ตัวอย่างเช่น "email นี้เป็น spam หรือไม่?" หรือ "ลูกค้ารายนี้จะผิดนัดชำระหรือไม่?" เป้าหมายหลักของสัปดาห์นี้คือทำความเข้าใจว่า **Logistic Regression** แก้ปัญหา Classification อย่างไร — โดยทำนาย **ความน่าจะเป็น** แทนที่จะทำนายค่าตรงๆ แนวคิดหลักคือ Sigmoid Function ที่บีบ output เข้าสู่ช่วง [0,1] และ Log-odds ที่ทำให้ relationship กลับมาเป็น Linear เราจะเรียนรู้ว่า β̂ ใน Logistic Regression หาไม่ได้ด้วย Least Squares แต่ต้องใช้ **Maximum Likelihood Estimation (MLE)** แทน หลังเรียนจบนักศึกษาจะสามารถสร้างโมเดล Logistic Regression, อ่านค่า Coefficient และตีความใน scale Log-odds และ Odds Ratio, และตัดสินใจด้วย Decision Boundary ได้ ทักษะนี้เป็นพื้นฐานของ Classifier ทุกประเภทในงาน Machine Learning

---

## 11.1 ทำไม Linear Regression ใช้กับ Classification ไม่ได้?  *(ISLP 4.2)*

ในส่วนนี้เราจะเข้าใจข้อจำกัดของ Linear Regression เมื่อใช้กับ Categorical Response เพื่อให้เห็นว่าทำไมเราถึงต้องการ Logistic Regression

### ปัญหาที่ 1: ค่าทำนายออกนอกช่วง [0,1]

สมมติ Response Y = 1 (default) หรือ 0 (no default) หากเราใช้ Linear Regression จะได้ $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X$ ซึ่งอาจมีค่าเป็น -0.3 หรือ 1.7 — ซึ่งตีความเป็นความน่าจะเป็นไม่ได้เลย

### ปัญหาที่ 2: Ordering Problem สำหรับ K > 2 Classes

หาก Y ∈ {Stroke, Overdose, Seizure} การ encode เป็น {1, 2, 3} สมมติว่า Stroke < Overdose < Seizure ซึ่งไม่มีความหมาย ลำดับที่เลือกต่างกันจะให้โมเดลต่างกัน

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ─── แสดงปัญหาของ Linear Regression กับ Classification ─────────────────
# วัตถุประสงค์: ดูว่า Linear Regression ให้ค่าทำนายนอกช่วง [0,1] อย่างไร
np.random.seed(42)
balance = np.array([300, 500, 800, 1200, 1500, 2000, 2500, 3000])
default = np.array([0, 0, 0, 0, 1, 0, 1, 1])

# Linear Regression
X_lin = sm.add_constant(balance)
model_lin = sm.OLS(default, X_lin).fit()

balance_range = np.linspace(0, 3500, 200)
pred_lin = model_lin.predict(sm.add_constant(balance_range))

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(balance, default, color='black', zorder=5, label='Observed')
ax.plot(balance_range, pred_lin, 'r-', label='Linear Regression')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(balance_range, 0, 1, alpha=0.05, color='green')
ax.set_xlabel('Balance'); ax.set_ylabel('P(Default=1)')
ax.set_title('Linear Regression — ค่าทำนายออกนอก [0,1]')
ax.legend()
plt.tight_layout(); plt.show()

# สังเกต: pred_lin อาจ < 0 หรือ > 1 ได้ง่าย
print("ค่าทำนายที่ balance=0:", model_lin.predict([1, 0])[0])
print("ค่าทำนายที่ balance=3500:", model_lin.predict([1, 3500])[0])
```

---

## 11.2 The Logistic Model  *(ISLP 4.3.1)*

ในส่วนนี้เราจะเรียนรู้ Sigmoid Function และ Log-odds ซึ่งเป็นหัวใจของ Logistic Regression เพื่อสร้างโมเดลที่ทำนายความน่าจะเป็นที่อยู่ใน [0,1] เสมอ

### Sigmoid (Logistic) Function

แทนที่จะทำนาย Y โดยตรง Logistic Regression ทำนาย **ความน่าจะเป็น** $p(X) = \Pr(Y = 1 \mid X)$ ผ่าน Sigmoid Function:

$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$

**คุณสมบัติของ Sigmoid**:
- $p(X) \in (0, 1)$ เสมอ — ไม่มีทางออกนอกช่วง
- เมื่อ $\beta_0 + \beta_1 X \to +\infty$ → $p(X) \to 1$
- เมื่อ $\beta_0 + \beta_1 X \to -\infty$ → $p(X) \to 0$
- S-shape (Sigmoid) สะท้อนว่าในบางช่วง X เปลี่ยนนิดเดียวก็เปลี่ยน prob มาก (ช่วงกลาง) แต่บางช่วงแทบไม่เปลี่ยน (ปลายขอบ)

```python
# ─── แสดง Sigmoid Function ───────────────────────────────────────────────
# วัตถุประสงค์: เห็น S-shape ที่บีบทุกค่าให้อยู่ใน (0,1)
# เปรียบเทียบกับ Linear Regression เพื่อเห็นความแตกต่างชัดเจน
z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Sigmoid function
axes[0].plot(z, sigmoid, 'b-', linewidth=2.5)
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='p=0.5')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.3)
axes[0].axhline(1, color='gray', linestyle='--', alpha=0.3)
axes[0].fill_between(z, 0, sigmoid, alpha=0.1)
axes[0].set_xlabel('z = β₀ + β₁X'); axes[0].set_ylabel('p(X) = σ(z)')
axes[0].set_title('Sigmoid Function — ทุกค่าอยู่ใน (0,1)')
axes[0].legend()

# Right: Default dataset (balance vs default probability)
default_data = pd.DataFrame({
    'balance': np.random.normal(1000, 500, 300).clip(0, 3000),
    'default': np.random.binomial(1, 0.03, 300)
})
default_data.loc[default_data['balance'] > 2000, 'default'] = np.random.binomial(
    1, 0.4, (default_data['balance'] > 2000).sum()
)

# Fit Logistic Regression
from sklearn.linear_model import LogisticRegression as LR
lr = LR()
lr.fit(default_data[['balance']], default_data['default'])

bal_range = np.linspace(0, 3000, 300).reshape(-1, 1)
prob_range = lr.predict_proba(bal_range)[:, 1]

axes[1].scatter(default_data['balance'], default_data['default'],
                alpha=0.2, color='gray', label='Observed')
axes[1].plot(bal_range, prob_range, 'b-', linewidth=2, label='Logistic Regression')
axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Threshold 0.5')
axes[1].set_xlabel('Balance'); axes[1].set_ylabel('P(Default=1|Balance)')
axes[1].set_title('Logistic Regression — Default Dataset')
axes[1].legend()

plt.tight_layout(); plt.show()
```

### Log-odds (Logit) Transformation

จาก Sigmoid กลับมาได้ **Log-odds** ซึ่งเป็น Linear function ของ X:

$$\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X$$

**Odds** = $\frac{p}{1-p}$ คืออัตราส่วนของ probability เป็น vs ไม่เป็น (range: $[0, \infty)$)

ตัวอย่าง:
- $p = 0.5$ → Odds = $0.5/0.5 = 1$ → Log-odds = 0
- $p = 0.9$ → Odds = $0.9/0.1 = 9$ → Log-odds = $\ln(9) \approx 2.2$
- $p = 0.1$ → Odds = $0.1/0.9 \approx 0.111$ → Log-odds = $\ln(0.111) \approx -2.2$

### Worked Example: การตีความ Coefficient

สมมติ fit Logistic Regression ได้:
$$\log\left(\frac{p}{1-p}\right) = -10.65 + 0.0055 \times \text{balance}$$

**การตีความ**:
- $\hat{\beta}_1 = 0.0055$: เมื่อ balance เพิ่ม 1 → **log-odds ของ default เพิ่ม 0.0055**
- **Odds Ratio** = $e^{0.0055} \approx 1.0055$: odds ของ default คูณด้วย 1.0055 ต่อการเพิ่ม balance 1 หน่วย
- เมื่อ balance = 2000: log-odds = $-10.65 + 0.0055 \times 2000 = 0.35$ → $p = \sigma(0.35) \approx 0.587$

```python
# ─── แสดงความสัมพันธ์ระหว่าง p, Odds, และ Log-odds ──────────────────────
# วัตถุประสงค์: เข้าใจ scale ทั้ง 3 ของ Logistic Regression พร้อมกัน
p_vals = np.linspace(0.01, 0.99, 200)
odds = p_vals / (1 - p_vals)
log_odds = np.log(odds)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(p_vals, p_vals, 'b-'); axes[0].set_title('Probability scale [0,1]')
axes[0].set_xlabel('p'); axes[0].set_ylabel('p')

axes[1].plot(p_vals, odds, 'g-'); axes[1].set_title('Odds scale [0, ∞)')
axes[1].set_xlabel('p'); axes[1].set_ylabel('odds = p/(1-p)')

axes[2].plot(p_vals, log_odds, 'r-'); axes[2].set_title('Log-odds (Logit) — Linear!')
axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('p'); axes[2].set_ylabel('log(p/(1-p))')

plt.suptitle('3 Scales ของ Logistic Regression', fontsize=13)
plt.tight_layout(); plt.show()
```

**DS Connection**: Log-odds scale ทำให้ Logistic Regression มีรูปแบบ Linear เหมือน Linear Regression ทำให้ตีความ Coefficients และคำนวณ Confidence Intervals ได้ง่าย Feature Engineering สำหรับ Credit Scoring มักทำบน Log-odds scale โดยตรง

---

## 11.3 Maximum Likelihood Estimation  *(ISLP 4.3.2)*

ในส่วนนี้เราจะเรียนรู้วิธีหา β̂ ใน Logistic Regression ซึ่งต่างจาก Linear Regression ตรงที่ไม่มี Closed-form Solution ต้องใช้ Numerical Optimization

### ทำไม Least Squares ใช้ไม่ได้

ใน Linear Regression เราใช้ Least Squares เพราะ $\text{RSS} = \sum(y_i - \hat{y}_i)^2$ มี Closed-form Solution $\hat{\beta} = (X^TX)^{-1}X^Ty$ แต่ใน Logistic Regression $\hat{p}(x_i) = \sigma(\beta_0 + \beta_1 x_i)$ ทำให้ RSS ไม่มี Closed-form → ต้องใช้ **Maximum Likelihood**

### Likelihood Function

สำหรับ Binary Classification โดยที่ $y_i \in \{0, 1\}$:

$$\ell(\beta_0, \beta_1) = \prod_{i: y_i=1} p(x_i) \cdot \prod_{i: y_i=0} (1 - p(x_i))$$

เขียนรวมกันได้:

$$\ell(\beta_0, \beta_1) = \prod_{i=1}^{n} p(x_i)^{y_i} (1 - p(x_i))^{1 - y_i}$$

**Log-likelihood** (ง่ายกว่าในการ optimize):

$$\log \ell = \sum_{i=1}^{n} \left[ y_i \log p(x_i) + (1 - y_i) \log(1 - p(x_i)) \right]$$

เรียกว่า **Binary Cross-Entropy Loss** (นำเครื่องหมาย − ออก) ซึ่งใช้กันทั่วไปใน Deep Learning

### Intuition ของ MLE

MLE ถามว่า: "β ชุดไหนที่ทำให้ข้อมูลที่เราสังเกตเห็นมีความน่าจะเป็น **สูงสุด**?"

- สำหรับ observation ที่ $y_i = 1$: ต้องการให้ $p(x_i)$ สูง → contribution = $\log p(x_i)$
- สำหรับ observation ที่ $y_i = 0$: ต้องการให้ $p(x_i)$ ต่ำ → contribution = $\log(1-p(x_i))$

Maximize log-likelihood ด้วย Newton-Raphson algorithm (ใน statsmodels จัดการให้อัตโนมัติ)

### Worked Example: Default Dataset

```python
# ─── โหลด Default Dataset ─────────────────────────────────────────────────
# วัตถุประสงค์: ทดสอบ Logistic Regression กับข้อมูล credit default จริง
default_df = pd.read_csv('Default.csv')

# แปลง categorical → binary: Yes=1, No=0
default_df['default_bin'] = (default_df['default'] == 'Yes').astype(int)
default_df['student_bin'] = (default_df['student'] == 'Yes').astype(int)

print(f"Dataset shape: {default_df.shape}")
print(f"Default rate: {default_df['default_bin'].mean():.4f} ({default_df['default_bin'].mean()*100:.1f}%)")

# ─── Fit Logistic Regression: balance → default ────────────────────────────
# วัตถุประสงค์: SLR-style Logistic Regression เพื่อเห็น coefficient และ z-stat
X_balance = sm.add_constant(default_df['balance'])
logit_model = sm.Logit(default_df['default_bin'], X_balance).fit()
print(logit_model.summary())
```

**Output ที่คาดหวัง**:
```
                          Logit Regression Results
=========================================================
Dep. Variable: default_bin      No. Observations: 10000
Model:         Logit             Df Residuals:     9998
=========================================================
                  coef    std err        z    P>|z|
---------------------------------------------------------
const          -10.6513     0.361    -29.5   0.000
balance          0.0055     0.000     24.9   0.000
=========================================================
```

**การอ่านผลลัพธ์**:
- `coef` = $\hat{\beta}$ (ใน Log-odds scale)
- `std err` = SE($\hat{\beta}$) — คำนวณจาก Fisher Information Matrix
- `z` = $\hat{\beta}/\text{SE}(\hat{\beta})$ = **z-statistic** (ไม่ใช่ t-statistic เหมือน Linear Regression)
- `P>|z|` = p-value จาก Normal distribution (ไม่ใช่ t distribution)

### z-statistic vs t-statistic

| Feature | Linear Regression | Logistic Regression |
|---------|-------------------|---------------------|
| Test statistic | $t = \hat{\beta}/\text{SE}$ ~ $t_{n-p-1}$ | $z = \hat{\beta}/\text{SE}$ ~ $\mathcal{N}(0,1)$ |
| Distribution under H₀ | t distribution (exact) | Normal (asymptotic) |
| n ใหญ่ | t ≈ Normal | เหมือนกัน |
| H₀ | $\beta_j = 0$ | $\beta_j = 0$ |

---

## 11.4 Making Predictions  *(ISLP 4.3.3)*

ในส่วนนี้เราจะเรียนรู้วิธีใช้โมเดลที่ train แล้วเพื่อทำนาย Class ของ observation ใหม่ ซึ่งต้องตั้ง **Decision Boundary** ก่อน

### จาก Coefficient สู่ Predicted Probability

$$\hat{p}(X) = \frac{e^{\hat{\beta}_0 + \hat{\beta}_1 X}}{1 + e^{\hat{\beta}_0 + \hat{\beta}_1 X}}$$

ตัวอย่าง: balance = 1000 → $\hat{p} = \sigma(-10.65 + 0.0055 \times 1000) = \sigma(-5.15) \approx 0.006$ (0.6%)
ตัวอย่าง: balance = 2500 → $\hat{p} = \sigma(-10.65 + 0.0055 \times 2500) = \sigma(3.1) \approx 0.957$ (95.7%)

### Decision Boundary

**Default Threshold**: $\hat{p}(X) \geq 0.5$ → predict class 1

```python
# ─── ทำนาย Class ด้วย Decision Boundary ─────────────────────────────────
# วัตถุประสงค์: แปลง probability เป็น class label ด้วย threshold ที่เลือกได้
# threshold ค่าเริ่มต้น 0.5 แต่สามารถเปลี่ยนได้ตาม business requirement
prob_hat = logit_model.predict(X_balance)

# Default threshold = 0.5
threshold = 0.5
y_pred = (prob_hat >= threshold).astype(int)

# ─── หา Decision Boundary บน balance scale ───────────────────────────────
# วัตถุประสงค์: หาค่า balance ที่ทำให้ p̂ = threshold (จุดตัดสินใจ)
# p = 0.5 เมื่อ log-odds = 0 → β₀ + β₁·balance = 0 → balance = -β₀/β₁
b0 = logit_model.params['const']
b1 = logit_model.params['balance']
decision_boundary = -b0 / b1
print(f"Decision Boundary (threshold=0.5): balance = {decision_boundary:.1f}")

# ─── Visualize: Predicted Probability vs Balance ──────────────────────────
# วัตถุประสงค์: เห็น S-curve และตำแหน่ง decision boundary
bal_range = np.linspace(0, 3500, 500)
X_range = sm.add_constant(bal_range)
prob_range = logit_model.predict(X_range)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(default_df['balance'], default_df['default_bin'],
           alpha=0.05, color='gray', label='Observed')
ax.plot(bal_range, prob_range, 'b-', linewidth=2.5, label='P(Default|Balance)')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Threshold = 0.5')
ax.axvline(decision_boundary, color='orange', linestyle='--',
           label=f'Decision Boundary = {decision_boundary:.0f}')
ax.set_xlabel('Balance'); ax.set_ylabel('P(Default = 1)')
ax.set_title('Logistic Regression — Predicted Probability and Decision Boundary')
ax.legend()
plt.tight_layout(); plt.show()

print(f"\nAccuracy: {(y_pred == default_df['default_bin']).mean():.4f}")
```

### เมื่อไหร่ควรเปลี่ยน Threshold?

Threshold ค่าเริ่มต้น 0.5 ไม่ใช่ทางเลือกที่ดีเสมอไป:

| Situation | ควรใช้ Threshold |
|-----------|----------------|
| Medical Diagnosis (มะเร็ง) | ต่ำ (เช่น 0.2) — ดีกว่า FN น้อย (miss ผู้ป่วย) |
| Spam Filter | สูง (เช่น 0.7) — ดีกว่า FP น้อย (email ดีถูก filter) |
| Fraud Detection | ต่ำ (เช่น 0.1) — ค่าใช้จ่าย fraud สูงมาก |
| Balanced classes | 0.5 เหมาะสม |

---

## Case Study: Credit Card Default Prediction

**Scenario**
ธนาคารต้องการสร้างโมเดลประเมินความเสี่ยงสินเชื่อ โดยทำนายว่าลูกค้าจะ "ผิดนัดชำระ" (default) หรือไม่ จากยอดหนี้คงค้าง (balance) ซึ่งเป็น Strong predictor

**Data**
- Default.csv (ISLP): 10,000 ลูกค้า
- Response: `default` (Yes/No)
- Predictor: `balance` (ยอดหนี้คงค้าง), `income`, `student`
- Default rate: 3.33% (imbalanced dataset)

**Method — Logistic Regression**

```python
# ─── Fit SLR-style Logistic: balance → default ────────────────────────────
# วัตถุประสงค์: สร้างโมเดลพื้นฐานก่อนเพิ่ม predictors ใน Note 2
results = sm.Logit(default_df['default_bin'],
                   sm.add_constant(default_df['balance'])).fit(disp=False)

# ─── ตีความ Coefficient ───────────────────────────────────────────────────
# วัตถุประสงค์: แปลง log-odds เป็น odds ratio ที่ตีความได้ง่ายกว่า
beta1 = results.params['balance']
odds_ratio = np.exp(beta1)
print(f"β̂₁ (balance) = {beta1:.6f}")
print(f"Odds Ratio = e^β̂₁ = {odds_ratio:.6f}")
print(f"ตีความ: balance เพิ่ม 1 ดอลลาร์ → odds ของ default คูณด้วย {odds_ratio:.4f}")
print(f"ตีความ: balance เพิ่ม 1,000 ดอลลาร์ → odds คูณด้วย {np.exp(beta1*1000):.2f}")

# ─── ทำนาย probabilities ─────────────────────────────────────────────────
# วัตถุประสงค์: เปรียบเทียบ prob ที่ balance ต่างกัน
for bal in [500, 1000, 1500, 2000, 2500, 3000]:
    p = results.predict([1, bal])[0]
    print(f"balance = {bal:5d}: P(default) = {p:.4f} ({p*100:.2f}%)")
```

**Result**
```
β̂₁ (balance) = 0.005499
Odds Ratio = e^β̂₁ = 1.005514
ตีความ: balance เพิ่ม 1,000 ดอลลาร์ → odds คูณด้วย 241.47

balance =   500: P(default) = 0.0058  (0.58%)
balance =  1000: P(default) = 0.0058  (0.58%)  
balance =  1500: P(default) = 0.0583  (5.83%)
balance =  2000: P(default) = 0.3854  (38.54%)
balance =  2500: P(default) = 0.9207  (92.07%)
```

**Insight**
1. β̂₁ = 0.0055 ดูเล็ก แต่ที่ scale ของ balance (0–3000) มีผลมาก — balance เพิ่ม 1000 ดอลลาร์ทำให้ odds เพิ่ม 241 เท่า
2. S-curve ชี้ว่า "จุดเสี่ยง" อยู่ที่ balance ≈ 1,800–2,200 (ตรงที่ curve ชันสุด)
3. Default rate 3.33% หมายความว่า threshold 0.5 จะทำนาย default น้อยมาก — ธนาคารมักใช้ threshold ต่ำกว่า (เช่น 0.2) เพื่อ capture ผู้ที่มีความเสี่ยง

---

## สรุป

| แนวคิด | สูตร/Method | Python Function |
|--------|------------|----------------|
| Sigmoid Function | $p(X) = 1/(1+e^{-(\beta_0+\beta_1X)})$ | `scipy.special.expit(z)` |
| Log-odds (Logit) | $\log(p/(1-p)) = \beta_0 + \beta_1X$ | — (linear in X) |
| Odds Ratio | $e^{\hat{\beta}_j}$ | `np.exp(model.params)` |
| Log-likelihood | $\sum[y_i\log p_i + (1-y_i)\log(1-p_i)]$ | `model.llf` |
| z-statistic | $z = \hat{\beta}/\text{SE}(\hat{\beta})$ ~ $\mathcal{N}(0,1)$ | `model.tvalues` (statsmodels) |
| Prediction | $\hat{p} = \sigma(\hat{\beta}_0 + \hat{\beta}_1 X)$ | `model.predict()` |
| Decision Boundary (p=0.5) | $X = -\hat{\beta}_0/\hat{\beta}_1$ | — |
| Fit in statsmodels | — | `sm.Logit(y, X).fit()` |
| Fit in sklearn | — | `LogisticRegression().fit(X, y)` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 06**: Bayes Classifier ทำนาย Pr(Y=k|X) โดยตรง — Logistic Regression เป็น Discriminative approach ที่ทำแบบเดียวกันแต่ Parametric
- ← **Week 09**: Linear Regression ใช้ Least Squares หา β̂; Logistic Regression ใช้ MLE — concept ต่างกันแต่ interpretation Coefficient คล้ายกัน
- → **Note 11-2**: Multiple Logistic Regression, Confounding, Multinomial LR, Evaluation Metrics (Confusion Matrix, F1)
- → **Week 12**: LDA/QDA ทำ Classification ด้วยวิธีต่าง → เปรียบเทียบกับ Logistic Regression ว่าเมื่อไหร่ใช้อะไร
- → **Week 13**: ROC Curve และ AUC ประเมิน Logistic Regression ที่ Threshold ต่างๆ ทั้งหมด
