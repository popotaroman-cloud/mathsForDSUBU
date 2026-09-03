# Note 2: Inference for SLR Coefficients, R², and RSE
> Week 8 | CLO3 | ISLP Reference: Ch.3 Section 3.1.2, 3.1.3

---

## บทนำ

หลังจากที่เราประมาณค่า $\hat{\beta}_0$ และ $\hat{\beta}_1$ ได้แล้ว คำถามต่อไปคือ: **"เราเชื่อถือค่าเหล่านี้ได้มากแค่ไหน?"** และ **"model ของเราดีแค่ไหน?"** สัปดาห์นี้เราจะเรียนเครื่องมือ 2 กลุ่ม: (1) **Inference** สำหรับสัมประสิทธิ์ — Standard Error, Confidence Interval, t-statistic, p-value และ (2) **Model Fit** — RSE และ R². ทักษะเหล่านี้จำเป็นในการอ่านและตีความ output ของ `statsmodels` ซึ่งเป็นทักษะหลักของ Data Scientist. การเข้าใจ p-value และ R² ช่วยให้ตัดสินใจได้ว่า model มีประโยชน์หรือไม่ และต้องปรับปรุงอะไร.

---

## 8.3 Assessing Accuracy of Coefficient Estimates  *(ISLP 3.1.2)*

ในส่วนนี้เราจะวัดความไม่แน่นอนของ $\hat{\beta}_0$ และ $\hat{\beta}_1$ เพื่อทำ inference ว่า X มีความสัมพันธ์จริงกับ Y หรือไม่

### Standard Errors (SE)

ถ้าเราสุ่มตัวอย่างซ้ำหลาย ๆ ครั้ง $\hat{\beta}_1$ จะได้ค่าต่างกันในแต่ละครั้ง SE วัดว่า $\hat{\beta}_1$ มีความแปรปรวนมากแค่ไหน:

$$\text{SE}(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$\text{SE}(\hat{\beta}_0)^2 = \sigma^2 \left[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}\right]$$

โดยเราประมาณ $\sigma^2$ ด้วย **Residual Standard Error (RSE)**:

$$\hat{\sigma} = \text{RSE} = \sqrt{\frac{\text{RSS}}{n-2}}$$

**สังเกต**: SE($\hat{\beta}_1$) ลดลงเมื่อ:
- $n$ มากขึ้น (ข้อมูลเยอะขึ้น)
- $\sum(x_i - \bar{x})^2$ มากขึ้น (X มี spread มาก)
- $\sigma^2$ น้อยลง (ข้อมูลมี noise น้อย)

### 95% Confidence Interval

ช่วงความเชื่อมั่น 95% สำหรับ $\beta_1$:

$$\hat{\beta}_1 \pm 2 \cdot \text{SE}(\hat{\beta}_1)$$

(ใช้ค่า $t_{n-2, 0.025}$ แทน 2 เมื่อต้องการความแม่นยำสูงกว่า)

**Interpretation**: ถ้าเราสุ่มตัวอย่างซ้ำ 100 ครั้ง ประมาณ 95 ครั้งที่ CI จะครอบ $\beta_1$ จริง

**ตัวอย่าง Advertising**: จาก ISLP Table 3.1  
$\hat{\beta}_1 = 0.0475$, SE = 0.0027  
95% CI: $0.0475 \pm 2(0.0027) = (0.0422, 0.0528)$  
→ เราค่อนข้างมั่นใจว่า TV budget เพิ่ม 1 พัน $ → Sales เพิ่ม 42.2–52.8 หน่วย

### Hypothesis Test: H₀: β₁ = 0

ถ้า $\beta_1 = 0$ แสดงว่า X ไม่มีความสัมพันธ์เชิงเส้นกับ Y

**t-statistic**:

$$t = \frac{\hat{\beta}_1 - 0}{\text{SE}(\hat{\beta}_1)} = \frac{\hat{\beta}_1}{\text{SE}(\hat{\beta}_1)}$$

ภายใต้ H₀: $t \sim t_{n-2}$

**p-value**: ความน่าจะเป็นที่จะเห็น |t| มากเท่านี้หรือมากกว่า ถ้า H₀ เป็นจริง

**Decision rule**: ถ้า p-value < 0.05 → reject H₀ → มีหลักฐานว่า $\beta_1 \neq 0$

**ตัวอย่าง Advertising**:  
$t = 0.0475/0.0027 = 17.67$  
p-value ≈ 0 → reject H₀ → TV มีความสัมพันธ์จริงกับ Sales อย่างมีนัยสำคัญ

### ตัวอย่าง: อ่าน statsmodels Summary Table

```
                 coef    std err          t      P>|t|     [0.025      0.975]
Intercept       7.0326      0.458     15.360      0.000      6.130       7.935
TV              0.0475      0.003     17.668      0.000      0.042       0.053
```

| คอลัมน์ | ความหมาย |
|--------|----------|
| `coef` | $\hat{\beta}$ — ค่าประมาณสัมประสิทธิ์ |
| `std err` | SE($\hat{\beta}$) |
| `t` | t-statistic = coef / std err |
| `P>|t|` | p-value (two-tailed) |
| `[0.025, 0.975]` | 95% CI (lower, upper) |

---

## 8.4 Assessing Accuracy of the Model  *(ISLP 3.1.3)*

ในส่วนนี้เราจะวัดว่า model โดยรวมดีแค่ไหน โดยใช้ RSE และ R²

### Residual Standard Error (RSE)

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n-2}} = \sqrt{\frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n-2}}$$

- RSE มีหน่วยเดียวกับ Y → interpret ได้ตรงกว่า
- RSE = average deviation ของ observation จาก regression line
- **ตัวอย่าง**: RSE = 3.26 (พันหน่วย Sales) → แต่ละ prediction ผิดเฉลี่ย 3,260 หน่วย
- Mean Sales = 14.02 พันหน่วย → percentage error = 3.26/14.02 ≈ 23.3%

### R-squared (R²)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$$

โดย **TSS** (Total Sum of Squares) = $\sum_{i=1}^{n}(y_i - \bar{y})^2$ วัดความแปรปรวนทั้งหมดใน Y

- $R^2 \in [0,1]$: fraction ของ variance ที่ model อธิบายได้
- $R^2 = 1$ → model perfect (RSS = 0)
- $R^2 = 0$ → model ไม่ดีกว่าการใช้ $\bar{y}$ เลย

**ใน SLR**: $R^2 = r^2$ (กำลังสองของ Pearson correlation coefficient)

**ตัวอย่าง Advertising**: $R^2 = 0.612$ → TV อธิบาย 61.2% ของ variance ใน Sales

### RSE vs R² — เมื่อไหรใช้อะไร?

| | RSE | R² |
|---|-----|-----|
| หน่วย | หน่วยเดียวกับ Y | ไม่มีหน่วย (0–1) |
| ใช้เมื่อ | ต้องการ absolute accuracy | ต้องการ relative fit |
| ค่าที่ "ดี" | ขึ้นกับ context | ใกล้ 1 ยิ่งดี |
| ข้อควรระวัง | เปรียบได้เฉพาะ dataset เดียวกัน | R² สูงไม่ได้แปลว่า model ดีเสมอ |

---

## Case Study: Advertising → Sales — การตีความผล

**Scenario**: ต้องการตัดสินใจว่า TV advertising มีผลต่อยอดขายจริงหรือไม่

**Data**: Advertising.csv — TV → Sales, n=200

**Method**: Fit SLR, ดู t-statistic, p-value, CI, RSE, R²

**Result** (จาก ISLP Table 3.1 และ 3.2):
- $\hat{\beta}_1 = 0.0475$, SE = 0.0027, t = 17.67, p ≈ 0 → **significant**
- 95% CI: (0.042, 0.053)
- RSE = 3.26 (หน่วย: พันหน่วย) — error เฉลี่ย ~23%
- $R^2 = 0.612$ — TV อธิบาย variance ได้ 61.2%

**Insight**: มีหลักฐานทางสถิติอย่างแข็งแกร่งว่า TV มีผลต่อ Sales แต่ R² = 0.612 บอกว่ายังขาด predictor อื่น ๆ → ต้องทำ MLR (Week 9)

```python
# ─── ประเมิน Model Fit: RSE และ R² ──────────────────────────────────────────
# วัตถุประสงค์: วัดว่า model ดีแค่ไหนทั้งในแง่ absolute (RSE) และ relative (R²)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('Advertising.csv')
model = smf.ols('Sales ~ TV', data=df).fit()

# ─── ดึงค่า RSE และ R² ────────────────────────────────────────────────────
# วัตถุประสงค์: เข้าใจว่า model explain variance ได้เท่าไหร่

rse = np.sqrt(model.ssr / model.df_resid)  # df_resid = n - 2
r_squared = model.rsquared
n = len(df)

print(f"n = {n}")
print(f"RSS = {model.ssr:.4f}")
print(f"RSE = {rse:.4f} (หน่วย Sales)")
print(f"R² = {r_squared:.4f}")
print(f"Mean Sales = {df['Sales'].mean():.4f}")
print(f"RSE เป็น % ของ mean = {rse/df['Sales'].mean()*100:.1f}%")

# ─── ดึงค่า t-statistic และ p-value ────────────────────────────────────────
# วัตถุประสงค์: ตรวจสอบว่า β₁ มีนัยสำคัญทางสถิติหรือไม่

tstat = model.tvalues['TV']
pval = model.pvalues['TV']
ci = model.conf_int().loc['TV']

print(f"\nt-statistic สำหรับ TV: {tstat:.4f}")
print(f"p-value: {pval:.6f}")
print(f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")

if pval < 0.05:
    print("→ Reject H₀: TV มีความสัมพันธ์กับ Sales อย่างมีนัยสำคัญ")
```

---

## สรุป

| แนวคิด | สูตร | Python |
|--------|------|--------|
| Standard Error ($\hat{\beta}_1$) | $\sigma/\sqrt{\sum(x_i-\bar{x})^2}$ | `model.bse['TV']` |
| t-statistic | $\hat{\beta}_1/\text{SE}(\hat{\beta}_1)$ | `model.tvalues['TV']` |
| p-value | $P(\|T\| > \|t\|)$ | `model.pvalues['TV']` |
| 95% CI | $\hat{\beta}_1 \pm 2 \cdot \text{SE}$ | `model.conf_int()` |
| RSE | $\sqrt{\text{RSS}/(n-2)}$ | `np.sqrt(model.ssr/model.df_resid)` |
| R² | $1 - \text{RSS}/\text{TSS}$ | `model.rsquared` |

---

## เชื่อมกับสัปดาห์อื่น

- **← Week 8 Note 1**: ประมาณค่า $\hat{\beta}_0, \hat{\beta}_1$ ด้วย Least Squares
- **← Week 5**: Probability distributions, t-distribution — พื้นฐานของ t-test
- **→ Week 9**: F-statistic ใน MLR — ทดสอบว่าทุก predictor ไม่มีผลพร้อมกัน
- **→ Week 10**: Residual diagnostics — ตรวจสอบ assumptions ของ SLR
