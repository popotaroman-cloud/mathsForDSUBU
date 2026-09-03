# Note 1: Simple Linear Regression — Model และ Least Squares Estimation
> Week 8 | CLO3 | ISLP Reference: Ch.3 Section 3.1, 3.1.1

---

## บทนำ

สัปดาห์นี้เราเข้าสู่หัวใจของ supervised learning สำหรับ continuous response — **Simple Linear Regression (SLR)** ซึ่งเป็นพื้นฐานของ predictive modeling ทั้งหมดใน Data Science. เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถสร้าง SLR model, ประเมินความแม่นยำของสัมประสิทธิ์ด้วย Standard Error, t-statistic, p-value และวัด model fit ด้วย R² และ RSE ได้ด้วยตนเอง. SLR มีประโยชน์ใน Data Science อย่างมาก — ตั้งแต่การทำนายยอดขายจาก budget การโฆษณา ไปจนถึงการประมาณราคาบ้านจากขนาดพื้นที่. **หมายเหตุสำคัญ**: สัปดาห์นี้มี **สอบกลางภาค** ด้วย ซึ่งครอบคลุม CLO1 (Linear Algebra) และ CLO2 (Statistics, Bias-Variance, EDA) — เนื้อหา Weeks 1–7 ทั้งหมด. นักศึกษาจึงต้องบริหารเวลาให้ดีระหว่างการเรียน SLR ใหม่กับการทบทวนสำหรับการสอบ.

---

## 8.1 Simple Linear Regression Model  *(ISLP 3.1)*

ในส่วนนี้เราจะนิยาม SLR model เพื่อให้เข้าใจโครงสร้างทางคณิตศาสตร์ก่อนเริ่มประมาณค่าสัมประสิทธิ์

สมมติว่าเรามี **response variable** Y (เช่น Sales) และ **predictor variable** X (เช่น TV advertising budget) เราสมมติว่ามีความสัมพันธ์เชิงเส้น:

$$Y \approx \beta_0 + \beta_1 X$$

หรือในรูปที่รวม error term:

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

โดย:
- $\beta_0$ = **intercept** — ค่าของ Y เมื่อ X = 0
- $\beta_1$ = **slope** — การเปลี่ยนแปลงเฉลี่ยของ Y เมื่อ X เพิ่มขึ้น 1 หน่วย
- $\varepsilon$ = **error term** — ส่วนที่ model ไม่สามารถอธิบายได้

### Assumptions ของ SLR

1. **Linearity**: $E[Y|X] = \beta_0 + \beta_1 X$ — ความสัมพันธ์เป็นเส้นตรง
2. **Independence**: ข้อผิดพลาด $\varepsilon_i$ เป็นอิสระจากกัน
3. **Homoscedasticity**: $\text{Var}(\varepsilon) = \sigma^2$ คงที่ ไม่ขึ้นกับ X
4. **Normality**: $\varepsilon \sim N(0, \sigma^2)$

### ตัวอย่างจริง: Advertising Dataset (ISLP)

ในตัวอย่างของ ISLP (Chapter 3) มีข้อมูล 200 ตลาด (markets) แต่ละตลาดมีข้อมูล:
- TV: งบโฆษณาทางโทรทัศน์ (พัน $)
- Radio: งบโฆษณาทางวิทยุ (พัน $)
- Sales: ยอดขาย (พันหน่วย)

เราต้องการหา: **TV → Sales** เป็นอย่างไร?

---

## 8.2 Estimating the Coefficients: Least Squares  *(ISLP 3.1.1)*

ในส่วนนี้เราจะเรียนวิธีประมาณค่า $\hat{\beta}_0$ และ $\hat{\beta}_1$ จากข้อมูลจริง เพื่อให้ได้สมการ regression ที่ดีที่สุด

เรามีข้อมูล $n$ คู่: $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$

ค่าที่ model ทำนาย (fitted value): $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$

**Residual** ของแต่ละข้อมูล: $e_i = y_i - \hat{y}_i$

### Residual Sum of Squares (RSS)

เราต้องการหา $\hat{\beta}_0, \hat{\beta}_1$ ที่ทำให้ **RSS น้อยที่สุด**:

$$\text{RSS} = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

### สูตร Least Squares Estimates (Closed Form)

นำ partial derivative ของ RSS เทียบกับ $\hat{\beta}_0$ และ $\hat{\beta}_1$ แล้วตั้งให้เท่ากับ 0:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

โดย $\bar{x} = \frac{1}{n}\sum x_i$ และ $\bar{y} = \frac{1}{n}\sum y_i$

### ความเชื่อมโยงกับ Week 3 (Normal Equations)

สูตรนี้เชื่อมโยงกับ **Normal Equations** ที่เรียนใน Week 3:

$$\mathbf{X}^\top \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^\top \mathbf{y}$$

ใน SLR, $\mathbf{X}$ มีขนาด $n \times 2$ (คอลัมน์ 1s และคอลัมน์ X) และ solution คือ closed-form เดียวกับ formula ข้างบน

### ตัวอย่างคำนวณ Step-by-Step

สมมติมีข้อมูล 5 ตลาดเล็ก:

| i | TV ($x_i$) | Sales ($y_i$) |
|---|-----------|--------------|
| 1 | 10 | 8.5 |
| 2 | 20 | 10.5 |
| 3 | 30 | 12.0 |
| 4 | 40 | 13.5 |
| 5 | 50 | 16.0 |

**Step 1**: คำนวณ mean  
$\bar{x} = (10+20+30+40+50)/5 = 30$  
$\bar{y} = (8.5+10.5+12.0+13.5+16.0)/5 = 12.1$

**Step 2**: คำนวณ numerator ของ $\hat{\beta}_1$  
$\sum(x_i-\bar{x})(y_i-\bar{y}) = (-20)(-3.6) + (-10)(-1.6) + (0)(-0.1) + (10)(1.4) + (20)(3.9)$  
$= 72 + 16 + 0 + 14 + 78 = 180$

**Step 3**: คำนวณ denominator  
$\sum(x_i-\bar{x})^2 = 400 + 100 + 0 + 100 + 400 = 1000$

**Step 4**: คำนวณ $\hat{\beta}_1$ และ $\hat{\beta}_0$  
$\hat{\beta}_1 = 180/1000 = 0.18$  
$\hat{\beta}_0 = 12.1 - 0.18 \times 30 = 12.1 - 5.4 = 6.7$

**Fitted model**: $\hat{\text{Sales}} = 6.7 + 0.18 \times \text{TV}$

**Interpretation**: เมื่อ budget TV เพิ่มขึ้น 1 พัน $ ยอดขายจะเพิ่มขึ้นเฉลี่ย 0.18 พันหน่วย (= 180 หน่วย)

---

## Case Study: Advertising → Sales (Advertising.csv)

**Scenario**: บริษัทต้องการรู้ว่าการลงทุนโฆษณาทางโทรทัศน์มีผลต่อยอดขายอย่างไร

**Data**: Advertising.csv — 200 markets, TV budget (พัน $), Sales (พันหน่วย)

**Method**: Simple Linear Regression — TV เป็น predictor, Sales เป็น response

**Result**: จาก ISLP (Table 3.1):
- $\hat{\beta}_0 = 7.03$ (intercept: ยอดขายพื้นฐานแม้ไม่โฆษณา TV)
- $\hat{\beta}_1 = 0.0475$ (slope: TV เพิ่ม 1 พัน $ → Sales เพิ่ม 47.5 หน่วย)
- $R^2 = 0.612$ — TV อธิบาย 61.2% ของ variance ใน Sales ได้

**Insight**: มีหลักฐานชัดเจนว่า TV advertising มีความสัมพันธ์เชิงบวกกับยอดขาย แต่ R² = 0.612 บ่งชี้ว่ายังมี predictor อื่น (เช่น Radio, Newspaper) ที่ควรนำมาพิจารณา → Week 9 MLR

```python
# ─── Case Study: SLR บน Advertising Dataset ─────────────────────────────────
# วัตถุประสงค์: ประมาณค่า β̂₀ และ β̂₁ จากข้อมูลจริง และแสดง fitted line

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# โหลดข้อมูล
df = pd.read_csv('Advertising.csv')

# Fit SLR ด้วย statsmodels
model = smf.ols('Sales ~ TV', data=df).fit()
print(model.summary())

# ─── ดึงค่าสัมประสิทธิ์ ────────────────────────────────────────────────────
# วัตถุประสงค์: แสดงค่า β̂₀ และ β̂₁ ที่ประมาณได้

beta_0 = model.params['Intercept']   # 7.03
beta_1 = model.params['TV']           # 0.0475
print(f"β̂₀ = {beta_0:.4f}")
print(f"β̂₁ = {beta_1:.4f}")

# ─── Plot Scatter + Fitted Line ────────────────────────────────────────────
# วัตถุประสงค์: แสดงภาพความสัมพันธ์ระหว่าง TV กับ Sales พร้อม regression line

plt.figure(figsize=(8, 5))
plt.scatter(df['TV'], df['Sales'], alpha=0.5, label='Data points')
x_range = np.linspace(df['TV'].min(), df['TV'].max(), 100)
plt.plot(x_range, beta_0 + beta_1 * x_range, 'r-', linewidth=2,
         label=f'ŷ = {beta_0:.2f} + {beta_1:.4f}·TV')
plt.xlabel('TV Budget (พัน $)')
plt.ylabel('Sales (พันหน่วย)')
plt.title('Simple Linear Regression: TV → Sales')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## สรุป

| แนวคิด | สูตร | Python |
|--------|------|--------|
| SLR Model | $Y = \beta_0 + \beta_1 X + \varepsilon$ | `smf.ols('Y ~ X', data=df)` |
| RSS | $\sum(y_i - \hat{y}_i)^2$ | `model.ssr` |
| $\hat{\beta}_1$ | $\text{Cov}(X,Y)/\text{Var}(X)$ | `model.params['X']` |
| $\hat{\beta}_0$ | $\bar{y} - \hat{\beta}_1\bar{x}$ | `model.params['Intercept']` |
| Fitted value | $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ | `model.fittedvalues` |

---

## เชื่อมกับสัปดาห์อื่น

- **← Week 3**: Normal Equations $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$ คือรากฐานของ Least Squares
- **← Week 7**: EDA (scatter plot, correlation) เตรียมข้อมูลก่อน fit model
- **→ Week 8 Note 2**: การประเมินความแม่นยำของ $\hat{\beta}$ ด้วย SE, t-statistic, R², RSE
- **→ Week 9**: Multiple Linear Regression — เพิ่ม predictor หลายตัว
