# Slide Deck: SLR Model และ Assumptions
> Week 08 | CLO3 | ISLP Ch.3.1 | 10 slides

---
## Slide 1 — Title
**Simple Linear Regression: Model และ Assumptions**  
Week 8 | CLO3 | ISLP Chapter 3.1  
LLo: สร้าง SLR และอธิบาย assumptions ของ model ได้

---
## Slide 2 — Week Overview
**Key Message**: SLR คือสะพานเชื่อมระหว่าง Linear Algebra ที่เรียนไปกับ Statistical Learning ที่กำลังจะเรียน

สัปดาห์นี้เราจะเรียนรู้ **Simple Linear Regression (SLR)** ซึ่งเป็นโมเดล supervised learning พื้นฐานที่สุดและสำคัญที่สุดใน Data Science SLR ตอบคำถามว่า "X มีความสัมพันธ์เชิงเส้นกับ Y หรือไม่ และถ้ามี ความสัมพันธ์นั้นแข็งแกร่งแค่ไหน?" เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถสร้าง SLR ได้ 3 วิธี (manual, sklearn, statsmodels) อ่านค่า SE, t-statistic, p-value, CI, R² และ RSE ได้อย่างถูกต้อง และนำผลไปตีความในเชิงธุรกิจได้ ทักษะนี้ใช้ในการวิเคราะห์ผลของงบโฆษณาต่อยอดขาย การพยากรณ์ราคา และการวิจัยทุกสาขาที่ต้องการเข้าใจความสัมพันธ์ระหว่างตัวแปร

**ภาพรวมสัปดาห์**:
- Slide 1–2: SLR model Y = β₀ + β₁X + ε
- Slide 3–5: Assumptions และความหมาย
- Slide 6–7: Geometric interpretation
- Slide 8–9: Case Study — Advertising
- Slide 10: Summary + Preview

---
## Slide 3 — SLR Model คืออะไร?
**Key Message**: SLR บอกว่า Y เปลี่ยนเป็นเส้นตรงตาม X บวกกับ random error

**Model**:
$$Y = \beta_0 + \beta_1 X + \varepsilon$$

| Symbol | ชื่อ | ความหมาย |
|--------|------|---------|
| Y | Response / Dependent variable | สิ่งที่ต้องการพยากรณ์ (เช่น Sales) |
| X | Predictor / Independent variable | ตัวแปรที่ใช้พยากรณ์ (เช่น TV budget) |
| β₀ | Intercept | ค่า Y เมื่อ X = 0 |
| β₁ | Slope | Y เปลี่ยนกี่หน่วยเมื่อ X เพิ่ม 1 หน่วย |
| ε | Error term | สิ่งที่ model ไม่สามารถอธิบายได้ |

**ตัวอย่าง Advertising**:
$$\text{Sales} \approx \beta_0 + \beta_1 \times \text{TV}$$

---
## Slide 4 — β₀ และ β₁ หมายความว่าอะไร?
**Key Message**: β₀ คือจุดเริ่มต้น, β₁ คือ rate of change — ทั้งสองมีความหมายจริงในโลกธุรกิจ

[FIGURE: กราฟเส้นตรง y = β₀ + β₁x พร้อม annotation]

**β₀ (Intercept)**:
- ค่า Y เมื่อ X = 0
- ตัวอย่าง: ถ้า TV budget = 0 → ยังมียอดขายพื้นฐาน β₀ ≈ 7,000 หน่วย
- บางครั้งไม่มีความหมายในทางปฏิบัติ (เช่น ราคาบ้านเมื่อพื้นที่ = 0 ตร.ม.)

**β₁ (Slope)**:
- ทุก ๆ X เพิ่ม 1 หน่วย → Y เปลี่ยน β₁ หน่วย (holding other factors constant)
- ตัวอย่าง: β̂₁ = 0.048 → ทุก $1,000 TV budget → Sales เพิ่ม 48 หน่วย
- ถ้า β₁ > 0: positive relationship | β₁ < 0: negative relationship | β₁ = 0: no relationship

---
## Slide 5 — 4 Assumptions ของ SLR
**Key Message**: ถ้า assumptions ไม่เป็นจริง inference (SE, p-value, CI) จะผิด — ต้อง verify เสมอ

**L-I-N-E**:

| Assumption | ชื่อ | ความหมาย | วิธีตรวจ |
|------------|------|---------|---------|
| **L**inearity | ความสัมพันธ์เป็นเส้นตรง | E[Y\|X] = β₀ + β₁X | scatter plot, residual vs fitted |
| **I**ndependence | errors เป็นอิสระต่อกัน | ε₁, ε₂, ..., εₙ independent | ถ้า time series → Durbin-Watson |
| **N**ormality | errors มี normal distribution | ε ~ N(0, σ²) | QQ-plot, Shapiro-Wilk test |
| **E**qual Variance | homoscedasticity | Var(εᵢ) = σ² สำหรับทุก i | residual plot (fan shape = violation) |

[FIGURE: 4 plots แสดง residuals ที่ดี vs ละเมิด assumptions แต่ละข้อ]

---
## Slide 6 — Geometric Interpretation
**Key Message**: fitted line คือเส้นที่ทำให้ระยะทางแนวตั้งรวมน้อยที่สุด

[FIGURE: scatter plot พร้อม:
- Data points (x₁,y₁), ..., (xₙ,yₙ) 
- Fitted line ŷ = β̂₀ + β̂₁x
- Residuals eᵢ แสดงเป็นเส้นแนวตั้งสีแดง
- Arrow แสดง "Least Squares minimizes Σeᵢ²"]

**Residual**: eᵢ = yᵢ − ŷᵢ (actual − predicted)

**Residual Sum of Squares (RSS)**:
$$RSS = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

**Least Squares**: หา β̂₀, β̂₁ ที่ทำให้ RSS ต่ำที่สุด

---
## Slide 7 — True Line vs Fitted Line
**Key Message**: β̂₀, β̂₁ คือ estimates ของ true β₀, β₁ ซึ่งเราไม่รู้ค่าจริง

[FIGURE: กราฟสองเส้น:
- True line: Y = β₀ + β₁X (เส้นประ สีดำ — ไม่รู้จริง)
- Fitted line: Ŷ = β̂₀ + β̂₁X (เส้นทึบ สีแดง — ประมาณจาก data)
- แสดงว่าถ้าเก็บ sample ใหม่ fitted line จะต่างออกไป]

**ความหมาย**:
- β₀, β₁ = true population parameters (unknown)
- β̂₀, β̂₁ = estimates จาก sample (ค่าที่ได้จาก data ที่มี)
- ถ้าเก็บ sample ใหม่ → β̂ ต่างออกไปเล็กน้อย (sampling variability)

**นำไปสู่**: เราต้องประเมิน "ค่า β̂ นี้แม่นยำแค่ไหน?" → SE, CI, t-test (Slide 3)

---
## Slide 8 — สร้าง SLR ด้วย Python
**Key Message**: ทั้ง 3 library ต้องให้ผลเท่ากัน — เลือกตาม use case

```python
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('Advertising.csv')

# ─── วิธีที่ 1: Manual (สูตร Least Squares) ────────────────
x_bar = df['TV'].mean()
y_bar = df['Sales'].mean()
beta1 = ((df['TV'] - x_bar) * (df['Sales'] - y_bar)).sum() / \
         ((df['TV'] - x_bar)**2).sum()
beta0 = y_bar - beta1 * x_bar
print(f'Manual: β̂₀={beta0:.4f}, β̂₁={beta1:.4f}')

# ─── วิธีที่ 2: sklearn ────────────────────────────────────
model_sk = LinearRegression().fit(df[['TV']], df['Sales'])
print(f'sklearn: β̂₀={model_sk.intercept_:.4f}, β̂₁={model_sk.coef_[0]:.4f}')

# ─── วิธีที่ 3: statsmodels (ให้ full inference) ───────────
model_sm = smf.ols('Sales ~ TV', data=df).fit()
print(model_sm.summary())
```

**ผลลัพธ์**: β̂₀ ≈ 7.03, β̂₁ ≈ 0.0475 ทั้ง 3 วิธี

---
## Slide 9 — Case Study: TV Advertising vs Sales
**Key Message**: SLR บอกว่า TV budget มีความสัมพันธ์กับยอดขายอย่างมีนัยสำคัญ

**Scenario**: บริษัท FMCG ต้องการทราบว่าควรเพิ่มงบ TV หรือไม่

**Data**: Advertising.csv — 200 ตลาด, TV budget (พัน$), Sales (พันหน่วย)

[FIGURE: Scatter plot TV vs Sales พร้อม fitted line สีแดง และ equation annotation]

**ผลลัพธ์จาก statsmodels**:
```
Sales = 7.03 + 0.0475 × TV
```
- β̂₁ = 0.0475: TV เพิ่ม $1,000 → Sales เพิ่ม ~47.5 หน่วย
- R² = 0.612: TV อธิบาย 61.2% ของ variance ใน Sales
- p-value < 0.001: TV มีผลต่อ Sales อย่างมีนัยสำคัญ

**Insight**: TV มีผลจริง แต่ R² = 0.612 หมายความว่า 38.8% ยังอธิบายไม่ได้ → ต้องการ MLR (Week 9)

---
## Slide 10 — Summary
**Key Message**: SLR = model ที่เรียบง่ายที่สุด แต่เป็นพื้นฐานของ regression ทุกประเภท

**สิ่งที่เรียนรู้วันนี้**:
- **SLR Model**: Y = β₀ + β₁X + ε — slope และ intercept มีความหมาย
- **β₁**: rate of change — ทุก X เพิ่ม 1 หน่วย → Y เปลี่ยน β₁ หน่วย
- **Assumptions (LINE)**: Linearity, Independence, Normality, Equal Variance
- **Geometric**: Least Squares หาเส้นที่ minimize RSS = Σeᵢ²

| Concept | สูตร | Python |
|---------|------|--------|
| SLR model | Y = β₀ + β₁X + ε | — |
| Residual | eᵢ = yᵢ − ŷᵢ | `y - model.predict(X)` |
| RSS | Σeᵢ² | `((y - y_hat)**2).sum()` |
| Fit model | — | `smf.ols('y ~ x', data=df).fit()` |

**สัปดาห์ต่อไป — Slide 2**: Least Squares Estimation — วิธีคำนวณ β̂₀, β̂₁ ด้วยสูตรปิด
