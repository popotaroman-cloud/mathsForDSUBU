# Slide Deck: Multiple Linear Regression Model
> Week 09 | CLO3 | ISLP Ch.3.2 | 10 slides

---
## Slide 1 — Title
**Multiple Linear Regression: โมเดลและการประมาณสัมประสิทธิ์**  
Week 9 | CLO3 | ISLP 3.2  
LLo: สร้าง MLR บน matrix form และตีความ coefficient แต่ละตัวได้

---
## Slide 2 — Week Overview
**Key Message**: MLR คือการขยาย SLR ให้รับ predictor ได้มากกว่า 1 ตัว — ช่วย control confounding

สัปดาห์นี้เราจะเรียนรู้ **Multiple Linear Regression (MLR)** ซึ่งเป็น extension ของ SLR ที่เพิ่ม predictor ได้หลายตัวพร้อมกัน ความแตกต่างสำคัญคือ β̂ⱼ ใน MLR หมายถึงผลของ Xⱼ ต่อ Y **โดยยึด predictor ตัวอื่นคงที่** ซึ่งต่างจาก SLR ที่ไม่มีการ control สัปดาห์นี้เรียนรู้ MLR model ในรูป matrix, F-statistic สำหรับ test ว่า predictor ทั้งหมดมีความสัมพันธ์กับ Y หรือไม่, variable selection, Adjusted R², และ CI vs PI สำหรับ prediction ทักษะเหล่านี้ใช้จริงในทุก domain เช่น พยากรณ์ราคาบ้านจากหลายปัจจัย หรือวิเคราะห์ปัจจัยส่งผลต่อผลการเรียน

**ภาพรวม**:
- Slide 1–2: MLR model + matrix form
- Slide 3–5: ตีความ coefficient + confounding
- Slide 6–8: Normal Equations + Python
- Slide 9: Case Study
- Slide 10: Summary

---
## Slide 3 — จาก SLR สู่ MLR
**Key Message**: SLR เพียงพอเมื่อ predictor เดียว แต่ถ้ามีหลาย predictors ต้องใช้ MLR

**ปัญหาของ SLR เดี่ยว ๆ**:
- Sales ~ TV ได้ R² = 0.612 — ยังมี 38.8% อธิบายไม่ได้
- Sales ~ Radio ได้ R² = 0.332
- รวม TV + Radio + Newspaper ใน model เดียวจะดีกว่า

**MLR model**:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon$$

**ตัวอย่าง Advertising**:
$$\text{Sales} = \beta_0 + \beta_1\text{TV} + \beta_2\text{Radio} + \beta_3\text{Newspaper} + \varepsilon$$

**ผลจาก statsmodels**:
- R² เพิ่มจาก 0.612 (SLR) → **0.897** (MLR) — อธิบายได้เกือบ 90%!

---
## Slide 4 — ตีความ Coefficient ใน MLR
**Key Message**: β̂ⱼ ใน MLR = ผลของ Xⱼ ต่อ Y **โดยถือว่า predictor ตัวอื่นคงที่**

**"Holding others fixed"** — นี่คือความแตกต่างสำคัญ:

| | SLR: Sales ~ Radio | MLR: Sales ~ TV + Radio + Newspaper |
|--|----|----|
| β̂_Radio | 0.2025 | **0.1885** |
| ความหมาย | Radio เพิ่ม $1k → Sales +202.5 หน่วย **ไม่ control อะไร** | Radio เพิ่ม $1k → Sales +188.5 หน่วย **ขณะ TV และ Newspaper คงที่** |

**ทำไม β̂_Radio เปลี่ยน?**  
เพราะ TV และ Radio มี correlation กัน (ตลาดที่ใช้งบ TV สูง มักใช้ Radio สูงด้วย)  
MLR "แยก" ผลของแต่ละตัวออกจากกัน → ค่า coefficient น่าเชื่อถือกว่า

---
## Slide 5 — Confounding Effect
**Key Message**: ถ้า predictors correlated กัน SLR จะ overestimate หรือ underestimate ผล — MLR แก้ได้

[FIGURE: Venn diagram แสดง TV, Radio overlap กับ Sales:
- SLR Radio absorbs TV effect ด้วย
- MLR แยก contribution ของแต่ละตัว]

**ตัวอย่าง Newspaper**:
```
SLR: Sales ~ Newspaper → β̂_News = 0.055 (p = 0.001) — significant!
MLR: Sales ~ TV + Radio + Newspaper → β̂_News = -0.001 (p = 0.860) — NOT significant
```

**เหตุผล**: Newspaper correlated กับ Radio → SLR ดูดผล Radio มาด้วย  
MLR เปิดเผยว่า Newspaper ไม่มีผลต่อ Sales เมื่อ control TV และ Radio แล้ว

---
## Slide 6 — Matrix Form: ทบทวน Week 3
**Key Message**: MLR = Normal Equations ที่เรียนใน Week 3 — math เหมือนกันทุกประการ

**Design Matrix**:
$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix}_{n \times (p+1)}$$

**Normal Equations** (Week 3):
$$\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$$
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Advertising**: n = 200, p = 3 → X is 200×4, β̂ is 4×1

---
## Slide 7 — Python: สร้าง MLR ด้วย 3 วิธี
**Key Message**: Normal Equations, sklearn, statsmodels — ให้ β̂ เดียวกัน

```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Advertising.csv')

# ─── วิธี 1: Normal Equations (Week 3) ──────────────────────
# วัตถุประสงค์: แสดงว่า MLR = XᵀXβ̂ = Xᵀy จาก Week 3
X = np.column_stack([np.ones(len(df)), df['TV'], df['Radio'], df['Newspaper']])
y = df['Sales'].values
beta = np.linalg.lstsq(X, y, rcond=None)[0]
print('Normal Eq β̂:', np.round(beta, 4))

# ─── วิธี 2: sklearn ────────────────────────────────────────
# วัตถุประสงค์: สะดวกสำหรับ prediction ใน pipeline
m = LinearRegression().fit(df[['TV','Radio','Newspaper']], df['Sales'])
print('sklearn β̂:', [round(m.intercept_,4)] + [round(c,4) for c in m.coef_])

# ─── วิธี 3: statsmodels (full inference) ─────────────────
# วัตถุประสงค์: ได้ F-stat, t, p-value, R²_adj ครบ
model = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit()
print(model.summary())
```

---
## Slide 8 — อ่าน MLR Summary Output
**Key Message**: summary table เหมือน SLR แต่เพิ่ม F-statistic และ Adj. R²

```
OLS Regression Results
====================================================================
Dep. Variable:         Sales   R-squared:                   0.897
No. Observations:        200   Adj. R-squared:              0.896
Df Residuals:            196   F-statistic:                 570.3
                               Prob (F-statistic):        1.58e-96
====================================================================
              coef    std err    t      P>|t|  [0.025   0.975]
--------------------------------------------------------------------
Intercept    2.939      0.312    9.42    0.000   2.324    3.554
TV           0.046      0.001   32.81    0.000   0.043    0.049
Radio        0.189      0.009   21.89    0.000   0.172    0.206
Newspaper   -0.001      0.006   -0.18    0.860  -0.013    0.011
====================================================================
```

**สังเกต**: Newspaper p = 0.860 → NOT significant ใน MLR

**ตีความค่าในตาราง**:
- **F-statistic = 570.3, Prob ≈ 0** → ปฏิเสธ H₀: β₁=β₂=β₃=0 ได้ขาด — อย่างน้อย 1 predictor มีผลจริงต่อ Sales
- **R² = 0.897 vs Adj. R² = 0.896** → ห่างกันน้อยมาก = ไม่มี predictor ไหนถูกยัดเข้ามาแบบไม่จำเป็น
- **coef ของแต่ละตัว** = ผลเมื่อ**คุมตัวแปรอื่นให้คงที่** (ต่างจาก SLR ที่ไม่คุม)
- **CI คร่อม 0** (Newspaper: −0.013 ถึง 0.011) → สรุปว่าไม่ significant ตรงกับค่า p

| ดี | ไม่ดี |
|---|---|
| F-test และ R² สูงมาก — โมเดลโดยรวมแข็งแรง | Newspaper: coef≈0, p=0.860, CI คร่อม 0 |
| TV, Radio significant ชัดเจน (p<0.001) CI แคบ | ตรงกับ Confounding ใน Slide 5 — Newspaper "ยืม" ผลจาก Radio |

**ข้อสรุป**: Significant ที่ F-test (ภาพรวม) ไม่ได้แปลว่าทุกตัวแปร significant รายตัว — ควรตัด Newspaper ออก แล้วเลือกโมเดลด้วย Adjusted R²/AIC/BIC (Note 2)

---
## Slide 9 — Case Study: เปรียบเทียบ SLR vs MLR
**Key Message**: MLR เปิดเผย true effect ของแต่ละ predictor ที่ SLR ไม่สามารถทำได้

| Metric | SLR (TV only) | MLR (TV+Radio+News) |
|--------|--------------|---------------------|
| R² | 0.612 | **0.897** |
| RSE | 3.26 | **1.69** |
| β̂_TV | 0.0475 | 0.046 |
| β̂_Radio | — | 0.189 |
| β̂_Newspaper | — | -0.001 (p=0.86) |

**Business Insight**:
1. TV และ Radio มีผลต่อ Sales อย่างมีนัยสำคัญ — ลงทุนทั้งสองช่องทาง
2. Newspaper ไม่มีผลเมื่อ control TV+Radio — ไม่ควรเพิ่มงบ Newspaper
3. MLR ดีกว่า SLR ชัดเจน — R² จาก 0.61 → 0.90

---
## Slide 10 — Summary
**Key Message**: MLR = matrix form ของ least squares — coefficient หมายถึงผล "net of others"

**สิ่งที่เรียนรู้วันนี้**:
- **MLR**: Y = β₀ + Σβⱼxⱼ + ε — เพิ่ม predictor → อธิบาย Y ได้มากขึ้น
- **β̂ⱼ interpretation**: ผลของ Xⱼ ต่อ Y **holding others fixed**
- **Matrix form**: β̂ = (XᵀX)⁻¹Xᵀy — Normal Equations จาก Week 3
- **Confounding**: SLR เดี่ยว ๆ อาจให้ estimate ที่ผิด — MLR แก้ด้วยการ control

| Concept | สูตร | Python |
|---------|------|--------|
| MLR | Y = Xβ + ε | — |
| Normal Eq | β̂ = (XᵀX)⁻¹Xᵀy | `np.linalg.lstsq(X, y)` |
| Full inference | — | `smf.ols('y~x1+x2', data).fit()` |

**สัปดาห์ต่อไป — Slide 2**: F-statistic — "predictor ทั้งหมดมีความสัมพันธ์กับ Y หรือไม่?"
