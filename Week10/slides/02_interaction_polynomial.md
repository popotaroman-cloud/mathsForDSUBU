# Slide Deck: Interaction Terms และ Polynomial Regression
> Week 10 | CLO3 | ISLP Ch.3.3.2 | 9 slides

---
## Slide 1 — Title
**Extensions: Interaction Terms และ Polynomial Regression**  
Week 10 | CLO3 | ISLP 3.3.2  
LLo: เพิ่ม interaction term และ polynomial feature เข้า model และตีความได้

---
## Slide 2 — ข้อจำกัดของ Additive Model
**Key Message**: additive model บอกว่า effect ของ X₁ ต่อ Y ไม่ขึ้นกับ X₂ — แต่โลกจริงมักไม่เป็นแบบนี้

**Additive assumption**:
$$\text{Sales} = \beta_0 + \beta_1\text{TV} + \beta_2\text{Radio} + \varepsilon$$
- effect ของ TV ต่อ Sales = β₁ เสมอ ไม่ว่า Radio จะเท่าไร
- effect ของ Radio ต่อ Sales = β₂ เสมอ ไม่ว่า TV จะเท่าไร

**ปัญหาจริง**: ถ้าทุ่มงบทั้ง TV และ Radio พร้อมกัน ยอดขายอาจเพิ่มมากกว่าผลรวมของทั้งสอง (synergy)

**ตัวอย่าง**: TV=0, Radio=100 → Sales น้อย | TV=100, Radio=0 → Sales น้อย  
แต่ TV=50, Radio=50 → Sales **มากกว่าผลรวม** → additive model จับ pattern นี้ไม่ได้

---
## Slide 3 — Interaction Term
**Key Message**: interaction term β₃X₁X₂ บอกว่า "effect ของ X₁ ต่อ Y เปลี่ยนตาม X₂"

**Model ที่มี Interaction**:
$$\text{Sales} = \beta_0 + \beta_1\text{TV} + \beta_2\text{Radio} + \beta_3(\text{TV}\times\text{Radio}) + \varepsilon$$

**ตีความ** — rewrite:
$$\text{Sales} = \beta_0 + (\beta_1 + \beta_3\text{Radio})\text{TV} + \beta_2\text{Radio}$$

- effect ของ TV = β₁ + β₃ × Radio
- ถ้า β₃ > 0: Radio สูง → effect ของ TV ยิ่งมาก (synergy)
- ถ้า β₃ < 0: Radio สูง → effect ของ TV น้อยลง (antagonism)

**Hierarchy Principle**: ถ้าใส่ interaction TV×Radio ต้องใส่ TV และ Radio ด้วยเสมอ แม้ p-value ของ main effects สูง

---
## Slide 4 — ผล: Advertising Interaction Model
**Key Message**: interaction TV×Radio ทำให้ R² กระโดดจาก 0.897 → 0.968 — synergy effect จริงมาก

```python
# ─── Fit interaction model ──────────────────────────────────
# วัตถุประสงค์: test ว่า TV×Radio interaction ช่วย Sales ไหม
model_int = smf.ols('Sales ~ TV + Radio + TV:Radio', data=df).fit()
print(model_int.summary())
```

```
              coef    std err    t      P>|t|
--------------------------------------------
Intercept    6.750      0.248   27.23   0.000
TV           0.019      0.002   12.70   0.000
Radio        0.029      0.009    3.11   0.002
TV:Radio     0.001      0.000   20.55   0.000
R-squared:   0.968  (จาก 0.897 !)
```

**ตีความ β_TV:Radio = 0.001**:
- ทุก 1,000 dollar เพิ่มใน Radio → effect ของ TV ต่อ Sales เพิ่ม 0.001 หน่วย
- (หรือ: ทุก 1,000 dollar เพิ่มใน TV → effect ของ Radio ต่อ Sales เพิ่ม 0.001 หน่วย)

---
## Slide 5 — Interaction กับ Dummy Variable
**Key Message**: interaction ระหว่าง dummy กับ numerical สร้าง non-parallel regression lines

**ตัวอย่าง**: ถ้า slope ของ income ต่างกันระหว่าง student/non-student

$$\text{Balance} = \beta_0 + \beta_1\text{Income} + \beta_2\text{Student} + \beta_3(\text{Income}\times\text{Student}) + \varepsilon$$

[FIGURE: scatter plot Balance vs Income แบ่งตาม Student:
- Non-student: slope = β₁
- Student: slope = β₁ + β₃
- Lines NOT parallel — สีต่างกัน]

```python
# ─── dummy × numeric interaction ──────────────────────────
# วัตถุประสงค์: แสดงว่า income effect ต่างกันระหว่าง student/non-student
model_int2 = smf.ols('Balance ~ Income * Student', data=credit).fit()
# Income * Student = Income + Student + Income:Student
```

---
## Slide 6 — Polynomial Regression
**Key Message**: polynomial regression = linear regression ที่เพิ่ม X² เป็น feature — ยัง linear ใน β

**ปัญหา**: scatter plot แสดง curve — linear fit ไม่ดี

**Polynomial model**:
$$\text{mpg} = \beta_0 + \beta_1\text{hp} + \beta_2\text{hp}^2 + \varepsilon$$

- ยัง **linear ใน β** (linear model!) — แค่ feature engineering X²
- เหมาะกับ U-shape หรือ diminishing returns

**Auto dataset** (horsepower vs mpg):

| Model | R² |
|-------|----|
| Linear: mpg ~ hp | 0.606 |
| Quadratic: mpg ~ hp + hp² | **0.688** |
| Degree 5 | 0.693 (เพิ่มน้อยมาก) |

---
## Slide 7 — Python: Polynomial Feature
**Key Message**: เพิ่ม X² ง่ายมาก — ทำ feature engineering ก่อน fit

```python
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

# ─── วิธีที่ 1: เพิ่ม column ใน DataFrame ─────────────────────
# วัตถุประสงค์: สร้าง hp² เป็น feature แยกต่างหาก
auto = pd.read_csv('Auto.csv', na_values='?').dropna()
auto['hp2'] = auto['horsepower'] ** 2

model_poly = smf.ols('mpg ~ horsepower + hp2', data=auto).fit()
print(model_poly.summary())

# ─── วิธีที่ 2: ใช้ I() ใน formula (ง่ายกว่า) ─────────────
# วัตถุประสงค์: statsmodels I() ช่วย compute transformation ใน formula โดยตรง
model_poly2 = smf.ols('mpg ~ horsepower + I(horsepower**2)', data=auto).fit()
print(f'R² = {model_poly2.rsquared:.4f}')

# ─── วิธีที่ 3: sklearn PolynomialFeatures ────────────────
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipe.fit(auto[['horsepower']], auto['mpg'])
```

---
## Slide 8 — Plot: Linear vs Quadratic Fit
**Key Message**: quadratic fit ตาม curve ของ data ได้ดีกว่า linear อย่างชัดเจน

[FIGURE: scatter plot horsepower vs mpg พร้อม:
- Linear fit (สีแดง เส้นตรง)
- Quadratic fit (สีน้ำเงิน curve)
- Degree-5 fit (สีเขียว)]

```python
# ─── visualize polynomial fits ──────────────────────────────
hp_range = np.linspace(auto['horsepower'].min(), auto['horsepower'].max(), 100)
hp_df = pd.DataFrame({'horsepower': hp_range, 'hp2': hp_range**2})

plt.scatter(auto['horsepower'], auto['mpg'], alpha=0.4, label='Data')

# Linear
m1 = smf.ols('mpg ~ horsepower', data=auto).fit()
plt.plot(hp_range, m1.predict(pd.DataFrame({'horsepower':hp_range})), 'r-', label='Linear')

# Quadratic
plt.plot(hp_range, model_poly.predict(hp_df), 'b-', lw=2, label='Quadratic')
plt.xlabel('Horsepower'); plt.ylabel('MPG')
plt.legend(); plt.show()
```

---
## Slide 9 — Summary
**Key Message**: interaction จับ synergy; polynomial จับ non-linearity — ทั้งสองยัง linear ใน β

**สิ่งที่เรียนรู้วันนี้**:
- **Interaction term**: β₃X₁X₂ — effect ของ X₁ ขึ้นกับ X₂
- **Hierarchy principle**: ถ้ามี interaction ต้องมี main effects ด้วย
- **Polynomial regression**: เพิ่ม X² เป็น feature — ยัง linear ใน β

| Extension | Formula (Python) | ใช้เมื่อ |
|-----------|-----------------|---------|
| Interaction | `y ~ x1 + x2 + x1:x2` หรือ `x1*x2` | effect ของ X₁ ขึ้นกับ X₂ |
| Polynomial | `y ~ x + I(x**2)` | scatter แสดง curve |
| Dummy × numeric | `y ~ x * C(cat)` | slope ต่างกันระหว่าง groups |

**สัปดาห์ต่อไป — Slide 3**: Regression Diagnostics — ตรวจ 6 potential problems
