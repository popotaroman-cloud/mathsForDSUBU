# Note 1: Qualitative Predictors, Interaction Terms และ Polynomial Regression
> Week 10 | CLO3 | ISLP Reference: Ch.3.3.1–3.3.2

---

## บทนำ

สัปดาห์นี้เราจะขยาย Linear Regression ให้ handle ข้อมูลชีวิตจริงได้มากขึ้น สองส่วนสำคัญคือ: (1) **Qualitative Predictors** (ตัวแปรจัดประเภท เช่น เพศ, ภูมิภาค) ที่ต้องแปลงเป็นตัวเลขก่อน และ (2) **Extensions** ได้แก่ Interaction Terms และ Polynomial Regression ที่ขยาย linear assumption เป้าหมายคือให้นักศึกษาสร้าง dummy variables ได้ถูกต้อง ตีความ coefficient ของ dummy และ interaction terms ได้ และเข้าใจว่า Polynomial Regression ยังคงเป็น "linear model" แม้ดูเหมือน non-linear ในชีวิตจริง ตัวแปร categorical อยู่ทุกที่: ลูกค้าเป็น student หรือไม่? สินค้าอยู่ใน category ไหน? ผู้ป่วยได้รับ treatment แบบใด? ทักษะนี้จำเป็นสำหรับงาน Data Science ทุกประเภท

---

## Section 1: Qualitative (Categorical) Predictors  *(ISLP 3.3.1)*

ในส่วนนี้เราจะเรียนวิธีรวม categorical variables เข้าใน regression ด้วย **Dummy Variables** เพื่อให้ model จับผลของ category ต่าง ๆ ได้ถูกต้อง

### 1.1 Binary Predictor: Dummy Variable

สำหรับ predictor ที่มี 2 categories (เช่น gender: male/female):

**สร้าง dummy variable** $x_i$:
$$x_i = \begin{cases} 1 & \text{ถ้า female} \\ 0 & \text{ถ้า male (reference level)} \end{cases}$$

**Model**:
$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i = \begin{cases} \hat{\beta}_0 + \hat{\beta}_1 & \text{female} \\ \hat{\beta}_0 & \text{male} \end{cases}$$

**ตีความ**:
- $\hat{\beta}_0$: average $y$ สำหรับ male (reference)
- $\hat{\beta}_1$: **ส่วนต่าง** ระหว่าง female และ male

**ตัวอย่าง** (Credit dataset):
$$\widehat{\text{Balance}} = 509.80 + 19.73 \cdot \mathbf{1}[\text{female}]$$

$\hat{\beta}_1 = 19.73$: women มี average credit card balance สูงกว่า men $19.73 — โดยค่าที่ตอบแทนปัจจัยอื่นคงที่

### 1.2 Multi-level Predictor: k-1 Dummies

สำหรับ predictor ที่มี $k$ categories ต้องสร้าง $k-1$ dummies

**ตัวอย่าง**: ethnicity มี 3 categories (Asian, Caucasian, African American)

$$x_{i,1} = \mathbf{1}[\text{Asian}], \quad x_{i,2} = \mathbf{1}[\text{Caucasian}]$$

(reference = African American)

**Model**:
$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_{i,1} + \hat{\beta}_2 x_{i,2}$$

| Ethnicity | Predicted Balance |
|-----------|-----------------|
| African American | $\hat{\beta}_0$ |
| Asian | $\hat{\beta}_0 + \hat{\beta}_1$ |
| Caucasian | $\hat{\beta}_0 + \hat{\beta}_2$ |

### 1.3 Dummy Variable Trap

**ทำไมต้องสร้างแค่ $k-1$ dummies?**

ถ้าสร้าง $k$ dummies ครบทุกหมวด:
$$\text{African Am.} + \text{Asian} + \text{Caucasian} = 1 \text{ (always!)}$$

→ Perfect multicollinearity กับ intercept column → $(\mathbf{X}^T\mathbf{X})$ singular → ไม่มี inverse!

```python
# ─── Dummy Variables ด้วย pandas ──────────────────────────────────────
# วัตถุประสงค์: แสดงวิธีสร้าง dummy variables อย่างถูกต้อง
import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 100

# สร้าง dataset ที่มี categorical feature
df = pd.DataFrame({
    'ethnicity': np.random.choice(['Asian', 'Caucasian', 'AfricanAm'], n),
    'income': np.random.normal(50000, 15000, n),
    'balance': 300 + np.random.normal(0, 50, n)
})

# สร้าง dummy variables — drop_first=True ป้องกัน dummy trap
# วัตถุประสงค์: drop_first ทิ้ง category แรก (alphabetical) เป็น reference
dummies = pd.get_dummies(df['ethnicity'], drop_first=True, dtype=float)
print("Dummy columns created:", dummies.columns.tolist())
print(dummies.head(10))

# รวมกับ features อื่น
X_with_dummies = pd.concat([df[['income']], dummies], axis=1)
X_with_dummies = sm.add_constant(X_with_dummies)

model = sm.OLS(df['balance'], X_with_dummies).fit()
print("\n", model.summary())

# ─── สำหรับ statsmodels formula API ──────────────────────────────────────
# วัตถุประสงค์: C() ใน formula จัดการ dummy อัตโนมัติ
import statsmodels.formula.api as smf
model_formula = smf.ols('balance ~ income + C(ethnicity)', data=df).fit()
print("\nFormula API (same result):")
print(model_formula.params.round(4))
```

**DS Connection**: ใน Machine Learning pipeline `pd.get_dummies()` หรือ `sklearn.preprocessing.OneHotEncoder` ทำ dummy encoding อัตโนมัติ แต่ต้องระวัง reference level ให้สอดคล้องกันระหว่าง train และ test set

---

## Section 2: Interaction Terms  *(ISLP 3.3.2)*

ในส่วนนี้เราจะเรียน **Interaction Terms** ซึ่งขยาย additive assumption ของ linear model เพื่อ capture ว่า effect ของ predictor หนึ่งขึ้นกับค่าของ predictor อีกตัว

### 2.1 Additive vs Non-additive

**Additive model** (standard MLR):
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$$

สมมติ: effect ของ $X_1$ บน $Y$ **ไม่ขึ้นกับ** ค่าของ $X_2$

$$\frac{\partial Y}{\partial X_1} = \beta_1 \quad \text{(คงที่ ไม่ขึ้นกับ } X_2\text{)}$$

**Non-additive model (Interaction)**:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$$

$$\frac{\partial Y}{\partial X_1} = \beta_1 + \beta_3 X_2 \quad \text{(ขึ้นกับ } X_2\text{!)}$$

**ตีความ $\beta_3$**:
- $\beta_3 > 0$: TV และ Radio มี **synergistic effect** — ลงทุน Radio สูงทำให้ผล TV ดีขึ้น
- $\beta_3 < 0$: **antagonistic** — ตัวแปรสองตัวขัดกัน

### 2.2 Advertising Interaction: TV × Radio

จาก ISLP (full dataset):

| Model | R² | RSE |
|-------|-----|-----|
| TV + Radio + Newspaper | 0.897 | 1.68 |
| TV + Radio + TV×Radio | **0.968** | **1.00** |

R² กระโดดจาก 0.897 → 0.968 เมื่อเพิ่ม interaction!

$$\widehat{\text{Sales}} = 6.75 + 0.019 \cdot \text{TV} + 0.029 \cdot \text{Radio} + 0.001086 \cdot \text{TV} \times \text{Radio}$$

**ตีความ**: ทุก \\$1K เพิ่มใน TV → Sales เพิ่ม $(0.019 + 0.001086 \cdot \text{Radio})$ K units  
ถ้า Radio = 20: effect ของ TV = 0.019 + 0.001086×20 = 0.041 K units  
ถ้า Radio = 50: effect ของ TV = 0.019 + 0.001086×50 = 0.073 K units

**Hierarchy Principle**: ถ้าใส่ interaction $X_1 X_2$ ต้องใส่ main effects $X_1$ และ $X_2$ ด้วยเสมอ แม้ว่า main effect จะ insignificant

```python
# ─── Interaction Term ใน Advertising ────────────────────────────────────
# วัตถุประสงค์: แสดงผลของ interaction term ต่อ R² และการตีความ
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

np.random.seed(42)
n = 200
TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
# True model: super-additive interaction
Sales = 6.75 + 0.019*TV + 0.029*Radio + 0.001086*TV*Radio + np.random.normal(0, 1.0, n)

df = pd.DataFrame({'TV': TV, 'Radio': Radio, 'Sales': Sales})

# Model ไม่มี interaction
m_add = smf.ols('Sales ~ TV + Radio', data=df).fit()
# Model มี interaction
m_int = smf.ols('Sales ~ TV * Radio', data=df).fit()  # TV * Radio = TV + Radio + TV:Radio

print("Additive Model:")
print(f"  R² = {m_add.rsquared:.4f}, RSE = {np.sqrt(m_add.mse_resid):.4f}")
print(f"\nInteraction Model:")
print(f"  R² = {m_int.rsquared:.4f}, RSE = {np.sqrt(m_int.mse_resid):.4f}")
print(f"\nCoefficients:")
print(m_int.params.round(6))

# ตีความ: effect ของ TV ที่ Radio = 20 vs Radio = 50
beta_tv = m_int.params['TV']
beta_interaction = m_int.params['TV:Radio']
print(f"\nEffect of TV when Radio=20: {beta_tv + beta_interaction*20:.4f}")
print(f"Effect of TV when Radio=50: {beta_tv + beta_interaction*50:.4f}")
```

---

## Section 3: Polynomial Regression  *(ISLP 3.3.2)*

ในส่วนนี้เราจะเรียน **Polynomial Regression** ซึ่งเป็นการขยาย linear model ให้ fit non-linear relationship โดยยังคงเป็น "linear model" ใน parameter

### 3.1 ทำไม Polynomial?

บางครั้ง relationship ระหว่าง $X$ และ $Y$ ไม่ linear — ตัวอย่าง: Auto dataset, mpg vs horsepower มีรูปร่าง U คว่ำ (เพิ่มขึ้นช้า ๆ แล้วลดลง)

### 3.2 Model

$$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_d X^d + \varepsilon$$

**Still a Linear Model!** เพราะ linear ใน parameters $\beta_j$ แม้ว่า $X^2$ จะไม่ linear ใน $X$

**Implementation**: เพิ่ม $X^2, X^3, \ldots$ เป็น features ใหม่:

```python
# ─── Polynomial Regression ─────────────────────────────────────────────
# วัตถุประสงค์: fit polynomial ด้วย PolynomialFeatures และเปรียบเทียบ degrees
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

np.random.seed(42)
n = 100
# True non-linear: mpg = 60 - 0.3*hp + 0.0008*hp² + noise
hp = np.random.uniform(50, 230, n)
mpg_true = 60 - 0.3*hp + 0.0008*hp**2
mpg = mpg_true + np.random.normal(0, 2, n)

hp_grid = np.linspace(50, 230, 200)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, degree in zip(axes, [1, 2, 5]):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(hp.reshape(-1, 1), mpg)
    y_pred = model.predict(hp_grid.reshape(-1, 1))
    r2 = model.score(hp.reshape(-1, 1), mpg)

    ax.scatter(hp, mpg, alpha=0.5, s=20)
    ax.plot(hp_grid, y_pred, 'r-', linewidth=2)
    ax.set_title(f'Degree {degree} (R²={r2:.3f})')
    ax.set_xlabel('Horsepower'); ax.set_ylabel('MPG')

plt.suptitle('Polynomial Regression: MPG vs Horsepower')
plt.tight_layout()
plt.savefig("polynomial_regression.png", dpi=100, bbox_inches='tight')
print("Plot saved!")

# ─── สำหรับ inference: ใช้ statsmodels ──────────────────────────────────
# วัตถุประสงค์: Polynomial ผ่าน statsmodels ให้ t-test สำหรับแต่ละ term
X_poly = np.column_stack([np.ones(n), hp, hp**2])
m_poly = sm.OLS(mpg, X_poly).fit()
print("\nQuadratic fit:")
print(f"  β₀={m_poly.params[0]:.4f}, β₁={m_poly.params[1]:.4f}, β₂={m_poly.params[2]:.6f}")
print(f"  R² = {m_poly.rsquared:.4f}")
```

**Case Study: Auto MPG — Polynomial vs Linear**

```python
# ─── เปรียบเทียบ Linear vs Quadratic fit ──────────────────────────────
# วัตถุประสงค์: แสดง F-test comparison ระหว่าง nested models
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 100
hp = np.random.uniform(50, 230, n)
mpg = 60 - 0.3*hp + 0.0008*hp**2 + np.random.normal(0, 2, n)

# Linear model (degree 1)
X1 = sm.add_constant(hp)
m1 = sm.OLS(mpg, X1).fit()

# Quadratic model (degree 2)
X2 = sm.add_constant(np.column_stack([hp, hp**2]))
m2 = sm.OLS(mpg, X2).fit()

print(f"Linear:    R²={m1.rsquared:.4f}, AIC={m1.aic:.2f}")
print(f"Quadratic: R²={m2.rsquared:.4f}, AIC={m2.aic:.2f}")
print(f"\nβ₂ (hp²): {m2.params[2]:.6f}, p-value: {m2.pvalues[2]:.4f}")
print(f"→ {'Quadratic term is significant!' if m2.pvalues[2] < 0.05 else 'Not significant'}")
```

**DS Connection**: Polynomial features คือรูปแบบ Feature Engineering ที่ง่ายที่สุด ใน `sklearn.preprocessing.PolynomialFeatures` ใช้กับ Pipeline ได้โดยตรง เป็นพื้นฐานก่อนเรียน Splines และ Kernel Methods

---

## สรุป (Summary)

| แนวคิด | การ implement | Python |
|--------|-------------|--------|
| Binary dummy | $\mathbf{1}[\text{category}] \in \{0,1\}$ | `pd.get_dummies(drop_first=True)` |
| k-level dummy | $k-1$ dummies | `pd.get_dummies(drop_first=True)` |
| Reference level | ตัวที่ถูก drop | ระบุด้วย `drop_first` |
| Interaction | $X_1 \cdot X_2$ term | `df['X1'] * df['X2']` หรือ `X1*X2` ใน formula |
| Polynomial degree $d$ | $X, X^2, \ldots, X^d$ | `PolynomialFeatures(d)` |
| Hierarchy | ใส่ main effects ด้วยเสมอ | ต้องระวังเอง |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 9**: MLR บน numerical features — สัปดาห์นี้ขยายไปสู่ categorical และ non-additive
- → **Week 10 (note 2)**: Diagnostic plots ตรวจสอบ assumptions ที่อาจถูกละเมิดในทุก extensions
- → **Week 11**: Logistic Regression ก็ใช้ dummy variables และ interactions แบบเดียวกัน
