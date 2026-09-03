# Note 1: Multiple Linear Regression — Model, Estimation และ F-statistic
> Week 9 | CLO3 | ISLP Reference: Ch.3.2 Multiple Linear Regression

---

## บทนำ

สัปดาห์นี้เราจะขยาย Simple Linear Regression (SLR) จากสัปดาห์ที่ 8 ไปสู่ **Multiple Linear Regression (MLR)** ซึ่งรับ predictor หลายตัวพร้อมกัน เหตุผลที่ SLR ไม่เพียงพอในชีวิตจริงคือปรากฏการณ์ที่เรียกว่า **Confounding** — เมื่อตัวแปรหลายตัวมี correlation กัน การ fit SLR ทีละตัวให้ค่า coefficient ที่คลาดเคลื่อนจากความเป็นจริง ตัวอย่างเช่น ใน Advertising dataset TV มี correlation กับ Radio ดังนั้น coefficient ของ TV จาก SLR (TV → Sales) จะ "ดูดซับ" ผลของ Radio ที่ไม่ได้รวมอยู่ใน model เมื่อใส่ Radio เข้า MLR ด้วย coefficient ของ TV จะเปลี่ยนไปอย่างมีนัยสำคัญ MLR แก้ปัญหานี้โดยประมาณ coefficient ของ $X_j$ **โดยยึดตัวแปรอื่นทั้งหมดคงที่** ซึ่งเป็นสิ่งที่ SLR ทำไม่ได้ เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถสร้าง MLR ได้จาก Normal Equations ที่เรียนใน Week 3, ตีความค่า $\hat{\beta}_j$ ว่าหมายความว่าอะไรเมื่อมี p predictors, อ่านและตีความ F-statistic ได้ว่าต่างอะไรกับ t-statistic, และเลือก model ที่เหมาะสมด้วย Adjusted R² ใน Data Science ทุกวัน: ราคาบ้านขึ้นกับพื้นที่, จำนวนห้อง, ทำเล, อายุอาคาร — นี่คือ MLR problem; ผลตอบแทนหุ้นขึ้นกับ market factor, sector, size — นี่ก็คือ MLR problem การเข้าใจ MLR เป็นพื้นฐานของ machine learning models ทุกประเภทที่เรียนในสัปดาห์ถัดไป

---

## Section 1: MLR Model และ Matrix Form  *(ISLP 3.2)*

ในส่วนนี้เราจะนิยาม MLR อย่างเป็นทางการและแสดงว่า Normal Equations จาก Week 3 คือเครื่องมือที่ใช้ estimate coefficients

### 1.1 MLR Model

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon$$

**ตีความ $\beta_j$**: ค่าเฉลี่ยที่ $Y$ เปลี่ยนไปเมื่อ $X_j$ เพิ่มขึ้น 1 หน่วย **โดยยึด $X_k$ ทุกตัว ($k \neq j$) คงที่**

ความแตกต่างจาก SLR: ใน SLR $\hat{\beta}_1^{SLR}$ คือ marginal effect ของ $X_1$ รวมทั้ง indirect effect ผ่าน $X_k$ ทั้งหมดที่ correlated กับ $X_1$ แต่ใน MLR $\hat{\beta}_1^{MLR}$ คือ **partial effect** ควบคุม $X_k$ แล้ว

### 1.2 Matrix (Vector) Form

สำหรับ $n$ observations และ $p$ predictors:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

**Design Matrix** $\mathbf{X}$ ขนาด $n \times (p+1)$:

$$\mathbf{X} = \begin{pmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix}$$

คอลัมน์แรกเป็น 1 ทั้งหมด สำหรับ intercept $\beta_0$

**OLS Estimator** (Normal Equations จาก Week 3):

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

---

## Section 2: Worked Example — Normal Equations 3×3  *(ISLP 3.2.1)*

ในส่วนนี้เราจะ compute $\hat{\boldsymbol{\beta}}$ ด้วยมือสำหรับ dataset เล็ก ๆ เพื่อเห็นสูตร Normal Equations ทำงานอย่างไร

### 2.1 Dataset

**Advertising (subset 5 ตลาด)**:

| TV ($x_1$) | Radio ($x_2$) | Sales ($y$) |
|-----------|-------------|-----------|
| 230.1 | 37.8 | 22.1 |
| 44.5 | 39.3 | 10.4 |
| 17.2 | 45.9 | 9.3 |
| 151.5 | 41.3 | 18.5 |
| 180.8 | 10.8 | 12.9 |

**Model**: $\text{Sales} = \beta_0 + \beta_1\text{TV} + \beta_2\text{Radio} + \varepsilon$

### 2.2 สร้าง Design Matrix $\mathbf{X}$ และ $\mathbf{y}$

$$\mathbf{X} = \begin{pmatrix} 1 & 230.1 & 37.8 \\ 1 & 44.5 & 39.3 \\ 1 & 17.2 & 45.9 \\ 1 & 151.5 & 41.3 \\ 1 & 180.8 & 10.8 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 22.1 \\ 10.4 \\ 9.3 \\ 18.5 \\ 12.9 \end{pmatrix}$$

### 2.3 คำนวณ $\mathbf{X}^T\mathbf{X}$ (ขนาด $3 \times 3$)

$$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} \sum 1 & \sum x_{1i} & \sum x_{2i} \\ \sum x_{1i} & \sum x_{1i}^2 & \sum x_{1i}x_{2i} \\ \sum x_{2i} & \sum x_{1i}x_{2i} & \sum x_{2i}^2 \end{pmatrix}$$

คำนวณ:
- $\sum 1 = 5$
- $\sum x_{1i} = 230.1 + 44.5 + 17.2 + 151.5 + 180.8 = 624.1$
- $\sum x_{2i} = 37.8 + 39.3 + 45.9 + 41.3 + 10.8 = 175.1$
- $\sum x_{1i}^2 = 230.1^2 + 44.5^2 + \cdots = 52946.01 + 1980.25 + 295.84 + 22952.25 + 32688.64 = 110863.0$
- $\sum x_{2i}^2 = 37.8^2 + \cdots = 1428.84 + 1544.49 + 2106.81 + 1705.69 + 116.64 = 6902.47$
- $\sum x_{1i}x_{2i} = 230.1(37.8) + 44.5(39.3) + 17.2(45.9) + 151.5(41.3) + 180.8(10.8)$
  $= 8697.78 + 1748.85 + 789.48 + 6256.95 + 1952.64 = 19445.7$

$$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 5 & 624.1 & 175.1 \\ 624.1 & 110863.0 & 19445.7 \\ 175.1 & 19445.7 & 6902.47 \end{pmatrix}$$

### 2.4 คำนวณ $\mathbf{X}^T\mathbf{y}$

$$\mathbf{X}^T\mathbf{y} = \begin{pmatrix} \sum y_i \\ \sum x_{1i}y_i \\ \sum x_{2i}y_i \end{pmatrix} = \begin{pmatrix} 73.2 \\ 230.1(22.1)+44.5(10.4)+17.2(9.3)+151.5(18.5)+180.8(12.9) \\ 37.8(22.1)+39.3(10.4)+45.9(9.3)+41.3(18.5)+10.8(12.9) \end{pmatrix}$$

$\sum x_{1i}y_i = 5085.21 + 462.80 + 159.96 + 2802.75 + 2332.32 = 10843.04$  
$\sum x_{2i}y_i = 835.38 + 408.72 + 426.87 + 764.05 + 139.32 = 2574.34$

$$\mathbf{X}^T\mathbf{y} = \begin{pmatrix} 73.2 \\ 10843.04 \\ 2574.34 \end{pmatrix}$$

### 2.5 แก้ $(\mathbf{X}^T\mathbf{X})\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$

```python
# ─── คำนวณ Normal Equations ด้วย NumPy ─────────────────────────────────
# วัตถุประสงค์: verify ผลที่คำนวณด้วยมือและแสดง matrix solution จริง
import numpy as np

# Design matrix และ response vector
X = np.array([[1, 230.1, 37.8],
              [1,  44.5, 39.3],
              [1,  17.2, 45.9],
              [1, 151.5, 41.3],
              [1, 180.8, 10.8]])
y = np.array([22.1, 10.4, 9.3, 18.5, 12.9])

# Normal Equations: β̂ = (X^T X)^{-1} X^T y
# วัตถุประสงค์: นี่คือ closed-form solution ที่ derive ใน Week 3
XTX = X.T @ X
XTy = X.T @ y
print("X^T X:\n", XTX.round(2))
print("X^T y:", XTy.round(2))

# แก้ด้วย np.linalg.solve (stable กว่า inv โดยตรง)
beta_hat = np.linalg.solve(XTX, XTy)
print(f"\nβ̂₀ = {beta_hat[0]:.4f}")
print(f"β̂₁ (TV) = {beta_hat[1]:.6f}")
print(f"β̂₂ (Radio) = {beta_hat[2]:.6f}")

# Fitted values และ Residuals
y_hat = X @ beta_hat
residuals = y - y_hat
RSS = np.sum(residuals**2)
print(f"\nRSS = {RSS:.4f}")
```

**ผลลัพธ์ที่คาดหวัง**: $\hat{\beta}_0 \approx 3.1, \hat{\beta}_1 \approx 0.047, \hat{\beta}_2 \approx 0.148$

---

## Section 3: ตีความ MLR Coefficients  *(ISLP 3.2.1)*

ในส่วนนี้เราจะเรียนวิธีตีความ $\hat{\beta}_j$ อย่างถูกต้อง ซึ่งเป็นทักษะสำคัญที่ต้องใช้ใน reporting ผล regression ทุกครั้ง

### 3.1 ตีความ Coefficients

จาก Advertising dataset (full 200 observations):
$$\widehat{\text{Sales}} = 2.939 + 0.046 \cdot \text{TV} + 0.189 \cdot \text{Radio} - 0.001 \cdot \text{Newspaper}$$

**ตีความ**:
- $\hat{\beta}_0 = 2.939$: เมื่อไม่มีงบโฆษณาเลย คาด Sales ≈ 2,939 หน่วย
- $\hat{\beta}_1 = 0.046$: TV เพิ่มขึ้น 1 K$ → Sales เพิ่ม 46 หน่วย **โดยยึด Radio และ Newspaper คงที่**
- $\hat{\beta}_2 = 0.189$: Radio เพิ่มขึ้น 1 K$ → Sales เพิ่ม 189 หน่วย **โดยยึด TV และ Newspaper คงที่**
- $\hat{\beta}_3 = -0.001$: Newspaper เพิ่มขึ้น 1 K$ → Sales ลด 1 หน่วย (ผลเล็กมาก)

### 3.2 เปรียบเทียบ SLR vs MLR

| | $\hat{\beta}_{TV}$ | $\hat{\beta}_{Radio}$ | $\hat{\beta}_{Newspaper}$ |
|-|-----------------|-------------------|----------------------|
| SLR (TV → Sales) | 0.048 | — | — |
| SLR (Radio → Sales) | — | 0.203 | — |
| SLR (Newspaper → Sales) | — | — | 0.055 |
| **MLR (TV+Radio+Newspaper → Sales)** | **0.046** | **0.189** | **-0.001** |

**สังเกต**: Newspaper ใน SLR ดูเหมือน positive (0.055) แต่ใน MLR กลับเป็น -0.001 เพราะ Newspaper มี correlation กับ Radio และ TV ใน SLR coefficient ดูดซับ effect ของ Radio ที่ไม่อยู่ใน model!

---

## Section 4: F-statistic  *(ISLP 3.2.2 Q1)*

ในส่วนนี้เราจะเรียนรู้ **F-statistic** ซึ่งทดสอบว่า predictors ทั้งหมดร่วมกันมีความสัมพันธ์กับ $Y$ หรือไม่ — ต่างจาก t-test ที่ทดสอบทีละตัว

### 4.1 F-test: ทดสอบ model โดยรวม

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0 \quad \text{(ไม่มี predictor ใด useful)}$$
$$H_1: \text{อย่างน้อยหนึ่งตัว} \neq 0$$

**F-statistic**:
$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)}$$

โดย:
- TSS (Total Sum of Squares) = $\sum(y_i - \bar{y})^2$
- RSS (Residual Sum of Squares) = $\sum(y_i - \hat{y}_i)^2$
- ตัวเศษ = average variance explained by the model (per predictor)
- ตัวส่วน = RSE² (average unexplained variance)

**ถ้า $H_0$ จริง**: $F \approx 1$ (model ไม่ได้ explain อะไรเลย)  
**ถ้า $H_0$ เท็จ**: $F >> 1$ (model explain variance ได้มาก)

Distribution: $F \sim F(p, n-p-1)$ ถ้า $H_0$ จริง

### 4.2 F vs t: ทำไมต้องมีทั้งสอง?

| | F-test | t-test |
|-|--------|--------|
| ทดสอบ | ทุก predictor ร่วมกัน | predictor ทีละตัว |
| $H_0$ | $\beta_1=\cdots=\beta_p=0$ | $\beta_j=0$ |
| ปัญหา | ไม่บอกว่า predictor ไหนสำคัญ | ถ้า $p$ ใหญ่ มี false positive |

**ทำไม F-test สำคัญ**: ถ้า $p = 100$ predictors และ test แต่ละตัวด้วย $\alpha = 0.05$ จะมี 5 ตัวที่ significant "โดยบังเอิญ" (Type I error) — F-test แก้ปัญหาโดยทดสอบ overall ก่อน

### 4.3 ตัวอย่าง Advertising (n=200, p=3)

จาก statsmodels output:
```
                            OLS Regression Results
F-statistic:                   570.3   Df Model:                3
Prob (F-statistic):          1.58e-96   Df Residuals:          196
R-squared:                    0.897   Adj. R-squared:          0.896
RSE:                          1.681
```

**การตีความ**:
- F = 570.3 >> 1 → reject $H_0$ อย่างชัดเจน
- Prob (F) = 1.58e-96 ≈ 0 → at least one predictor is related to Sales
- R² = 0.897 → model อธิบาย 89.7% ของ variance ใน Sales

```python
# ─── Full MLR ด้วย statsmodels ─────────────────────────────────────────
# วัตถุประสงค์: แสดง complete output รวม F-statistic, t-statistics, CI ทั้งหมด
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 200

# สร้าง Advertising data (synthetic)
TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
Newspaper = np.random.uniform(0.3, 114.0, n)
Sales = 2.939 + 0.046*TV + 0.189*Radio - 0.001*Newspaper + np.random.normal(0, 1.5, n)

# Design matrix พร้อม intercept
# วัตถุประสงค์: sm.add_constant เพิ่ม column ของ 1 สำหรับ β₀
X = sm.add_constant(np.column_stack([TV, Radio, Newspaper]))
y = Sales

# Fit OLS model
# วัตถุประสงค์: statsmodels.OLS ให้ full statistical output ที่ sklearn ไม่มี
model = sm.OLS(y, X).fit()
print(model.summary())

# ดึงค่าสำคัญออกมา
print("\n=== Key Statistics ===")
print(f"F-statistic = {model.fvalue:.2f}, p-value = {model.f_pvalue:.2e}")
print(f"R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
print(f"RSE = {np.sqrt(model.mse_resid):.4f}")
print("\nCoefficients with 95% CI:")
print(model.conf_int().round(4))
```

---

## Case Study: House Price Prediction ด้วย MLR

**Scenario**  
บริษัท Real Estate ต้องการ model ราคาบ้าน โดยใช้หลาย features ไม่ใช่แค่พื้นที่

**Data (สมมติ)**: 100 บ้าน × 3 features

| Feature | ชื่อ | ช่วงค่า |
|---------|-----|---------|
| $X_1$ | พื้นที่ (ตร.ม.) | 50–300 |
| $X_2$ | จำนวนห้องนอน | 1–5 |
| $X_3$ | อายุอาคาร (ปี) | 0–50 |

**Method**

```python
# ─── House Price MLR ─────────────────────────────────────────────────────
# วัตถุประสงค์: แสดง MLR บน house price data พร้อมตีความผล
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 100
area = np.random.uniform(50, 300, n)
rooms = np.random.randint(1, 6, n).astype(float)
age = np.random.uniform(0, 50, n)

# True model: price = 1000 + 15*area + 80*rooms - 10*age + noise
price = 1000 + 15*area + 80*rooms - 10*age + np.random.normal(0, 200, n)

X = sm.add_constant(np.column_stack([area, rooms, age]))
model = sm.OLS(price, X).fit()

print("Coefficients:")
print(f"  Intercept: {model.params[0]:.2f}")
print(f"  Area: {model.params[1]:.2f} (เพิ่ม 1 ตร.ม. → ราคา +{model.params[1]:.0f} บาท)")
print(f"  Rooms: {model.params[2]:.2f} (เพิ่ม 1 ห้อง → ราคา +{model.params[2]:.0f} บาท)")
print(f"  Age: {model.params[3]:.2f} (เก่าขึ้น 1 ปี → ราคา {model.params[3]:.0f} บาท)")
print(f"\nF-stat: {model.fvalue:.1f} (p={model.f_pvalue:.2e})")
print(f"R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}")
```

**Insight**  
- Area มี effect ใหญ่ที่สุด: +15 บาท/ตร.ม.
- Age มี negative effect: อาคารเก่าขึ้น → ราคาลง
- F-test จะ significant มาก เพราะทุก feature มีผลจริง

---

## สรุป (Summary)

| แนวคิด | สูตร | Python |
|--------|------|--------|
| MLR Model | $Y = \beta_0 + \sum\beta_j X_j + \varepsilon$ | `sm.OLS(y, X).fit()` |
| Normal Equations | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | `np.linalg.solve(XTX, XTy)` |
| TSS | $\sum(y_i-\bar{y})^2$ | `((y-y.mean())**2).sum()` |
| RSS | $\sum(y_i-\hat{y}_i)^2$ | `((y-y_hat)**2).sum()` |
| RSE | $\sqrt{\text{RSS}/(n-p-1)}$ | `np.sqrt(model.mse_resid)` |
| R² | $1-\text{RSS}/\text{TSS}$ | `model.rsquared` |
| F-stat | $\frac{(\text{TSS}-\text{RSS})/p}{\text{RSS}/(n-p-1)}$ | `model.fvalue` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 3**: Normal Equations $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ คือ Least Squares ที่เรียนมาแล้ว
- ← **Week 7**: F-statistic ใช้ F-distribution ที่เกี่ยวกับ ratio ของ χ² — statistical inference จาก Week 7
- ← **Week 8**: SLR คือ MLR กรณีพิเศษที่ $p=1$
- → **Week 9 (note 2)**: Variable selection: เมื่อมี predictors มาก จะเลือกอย่างไร?
- → **Week 10**: MLR assumptions — เมื่อ assumptions ถูกละเมิด model ทำงานอย่างไร
