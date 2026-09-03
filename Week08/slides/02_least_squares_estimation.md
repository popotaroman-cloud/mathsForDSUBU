# Slide Deck: Least Squares Estimation
> Week 08 | CLO3 | ISLP Ch.3.1.1 | 10 slides

---
## Slide 1 — Title
**Least Squares Estimation: คำนวณ β̂₀ และ β̂₁**  
Week 8 | CLO3 | ISLP 3.1.1  
LLo: คำนวณ β̂₀ β̂₁ ด้วยสูตร Least Squares และเชื่อมกับ Normal Equations (Week 3)

---
## Slide 2 — ทบทวน: เราต้องการอะไร?
**Key Message**: เราต้องการ β̂₀, β̂₁ ที่ทำให้เส้นตรงอยู่ใกล้ data points มากที่สุด

**ปัญหา**: มี data (x₁,y₁), ..., (xₙ,yₙ) → หา β₀, β₁ ที่ดีที่สุด

[FIGURE: scatter plot + หลายเส้นตรงที่เป็นไปได้ — แสดงว่าเส้นใดดีที่สุด?]

**เกณฑ์ "ดีที่สุด"** — Least Squares:
$$\text{Minimize RSS} = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2$$

**ทำไมไม่ minimize Σ|eᵢ|?**
- RSS differentiable → หา closed form ได้
- RSS ลงโทษ outlier หนักกว่า (squared)
- Connect กับ Normal Equations จาก Week 3

---
## Slide 3 — Derivation: Minimize RSS
**Key Message**: หา β̂ โดย set partial derivatives = 0 → ได้สูตรปิด

**ขั้นตอน** (Calculus):

$$\frac{\partial RSS}{\partial \beta_0} = -2\sum(y_i - \beta_0 - \beta_1 x_i) = 0$$
$$\frac{\partial RSS}{\partial \beta_1} = -2\sum x_i(y_i - \beta_0 - \beta_1 x_i) = 0$$

**แก้ระบบสมการ** (Normal Equations):
$$\sum y_i = n\beta_0 + \beta_1\sum x_i$$
$$\sum x_i y_i = \beta_0\sum x_i + \beta_1\sum x_i^2$$

**ผลลัพธ์**:
$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$

---
## Slide 4 — สูตรปิด: ความหมายแต่ละส่วน
**Key Message**: β̂₁ = Cov(X,Y)/Var(X) — slope คือ covariance หารด้วย variance ของ X

**β̂₁**:
$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\widehat{Cov}(X,Y)}{\widehat{Var}(X)}$$

- ตัวเศษ: Cov(X,Y) — X และ Y เคลื่อนไปด้วยกันแค่ไหน?
- ตัวส่วน: Var(X) — X กระจายแค่ไหน?
- ถ้า Cov > 0 → β̂₁ > 0 (positive slope)

**β̂₀**:
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

- fitted line ต้องผ่าน point (x̄, ȳ) เสมอ
- β̂₀ adjust ให้เส้นผ่าน centroid ของ data

---
## Slide 5 — Connection: Normal Equations (Week 3)
**Key Message**: Least Squares formula สอดคล้องกับ Normal Equations AᵀAβ̂ = Aᵀy ที่เรียนใน Week 3

**Week 3 — Least Squares ทั่วไป**:
$$A^T A\hat{\boldsymbol{\beta}} = A^T \mathbf{y} \quad\Rightarrow\quad \hat{\boldsymbol{\beta}} = (A^TA)^{-1}A^T\mathbf{y}$$

**SLR — Design Matrix**:
$$A = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}, \quad \hat{\boldsymbol{\beta}} = \begin{bmatrix}\hat{\beta}_0 \\ \hat{\beta}_1\end{bmatrix}$$

**ผล**: สูตรปิดจาก calculus = formula จาก Normal Equations = `np.linalg.lstsq()` ทั้งหมดให้ผลเดียวกัน

**DS Insight**: ใน MLR (Week 9) เราจะใช้ normal equations กับ p predictors — SLR เป็นแค่ p=1

---
## Slide 6 — ตัวอย่างเลข: คำนวณด้วยมือ
**Key Message**: ลองคำนวณ β̂ ด้วยมือบน mini dataset เพื่อให้เข้าใจสูตร

**ข้อมูล** (4 ตลาด):

| i | TV (x) | Sales (y) | xᵢ−x̄ | yᵢ−ȳ | (xᵢ−x̄)(yᵢ−ȳ) | (xᵢ−x̄)² |
|---|--------|-----------|--------|-------|---------------|---------|
| 1 | 10 | 8 | −10 | −4 | 40 | 100 |
| 2 | 20 | 12 | 0 | 0 | 0 | 0 |
| 3 | 30 | 14 | 10 | 2 | 20 | 100 |
| 4 | 20 | 14 | 0 | 2 | 0 | 0 |
| **Σ** | **80** | **48** | — | — | **60** | **200** |

x̄ = 20, ȳ = 12

$$\hat{\beta}_1 = \frac{60}{200} = 0.3, \quad \hat{\beta}_0 = 12 - 0.3 \times 20 = 6$$

**Model**: Sales = 6 + 0.3 × TV

---
## Slide 7 — Python: 3 วิธี
**Key Message**: manual, sklearn, statsmodels ต้องให้ β̂ เดียวกัน — verify ก่อนเสมอ

```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Advertising.csv')

# ─── วิธี 1: Manual formula ────────────────────────────────
# วัตถุประสงค์: verify ว่าเข้าใจสูตรจริง
x_bar = df['TV'].mean();  y_bar = df['Sales'].mean()
num = ((df['TV'] - x_bar) * (df['Sales'] - y_bar)).sum()
den = ((df['TV'] - x_bar)**2).sum()
b1 = num / den;  b0 = y_bar - b1 * x_bar
print(f'Manual:  β̂₀={b0:.4f}, β̂₁={b1:.4f}')

# ─── วิธี 2: sklearn ───────────────────────────────────────
# วัตถุประสงค์: ใช้งานได้เร็ว เหมาะกับ prediction pipeline
m = LinearRegression().fit(df[['TV']], df['Sales'])
print(f'sklearn: β̂₀={m.intercept_:.4f}, β̂₁={m.coef_[0]:.4f}')

# ─── วิธี 3: statsmodels ──────────────────────────────────
# วัตถุประสงค์: ได้ SE, t, p-value, CI ครบถ้วนสำหรับ inference
model = smf.ols('Sales ~ TV', data=df).fit()
print(model.params)  # β̂₀, β̂₁
```

**ผลลัพธ์**: β̂₀ ≈ 7.0326, β̂₁ ≈ 0.04754 ทุกวิธี

---
## Slide 8 — Visualize: Scatter + Fitted Line
**Key Message**: plot เสมอ ก่อนอ่านตัวเลข — ตาดูก่อน แล้วค่อยอ่าน output

```python
import matplotlib.pyplot as plt

# ─── สร้าง scatter plot + fitted line ──────────────────────
# วัตถุประสงค์: ดูว่าเส้นตรง fit data ได้ดีแค่ไหนก่อนอ่าน R²
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['TV'], df['Sales'], alpha=0.6, label='Data')

# วาด fitted line โดยใช้ range ของ TV
x_line = np.linspace(df['TV'].min(), df['TV'].max(), 100)
y_line = b0 + b1 * x_line
ax.plot(x_line, y_line, 'r-', lw=2, label=f'Fit: y={b0:.2f}+{b1:.4f}x')

ax.set_xlabel('TV Budget (พัน $)')
ax.set_ylabel('Sales (พันหน่วย)')
ax.set_title('TV Budget vs Sales — Simple Linear Regression')
ax.legend()
plt.tight_layout()
plt.show()
```

[FIGURE: scatter plot + red fitted line, equation annotation top-left corner]

---
## Slide 9 — RSS บน Advertising Data
**Key Message**: RSS คือตัววัดความ "ไม่ดี" ของโมเดล — ยิ่งน้อยยิ่งดี แต่ไม่ใช่ 0 เพราะมี ε

```python
# ─── คำนวณ RSS ด้วยมือ ──────────────────────────────────────
# วัตถุประสงค์: เข้าใจว่า RSS คืออะไรก่อนนำไปคำนวณ R² และ RSE
y_hat = b0 + b1 * df['TV']
residuals = df['Sales'] - y_hat
RSS = (residuals**2).sum()
print(f'RSS = {RSS:.2f}')   # ≈ 2102.53

# ─── visualize residuals ───────────────────────────────────
# วัตถุประสงค์: ดู pattern ของ residuals — ถ้ามี pattern แสดงว่า assumptions ผิด
plt.scatter(y_hat, residuals, alpha=0.6)
plt.axhline(0, color='r', lw=1.5)
plt.xlabel('Fitted values (ŷ)')
plt.ylabel('Residuals (e = y − ŷ)')
plt.title('Residual Plot — ควรกระจายแบบสุ่มรอบ y=0')
plt.show()
```

**RSS ≈ 2102.53**: รวมกำลังสองของ residuals ทั้ง 200 ตลาด

---
## Slide 10 — Summary
**Key Message**: Least Squares คือหัวใจของ regression — สูตรปิดใช้ได้เพราะ RSS เป็น quadratic

**สิ่งที่เรียนรู้วันนี้**:
- **RSS**: Σ(yᵢ − β̂₀ − β̂₁xᵢ)² — สิ่งที่ minimize
- **β̂₁** = Cov(X,Y)/Var(X) — slope เป็น ratio ของ covariance ต่อ variance
- **β̂₀** = ȳ − β̂₁x̄ — fitted line ผ่าน centroid (x̄, ȳ) เสมอ
- **Connection**: Normal Equations Week 3 = Least Squares = sklearn = statsmodels

| สูตร | Python | ความหมาย |
|------|--------|---------|
| β̂₁ = Cov/Var | `(cov(x,y)/var(x))` | slope |
| β̂₀ = ȳ − β̂₁x̄ | `y_bar - b1*x_bar` | intercept |
| RSS = Σeᵢ² | `((y-yhat)**2).sum()` | total squared error |

**สัปดาห์ต่อไป — Slide 3**: Coefficient Accuracy — SE, CI, t-test: "β̂₁ แม่นยำแค่ไหน?"
