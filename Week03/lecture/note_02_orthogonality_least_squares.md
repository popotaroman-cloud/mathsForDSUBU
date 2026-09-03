# Note 2: Orthogonality, Projection, Least Squares และ QR Decomposition
> Week 3 | CLO1 | Strang Reference: Ch.4 (Orthogonality)

---

## บทนำ

บันทึกนี้เป็นส่วนที่สำคัญที่สุดของ Linear Algebra สำหรับ Data Science: **Least Squares** ซึ่งเป็นหัวใจทางคณิตศาสตร์ของ Linear Regression เมื่อระบบสมการ $\mathbf{Ax} = \mathbf{b}$ ไม่มีคำตอบที่แน่นอน (ข้อมูลมีมากกว่าพารามิเตอร์) เราจะหา "คำตอบดีที่สุด" $\hat{\mathbf{x}}$ ที่ทำให้ $\|\mathbf{Ax} - \mathbf{b}\|^2$ น้อยที่สุด — นี่คือ **Projection** ของ $\mathbf{b}$ ลงบน Column Space ของ $\mathbf{A}$ เป้าหมายของบันทึกนี้คือให้นักศึกษาเข้าใจ Geometric Intuition ของ Orthogonality และ Projection สามารถ derive Normal Equations ได้ และรู้จัก QR Decomposition ซึ่งเป็น numerically stable method สำหรับแก้ Least Squares ในชีวิตจริง

---

## Section 1: Orthogonality  *(Strang 4.1)*

ในส่วนนี้เราจะเรียนรู้ว่า **Orthogonality** (ตั้งฉากกัน) หมายความว่าอะไรใน Vector Space และทำไมถึงสำคัญใน Machine Learning

### 1.1 Orthogonal Vectors

**นิยาม**: vectors $\mathbf{u}$ และ $\mathbf{v}$ **ตั้งฉากกัน** ($\mathbf{u} \perp \mathbf{v}$) ถ้า:
$$\mathbf{u}^T \mathbf{v} = \mathbf{u} \cdot \mathbf{v} = 0$$

**ตัวอย่าง**:
$$\mathbf{u} = \begin{pmatrix}3\\0\end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix}0\\2\end{pmatrix} \quad \Rightarrow \quad \mathbf{u}^T\mathbf{v} = 0+0 = 0 \perp \checkmark$$

$$\mathbf{u} = \begin{pmatrix}1\\2\\-1\end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix}2\\0\\2\end{pmatrix} \quad \Rightarrow \quad \mathbf{u}^T\mathbf{v} = 2+0-2 = 0 \perp \checkmark$$

**Pythagorean Theorem** (เมื่อ $\mathbf{u} \perp \mathbf{v}$):
$$\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$

พิสูจน์: $\|\mathbf{u}+\mathbf{v}\|^2 = (\mathbf{u}+\mathbf{v})^T(\mathbf{u}+\mathbf{v}) = \mathbf{u}^T\mathbf{u} + 2\mathbf{u}^T\mathbf{v} + \mathbf{v}^T\mathbf{v} = \|\mathbf{u}\|^2 + 0 + \|\mathbf{v}\|^2$

### 1.2 Orthogonality ของ Four Subspaces

(จาก note 1) ความสัมพันธ์สำคัญ:
$$C(\mathbf{A}) \perp N(\mathbf{A}^T) \quad \text{และ} \quad C(\mathbf{A}^T) \perp N(\mathbf{A})$$

**ความหมายปฏิบัติ**: ถ้า $\mathbf{Ax} = \mathbf{b}$ ไม่มีคำตอบ ($\mathbf{b} \notin C(\mathbf{A})$) เราสามารถ split $\mathbf{b}$ เป็น:
$$\mathbf{b} = \mathbf{p} + \mathbf{e}$$
โดย $\mathbf{p} \in C(\mathbf{A})$ (projection) และ $\mathbf{e} \in N(\mathbf{A}^T)$ (error ที่ตั้งฉากกับ column space)

**DS Connection**: ใน Linear Regression, Residuals $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ ต้อง **ตั้งฉากกับ feature space** — นี่คือ "Orthogonality Condition" ที่ทำให้ OLS เป็น Best Linear Unbiased Estimator (BLUE)

---

## Section 2: Projection  *(Strang 4.2)*

ในส่วนนี้เราจะเรียนรู้ **Projection** ซึ่งเป็นการฉาย vector ลงบนพื้นที่ย่อย เพื่อสร้าง Geometric Intuition ของ Least Squares

### 2.1 Projection onto a Line

**ปัญหา**: ฉาย $\mathbf{b}$ ลงบนเส้นตรงที่มีทิศทาง $\mathbf{a}$

**สูตร**: Projection ของ $\mathbf{b}$ บน $\mathbf{a}$ คือ:
$$\mathbf{p} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}} \mathbf{a} = \hat{x}\mathbf{a} \quad \text{โดย} \quad \hat{x} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}$$

**Derivation**: $\mathbf{e} = \mathbf{b} - \mathbf{p} = \mathbf{b} - \hat{x}\mathbf{a}$ ต้องตั้งฉากกับ $\mathbf{a}$:
$$\mathbf{a}^T\mathbf{e} = 0 \quad \Rightarrow \quad \mathbf{a}^T(\mathbf{b} - \hat{x}\mathbf{a}) = 0 \quad \Rightarrow \quad \hat{x} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}$$

**Projection Matrix** $\mathbf{P}$:
$$\mathbf{P} = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}$$

โดยที่ $\mathbf{p} = \mathbf{P}\mathbf{b}$

**คุณสมบัติของ Projection Matrix**:
- $\mathbf{P}^2 = \mathbf{P}$ (**Idempotent** — ฉายซ้ำได้ผลเหมือนเดิม)
- $\mathbf{P}^T = \mathbf{P}$ (**Symmetric**)
- $(\mathbf{I}-\mathbf{P})$ คือ projection onto perpendicular complement

**ตัวอย่าง step-by-step**:
$$\mathbf{a} = \begin{pmatrix}1\\2\\2\end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix}1\\1\\1\end{pmatrix}$$

$$\hat{x} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}} = \frac{1+2+2}{1+4+4} = \frac{5}{9}$$

$$\mathbf{p} = \frac{5}{9}\begin{pmatrix}1\\2\\2\end{pmatrix} = \begin{pmatrix}5/9\\10/9\\10/9\end{pmatrix}$$

$$\mathbf{e} = \mathbf{b} - \mathbf{p} = \begin{pmatrix}1-5/9\\1-10/9\\1-10/9\end{pmatrix} = \begin{pmatrix}4/9\\-1/9\\-1/9\end{pmatrix}$$

ตรวจสอบ: $\mathbf{a}^T\mathbf{e} = 4/9 - 2/9 - 2/9 = 0$ ✓

### 2.2 Projection onto a Subspace

**ปัญหา**: ฉาย $\mathbf{b}$ ลงบน Column Space ของ $\mathbf{A}$ (subspace ที่กำหนดโดยหลายคอลัมน์)

**Projection Matrix**:
$$\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

**Derivation**: projection $\mathbf{p} = \mathbf{A}\hat{\mathbf{x}}$ ต้องทำให้ error $\mathbf{e} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$ ตั้งฉากกับทุก column ของ $\mathbf{A}$:
$$\mathbf{A}^T\mathbf{e} = \mathbf{0} \quad \Rightarrow \quad \mathbf{A}^T(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0} \quad \Rightarrow \quad \mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$$

---

## Section 3: Least Squares  *(Strang 4.3)*

ในส่วนนี้เราจะ derive **Least Squares Solution** ซึ่งเป็นคำตอบที่ดีที่สุดเมื่อ $\mathbf{Ax} = \mathbf{b}$ ไม่มีคำตอบแน่นอน นี่คือพื้นฐานทางคณิตศาสตร์ของ Linear Regression ทั้งหมด

### 3.1 Normal Equations

**ปัญหา**: minimize $\|\mathbf{Ax} - \mathbf{b}\|^2$ หา $\hat{\mathbf{x}}$

**Normal Equations**:
$$\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$$

**Least Squares Solution**:
$$\hat{\mathbf{x}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b} \quad \text{(ถ้า } \mathbf{A}^T\mathbf{A} \text{ invertible)}$$

**Intuition**: จาก Projection — $\hat{\mathbf{x}}$ คือค่าที่ทำให้ projection ของ $\mathbf{b}$ ลงบน $C(\mathbf{A})$ ซึ่ง $\mathbf{e} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$ ตั้งฉากกับ $C(\mathbf{A})$

### 3.2 ตัวอย่าง: Fit เส้นตรงผ่าน Data Points

**ปัญหา**: หาเส้น $y = c + dt$ ที่ดีที่สุดผ่าน 3 จุด $(1, 1), (2, 2), (3, 2)$

**3 สมการ 2 unknowns** ($c, d$):
$$c + d(1) = 1, \quad c + d(2) = 2, \quad c + d(3) = 2$$

เขียนเป็น Matrix:
$$\mathbf{A} = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix}, \quad \hat{\mathbf{x}} = \begin{pmatrix}c\\d\end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix}1\\2\\2\end{pmatrix}$$

**Normal Equations**:
$$\mathbf{A}^T\mathbf{A} = \begin{pmatrix}1&1&1\\1&2&3\end{pmatrix}\begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix} = \begin{pmatrix}3&6\\6&14\end{pmatrix}$$

$$\mathbf{A}^T\mathbf{b} = \begin{pmatrix}1&1&1\\1&2&3\end{pmatrix}\begin{pmatrix}1\\2\\2\end{pmatrix} = \begin{pmatrix}5\\11\end{pmatrix}$$

แก้สมการ $\begin{pmatrix}3&6\\6&14\end{pmatrix}\begin{pmatrix}c\\d\end{pmatrix} = \begin{pmatrix}5\\11\end{pmatrix}$:

$$\det = 3(14) - 6(6) = 42 - 36 = 6 \neq 0$$
$$\hat{c} = \frac{14(5)-6(11)}{6} = \frac{70-66}{6} = \frac{4}{6} = \frac{2}{3}$$
$$\hat{d} = \frac{3(11)-6(5)}{6} = \frac{33-30}{6} = \frac{3}{6} = \frac{1}{2}$$

**คำตอบ**: $y = \frac{2}{3} + \frac{1}{2}t$

**Residuals**: $\mathbf{e} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}} = (1-\frac{7}{6},\ 2-\frac{5}{3},\ 2-\frac{13}{6}) = (-\frac{1}{6},\ \frac{1}{3},\ -\frac{1}{6})$

ตรวจสอบ: $\mathbf{A}^T\mathbf{e} = \mathbf{0}$ (residuals ตั้งฉากกับ column space) ✓

```python
# ─── Least Squares ด้วย NumPy ──────────────────────────────────────────
# วัตถุประสงค์: fit เส้นตรง y = c + dt ผ่าน data points ด้วย Normal Equations
import numpy as np

# Design matrix: [1, t] สำหรับแต่ละ data point
# วัตถุประสงค์: คอลัมน์แรกเป็น 1 สำหรับ intercept, คอลัมน์ที่สองเป็น t
A = np.array([[1, 1],
              [1, 2],
              [1, 3]], dtype=float)
b = np.array([1, 2, 2], dtype=float)

# แก้ด้วย Normal Equations ด้วยตนเอง
# วัตถุประสงค์: แสดง derivation จริงที่อยู่เบื้องหลัง regression
ATA = A.T @ A
ATb = A.T @ b
x_hat_manual = np.linalg.solve(ATA, ATb)
print("Normal Equations solution:", x_hat_manual.round(4))

# แก้ด้วย np.linalg.lstsq (วิธีที่แนะนำสำหรับ production)
# วัตถุประสงค์: lstsq ใช้ SVD ภายใน — stable กว่าการ invert (XTX) โดยตรง
x_hat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print("lstsq solution:", x_hat.round(4))
print("Matrix rank:", rank)

# คำนวณ residuals และตรวจสอบ orthogonality
e = b - A @ x_hat
print(f"\nResiduals e: {e.round(4)}")
print(f"A^T @ e (ต้องใกล้ 0): {(A.T @ e).round(10)}")

# แสดงผลเป็น plot
import matplotlib.pyplot as plt
t = np.linspace(0.5, 3.5, 100)
y_fit = x_hat[0] + x_hat[1] * t

plt.figure(figsize=(6, 4))
# จุด data จริง
plt.scatter([1, 2, 3], [1, 2, 2], color='red', zorder=5, label='Data points')
# เส้น Least Squares
plt.plot(t, y_fit, 'b-', label=f'Least Squares: y = {x_hat[0]:.2f} + {x_hat[1]:.2f}t')
# Residuals (เส้นตั้งฉาก)
for xi, yi, pi in zip([1, 2, 3], [1, 2, 2], A @ x_hat):
    plt.vlines(xi, min(yi, pi), max(yi, pi), colors='gray', linestyles='--', alpha=0.7)
plt.legend(); plt.grid(True); plt.title("Least Squares Fit")
plt.savefig("least_squares_fit.png", dpi=100, bbox_inches='tight')
print("\nPlot saved!")
```

**DS Connection**: `np.linalg.lstsq()` คือ Least Squares solver ที่ใช้ SVD ภายใน `sklearn.linear_model.LinearRegression` ก็ใช้ Least Squares เช่นกัน แต่ optimize ด้วย Coordinate Descent สำหรับ sparse data

---

## Section 4: Gram-Schmidt และ QR Decomposition  *(Strang 4.4)*

ในส่วนนี้เราจะเรียนรู้การ**สร้างฐาน Orthonormal** จาก vectors ชุดหนึ่ง ซึ่งนำไปสู่ QR Decomposition ที่ stable กว่าการ invert matrix โดยตรง

### 4.1 Orthonormal Basis

**นิยาม**: เซต $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ เป็น Orthonormal ถ้า:
$$\mathbf{q}_i^T \mathbf{q}_j = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

เขียนรวมกัน: $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$ (ถ้า $\mathbf{Q}$ เป็น square matrix แล้ว $\mathbf{Q}^T = \mathbf{Q}^{-1}$)

### 4.2 Gram-Schmidt Process

**ปัญหา**: แปลง $k$ vectors ที่ linearly independent $\{\mathbf{a}_1, \ldots, \mathbf{a}_k\}$ เป็น orthonormal basis

**ขั้นตอน**:

**Step 1**: $\mathbf{A}_1 = \mathbf{a}_1$, normalize: $\mathbf{q}_1 = \mathbf{A}_1 / \|\mathbf{A}_1\|$

**Step 2**: ลบ projection ของ $\mathbf{a}_2$ บน $\mathbf{q}_1$ ออก:
$$\mathbf{A}_2 = \mathbf{a}_2 - (\mathbf{q}_1^T\mathbf{a}_2)\mathbf{q}_1, \quad \mathbf{q}_2 = \mathbf{A}_2 / \|\mathbf{A}_2\|$$

**Step k**: ลบ projections บน $\mathbf{q}_1, \ldots, \mathbf{q}_{k-1}$ ทั้งหมดออก:
$$\mathbf{A}_k = \mathbf{a}_k - \sum_{j=1}^{k-1}(\mathbf{q}_j^T\mathbf{a}_k)\mathbf{q}_j, \quad \mathbf{q}_k = \mathbf{A}_k / \|\mathbf{A}_k\|$$

**ตัวอย่าง** (2 vectors ใน $\mathbb{R}^3$):
$$\mathbf{a}_1 = \begin{pmatrix}1\\0\\1\end{pmatrix}, \quad \mathbf{a}_2 = \begin{pmatrix}1\\1\\0\end{pmatrix}$$

Step 1: $\|\mathbf{a}_1\| = \sqrt{2}$, $\mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\1\end{pmatrix}$

Step 2: $\mathbf{q}_1^T\mathbf{a}_2 = \frac{1}{\sqrt{2}}(1+0+0) = \frac{1}{\sqrt{2}}$

$$\mathbf{A}_2 = \begin{pmatrix}1\\1\\0\end{pmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\1\end{pmatrix} = \begin{pmatrix}1-1/2\\1\\0-1/2\end{pmatrix} = \begin{pmatrix}1/2\\1\\-1/2\end{pmatrix}$$

$\|\mathbf{A}_2\| = \sqrt{1/4 + 1 + 1/4} = \sqrt{3/2}$

$$\mathbf{q}_2 = \frac{1}{\sqrt{3/2}}\begin{pmatrix}1/2\\1\\-1/2\end{pmatrix} = \frac{1}{\sqrt{6}}\begin{pmatrix}1\\2\\-1\end{pmatrix}$$

ตรวจสอบ: $\mathbf{q}_1^T\mathbf{q}_2 = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{6}}(1 \cdot 1 + 0 \cdot 2 + 1 \cdot (-1)) = \frac{0}{\sqrt{12}} = 0$ ✓

### 4.3 QR Decomposition

**ผล** จาก Gram-Schmidt: $\mathbf{A} = \mathbf{QR}$

- $\mathbf{Q}$ = orthonormal basis จาก Gram-Schmidt (columns = $\mathbf{q}_1, \ldots, \mathbf{q}_k$)
- $\mathbf{R}$ = upper triangular matrix (เก็บ projections ที่ลบออก)

$$\begin{pmatrix}| & | \\ \mathbf{a}_1 & \mathbf{a}_2 \\ | & |\end{pmatrix} = \begin{pmatrix}| & | \\ \mathbf{q}_1 & \mathbf{q}_2 \\ | & |\end{pmatrix}\begin{pmatrix}r_{11} & r_{12} \\ 0 & r_{22}\end{pmatrix}$$

**Least Squares ด้วย QR** (stable version):

แทน $\mathbf{A} = \mathbf{QR}$ เข้าใน Normal Equations:
$$\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$$
$$(\mathbf{QR})^T(\mathbf{QR})\hat{\mathbf{x}} = (\mathbf{QR})^T\mathbf{b}$$
$$\mathbf{R}^T\underbrace{\mathbf{Q}^T\mathbf{Q}}_{=\mathbf{I}}\mathbf{R}\hat{\mathbf{x}} = \mathbf{R}^T\mathbf{Q}^T\mathbf{b}$$
$$\mathbf{R}\hat{\mathbf{x}} = \mathbf{Q}^T\mathbf{b}$$

→ แก้ Upper Triangular system ด้วย back substitution (ไม่ต้องหา $(\mathbf{A}^T\mathbf{A})^{-1}$!)

```python
# ─── QR Decomposition ด้วย NumPy ──────────────────────────────────────
# วัตถุประสงค์: แสดง QR เป็นวิธีที่ numerically stable สำหรับ Least Squares
import numpy as np

A = np.array([[1, 1],
              [1, 2],
              [1, 3]], dtype=float)
b = np.array([1, 2, 2], dtype=float)

# QR decomposition
# วัตถุประสงค์: แยก A = QR เพื่อหลีกเลี่ยงการคำนวณ (A^T A)^{-1}
Q, R = np.linalg.qr(A)

print("Q (orthonormal columns):\n", Q.round(4))
print("R (upper triangular):\n", R.round(4))

# ตรวจสอบ Q^T Q = I
# วัตถุประสงค์: verify orthonormality ของ Q columns
print("\nQ^T Q =\n", (Q.T @ Q).round(10))

# Least Squares ด้วย QR: แก้ Rx = Q^T b
# วัตถุประสงค์: stable กว่าการคำนวณ inverse โดยตรง
QTb = Q.T @ b
x_hat_qr = np.linalg.solve(R, QTb)
print("\nLeast Squares via QR:", x_hat_qr.round(4))

# เปรียบเทียบกับ Normal Equations
x_hat_normal = np.linalg.solve(A.T @ A, A.T @ b)
print("Least Squares via Normal Eq:", x_hat_normal.round(4))
print("Difference:", np.abs(x_hat_qr - x_hat_normal).max())
```

**DS Connection**: `np.linalg.lstsq()` และ `sklearn.linear_model.LinearRegression` ใช้ QR (หรือ SVD) ภายใน ไม่ใช่การ invert $(X^TX)$ โดยตรง เพราะ Condition Number ของ $X^TX$ = square ของ Condition Number ของ $X$ — ทำให้ numerical error รุนแรงขึ้น 2 เท่า

---

## Case Study: OLS Regression บน Advertising Dataset

**Scenario**  
ต้องการอธิบาย Sales โดยใช้งบโฆษณา TV และ Radio โดยใช้ Least Squares

**Data**: Advertising.csv — 200 ตลาด, TV (0–296.4 K$), Radio (0–49.6 K$), Sales (1.6–27.0 K units)

**Method — Least Squares**

```python
# ─── OLS Regression ด้วย Least Squares ────────────────────────────────
# วัตถุประสงค์: แสดงว่า Linear Regression = Least Squares บน Design Matrix
import numpy as np
import pandas as pd

# สมมติข้อมูลตัวอย่าง (subset ของ Advertising.csv)
np.random.seed(42)
n = 50
TV = np.random.uniform(0, 300, n)
Radio = np.random.uniform(0, 50, n)
# Sales จาก ground truth β₀=2, β_TV=0.05, β_Radio=0.1 + noise
Sales = 2 + 0.05*TV + 0.1*Radio + np.random.normal(0, 1.5, n)

# Design matrix: intercept + TV + Radio
# วัตถุประสงค์: X ขนาด n×3, แต่ละแถวคือ [1, TV_i, Radio_i]
X = np.column_stack([np.ones(n), TV, Radio])
print(f"Design matrix X shape: {X.shape}")
print(f"Rank of X: {np.linalg.matrix_rank(X)} (ต้องเป็น 3)")

# แก้ด้วย lstsq (QR internally)
# วัตถุประสงค์: หา β̂ ที่ minimize ‖Xβ - y‖²
beta_hat, _, _, _ = np.linalg.lstsq(X, Sales, rcond=None)
print(f"\nβ̂ = {beta_hat.round(4)}")
print(f"β₀ (intercept) = {beta_hat[0]:.3f}")
print(f"β_TV = {beta_hat[1]:.4f}")
print(f"β_Radio = {beta_hat[2]:.4f}")

# R² = 1 - SS_res / SS_tot
residuals = Sales - X @ beta_hat
SS_res = np.sum(residuals**2)
SS_tot = np.sum((Sales - Sales.mean())**2)
R2 = 1 - SS_res/SS_tot
print(f"\nR² = {R2:.4f}")
```

**Result**  
$\hat{\beta}_{TV} \approx 0.050$, $\hat{\beta}_{Radio} \approx 0.100$, $R^2 \approx 0.90$

**Insight**  
- Least Squares คือการ **project $\mathbf{y}$ ลงบน Column Space ของ $\mathbf{X}$** อย่างแม่นยำ
- $R^2$ บอกว่าเราอธิบาย variance ของ Sales ได้ 90% — ดีมาก
- ทุกค่าที่ sklearn คำนวณใน `model.coef_` มาจากกระบวนการนี้ทั้งสิ้น

---

## สรุป (Summary)

| แนวคิด | สูตร | Python |
|--------|------|--------|
| Orthogonality | $\mathbf{u}^T\mathbf{v} = 0$ | `np.dot(u, v)` |
| Projection (line) | $\hat{x} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}$ | manual |
| Projection Matrix | $\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$ | `A @ np.linalg.solve(A.T@A, A.T)` |
| Normal Equations | $\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$ | `np.linalg.solve(A.T@A, A.T@b)` |
| Least Squares | $\hat{\mathbf{x}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$ | `np.linalg.lstsq(A, b)` |
| Gram-Schmidt | ลบ projections ออกทีละตัว | manual |
| QR Decomposition | $\mathbf{A} = \mathbf{QR}$ | `np.linalg.qr(A)` |
| Orthonormal Q | $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$ | verify: `Q.T @ Q` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 3 (note 1)**: Column Space และ Orthogonality ของ Four Subspaces เป็นรากฐานของ Projection
- ← **Week 2**: Normal Equations $(\mathbf{A}^T\mathbf{A})\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$ ใช้ Matrix Operations และ Inverse จาก Week 2
- → **Week 4**: QR เชื่อมกับ SVD — $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ เป็น "ultimate decomposition" ที่รวม Orthogonality และ Rank เข้าด้วยกัน
- → **Week 8**: Normal Equation ของ Linear Regression $(\mathbf{X}^T\mathbf{X})\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$ คือ Least Squares ที่เรียนในสัปดาห์นี้ทุกประการ
