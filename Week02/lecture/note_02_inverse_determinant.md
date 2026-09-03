# Note 2: Inverse Matrix และ Determinant
> Week 2 | CLO1 | Strang Reference: Ch.2.5 (Inverse), Ch.5 (Determinants)

---

## บทนำ

ในบันทึกนี้เราจะต่อจาก Gaussian Elimination มาเรียนเครื่องมือสำคัญอีกสองอย่างของ Linear Algebra ได้แก่ **Inverse Matrix** (แมทริกซ์ผกผัน) และ **Determinant** ซึ่งทั้งสองเชื่อมกันอย่างลึกซึ้ง เป้าหมายคือให้นักศึกษาเข้าใจว่าแมทริกซ์ตัวไหน "หาผกผันได้" และตัวไหน "หาไม่ได้" เพราะใน Machine Learning ถ้า Feature Matrix มีปัญหา (Singular) โมเดลจะ **break** ทันที Determinant เป็นเครื่องทดสอบที่รวดเร็วว่าแมทริกซ์นั้นใช้งานได้หรือไม่ และยังมีความหมายเชิงเรขาคณิตที่สวยงามมาก ในชีวิตจริง Inverse Matrix ใช้ในการแก้สมการ Normal Equation ของ Regression ส่วน Determinant ใช้ตรวจสอบ Multicollinearity ใน Feature Engineering

---

## Section 1: Inverse Matrix  *(Strang 2.5)*

ในส่วนนี้เราจะเรียนว่า **Inverse Matrix คืออะไร** เมื่อไหรมีอยู่ และคำนวณอย่างไร เพื่อให้สามารถแก้สมการ $\mathbf{Ax} = \mathbf{b}$ ในรูปแบบ $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ ได้

### 1.1 นิยามและเงื่อนไขการมีอยู่

**นิยาม**: $\mathbf{A}^{-1}$ คือแมทริกซ์ที่ทำให้:
$$\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$$

**เงื่อนไขการมี Inverse (Invertible / Non-singular)**:
1. $\mathbf{A}$ ต้องเป็น **Square Matrix** ($n \times n$)
2. $\det(\mathbf{A}) \neq 0$ (จะเรียนใน Section 2)
3. Columns ของ $\mathbf{A}$ ต้อง **Linearly Independent**
4. Gaussian Elimination บน $\mathbf{A}$ ต้องได้ pivot ทุกแถว (ไม่มีแถว 0)

**Singular Matrix**: แมทริกซ์ที่ไม่มี Inverse เรียกว่า **Singular** หรือ **Degenerate**

ตัวอย่าง Singular Matrix:
$$\mathbf{S} = \begin{pmatrix}1&2\\2&4\end{pmatrix} \quad \text{เหตุผล: แถวที่ 2 = 2 × แถวที่ 1 (linearly dependent)}$$

### 1.2 สูตรสำหรับ 2×2

สำหรับ $\mathbf{A} = \begin{pmatrix}a&b\\c&d\end{pmatrix}$:

$$\mathbf{A}^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$$

โดย $ad - bc = \det(\mathbf{A})$ (ถ้า $\det = 0$ ไม่มี inverse)

**ตัวอย่าง step-by-step**:
$$\mathbf{A} = \begin{pmatrix}2&1\\5&3\end{pmatrix}$$

$$\det(\mathbf{A}) = (2)(3) - (1)(5) = 6 - 5 = 1$$

$$\mathbf{A}^{-1} = \frac{1}{1}\begin{pmatrix}3&-1\\-5&2\end{pmatrix} = \begin{pmatrix}3&-1\\-5&2\end{pmatrix}$$

**ตรวจสอบ** $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$:
$$\begin{pmatrix}3&-1\\-5&2\end{pmatrix}\begin{pmatrix}2&1\\5&3\end{pmatrix} = \begin{pmatrix}6-5&3-3\\-10+10&-5+6\end{pmatrix} = \begin{pmatrix}1&0\\0&1\end{pmatrix} = \mathbf{I} \checkmark$$

### 1.3 Gauss-Jordan Elimination สำหรับ n×n

**ขั้นตอน**: สร้าง augmented matrix $[\mathbf{A}|\mathbf{I}]$ แล้วทำ row operations จนซ้ายกลายเป็น $\mathbf{I}$ → ขวาจะกลายเป็น $\mathbf{A}^{-1}$

$$[\mathbf{A}|\mathbf{I}] \xrightarrow{\text{row ops}} [\mathbf{I}|\mathbf{A}^{-1}]$$

**ตัวอย่าง** $3 \times 3$:
$$\mathbf{A} = \begin{pmatrix}1&2&0\\0&1&0\\0&0&2\end{pmatrix}$$

เริ่ม:
$$\left[\begin{array}{ccc|ccc}1&2&0&1&0&0\\0&1&0&0&1&0\\0&0&2&0&0&1\end{array}\right]$$

$R_1 \leftarrow R_1 - 2R_2$:
$$\left[\begin{array}{ccc|ccc}1&0&0&1&-2&0\\0&1&0&0&1&0\\0&0&2&0&0&1\end{array}\right]$$

$R_3 \leftarrow R_3 / 2$:
$$\left[\begin{array}{ccc|ccc}1&0&0&1&-2&0\\0&1&0&0&1&0\\0&0&1&0&0&1/2\end{array}\right]$$

$$\therefore \mathbf{A}^{-1} = \begin{pmatrix}1&-2&0\\0&1&0\\0&0&1/2\end{pmatrix}$$

### 1.4 คุณสมบัติของ Inverse

$$(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1} \quad \text{(สังเกตว่า order กลับ!)}$$

$$(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$$

$$(\mathbf{A}^{-1})^{-1} = \mathbf{A}$$

**DS Connection**: ใน OLS Regression สูตร Normal Equation คือ:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

ถ้า $\mathbf{X}^T\mathbf{X}$ เป็น **Singular** (det = 0) — หมายความว่ามี features ที่ **linearly dependent** กัน (Multicollinearity) และหา $\hat{\boldsymbol{\beta}}$ ไม่ได้! นี่คือเหตุผลที่ต้องเช็ค correlation matrix ของ features ก่อน fit model

```python
# ─── คำนวณ Inverse Matrix ด้วย NumPy ────────────────────────────────
# วัตถุประสงค์: แสดงการใช้ inverse ในบริบท Data Science (OLS regression)
import numpy as np

# สร้าง Design Matrix X (สมมติ feature TV และ Radio จาก Advertising data)
# วัตถุประสงค์: แต่ละแถวคือ data point, แต่ละคอลัมน์คือ feature
X = np.array([[1, 230.1, 37.8],  # intercept + TV + Radio
              [1,  44.5, 39.3],
              [1,  17.2, 45.9],
              [1, 151.5, 41.3],
              [1, 180.8, 10.8]], dtype=float)

y = np.array([22.1, 10.4, 9.3, 18.5, 12.9])

# คำนวณ (X^T X)^{-1} X^T y — Normal Equation
# วัตถุประสงค์: หา beta hat โดยตรงจาก closed-form solution ของ OLS
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
beta_hat = XTX_inv @ X.T @ y
print("β̂:", beta_hat.round(4))

# ตรวจสอบว่า X^T X หา inverse ได้ (det ≠ 0)
# วัตถุประสงค์: ป้องกัน singular matrix ก่อนคำนวณ
det_val = np.linalg.det(XTX)
print(f"det(X^T X) = {det_val:.4f}")
print(f"Matrix {'is invertible' if abs(det_val) > 1e-10 else 'is SINGULAR — cannot invert!'}")
```

---

## Section 2: Determinant  *(Strang Ch.5)*

ในส่วนนี้เราจะเรียนรู้ว่า **Determinant** คืออะไร คำนวณอย่างไร และมีความหมายเชิงเรขาคณิตที่สวยงามอย่างไร เพราะ Determinant เป็นเครื่องตรวจสอบ "ความ invertible" ของ Matrix ในรูปแบบตัวเลขตัวเดียว

### 2.1 นิยามและการคำนวณ

#### 2×2 Determinant

$$\det\begin{pmatrix}a&b\\c&d\end{pmatrix} = ad - bc$$

**ตัวอย่าง**:
$$\det\begin{pmatrix}3&1\\2&4\end{pmatrix} = (3)(4) - (1)(2) = 12 - 2 = 10$$

$$\det\begin{pmatrix}2&4\\1&2\end{pmatrix} = (2)(2) - (4)(1) = 4 - 4 = 0 \quad \Rightarrow \text{Singular!}$$

#### 3×3 Determinant (Cofactor Expansion)

ขยายตามแถวแรก:

$$\det(\mathbf{A}) = a_{11} C_{11} + a_{12} C_{12} + a_{13} C_{13}$$

โดย **Cofactor** $C_{ij} = (-1)^{i+j} M_{ij}$ และ **Minor** $M_{ij}$ คือ determinant ของ submatrix ที่ตัดแถว $i$ คอลัมน์ $j$ ออก

**ตัวอย่าง step-by-step**:
$$\mathbf{A} = \begin{pmatrix}1&2&3\\0&4&5\\1&0&6\end{pmatrix}$$

ขยายตามแถวที่ 1:
$$\det(\mathbf{A}) = 1 \cdot \det\begin{pmatrix}4&5\\0&6\end{pmatrix} - 2 \cdot \det\begin{pmatrix}0&5\\1&6\end{pmatrix} + 3 \cdot \det\begin{pmatrix}0&4\\1&0\end{pmatrix}$$

$$= 1 \cdot (24-0) - 2 \cdot (0-5) + 3 \cdot (0-4)$$

$$= 24 - 2(-5) + 3(-4) = 24 + 10 - 12 = 22$$

### 2.2 คุณสมบัติสำคัญของ Determinant

| คุณสมบัติ | สูตร | ความหมาย |
|----------|------|---------|
| Multiplicative | $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$ | product ของ det |
| Transpose | $\det(\mathbf{A}^T) = \det(\mathbf{A})$ | transpose ไม่เปลี่ยน det |
| Row swap | $\det$ เปลี่ยนเครื่องหมาย | สลับแถว → คูณด้วย -1 |
| Row scale | $\det(c\mathbf{A}) = c^n \det(\mathbf{A})$ | คูณทั้งแถวด้วย $c$ |
| Singular | $\det(\mathbf{A}) = 0 \Leftrightarrow$ Singular | แถวขึ้นอยู่กัน |
| Identity | $\det(\mathbf{I}) = 1$ | — |
| Inverse | $\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$ | — |

**ความสัมพันธ์กับ LU**:
$$\det(\mathbf{A}) = \det(\mathbf{L}) \cdot \det(\mathbf{U}) = 1 \cdot \prod_{i=1}^{n} U_{ii} = \prod_{i=1}^{n} U_{ii}$$

(เพราะ $\det(\mathbf{L}) = 1$ จาก diagonal ของ L เป็น 1 ทั้งหมด)

### 2.3 ความหมายเชิงเรขาคณิต

$|\det(\mathbf{A})|$ คือ **ปัจจัยการขยาย (scaling factor)** ของพื้นที่/ปริมาตรที่ Matrix $\mathbf{A}$ กระทำต่อ space

**ใน 2D**: $|\det(\mathbf{A})|$ = พื้นที่ของ parallelogram ที่สร้างจาก column vectors ของ $\mathbf{A}$

$$\mathbf{A} = \begin{pmatrix}3&1\\0&2\end{pmatrix} \quad \Rightarrow \quad |\det| = |6-0| = 6$$

ตีความ: transformation $\mathbf{A}$ ขยายพื้นที่ทุกรูป 6 เท่า

**เมื่อ $\det = 0$**: ทุกรูปทรงถูก "บีบ" ให้เหลือมิติที่ต่ำกว่า (จาก 2D → เส้น, จาก 3D → ระนาบ) ข้อมูลสูญหาย → ไม่สามารถย้อนกลับได้ → ไม่มี Inverse!

**เครื่องหมายของ $\det$**:
- $\det > 0$: transformation รักษา **orientation** (ด้าน)
- $\det < 0$: transformation **สะท้อน** (flip)

### 2.4 Cramer's Rule (แนวคิด)

สำหรับระบบสมการ $\mathbf{Ax} = \mathbf{b}$ (ถ้า $\det(\mathbf{A}) \neq 0$):

$$x_j = \frac{\det(\mathbf{A}_j)}{\det(\mathbf{A})}$$

โดย $\mathbf{A}_j$ คือ $\mathbf{A}$ ที่แทนคอลัมน์ $j$ ด้วย $\mathbf{b}$

**หมายเหตุ**: Cramer's Rule มีประโยชน์เชิงทฤษฎี แต่ไม่ใช้ในทางปฏิบัติเพราะช้ากว่า Gaussian Elimination มาก (complexity $O(n!)$ vs $O(n^3)$)

```python
# ─── คำนวณ Determinant และตรวจสอบ Singularity ──────────────────────
# วัตถุประสงค์: แสดง workflow จริงที่ใช้ detect multicollinearity ใน ML
import numpy as np

# สร้าง matrix ที่ปกติ (invertible)
A_good = np.array([[2, 1, -1],
                   [-3, -1, 2],
                   [-2, 1, 2]], dtype=float)

# สร้าง matrix ที่มีปัญหา (singular — คอลัมน์ที่ 3 = คอลัมน์ 1 + คอลัมน์ 2)
A_bad = np.array([[1, 2, 3],
                  [4, 5, 9],   # row_3 = row_1 + row_2
                  [7, 8, 15]], dtype=float)

det_good = np.linalg.det(A_good)
det_bad = np.linalg.det(A_bad)

print(f"det(A_good) = {det_good:.4f}")  # non-zero
print(f"det(A_bad)  = {det_bad:.6f}")  # ≈ 0 (singular)

# ตรวจสอบด้วย condition number (วิธีที่ robust กว่า det)
# วัตถุประสงค์: det ≈ 0 อาจเป็น numerical issue, cond number บอก "ความใกล้ singular"
cond_good = np.linalg.cond(A_good)
cond_bad = np.linalg.cond(A_bad)

print(f"\ncond(A_good) = {cond_good:.2f}")
print(f"cond(A_bad)  = {cond_bad:.2e}")  # ขนาดใหญ่มาก = near singular

# แสดงความหมายเรขาคณิต: det คือ scaling factor ของพื้นที่
# วัตถุประสงค์: verify ว่า |det| = พื้นที่ parallelogram จาก column vectors
v1 = np.array([3.0, 0.0])
v2 = np.array([1.0, 2.0])
M_2d = np.column_stack([v1, v2])
area = abs(np.linalg.det(M_2d))
print(f"\nColumn vectors: {v1}, {v2}")
print(f"|det| = พื้นที่ parallelogram = {area:.2f} ตารางหน่วย")
```

---

## Case Study: Multicollinearity Detection — เมื่อ Feature Matrix เป็น Singular

**Scenario**  
ทีม Data Science กำลัง fit โมเดลพยากรณ์ราคาบ้าน โดยใช้ features 3 ตัว: พื้นที่ (ตร.ม.), จำนวนห้อง, และ "พื้นที่ต่อห้อง" ที่คำนวณเองจากสองตัวแรก

**Data**  
```
Feature 1: area = [100, 150, 200]
Feature 2: rooms = [2, 3, 4]  
Feature 3: area_per_room = [50, 50, 50]  ← area / rooms ← สร้างจาก 2 ตัวแรก!
```

**Method**

```python
# ─── ตรวจสอบ Multicollinearity ด้วย Determinant ────────────────────
# วัตถุประสงค์: แสดงว่า feature ที่ linearly dependent ทำให้ matrix singular
import numpy as np

# Design matrix X (พร้อม intercept column)
X = np.array([[1, 100, 2, 50],
              [1, 150, 3, 50],
              [1, 200, 4, 50]], dtype=float)

# XTX ขนาด 4x4
XTX = X.T @ X

det_XTX = np.linalg.det(XTX)
print(f"det(X^T X) = {det_XTX:.6f}")  # ≈ 0!

# รัน regression โดยไม่ตรวจสอบก่อน
try:
    beta = np.linalg.inv(XTX) @ X.T @ np.array([500, 700, 900], dtype=float)
except np.linalg.LinAlgError as e:
    print(f"ERROR: {e}")

# วิธีแก้: ตัด feature ที่ redundant ออก
# วัตถุประสงค์: ใช้เฉพาะ feature ที่ independent กัน
X_clean = X[:, :3]  # ตัด area_per_room ออก
XTX_clean = X_clean.T @ X_clean
print(f"det(X_clean^T X_clean) = {np.linalg.det(XTX_clean):.4f}")  # non-zero
```

**Result**  
`det(X^T X) ≈ 0` → `LinAlgError: Singular matrix`

**Insight**
- เมื่อ features มี linear dependency → Design Matrix เป็น Singular → ไม่มี Inverse → หา $\hat{\boldsymbol{\beta}}$ ไม่ได้
- ใน sklearn, `LinearRegression` ใช้ pseudoinverse (SVD) แทน จึงไม่ error แต่ค่า $\hat{\boldsymbol{\beta}}$ จะไม่มีความหมาย
- เครื่องมือที่ใช้จริงใน production: **Variance Inflation Factor (VIF)** และ **Condition Number** สำหรับ detect multicollinearity ก่อน fit model

---

## สรุป (Summary)

| แนวคิด | สูตร | Python Function |
|--------|------|----------------|
| Inverse (2×2) | $\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$ | `np.linalg.inv(A)` |
| Gauss-Jordan | $[\mathbf{A}|\mathbf{I}] \to [\mathbf{I}|\mathbf{A}^{-1}]$ | manual |
| $(AB)^{-1}$ | $\mathbf{B}^{-1}\mathbf{A}^{-1}$ | ต้องระวังลำดับ |
| det (2×2) | $ad - bc$ | `np.linalg.det(A)` |
| det จาก LU | $\prod U_{ii}$ | `scipy.linalg.lu` |
| $\det(\mathbf{AB})$ | $\det(\mathbf{A})\det(\mathbf{B})$ | — |
| Geometric det | scaling factor ของพื้นที่/ปริมาตร | — |
| Singular test | $\det = 0 \Leftrightarrow$ ไม่มี inverse | `np.linalg.cond(A)` |
| Condition Number | ยิ่งสูงยิ่ง near-singular | `np.linalg.cond(A)` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 2 (note 1)**: Gaussian Elimination สร้าง LU ซึ่งนำมาใช้คำนวณ det ได้ $\det(\mathbf{A}) = \prod U_{ii}$
- → **Week 3**: Rank ของ Matrix — ถ้า $\text{rank}(\mathbf{A}) < n$ แล้ว $\det = 0$ และ Column Space ไม่เต็ม $\mathbb{R}^n$
- → **Week 4**: $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$ คือ **Characteristic Equation** ที่ใช้หา Eigenvalues — det เป็นหัวใจของ Eigenvalue problem
- → **Week 8**: Normal Equation $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ — ต้องการ $\det(\mathbf{X}^T\mathbf{X}) \neq 0$ จึงจะ fit Regression ได้
