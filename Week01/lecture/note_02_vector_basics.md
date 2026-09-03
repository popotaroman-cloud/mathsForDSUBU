# Note 2: Scalar, Vector และการสร้าง Vector ด้วย NumPy
> Week 1 | CLO1 | Strang Reference: Ch.1 (Vectors and Linear Combinations)

---

## บทนำ

สัปดาห์นี้เราเรียนรู้ว่า **Vector คืออะไรและทำไมมันถึงสำคัญสำหรับ Data Science** ในบันทึกการสอนฉบับนี้เราจะลงลึกในเรื่อง **NumPy** ซึ่งเป็น library Python ที่ใช้จัดการ Vector และ Matrix ทุกอย่างใน Machine Learning เป้าหมาย (LLo) คือให้นักศึกษา **สร้าง Vector ได้** และทำการดำเนินการพื้นฐานทาง Python ได้อย่างถูกต้อง การเรียน NumPy อย่างลึกซึ้งตั้งแต่สัปดาห์แรกจะช่วยให้การทำ Lab ในสัปดาห์ต่อ ๆ ไปรวดเร็วและไม่เสียเวลากับ syntax การนำไปใช้ใน Data Science ได้แก่ การสร้าง feature matrix, การประมวลผลภาพ, และการทำ vectorized computation ซึ่งเร็วกว่า Python loop ธรรมดาหลายร้อยเท่า

---

## Section 1: แนะนำ NumPy  *(NumPy Fundamentals)*

ในส่วนนี้เราจะติดตั้งและทดลองใช้ **NumPy** เพื่อเข้าใจว่า Python จัดการ Vector อย่างไรและทำไมต้องใช้ NumPy แทน Python list ธรรมดา

### 1.1 ทำไมต้อง NumPy?

| Python List | NumPy Array |
|-------------|-------------|
| ช้า (loop ทีละ element) | เร็ว (vectorized C-code) |
| ไม่รองรับ math operations โดยตรง | รองรับทุก mathematical operation |
| ไม่มี shape/dtype | มี shape, dtype, ndim |
| หน่วยความจำไม่ต่อเนื่อง | หน่วยความจำต่อเนื่อง (cache-friendly) |

**ตัวอย่างความต่าง**: `[1, 2, 3] + [4, 5, 6]` → `[1, 2, 3, 4, 5, 6]` (concatenate!)
แต่ `np.array([1,2,3]) + np.array([4,5,6])` → `[5, 7, 9]` (element-wise add ✓)

### 1.2 การสร้าง ndarray

```python
# ─── สร้าง NumPy Array พื้นฐาน ─────────────────────────────────────
# วัตถุประสงค์: เรียนรู้วิธีสร้าง array จาก list และ special functions
# เพื่อสร้าง vector/matrix ที่ใช้ใน ML ได้หลายรูปแบบ

import numpy as np

# วิธีที่ 1: จาก Python list
v = np.array([1, 2, 3, 4, 5])
print("Vector v:", v)
print("Shape:", v.shape)   # (5,) — 1D array, 5 elements
print("Dtype:", v.dtype)   # int64 (หรือ int32 ขึ้นกับ OS)
print("Ndim:", v.ndim)     # 1 — เป็น 1 มิติ

# วิธีที่ 2: ระบุ dtype
v_float = np.array([1.0, 2.5, 3.7], dtype=np.float64)
print("\nFloat vector:", v_float)

# วิธีที่ 3: zeros และ ones
zeros_vec = np.zeros(5)       # [0. 0. 0. 0. 0.]
ones_vec  = np.ones(4)        # [1. 1. 1. 1.]
print("\nZeros:", zeros_vec)
print("Ones:", ones_vec)

# วิธีที่ 4: arange และ linspace
v_range  = np.arange(0, 10, 2)          # [0 2 4 6 8]
v_linsp  = np.linspace(0, 1, 5)         # [0.   0.25 0.5  0.75 1.  ]
print("\nArange:", v_range)
print("Linspace:", v_linsp)

# วิธีที่ 5: random vectors (ใช้บ่อยมากใน ML initialization)
np.random.seed(42)                        # กำหนด seed เพื่อ reproducibility
v_rand = np.random.randn(5)              # random จาก standard normal
print("\nRandom (normal):", v_rand)
```

### 1.3 Column Vector vs Row Vector

```python
# ─── Column Vector และ Row Vector ─────────────────────────────────
# วัตถุประสงค์: แสดงความแตกต่างระหว่าง 1D array, column vector, row vector
# เพราะใน Linear Algebra convention สำคัญมากเรื่อง shape

import numpy as np

# 1D array (flat) — ไม่มีทิศทางชัดเจน
v_1d = np.array([1, 2, 3])
print("1D array:", v_1d, "| shape:", v_1d.shape)  # (3,)

# Column Vector — shape (n, 1)
v_col = v_1d.reshape(-1, 1)          # หรือ v_1d[:, np.newaxis]
print("\nColumn vector:\n", v_col, "| shape:", v_col.shape)  # (3, 1)

# Row Vector — shape (1, n)
v_row = v_1d.reshape(1, -1)
print("\nRow vector:\n", v_row, "| shape:", v_row.shape)  # (1, 3)

# Transpose: สลับ row ↔ column
print("\nColumn.T =", v_col.T, "| shape:", v_col.T.shape)  # (1, 3)
```

---

## Section 2: Vector Operations ด้วย NumPy  *(Strang 1.1)*

ในส่วนนี้เราจะใช้ NumPy ทำ **Vector Operations** ทั้งหมดที่เรียนใน Note 1 เพื่อยืนยันว่า NumPy ทำงานตรงกับคณิตศาสตร์

### 2.1 Vector Addition และ Scalar Multiplication

```python
# ─── Vector Addition & Scalar Multiplication ───────────────────────
# วัตถุประสงค์: แสดงว่า NumPy ทำ element-wise operations ได้โดยตรง
# ซึ่งเร็วกว่าการเขียน loop มาก — สำคัญมากสำหรับ big data

import numpy as np

u = np.array([2, 3, 1])
v = np.array([1, -1, 4])

# การบวก vector
result_add = u + v
print("u + v =", result_add)   # [3, 2, 5]

# การลบ vector
result_sub = u - v
print("u - v =", result_sub)   # [1, 4, -3]

# คูณ scalar
c = 3
result_scale = c * u
print(f"{c} * u =", result_scale)  # [6, 9, 3]

# Linear combination: 2u + (-1)v
lc = 2 * u + (-1) * v
print("2u + (-1)v =", lc)      # [3, 7, -2]
```

### 2.2 Dot Product และ Norm

```python
# ─── Dot Product และ Norm ──────────────────────────────────────────
# วัตถุประสงค์: คำนวณ dot product และ norm ซึ่งเป็นพื้นฐาน
# ของ cosine similarity, projection, และ optimization ทุกประเภท

import numpy as np

v = np.array([1.0, 2.0, 3.0])
w = np.array([4.0, 5.0, 6.0])

# Dot Product — 3 วิธีที่ได้ผลเหมือนกัน
dot1 = np.dot(v, w)           # วิธีที่ 1: np.dot()
dot2 = v @ w                  # วิธีที่ 2: @ operator (Python 3.5+)
dot3 = np.sum(v * w)          # วิธีที่ 3: element-wise multiply แล้ว sum
print(f"v·w = {dot1} = {dot2} = {dot3}")  # ทุกอันได้ 32.0

# Norm (Euclidean norm = L2 norm)
norm_v = np.linalg.norm(v)
print(f"||v|| = {norm_v:.4f}")  # 3.7417

# Unit Vector
unit_v = v / np.linalg.norm(v)
print(f"Unit vector v̂ = {unit_v}")
print(f"||v̂|| = {np.linalg.norm(unit_v):.6f}")  # ควรได้ 1.0

# Cosine Similarity
cos_sim = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
print(f"Cosine similarity(v, w) = {cos_sim:.4f}")

# ตรวจสอบ: ถ้า w = 2v ควรได้ cos = 1.0
w2 = 2 * v
cos_parallel = np.dot(v, w2) / (np.linalg.norm(v) * np.linalg.norm(w2))
print(f"Cosine similarity(v, 2v) = {cos_parallel:.4f}")  # 1.0
```

---

## Section 3: Broadcasting ใน NumPy  *(NumPy Advanced)*

ในส่วนนี้เราจะเรียน **Broadcasting** ซึ่งเป็นคุณสมบัติพิเศษของ NumPy ที่ทำให้ทำ operations บน arrays ที่ shape ต่างกันได้อย่างชาญฉลาด เพื่อให้เขียน code ที่สั้น อ่านง่าย และเร็ว

### 3.1 Broadcasting Rules

```python
# ─── Broadcasting ─────────────────────────────────────────────────
# วัตถุประสงค์: แสดง broadcasting — NumPy ขยาย array อัตโนมัติ
# เพื่อให้ operations บน arrays ขนาดต่างกันทำงานได้
# ใช้บ่อยมากใน feature normalization ใน ML preprocessing

import numpy as np

# ตัวอย่าง 1: scalar กับ vector
v = np.array([1, 2, 3, 4, 5])
print("v + 10 =", v + 10)     # [11, 12, 13, 14, 15]  → scalar ถูก broadcast
print("v * 2  =", v * 2)      # [2, 4, 6, 8, 10]

# ตัวอย่าง 2: normalize vector (ลบ mean, หาร std)
data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
mean = data.mean()
std  = data.std()
normalized = (data - mean) / std     # broadcasting: scalar ถูกนำไปทำทุก element
print(f"\nOriginal: {data}")
print(f"Mean: {mean}, Std: {std:.2f}")
print(f"Normalized: {normalized.round(3)}")

# ตัวอย่าง 3: ระยะห่างระหว่าง vector หลาย ๆ ตัว (vectorized)
# dataset: 4 data points ใน 3 มิติ
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [1, 0, 1]])
query = np.array([3, 3, 3])

# คำนวณ Euclidean distance จาก query ถึงทุก data point พร้อมกัน
diff = X - query                        # (4, 3) - (3,) → broadcast → (4, 3)
distances = np.linalg.norm(diff, axis=1)  # norm ตาม axis=1 (ตาม row)
print(f"\nDistances from query {query}: {distances.round(3)}")
```

---

## Section 4: Linear Combinations และ Span  *(Strang 1.3)*

ในส่วนนี้เราจะทำความเข้าใจ **Linear Combination** และ **Span** ซึ่งนำไปสู่แนวคิดสำคัญมากคือ Column Space ในสัปดาห์ที่ 3

**นิยาม (Span)**: Span ของเซต $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ คือเซตของ vector ทุกตัวที่สร้างได้จาก linear combination:
$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k \mid c_i \in \mathbb{R}\}$$

**ตัวอย่าง**: $\mathbf{v}_1 = \begin{pmatrix}1\\0\end{pmatrix}$, $\mathbf{v}_2 = \begin{pmatrix}0\\1\end{pmatrix}$
$\Rightarrow$ span$\{\mathbf{v}_1, \mathbf{v}_2\} = \mathbb{R}^2$ (ทั้งระนาบ)

**ตัวอย่าง**: $\mathbf{v}_1 = \begin{pmatrix}1\\2\end{pmatrix}$, $\mathbf{v}_2 = \begin{pmatrix}2\\4\end{pmatrix} = 2\mathbf{v}_1$
$\Rightarrow$ span$\{\mathbf{v}_1, \mathbf{v}_2\}$ = เส้นตรง $y = 2x$ เท่านั้น! (ไม่ใช่ทั้งระนาบ)

```python
# ─── Visualize Linear Combinations ────────────────────────────────
# วัตถุประสงค์: วาดภาพ span ของ vector 2 ตัวใน R^2
# เพื่อให้เห็นภาพว่า linear combination "สร้าง" พื้นที่อย่างไร

import numpy as np
import matplotlib.pyplot as plt

# Vector พื้นฐาน
v1 = np.array([1, 2])
v2 = np.array([3, 1])
origin = np.array([0, 0])

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# วาด v1 และ v2
ax.annotate('', xy=v1, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=v2, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# วาด linear combination: 1*v1 + 1*v2
lc = v1 + v2
ax.annotate('', xy=lc, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='dashed'))

ax.text(v1[0]+0.1, v1[1], 'v₁=(1,2)', color='blue', fontsize=12)
ax.text(v2[0]+0.1, v2[1], 'v₂=(3,1)', color='red', fontsize=12)
ax.text(lc[0]+0.1, lc[1], 'v₁+v₂=(4,3)', color='green', fontsize=12)

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.grid(True, alpha=0.3)
ax.set_title('Linear Combination ของ Vector', fontsize=14)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')

plt.tight_layout()
plt.savefig('linear_combination.png', dpi=100)
plt.show()
print("บันทึก linear_combination.png แล้ว")
```

---

## สรุป (Summary)

| แนวคิด | สูตร | Python Function |
|--------|------|----------------|
| สร้าง Vector | — | `np.array([...])` |
| Shape / Dtype | — | `.shape`, `.dtype`, `.ndim` |
| Column Vector | $(n, 1)$ | `.reshape(-1, 1)` |
| Vector Addition | $\mathbf{u}+\mathbf{v}$ | `u + v` |
| Scalar Multiply | $c\mathbf{v}$ | `c * v` |
| Dot Product | $\sum v_i w_i$ | `np.dot(v, w)` หรือ `v @ w` |
| Norm | $\sqrt{\sum v_i^2}$ | `np.linalg.norm(v)` |
| Unit Vector | $\mathbf{v}/\|\mathbf{v}\|$ | `v / np.linalg.norm(v)` |
| Cosine Similarity | $\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}$ | manual หรือ `sklearn` |
| Broadcasting | — | ทำงานอัตโนมัติ |
| Zeros / Ones | — | `np.zeros(n)`, `np.ones(n)` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **ก่อนหน้า**: Note 1 — นิยาม scalar, vector, และ cosine similarity (ทฤษฎี)
- → **Week 2**: Vector หลาย ๆ ตัวรวมเป็น Matrix — เราจะเรียน Matrix multiplication และ Gaussian Elimination
- → **Week 3**: Dot product นำไปสู่ Orthogonality และ Projection — พื้นฐานของ Least Squares
- → **Week 4**: Unit vector และ basis เชื่อมกับ Eigenvector และ Principal Components ใน PCA
