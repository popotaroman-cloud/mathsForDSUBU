# Slide Deck: แนะนำ NumPy — Array และ ndarray พื้นฐาน
> Week 01 | CLO1 | 10 slides

---
## Slide 1 — Title
**NumPy: เครื่องมือหลักสำหรับ Vector/Matrix ใน Python**  
Week 1 | CLO1 | Lab Preparation | LLo: ใช้ NumPy สร้างและดำเนินการกับ Vector ได้

---
## Slide 2 — NumPy คืออะไร และทำไมถึงเร็ว?
**Key Message**: NumPy เร็วกว่า Python list 10–100 เท่าเพราะใช้ C ข้างหลัง

**NumPy** = Numerical Python — library สำหรับ scientific computing

```python
import numpy as np
import time

n = 1_000_000
lst = list(range(n))
arr = np.arange(n)

# Python list
t0 = time.time()
total = sum(x**2 for x in lst)
print(f"List: {time.time()-t0:.3f}s")

# NumPy
t0 = time.time()
total = (arr**2).sum()
print(f"NumPy: {time.time()-t0:.3f}s")
# NumPy เร็วกว่า ~50x
```

- C memory layout → no Python overhead
- Vectorized operations → no Python loop
- BLAS/LAPACK → optimized linear algebra routines

---
## Slide 3 — สร้าง Array: 5 วิธีหลัก
**Key Message**: เลือกวิธีสร้าง array ให้ถูกกับ use case

```python
# 1. จาก Python list
v1 = np.array([1, 2, 3, 4])

# 2. Special arrays
v2 = np.zeros(5)          # [0. 0. 0. 0. 0.]
v3 = np.ones(4)           # [1. 1. 1. 1.]
v4 = np.eye(3)            # identity matrix 3×3

# 3. ช่วงตัวเลข
v5 = np.arange(0, 10, 2)  # [0 2 4 6 8]
v6 = np.linspace(0, 1, 5) # [0.   0.25 0.5  0.75 1.  ]

# 4. Random
np.random.seed(42)
v7 = np.random.randn(4)   # standard normal
v8 = np.random.uniform(0, 1, 4)

# 5. จาก Pandas DataFrame
# arr = df['column'].values
```

---
## Slide 4 — Attributes สำคัญ: shape, dtype, ndim
**Key Message**: ตรวจสอบ shape และ dtype ก่อนทำ operation ทุกครั้ง

```python
v = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

print(v.shape)   # (2, 3) — 2 rows, 3 columns
print(v.dtype)   # float64
print(v.ndim)    # 2 — 2 dimensions
print(v.size)    # 6 — total elements
print(v.nbytes)  # 48 — bytes in memory
```

**Common dtypes**:
| dtype | ใช้สำหรับ | size |
|-------|---------|------|
| int64 | integer counts, indices | 8 bytes |
| float64 | ML weights, features | 8 bytes |
| bool | masks, conditions | 1 byte |

---
## Slide 5 — Indexing และ Slicing
**Key Message**: Slicing ใน NumPy ส่งคืน view (ไม่ copy) — ระวัง side effects

```python
v = np.array([10, 20, 30, 40, 50])

# Basic indexing
print(v[0])     # 10  (first)
print(v[-1])    # 50  (last)
print(v[1:4])   # [20, 30, 40]
print(v[::2])   # [10, 30, 50] (every other)

# 2D array
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(A[1, 2])   # 6  (row 1, col 2)
print(A[:, 0])   # [1, 4, 7]  (column 0)
print(A[0, :])   # [1, 2, 3]  (row 0)

# Boolean masking
v = np.array([1, -2, 3, -4, 5])
print(v[v > 0])  # [1, 3, 5]
```

---
## Slide 6 — Vectorized Operations: ไม่ต้องใช้ Loop
**Key Message**: Operations บน NumPy arrays ทำงานแบบ element-wise โดยอัตโนมัติ

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)    # [11 22 33 44]
print(a * b)    # [10 40 90 160]
print(a ** 2)   # [1  4  9 16]
print(np.sqrt(a))  # [1. 1.41 1.73 2.]

# Broadcasting: scalar กับ array
print(a * 3)    # [3  6  9  12]
print(a + 100)  # [101 102 103 104]

# Math functions
print(np.log(a))   # natural log
print(np.exp(a))   # e^x
print(np.sin(a))   # sine
```

[FIGURE: diagram แสดง element-wise operation]

---
## Slide 7 — Linear Algebra ใน NumPy
**Key Message**: `np.linalg` คือ module สำหรับ linear algebra operations

```python
u = np.array([3.0, 4.0])
v = np.array([1.0, 2.0])

# Dot product (inner product)
dot = np.dot(u, v)           # 3*1 + 4*2 = 11
# หรือ: u @ v, np.inner(u, v)

# Norm (magnitude)
norm_u = np.linalg.norm(u)   # sqrt(9+16) = 5.0
norm_v = np.linalg.norm(v)   # sqrt(1+4) = 2.236

# Cosine similarity
cos = dot / (norm_u * norm_v)  # 11 / (5 * 2.236) = 0.984

# Matrix operations
A = np.array([[1, 2], [3, 4]])
print(np.linalg.det(A))       # -2.0
print(np.linalg.inv(A))       # inverse matrix
eigvals = np.linalg.eigvals(A)  # eigenvalues
```

---
## Slide 8 — Common Mistakes ที่ต้องระวัง
**Key Message**: shape mismatch คือ error ที่พบบ่อยที่สุด — check shape ก่อนทำ operation

```python
# ❌ Mistake 1: shape mismatch
a = np.array([1, 2, 3])    # shape (3,)
b = np.array([[1, 2, 3]])  # shape (1,3)
# a + b → Broadcasting ให้ shape (1,3) ไม่ใช่ error แต่อาจผิดที่คาดหวัง

# ❌ Mistake 2: View vs Copy
a = np.array([1, 2, 3])
b = a[1:]          # b คือ VIEW ของ a
b[0] = 99          # เปลี่ยน a ด้วย!
print(a)           # [1, 99, 3] — ไม่ใช่ [1, 2, 3]

# ✅ Fix: ใช้ copy()
b = a[1:].copy()   # b คือ copy แยกต่างหาก

# ❌ Mistake 3: integer division
v = np.array([1, 2, 3])
print(v / 2)   # [0.5 1.  1.5] — OK (float division)
# ใน Python 2 จะได้ [0, 1, 1] — ระวังถ้าใช้ Python 2 code เก่า
```

---
## Slide 9 — From Array to Data Science
**Key Message**: NumPy คือ foundation — Pandas, sklearn, PyTorch ล้วนสร้างบน NumPy

```
NumPy ndarray
     │
     ├── Pandas Series / DataFrame
     │     └── การจัดการ tabular data
     │
     ├── scikit-learn (sklearn)
     │     └── X_train, y_train เป็น numpy arrays
     │
     ├── statsmodels
     │     └── endog, exog เป็น numpy arrays
     │
     └── PyTorch / TensorFlow
           └── Tensor คือ NumPy array บน GPU
```

- `X_train = df[features].values` → numpy array พร้อมใช้กับ sklearn
- `model.fit(X_train, y_train)` → X_train ต้องเป็น shape (n_samples, n_features)

---
## Slide 10 — Summary & Lab Preview
**Key Message**: NumPy คือ backbone ของ data science ecosystem ทั้งหมด — master NumPy แล้ว Pandas, sklearn, PyTorch จะไม่ใช่เรื่องยากอีกต่อไป

**สิ่งที่เรียนรู้วันนี้**:
- NumPy เร็วกว่า Python list ด้วย vectorized operations
- `np.array()`, `.shape`, `.dtype`, `.ndim` — attribute พื้นฐาน
- Indexing, Slicing, Boolean masking
- `np.dot()`, `np.linalg.norm()`, `np.linalg.inv()` สำหรับ linear algebra
- View vs Copy — ระวัง side effects

**Lab 01 Preview — 3 TODOs**:
- **TODO 1** (Easy): สร้าง feature vector ของบ้านและ compute unit vector
- **TODO 2** (Medium): Cosine similarity ระหว่าง word vectors
- **TODO 3** (Hard): Similarity matrix + heatmap visualization

**สัปดาห์ต่อไป**: Matrix ขนาด m×n, Matrix multiplication, Gaussian Elimination
