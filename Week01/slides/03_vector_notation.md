# Slide Deck: สัญลักษณ์และการเขียน Vector ในรูปแบบต่าง ๆ
> Week 01 | CLO1 | 8 slides

---
## Slide 1 — Title
**Vector Notation: อ่านสัญลักษณ์คณิตศาสตร์ให้เป็น**  
Week 1 | CLO1 | สิ่งสำคัญ: อ่าน paper และ textbook ได้โดยไม่งง

---
## Slide 2 — ทำไม Notation ถึงสำคัญ?
**Key Message**: Notation ที่ถูกต้องช่วยให้อ่าน paper และ code ได้เร็วขึ้น

ปัญหาที่พบบ่อย: นักศึกษาเข้าใจ concept แต่อ่าน paper ไม่ออกเพราะ notation  

- สัญลักษณ์เดียวกันอาจหมายถึงสิ่งต่างกันใน context ต่างกัน
- Math notation ↔ Python code ↔ Paper notation มักไม่เหมือนกัน
- สัปดาห์นี้เราจะ map notation ทั้ง 3 แบบให้ตรงกัน

---
## Slide 3 — Notation: Scalar, Vector, Matrix
**Key Message**: ขนาดตัวอักษรและ style บอกประเภทของ object

| Object | Math notation | ตัวอย่าง |
|--------|--------------|---------|
| Scalar | ตัวเล็ก italic: *c*, *α*, *λ* | *n* = 100, *α* = 0.01 |
| Vector | ตัวเล็กหนา: **v**, **x**, **w** | **x** ∈ ℝⁿ |
| Vector (alternative) | ตัวเล็กมีลูกศร: $\vec{v}$ | $\vec{w}$ = weights |
| Matrix | ตัวใหญ่หนา: **A**, **X**, **W** | **X** ∈ ℝⁿˣᵖ |
| Random variable | ตัวใหญ่ italic: *X*, *Y* | *Y* = target variable |

```python
# Python mapping:
c = 0.01                      # scalar
v = np.array([1, 2, 3])       # vector (1D)
A = np.array([[1,2],[3,4]])   # matrix (2D)
```

---
## Slide 4 — การเข้าถึง Elements ของ Vector
**Key Message**: Index เริ่มต้นต่างกันระหว่าง Math (1) และ Python (0)

**Math notation**:  
$\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ — index เริ่มที่ 1

**Python**:  
`v[0], v[1], ..., v[n-1]` — index เริ่มที่ 0

```python
v = np.array([10, 20, 30, 40])
# v₁ ใน math = v[0] ใน Python = 10
# v₄ ใน math = v[3] ใน Python = 40
print(v[0], v[-1])  # 10 40 (last element)
print(v[1:3])       # [20, 30] (slice)
```

[FIGURE: แสดง index mapping ระหว่าง math และ Python]

---
## Slide 5 — Column Vector vs Row Vector
**Key Message**: ใน math เราเขียน column vector, ใน NumPy ต้องระวัง shape

**Column vector** (n×1):
$$\mathbf{v} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$$

**Row vector** (1×n):
$$\mathbf{v}^T = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$$

```python
v = np.array([1, 2, 3])       # shape (3,) — 1D, ไม่ใช่ column/row
col = v.reshape(-1, 1)         # shape (3,1) — column vector
row = v.reshape(1, -1)         # shape (1,3) — row vector

print(v.shape)    # (3,)
print(col.shape)  # (3, 1)
print(row.shape)  # (1, 3)
```

---
## Slide 6 — Summation Notation
**Key Message**: $\sum$ notation กระชับ แต่แปลงเป็น Python ได้ตรง ๆ

**Math**: $\displaystyle\sum_{i=1}^{n} v_i = v_1 + v_2 + \cdots + v_n$

```python
# สามวิธีที่เทียบเท่ากัน:
v = np.array([1, 2, 3, 4, 5])

# 1. Python loop (ช้า):
total = sum(v_i for v_i in v)

# 2. NumPy (เร็วกว่า 100x):
total = np.sum(v)

# 3. Even shorter:
total = v.sum()

print(total)  # 15
```

**ใน ML**: RSS = $\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ → `np.sum((y - y_hat)**2)`

---
## Slide 7 — Notation ใน ISLP และ Strang
**Key Message**: textbook สองเล่มที่ใช้ในวิชานี้มี convention ต่างกันเล็กน้อย

| Concept | Strang (W1–4) | ISLP (W5–15) |
|---------|--------------|--------------|
| Data matrix | **A** ∈ ℝᵐˣⁿ | **X** ∈ ℝⁿˣᵖ |
| Response | **b** | **Y** |
| Coefficients | **x** (solution) | **β** |
| Sample size | m (rows) | n |
| Features | n (columns) | p |
| Predicted value | **Ax** | $\hat{Y}$ |

→ ในวิชานี้เราจะใช้ ISLP convention ตั้งแต่ Week 5 เป็นต้นไป

---
## Slide 8 — Summary
**Key Message**: Notation ที่ถูกต้องคือกุญแจสู่การอ่าน paper และ textbook — ชั่วโมงที่ใช้เรียน notation วันนี้จะประหยัดเวลาหลายชั่วโมงในอนาคต

**สิ่งที่เรียนรู้วันนี้**:
- Scalar (*c*), Vector (**v**), Matrix (**A**) — แยกด้วย font style
- Python index เริ่ม 0, Math index เริ่ม 1
- shape (3,) ≠ shape (3,1) ใน NumPy — ระวังการ multiply
- $\sum$ notation แปลเป็น `np.sum()` โดยตรง

**Quick Reference Card**:
```
Math   →   Python
vᵢ    →   v[i-1]
||v|| →   np.linalg.norm(v)
u·v   →   np.dot(u, v)
cA    →   c * A
Aᵀ    →   A.T
```

**สัปดาห์ต่อไป**: Matrix notation: **A**ᵢⱼ, **AB**, **Aᵀ**, det(**A**)
