# Slide Deck: Inverse Matrix and Determinant
> Week 02 | CLO1 | Strang 2.5, Ch.5 | 10 slides

---
## Slide 1 — Title
**Inverse Matrix และ Determinant: เมื่อไหร่ที่ Ax = b มี Solution เดียว?**  
Week 2 | CLO1 | เชื่อมกับ Multicollinearity และ Singular System

---
## Slide 2 — Inverse Matrix คืออะไร?
**Key Message**: A⁻¹ มีอยู่เมื่อ A เป็น square full-rank matrix

**นิยาม**: **AA⁻¹ = A⁻¹A = I**

**ใช้แก้ Ax = b**: x = A⁻¹b

```python
A = np.array([[2, 1], [1, 3]], dtype=float)
A_inv = np.linalg.inv(A)
print('A_inv =\n', A_inv)
print('A @ A_inv =\n', A @ A_inv)  # ≈ Identity

# Solve Ax = b
b = np.array([5, 10])
x = A_inv @ b
print('x =', x)
```

**ข้อควรระวัง**: `np.linalg.solve(A, b)` เร็วกว่าและ stable กว่า `inv(A) @ b`

---
## Slide 3 — Gauss-Jordan Elimination หา A⁻¹
**Key Message**: augment [A|I] แล้วทำ row operations จนได้ [I|A⁻¹]

$$[\mathbf{A}|\mathbf{I}] \xrightarrow{\text{row ops}} [\mathbf{I}|\mathbf{A}^{-1}]$$

**ตัวอย่าง**:
$$\left[\begin{array}{cc|cc} 2 & 1 & 1 & 0 \\ 1 & 3 & 0 & 1 \end{array}\right] \rightarrow \left[\begin{array}{cc|cc} 1 & 0 & 0.6 & -0.2 \\ 0 & 1 & -0.2 & 0.4 \end{array}\right]$$

```python
# Verify manually:
A = np.array([[2., 1.], [1., 3.]])
A_inv = np.array([[0.6, -0.2], [-0.2, 0.4]])
print(np.allclose(A @ A_inv, np.eye(2)))  # True
```

---
## Slide 4 — Properties ของ Inverse
**Key Message**: ลำดับ inverse ต้องกลับกัน — ฝึก proof เพื่อไม่ผิด

$$(AB)^{-1} = B^{-1}A^{-1}$$
$$(A^T)^{-1} = (A^{-1})^T$$
$$(cA)^{-1} = \frac{1}{c}A^{-1}$$

**ตัวอย่าง**:
```python
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)
lhs = np.linalg.inv(A @ B)
rhs = np.linalg.inv(B) @ np.linalg.inv(A)
print(np.allclose(lhs, rhs))  # True
```

**ใน Normal Equations**: (XᵀX)⁻¹ = (X⁻¹)(Xᵀ)⁻¹ = (X⁻¹)(X⁻ᵀ)

---
## Slide 5 — Singular Matrix: เมื่อ A⁻¹ ไม่มี
**Key Message**: Singular matrix ↔ linearly dependent rows/columns ↔ det = 0

```python
# Singular: row 2 = 2 × row 1
B = np.array([[1, 2], [2, 4]], dtype=float)
print('det(B) =', np.linalg.det(B))  # ≈ 0
print('rank(B) =', np.linalg.matrix_rank(B))  # 1

try:
    np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print('Error:', e)
```

**ใน DS**: XᵀX singular หมายถึง features บางตัวเป็น linear combination ของตัวอื่น (multicollinearity) → Ridge Regression แก้ได้

---
## Slide 6 — Determinant: คืออะไร?
**Key Message**: det(A) = signed volume ที่ matrix A scale พื้นที่ unit square/cube

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

```python
A = np.array([[3, 1], [1, 2]])
print(np.linalg.det(A))  # 3*2 - 1*1 = 5
```

[FIGURE: แสดง unit square transform โดย A และ area = |det(A)|]

**Geometric interpretation**:
- |det(A)| = scaling factor ของ area/volume
- det(A) > 0: preserves orientation
- det(A) < 0: flips orientation
- det(A) = 0: collapses to lower dimension (singular!)

---
## Slide 7 — Properties ของ Determinant
**Key Message**: det(AB) = det(A)det(B) — สำคัญมากในการ prove theorems

| Property | สูตร |
|---------|------|
| Product | det(AB) = det(A)det(B) |
| Transpose | det(Aᵀ) = det(A) |
| Inverse | det(A⁻¹) = 1/det(A) |
| Scalar | det(cA) = cⁿ det(A) for n×n |
| Singular | det(A) = 0 ⟺ A singular |

```python
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)
print(np.allclose(np.linalg.det(A@B),
                  np.linalg.det(A) * np.linalg.det(B)))  # True
```

---
## Slide 8 — Cofactor Expansion
**Key Message**: สำหรับ 3×3: expand along row/column ที่มี zeros มากที่สุด

$$\det(\mathbf{A}) = \sum_{j=1}^{n} a_{ij}(-1)^{i+j} M_{ij}$$

**3×3 ตัวอย่าง** (expand along row 1):
$$\det\begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{pmatrix} = 1\det\begin{pmatrix}4&5\\0&6\end{pmatrix} - 2\det\begin{pmatrix}0&5\\1&6\end{pmatrix} + 3\det\begin{pmatrix}0&4\\1&0\end{pmatrix}$$
$$= 1(24) - 2(-5) + 3(-4) = 24 + 10 - 12 = 22$$

```python
A = np.array([[1,2,3],[0,4,5],[1,0,6]])
print(np.linalg.det(A))  # 22
```

---
## Slide 9 — Condition Number และ Numerical Stability
**Key Message**: Condition number วัดว่า matrix "ใกล้ singular" แค่ไหน

$$\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

```python
A_good = np.array([[2, 0], [0, 3]])      # diagonal, well-conditioned
A_bad  = np.array([[1, 1], [1, 1.001]]) # near-singular

print('cond(A_good):', np.linalg.cond(A_good))  # 1.5
print('cond(A_bad): ', np.linalg.cond(A_bad))   # ~2000
```

**Rule of thumb**: κ > 10^k → solutions มี ~k digits ของ round-off error

---
## Slide 10 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- A⁻¹ มีเมื่อ det(A) ≠ 0 (full rank, non-singular)
- Gauss-Jordan: [A|I] → [I|A⁻¹]
- det = 0 ↔ singular ↔ linearly dependent columns
- Condition number วัด numerical sensitivity

**สัปดาห์ต่อไป — Week 3**: Vector Spaces, Null Space, Column Space, Rank
