# Slide Deck: LU Decomposition
> Week 02 | CLO1 | Strang 2.6 | 9 slides

---
## Slide 1 — Title
**LU Decomposition: A = LU**  
Week 2 | CLO1 | พื้นฐาน solver ทุกตัวใน NumPy/MATLAB/R

---
## Slide 2 — ทำไมต้องมี LU?
**Key Message**: LU ช่วยให้แก้ Ax = b ได้เร็วขึ้นเมื่อมีหลาย b

**ปัญหา**: ถ้าต้องแก้ Ax = b₁, Ax = b₂, ..., Ax = b₁₀₀  
→ Gaussian Elimination ซ้ำทุกครั้ง = ช้ามาก

**ทางแก้ LU**: แยก A = LU ครั้งเดียว (O(n³))  
แล้วแก้ Ax = b ด้วย forward/back substitution (O(n²)) แต่ละครั้ง

```python
# Factorize A ครั้งเดียว
P, L, U = linalg.lu(A)

# แก้หลาย b ได้เร็ว
for b_i in [b1, b2, b3]:
    x = linalg.solve_triangular(U, 
        linalg.solve_triangular(L, P @ b_i, lower=True))
```

---
## Slide 3 — LU Decomposition คืออะไร?
**Key Message**: A = LU แยก matrix เป็น Lower × Upper triangular

$$\mathbf{A} = \mathbf{L}\mathbf{U}$$

$$\begin{pmatrix} 2 & 1 & -1 \\ -3 & -1 & 2 \\ -2 & 1 & 2 \end{pmatrix} = \underbrace{\begin{pmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{pmatrix}}_{\mathbf{L}} \underbrace{\begin{pmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{pmatrix}}_{\mathbf{U}}$$

- **L** (Lower triangular): diagonal = 1, เก็บ elimination multipliers
- **U** (Upper triangular): ผลลัพธ์ของ forward elimination

---
## Slide 4 — สร้าง L จาก Elimination Multipliers
**Key Message**: multiplier ที่ใช้ใน forward elimination กลายเป็น elements ของ L

**Forward elimination ของ A**:
```
Row2 ← Row2 - m₂₁ × Row1,  m₂₁ = -3/2 = -1.5
Row3 ← Row3 - m₃₁ × Row1,  m₃₁ = -2/2 = -1
Row3 ← Row3 - m₃₂ × Row2,  m₃₂ = 2/0.5 = 4
```

**L**:
$$\mathbf{L} = \begin{pmatrix} 1 & 0 & 0 \\ -1.5 & 1 & 0 \\ -1 & 4 & 1 \end{pmatrix}$$

Multipliers ไปอยู่ตำแหน่งใต้ diagonal ของ L ตรง ๆ — ไม่ต้องคำนวณใหม่!

---
## Slide 5 — Solving Ax = b via LU
**Key Message**: แก้สองขั้นตอน Ly = b แล้ว Ux = y — ทั้งสองอย่าง triangular (เร็ว)

**Ax = b → LUx = b → Let y = Ux**

**Step 1**: Solve **Ly = b** (forward substitution)
$$y_i = b_i - \sum_{j=1}^{i-1} l_{ij} y_j$$

**Step 2**: Solve **Ux = y** (back substitution)
$$x_i = \frac{1}{u_{ii}}\left(y_i - \sum_{j=i+1}^{n} u_{ij} x_j\right)$$

```python
P, L, U = linalg.lu(A)
b_perm = P @ b              # apply permutation
y = linalg.solve_triangular(L, b_perm, lower=True)
x = linalg.solve_triangular(U, y)
print(x)  # same as np.linalg.solve(A, b)
```

---
## Slide 6 — PA = LU: Partial Pivoting
**Key Message**: Permutation matrix P swap rows เพื่อ numerical stability

**PA = LU** (with pivoting)  
P = permutation matrix (ระบุ row swaps)

```python
from scipy import linalg
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
P, L, U = linalg.lu(A)

print('P =\n', P)
print('L =\n', L)
print('U =\n', U)
print('PA == LU?', np.allclose(P @ A, L @ U))  # True
```

[FIGURE: แสดง P matrix ว่า row swap ใดเกิดขึ้น]

---
## Slide 7 — LU vs Other Factorizations
**Key Message**: เลือก factorization ตาม structure ของ matrix

| Factorization | ใช้เมื่อ | ข้อดี |
|--------------|---------|------|
| **LU** | General square matrix | General purpose |
| **Cholesky (LLᵀ)** | Positive definite (e.g., XᵀX) | เร็วกว่า LU 2x |
| **QR** | Least squares, overdetermined | Numerically stable |
| **SVD** | Rank deficient, ill-conditioned | Most robust |

**ใน Sklearn**: LinearRegression ใช้ SVD ของ X (ไม่ใช่ LU) เพราะ X มักไม่ square

---
## Slide 8 — LDU Decomposition
**Key Message**: A = LDU แยก scaling (D) ออกจาก shape (L, U)

$$\mathbf{A} = \mathbf{L}\mathbf{D}\mathbf{U}$$

โดย D = diagonal matrix ของ pivot values, L และ U มี unit diagonal

```python
P, L, U = linalg.lu(A)
D = np.diag(np.diag(U))    # extract diagonal of U
U_normalized = np.diag(1/np.diag(U)) @ U  # unit upper triangular
print('A ≈ L @ D @ U_normalized?', np.allclose(A, L @ D @ U_normalized))
```

**ใน Stats**: Cholesky = LDLᵀ สำหรับ covariance matrix decomposition

---
## Slide 9 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- A = LU แยก matrix เป็น Lower × Upper triangular
- L เก็บ elimination multipliers, U เป็นผลจาก elimination
- Solving Ax = b: Ly = b (forward sub) → Ux = y (back sub)
- PA = LU (with partial pivoting) สำหรับ numerical stability

**Python**:
```python
from scipy import linalg
P, L, U = linalg.lu(A)   # factorize
x = np.linalg.solve(A, b)  # ใช้ LU internally
```

**สัปดาห์ต่อไป**: Inverse Matrix และ Determinant
