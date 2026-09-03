# Slide Deck: Least Squares, Normal Equations, QR Decomposition
> Week 03 | CLO1 | Strang 4.3–4.4 | 10 slides

---
## Slide 1 — Title
**Least Squares: คณิตศาสตร์เบื้องหลัง Linear Regression**  
Week 3 | CLO1 | Connection: → Week 8 (SLR), → Week 9 (MLR)

---
## Slide 2 — ปัญหา: Ax = b ไม่มีคำตอบ
**Key Message**: Dataset จริงมี n >> p — overdetermined system ไม่มี exact solution

**Regression setup**:
- n = 100 data points, p = 3 features
- A ∈ ℝ¹⁰⁰ˣ³, b ∈ ℝ¹⁰⁰ → overdetermined
- b ∉ C(A) generally → ไม่มี **x** ที่ทำให้ Ax = b

**Least Squares Goal**: หา x̂ ที่ minimize ||Ax − b||²

$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 = \min_{\mathbf{x}} \sum_{i=1}^n (a_i^T x - b_i)^2$$

---
## Slide 3 — Geometric View ของ Least Squares
**Key Message**: หา x̂ ที่ทำให้ Ax̂ เป็น projection ของ b ลง C(A)

[FIGURE: b ∈ ℝᵐ; p = Ax̂ = projection onto C(A); e = b - p ⊥ C(A)]

- Ax̂ = p = projection ของ b ลง C(A)
- Error e = b − Ax̂ ⊥ C(A)
- → Aᵀe = 0 → Aᵀ(b − Ax̂) = 0 → **Normal Equations**

$$\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$$

---
## Slide 4 — Normal Equations
**Key Message**: Aᵀ(b − Ax̂) = 0 ↔ Normal Equations ↔ x̂ = (AᵀA)⁻¹Aᵀb

$$(\mathbf{A}^T\mathbf{A})\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}$$

$$\hat{\mathbf{x}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$

```python
A = np.column_stack([np.ones(n), x])  # design matrix [1 | x]
b = y

# Normal Equations
ATA = A.T @ A
ATb = A.T @ b
x_hat = np.linalg.solve(ATA, ATb)
print('β̂ =', x_hat)
```

**ข้อจำกัด**: AᵀA ต้องเป็น invertible → rank(A) = p → ไม่มี multicollinearity

---
## Slide 5 — Least Squares = Linear Regression
**Key Message**: Linear Regression คือ Least Squares บน [1, X] design matrix

**Model**: ŷ = β₀ + β₁x₁ + ... + βₚxₚ

**Design Matrix** A = **X** ∈ ℝⁿˣ⁽ᵖ⁺¹⁾:

$$\mathbf{X} = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & & & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}$$

→ β̂ = (XᵀX)⁻¹Xᵀy = `np.linalg.lstsq(X, y)`

**Connection**: Week 8 จะใช้ SLR (p=1), Week 9 MLR (p>1) — math เหมือนกัน

---
## Slide 6 — np.linalg.lstsq vs Normal Equations
**Key Message**: lstsq ใช้ SVD ดีกว่า Normal Equations เมื่อ AᵀA near-singular

| วิธี | Python | ข้อดี | ข้อเสีย |
|------|--------|-------|--------|
| Normal Equations | `inv(A.T@A) @ A.T@b` | เข้าใจง่าย | ไม่ stable เมื่อ cond(A) สูง |
| `np.linalg.solve` | `solve(A.T@A, A.T@b)` | stable กว่า inv | ยังใช้ AᵀA |
| `np.linalg.lstsq` | `lstsq(A, b)` | SVD: robust, handles rank deficient | ช้ากว่าเล็กน้อย |

```python
# All give same result for well-conditioned A:
x1 = np.linalg.inv(A.T@A) @ A.T @ b
x2 = np.linalg.solve(A.T@A, A.T@b)
x3, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print(np.allclose(x1, x2), np.allclose(x2, x3))  # True True
```

---
## Slide 7 — Gram-Schmidt Orthogonalization
**Key Message**: แปลง independent vectors เป็น orthonormal vectors ทีละตัว

**Algorithm**:
1. q₁ = a₁ / ||a₁||
2. v₂ = a₂ - (a₂ᵀq₁)q₁; q₂ = v₂ / ||v₂||
3. v₃ = a₃ - (a₃ᵀq₁)q₁ - (a₃ᵀq₂)q₂; q₃ = v₃ / ||v₃||

[FIGURE: step-by-step diagram ของ Gram-Schmidt ใน 2D]

```python
# Gram-Schmidt step-by-step สำหรับ 2 vectors
a1 = np.array([1., 1., 0.])
a2 = np.array([1., 0., 1.])

q1 = a1 / np.linalg.norm(a1)
v2 = a2 - np.dot(a2, q1) * q1
q2 = v2 / np.linalg.norm(v2)
print('q1 · q2 =', np.dot(q1, q2).round(10))  # ≈ 0
```

---
## Slide 8 — QR Decomposition
**Key Message**: A = QR เก็บ orthogonal (Q) และ upper triangular (R) ไว้แยกกัน

$$\mathbf{A} = \mathbf{Q}\mathbf{R}$$

โดย Q orthogonal (QᵀQ = I), R upper triangular

**Solving Least Squares via QR**:
$$\hat{x} = \mathbf{R}^{-1}\mathbf{Q}^T\mathbf{b}$$

```python
Q, R = np.linalg.qr(A)     # A = QR
x_qr = np.linalg.solve(R, Q.T @ b)  # more stable than (AᵀA)⁻¹Aᵀb
print('condition(A):', np.linalg.cond(A))
print('condition(R):', np.linalg.cond(R))  # R: cond ≈ sqrt(cond(A))
```

---
## Slide 9 — R² และ RSE
**Key Message**: วัดความดีของ fit ด้วย R² (relative) และ RSE (absolute)

$$RSS = \|\mathbf{b} - \mathbf{A}\hat{x}\|^2 = \sum_i e_i^2$$
$$TSS = \|\mathbf{b} - \bar{b}\mathbf{1}\|^2, \quad R^2 = 1 - \frac{RSS}{TSS}$$
$$RSE = \sqrt{\frac{RSS}{n-p}}$$

```python
y_hat = A @ x_hat
e = y - y_hat
RSS = np.sum(e**2)
TSS = np.sum((y - y.mean())**2)
R2  = 1 - RSS / TSS
RSE = np.sqrt(RSS / (len(y) - A.shape[1]))
print(f'R² = {R2:.4f}, RSE = {RSE:.4f}')
```

---
## Slide 10 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- Overdetermined Ax ≈ b → minimize ||Ax − b||²
- Normal Equations: AᵀAx̂ = Aᵀb → x̂ = (AᵀA)⁻¹Aᵀb
- lstsq ใช้ SVD: robust กว่าสำหรับ ill-conditioned matrix
- QR decomposition: stable solver ผ่าน x̂ = R⁻¹Qᵀb
- R², RSE: วัด goodness-of-fit

**Connection**: Week 8–9 = Least Squares on Advertising/Housing data
