# Slide Deck: The Four Fundamental Subspaces
> Week 03 | CLO1 | Strang Ch.3 | 9 slides

---
## Slide 1 — Title
**Four Fundamental Subspaces: Strang's Big Picture**  
Week 3 | CLO1 | ภาพรวมสมบูรณ์ของ structure ใน matrix

---
## Slide 2 — The Big Picture
**Key Message**: Matrix A สร้าง 4 subspaces ที่เชื่อมกันอย่างสวยงาม

[FIGURE: Strang's Big Picture diagram — R^n (domain) ซ้าย, R^m (codomain) ขวา]

$$\mathbf{A}: \mathbb{R}^n \rightarrow \mathbb{R}^m$$

| Subspace | อยู่ใน | Dimension |
|---------|--------|-----------|
| Column Space C(A) | ℝᵐ | r |
| Left Null Space N(Aᵀ) | ℝᵐ | m − r |
| Row Space C(Aᵀ) | ℝⁿ | r |
| Null Space N(A) | ℝⁿ | n − r |

---
## Slide 3 — Row Space C(Aᵀ)
**Key Message**: C(Aᵀ) = span ของ rows ของ A = column space ของ Aᵀ

$$C(\mathbf{A}^T) = \text{span of rows of A}$$

- dim C(Aᵀ) = r = rank(A)
- C(Aᵀ) อยู่ใน ℝⁿ
- Basis: pivot rows จาก RREF

**Key fact**: C(A) ⊥ N(Aᵀ) และ C(Aᵀ) ⊥ N(A)

```python
# Basis ของ row space = pivot rows
U, s, Vt = np.linalg.svd(A)
r = np.linalg.matrix_rank(A)
row_space_basis = Vt[:r]   # first r rows of Vt
null_space_basis = Vt[r:]  # last n-r rows of Vt
```

---
## Slide 4 — Left Null Space N(Aᵀ)
**Key Message**: N(Aᵀ) = {y: Aᵀy = 0} = ตั้งฉากกับ C(A)

$$N(\mathbf{A}^T) = \{\mathbf{y} : \mathbf{A}^T\mathbf{y} = \mathbf{0}\}$$

- dim N(Aᵀ) = m − r
- N(Aᵀ) อยู่ใน ℝᵐ
- **C(A) ⊥ N(Aᵀ)**: C(A) + N(Aᵀ) = ℝᵐ

**สำคัญใน Regression**: residuals e = y − Ŷ อยู่ใน N(Aᵀ) (ตั้งฉากกับ C(A))

```python
U, s, Vt = np.linalg.svd(A)
left_null = U[:, r:]  # left null space basis
print('A.T @ left_null ≈ 0?', np.allclose(A.T @ left_null, 0))
```

---
## Slide 5 — Orthogonality ของ 4 Subspaces
**Key Message**: Domain และ Codomain แต่ละส่วนตั้งฉากกัน

**ใน ℝⁿ (domain)**:
$$C(\mathbf{A}^T) \perp N(\mathbf{A}), \quad C(\mathbf{A}^T) \oplus N(\mathbf{A}) = \mathbb{R}^n$$

**ใน ℝᵐ (codomain)**:
$$C(\mathbf{A}) \perp N(\mathbf{A}^T), \quad C(\mathbf{A}) \oplus N(\mathbf{A}^T) = \mathbb{R}^m$$

[FIGURE: แสดง orthogonal decomposition ของ ℝⁿ เป็น row space ⊕ null space]

**ทำไมถึงสำคัญ**: ทุก vector ใน ℝⁿ แยกได้ unique เป็น component ใน C(Aᵀ) + N(A)

---
## Slide 6 — Counting Dimensions
**Key Message**: Rank-Nullity ใช้กับทั้ง domain และ codomain

**ใน domain (ℝⁿ)**:
$$\dim C(\mathbf{A}^T) + \dim N(\mathbf{A}) = r + (n-r) = n$$

**ใน codomain (ℝᵐ)**:
$$\dim C(\mathbf{A}) + \dim N(\mathbf{A}^T) = r + (m-r) = m$$

**ตัวอย่าง**: A ∈ ℝ⁴ˣ⁶, rank = 3
- Null Space dim = 6 - 3 = 3 (ใน ℝ⁶)
- Left Null Space dim = 4 - 3 = 1 (ใน ℝ⁴)
- Row Space dim = 3 (ใน ℝ⁶)
- Column Space dim = 3 (ใน ℝ⁴)

---
## Slide 7 — Complete Solution ของ Ax = b
**Key Message**: complete solution = particular solution + null space

ถ้า Ax = b มี solution:
$$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_n, \quad \mathbf{x}_p \in C(\mathbf{A}^T), \quad \mathbf{x}_n \in N(\mathbf{A})$$

```
Ax_p = b   (particular solution)
Ax_n = 0   (null space)
A(x_p + x_n) = b + 0 = b  ✓
```

**ตัวอย่าง**: Ax = b, rank = 2, n = 4
- 1 particular solution + 2-dimensional null space
- → infinite solutions (family)

**ใน Regression**: จาก overdetermined Ax ≈ b, เราเลือก x̂ ที่อยู่ใน row space (minimum norm)

---
## Slide 8 — Case Study: XᵀX Decomposition
**Key Message**: ใน Regression, rank(X) determines everything

```python
X = np.array([[1, 2, 4],  # design matrix
              [1, 3, 6],  # note: col3 = 2×col2
              [1, 4, 8],
              [1, 5, 10]])

r = np.linalg.matrix_rank(X)
print('rank(X) =', r)  # 2 (multicollinearity!)
print('rank(XᵀX) =', np.linalg.matrix_rank(X.T @ X))  # 2

# XᵀX singular → can't invert → use lstsq or Ridge
beta, _, _, _ = np.linalg.lstsq(X, np.array([1,2,3,4]), rcond=None)
print('lstsq solution:', beta)
```

---
## Slide 9 — Summary
**Strang's Big Picture (summary)**:

```
        ℝⁿ (domain)                ℝᵐ (codomain)
   ┌─────────────────┐        ┌─────────────────┐
   │  Row Space       │ ──A──▶ │  Column Space    │
   │  C(Aᵀ): dim r   │        │  C(A): dim r     │
   ├─────────────────┤        ├─────────────────┤
   │  Null Space      │ ──A──▶ │  Left Null Space │
   │  N(A): dim n-r  │   (=0) │  N(Aᵀ): dim m-r  │
   └─────────────────┘        └─────────────────┘
```

**สัปดาห์ต่อไป**: Orthogonality และ Projection — p = A(AᵀA)⁻¹Aᵀb
