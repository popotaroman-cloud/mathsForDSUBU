# Slide Deck: Vector Spaces and Subspaces
> Week 03 | CLO1 | Strang Ch.3 | 9 slides

---
## Slide 1 — Title
**Vector Spaces และ Subspaces: โครงสร้างเบื้องหลัง Data**  
Week 3 | CLO1 | LLo: อธิบาย Vector Space, Subspace, Rank ได้

---
## Slide 2 — Week Overview
**Key Message**: Vector Space กำหนด "กฎ" ว่า vector operations ทำได้อะไรบ้าง

สัปดาห์นี้เราจะเรียนรู้แนวคิดที่เป็นหัวใจของ Linear Algebra ทั้งหมด นั่นคือ Vector Space และ Subspace ซึ่งช่วยให้เราเข้าใจว่า solution set ของ Ax = b มี structure อย่างไร เป้าหมายคือให้นักศึกษาสามารถหา rank, Column Space, Null Space ของ matrix ได้ รวมถึงนำ Least Squares ไปแก้ปัญหา overdetermined systems ได้ ทักษะเหล่านี้เป็นพื้นฐานก่อน Eigenvalues (Week 4) และ Linear Regression (Week 8–9)

**หัวข้อวันนี้**: Vector Space axioms, Column Space, Null Space, Rank, Basis, Dimension

---
## Slide 3 — Vector Space: นิยาม 8 Axioms
**Key Message**: Vector Space คือ set ที่ปิดภายใต้ + และ × พร้อม 8 axioms

**V** คือ Vector Space ถ้า: สำหรับ **u**, **v** ∈ V และ scalar c:
1. **u + v** ∈ V (closure under addition)
2. c**v** ∈ V (closure under scalar multiplication)
3. **u + v = v + u** (commutativity)
4. (**u + v**) + **w** = **u** + (**v + w**) (associativity)
5. **0** ∈ V (zero vector)
6. **v** + (-**v**) = **0** (inverse)
7. 1·**v** = **v** (identity scalar)
8. c(d**v**) = (cd)**v** (associativity scalar)

**ตัวอย่าง Vector Spaces**: ℝⁿ, polynomials, continuous functions  
**ตัวอย่าง ที่ไม่ใช่**: line ที่ไม่ผ่าน origin (axiom 5 fails)

---
## Slide 4 — Subspace
**Key Message**: Subspace ต้องผ่าน origin และปิดภายใต้ + และ ×

**Subspace** W ⊆ V ถ้า:
1. **0** ∈ W
2. ปิดภายใต้ +: **u**, **v** ∈ W → **u + v** ∈ W
3. ปิดภายใต้ ×: **v** ∈ W, c scalar → c**v** ∈ W

**ตัวอย่างใน ℝ²/ℝ³**:
- {**0**}: trivial subspace ✓
- เส้นผ่าน origin: {c**v** : c ∈ ℝ} ✓
- ℝ³ ทั้งหมด ✓
- เส้นที่ไม่ผ่าน origin ✗ (no zero vector)
- Circle (unit circle) ✗ (not closed under +)

---
## Slide 5 — Column Space C(A) และ Null Space N(A)
**Key Message**: C(A) กำหนดว่า Ax = b มี solution เมื่อไหร่; N(A) กำหนด freedom ของ solution

$$C(\mathbf{A}) = \{\mathbf{A}\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} = \text{span of columns of A}$$
$$N(\mathbf{A}) = \{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$$

```python
A = np.array([[1, 2, 3],
              [2, 4, 6]])  # rank 1: col 2 = 2×col 1, col 3 = 3×col 1

print('rank:', np.linalg.matrix_rank(A))  # 1
print('dim C(A) = 1, dim N(A) = 2')

# Null space vector: Ax = 0
_, _, Vt = np.linalg.svd(A)
print('null vector:', Vt[-1])  # verify A @ v ≈ 0
```

**Ax = b มี solution ↔ b ∈ C(A)**

---
## Slide 6 — Rank: Dimension ของ Column Space
**Key Message**: rank(A) = dim C(A) = dim C(Aᵀ) = number of pivots

$$r = \text{rank}(A)$$

- **rank = n**: full column rank → unique solution (only if m ≥ n)
- **rank = m**: full row rank → solution always exists
- **rank < min(m,n)**: rank deficient → multicollinearity ใน ML

```python
A1 = np.array([[1, 0], [0, 1], [0, 0]])  # 3×2, rank 2
A2 = np.array([[1, 2], [2, 4]])           # 2×2, rank 1 (singular)

print('rank(A1):', np.linalg.matrix_rank(A1))  # 2
print('rank(A2):', np.linalg.matrix_rank(A2))  # 1
```

---
## Slide 7 — Rank-Nullity Theorem
**Key Message**: rank + nullity = n (number of columns)

$$\text{rank}(A) + \text{nullity}(A) = n$$

**ตัวอย่าง**: A ∈ ℝ³ˣ⁴, rank = 2
- Null Space dim = 4 - 2 = 2 → สองพารามิเตอร์อิสระ
- Complete solution: x = x_particular + c₁v₁ + c₂v₂

**ใน DS**: XᵀX มี rank < p → multicollinearity → XᵀX singular → regression ล้มเหลว

---
## Slide 8 — Linear Independence และ Basis
**Key Message**: Basis คือ independent spanning set ที่เล็กที่สุด

**Linear Independence**: ไม่มี vector = linear combo ของตัวอื่น  
→ c₁v₁ + c₂v₂ + ... + cₖvₖ = **0** implies c₁ = c₂ = ... = cₖ = 0

**Basis ของ C(A)**: columns ของ A ที่เป็น pivot columns (จาก RREF)

```python
# Standard basis ของ R³
e1 = [1, 0, 0]
e2 = [0, 1, 0]
e3 = [0, 0, 1]
# Independent: no one is combo of others
```

---
## Slide 9 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- Vector Space = set ปิดภายใต้ + และ × พร้อม zero vector
- Subspace ต้องผ่าน origin
- C(A) = span ของ columns; N(A) = solutions ของ Ax = 0
- rank + nullity = n (Rank-Nullity Theorem)
- Multicollinearity ↔ rank deficient XᵀX

**สัปดาห์ต่อไป**: Four Fundamental Subspaces — Strang's Big Picture
