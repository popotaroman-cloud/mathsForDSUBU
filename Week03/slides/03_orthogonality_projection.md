# Slide Deck: Orthogonality and Projection
> Week 03 | CLO1 | Strang Ch.4 | 9 slides

---
## Slide 1 — Title
**Orthogonality และ Projection: การ "ส่าย" ข้อมูลลง Subspace**  
Week 3 | CLO1 | พื้นฐาน Least Squares และ PCA

---
## Slide 2 — Orthogonal Vectors
**Key Message**: Orthogonal = dot product เป็น 0 = ตั้งฉากกัน = "ไม่เกี่ยวข้องกัน"

$$\mathbf{u} \perp \mathbf{v} \iff \mathbf{u}^T\mathbf{v} = 0$$

```python
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
w = np.array([1, 1, 0])

print(np.dot(u, v))  # 0 → orthogonal ✓
print(np.dot(u, w))  # 1 → not orthogonal
```

**ใน DS**:
- Orthogonal features = uncorrelated (ไม่มี multicollinearity)
- PCA สร้าง orthogonal principal components
- Orthonormal matrix Q: QᵀQ = I

---
## Slide 3 — Projection onto a Line
**Key Message**: project b ลงบน line ผ่าน a คือ scalar multiple ของ a

**สูตร**: $\mathbf{p} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\mathbf{a} = \mathbf{P}\mathbf{b}$

โดย Projection Matrix: $\mathbf{P} = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}$

```python
a = np.array([1, 1, 1], dtype=float)  # line direction
b = np.array([1, 2, 3], dtype=float)  # vector to project

P_line = np.outer(a, a) / np.dot(a, a)
p = P_line @ b
e = b - p
print('p =', p)         # [2, 2, 2] — projected
print('e =', e)         # [-1, 0, 1] — error
print('e ⊥ a?', np.allclose(np.dot(e, a), 0))  # True
```

[FIGURE: 2D diagram แสดง b, p, e และ line a]

---
## Slide 4 — Projection Properties
**Key Message**: P² = P และ Pᵀ = P — นี่คือ idempotent symmetric matrix

$$\mathbf{P}^2 = \mathbf{P}, \quad \mathbf{P}^T = \mathbf{P}$$

**ทำไม P² = P ถึงสมเหตุสมผล**:  
Project ครั้งที่สอง ผลไม่เปลี่ยนแปลง (p อยู่บน subspace แล้ว)

```python
P = np.outer(a, a) / np.dot(a, a)
print('P² == P?', np.allclose(P @ P, P))  # True
print('Pᵀ == P?', np.allclose(P.T, P))   # True
print('I-P symmetric?', np.allclose((np.eye(3)-P).T, np.eye(3)-P))  # True
```

Note: **I − P** คือ projection onto orthogonal complement (error direction)

---
## Slide 5 — Projection onto Subspace
**Key Message**: project b ลงบน C(A) ด้วย P = A(AᵀA)⁻¹Aᵀ

$$\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

```python
A = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
P = A @ np.linalg.inv(A.T @ A) @ A.T  # (3×3)

b = np.array([1, 2, 2])
p = P @ b            # projection of b onto C(A)
e = b - p            # perpendicular component

print('P² == P?', np.allclose(P @ P, P))  # True
print('e ⊥ each col of A?')
for i in range(A.shape[1]):
    print(f'  col {i}:', np.allclose(np.dot(e, A[:, i]), 0))  # True
```

---
## Slide 6 — Geometric Interpretation
**Key Message**: Projection คือ "เงา" ที่ใกล้ที่สุดบน subspace

[FIGURE: แสดง b = p + e ใน 3D โดย p ∈ C(A) และ e ⊥ C(A)]

**เรขาคณิต**:
- p = Pb คือจุดที่ใกล้ b มากที่สุดใน C(A)
- ||e|| = ||b − p|| = minimum distance from b to C(A)
- This is why Least Squares works!

$$\min_{\hat{x}} \|\mathbf{A}\hat{x} - \mathbf{b}\|^2 \Leftrightarrow \hat{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$

---
## Slide 7 — Orthonormal Basis
**Key Message**: Orthonormal basis ทำให้คำนวณง่ายขึ้นมาก

**Orthonormal**: qᵢᵀqⱼ = 0 (i≠j), qᵢᵀqᵢ = 1

$$\mathbf{Q}^T\mathbf{Q} = \mathbf{I} \quad (\text{if Q is square: } \mathbf{Q}^T = \mathbf{Q}^{-1})$$

**Projection ง่ายขึ้น**: ถ้า A มี orthonormal columns Q
$$\mathbf{P} = \mathbf{Q}\mathbf{Q}^T \quad (\text{no inverse needed!})$$
$$\hat{\mathbf{x}} = \mathbf{Q}^T\mathbf{b} \quad (\text{coordinate computation})$$

```python
Q, R = np.linalg.qr(A)  # A = QR
P_qr = Q @ Q.T           # projection via Q
print('P == P_qr?', np.allclose(P, P_qr))  # True
```

---
## Slide 8 — Connection to Gram-Schmidt
**Key Message**: Gram-Schmidt สร้าง orthonormal basis จาก any independent vectors

```python
def gram_schmidt(A):
    """สร้าง orthonormal basis ด้วย Gram-Schmidt"""
    Q = np.zeros_like(A, dtype=float)
    for i in range(A.shape[1]):
        q = A[:, i].copy()
        for j in range(i):
            q -= np.dot(q, Q[:, j]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)
    return Q

A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
Q = gram_schmidt(A)
print('QᵀQ =\n', Q.T @ Q)  # ≈ Identity
```

---
## Slide 9 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- **u** ⊥ **v** ↔ **u**ᵀ**v** = 0
- Projection onto line: P = aaᵀ/(aᵀa)
- Projection onto subspace: P = A(AᵀA)⁻¹Aᵀ
- Properties: P² = P, Pᵀ = P (symmetric idempotent)
- Orthonormal Q: QᵀQ = I, projection = QQᵀ

**สัปดาห์ต่อไป**: Least Squares — Normal Equations และ QR
