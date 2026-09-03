# Note 1: Vector Spaces, Subspaces, Rank และ Four Fundamental Subspaces
> Week 3 | CLO1 | Strang Reference: Ch.3 (Vector Spaces and Subspaces)

---

## บทนำ

สัปดาห์นี้เราจะก้าวจาก "การคำนวณ" ไปสู่ "การเข้าใจโครงสร้าง" ของ Linear Algebra โดยเรียนรู้ว่า **Vector Space** และ **Subspace** คืออะไร และ **Rank** ของแมทริกซ์บอกเราว่าอะไร เป้าหมายคือให้นักศึกษาสามารถอธิบาย Column Space, Null Space, Row Space และ Left Null Space ของแมทริกซ์ได้ รวมถึงใช้ Rank-Nullity Theorem เพื่อวิเคราะห์ว่าระบบสมการมีคำตอบกี่ตัว แนวคิดเหล่านี้ดูเป็นนามธรรมในตอนแรก แต่ในชีวิตจริง Rank ของ Feature Matrix บอกเราว่า features มี multicollinearity หรือเปล่า และ Null Space บอกว่ามีข้อมูลสูญหายหรือ redundant อยู่ในชุดข้อมูล — ความเข้าใจนี้จะเปลี่ยนวิธีที่เราออกแบบ Machine Learning models ไปตลอด

---

## Section 1: Vector Space และ Subspace  *(Strang 3.1)*

ในส่วนนี้เราจะเรียนรู้ว่า **Vector Space** คือ "สนาม" ที่ Vector อาศัยอยู่ และ **Subspace** คือส่วนย่อยของสนามนั้น เพื่อสร้างภาษาที่ใช้อธิบายโครงสร้างของข้อมูลได้อย่างแม่นยำ

### 1.1 Vector Space (ปริภูมิเวกเตอร์)

**นิยาม**: Vector Space $V$ คือเซตของ vectors ที่สามารถบวกกัน (vector addition) และคูณด้วย scalar (scalar multiplication) โดยผลลัพธ์ยังอยู่ใน $V$ เสมอ

**8 Axioms** (คุณสมบัติบังคับ):

| # | Axiom | ตัวอย่างใน $\mathbb{R}^n$ |
|---|-------|-------------------------|
| 1 | Closure under + | $\mathbf{u} + \mathbf{v} \in V$ |
| 2 | Closure under × | $c\mathbf{v} \in V$ |
| 3 | Commutativity | $\mathbf{u}+\mathbf{v} = \mathbf{v}+\mathbf{u}$ |
| 4 | Associativity (+) | $(\mathbf{u}+\mathbf{v})+\mathbf{w} = \mathbf{u}+(\mathbf{v}+\mathbf{w})$ |
| 5 | Zero vector | $\mathbf{v}+\mathbf{0}=\mathbf{v}$ |
| 6 | Additive inverse | $\mathbf{v}+(-\mathbf{v})=\mathbf{0}$ |
| 7 | Distributive | $c(\mathbf{u}+\mathbf{v}) = c\mathbf{u}+c\mathbf{v}$ |
| 8 | Scalar assoc. | $(cd)\mathbf{v} = c(d\mathbf{v})$ |

**ตัวอย่างที่เป็น Vector Space**: $\mathbb{R}^n$, ระนาบผ่าน origin, เส้นตรงผ่าน origin

**ตัวอย่างที่ไม่ใช่ Vector Space**:
- เซตของ vectors ที่ทุกสมาชิกมีค่า $\geq 0$ — ไม่ปิดภายใต้ scalar multiplication (คูณด้วย $-1$ ออกนอกเซต)
- เส้นตรงที่ **ไม่ผ่าน origin** — บวกสองจุดบนเส้นได้จุดที่ไม่อยู่บนเส้น

### 1.2 Subspace (ปริภูมิย่อย)

**นิยาม**: $S$ เป็น Subspace ของ $V$ ถ้า $S \subseteq V$ และ $S$ เองเป็น Vector Space

**เงื่อนไขตรวจสอบ 3 ข้อ** (compact version):
1. $\mathbf{0} \in S$ (มี zero vector)
2. ถ้า $\mathbf{u}, \mathbf{v} \in S$ แล้ว $\mathbf{u}+\mathbf{v} \in S$ (ปิดภายใต้การบวก)
3. ถ้า $\mathbf{v} \in S$ และ $c \in \mathbb{R}$ แล้ว $c\mathbf{v} \in S$ (ปิดภายใต้ scalar mult)

**ตัวอย่างใน $\mathbb{R}^3$**:
- $S_1 = \{(0,0,0)\}$ — subspace เล็กสุด (zero subspace)
- $S_2 = \{(x, 2x, 0) : x \in \mathbb{R}\}$ — เส้นตรงผ่าน origin ✓
- $S_3 = \{(x, y, 0) : x,y \in \mathbb{R}\}$ — ระนาบ $xy$ ✓
- $S_4 = \{(x, y, 1) : x,y \in \mathbb{R}\}$ — ระนาบที่ไม่ผ่าน origin ✗

---

## Section 2: Column Space และ Null Space  *(Strang 3.2–3.3)*

ในส่วนนี้เราจะเรียนรู้ subspace ที่สำคัญที่สุดสองตัวที่เกิดขึ้นจากแมทริกซ์ $\mathbf{A}$ — ซึ่งตอบคำถามว่าระบบสมการ $\mathbf{Ax} = \mathbf{b}$ มีคำตอบเมื่อไหร่ และ "space" ของคำตอบเหล่านั้นมีรูปร่างอย่างไร

### 2.1 Column Space C(A)

**นิยาม**: $C(\mathbf{A})$ คือ set ของ vectors ทั้งหมดที่เขียนเป็น linear combination ของ columns ของ $\mathbf{A}$

$$C(\mathbf{A}) = \{\mathbf{A}\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} = \text{span(columns of } \mathbf{A})$$

**ความหมาย**: $\mathbf{Ax} = \mathbf{b}$ มีคำตอบ **ก็ต่อเมื่อ** $\mathbf{b} \in C(\mathbf{A})$

**ตัวอย่าง**:
$$\mathbf{A} = \begin{pmatrix}1&2\\3&6\end{pmatrix}, \quad C(\mathbf{A}) = \text{span}\left\{\begin{pmatrix}1\\3\end{pmatrix}, \begin{pmatrix}2\\6\end{pmatrix}\right\}$$

สังเกตว่าคอลัมน์ที่ 2 = 2 × คอลัมน์ที่ 1 ดังนั้น $C(\mathbf{A})$ เป็นแค่เส้นตรงใน $\mathbb{R}^2$ ไม่ใช่ระนาบทั้งหมด

### 2.2 Null Space N(A)

**นิยาม**: $N(\mathbf{A})$ คือ set ของ vectors $\mathbf{x}$ ทั้งหมดที่ทำให้ $\mathbf{Ax} = \mathbf{0}$

$$N(\mathbf{A}) = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A}\mathbf{x} = \mathbf{0}\}$$

**ความหมาย**: $N(\mathbf{A})$ บอกว่า "input ไหน" ที่ทำให้ output เป็น 0 — คือข้อมูลที่ "สูญหาย" จาก transformation

**วิธีหา N(A)**: ทำ RREF ของ $\mathbf{A}$ แล้วตั้ง free variables เป็น parameter

**ตัวอย่าง step-by-step**:
$$\mathbf{A} = \begin{pmatrix}1&2&2&2\\2&4&6&8\\3&6&8&10\end{pmatrix}$$

ทำ RREF:
$$\text{RREF}(\mathbf{A}) = \begin{pmatrix}1&2&0&-2\\0&0&1&2\\0&0&0&0\end{pmatrix}$$

Pivot columns: 1, 3 → Pivot variables: $x_1, x_3$  
Free columns: 2, 4 → Free variables: $x_2 = s$, $x_4 = t$ (เลือกเองได้)

จาก RREF:
- $x_1 = -2s + 2t$ (จากแถว 1: $x_1 + 2x_2 - 2x_4 = 0$)
- $x_3 = -2t$ (จากแถว 2: $x_3 + 2x_4 = 0$)

**Special Solutions**:

ตั้ง $(s=1, t=0)$:
$$\mathbf{s}_1 = \begin{pmatrix}-2\\1\\0\\0\end{pmatrix}$$

ตั้ง $(s=0, t=1)$:
$$\mathbf{s}_2 = \begin{pmatrix}2\\0\\-2\\1\end{pmatrix}$$

$$N(\mathbf{A}) = \text{span}\{\mathbf{s}_1, \mathbf{s}_2\} \quad \text{(เส้นตรงสองเส้นที่ตัดกันที่ origin ใน } \mathbb{R}^4\text{)}$$

### 2.3 คำตอบสมบูรณ์ของ Ax = b

ถ้า $\mathbf{b} \in C(\mathbf{A})$ (มีคำตอบ):

$$\mathbf{x} = \mathbf{x}_{\text{particular}} + \mathbf{x}_{\text{null}}$$

- $\mathbf{x}_{\text{particular}}$ คือ **คำตอบเฉพาะ** หนึ่งตัวที่ตอบ $\mathbf{Ax}_p = \mathbf{b}$ (ตั้ง free variables = 0)
- $\mathbf{x}_{\text{null}}$ คือ **คำตอบ null** ใดก็ได้จาก $N(\mathbf{A})$

เหตุผล: $\mathbf{A}(\mathbf{x}_p + \mathbf{x}_n) = \mathbf{Ax}_p + \mathbf{Ax}_n = \mathbf{b} + \mathbf{0} = \mathbf{b}$ ✓

**DS Connection**: ใน Underdetermined Systems (มีตัวแปรมากกว่าสมการ) — เช่น Neural Network ที่มี parameters มากกว่า training samples — Null Space ของ Weight Matrix บอกว่ามี "freedom" ในการเลือก weights ที่ต่างกันแต่ให้ผลลัพธ์เหมือนกัน นี่คือเหตุผลที่ Neural Network สามารถ overfit ได้ง่าย

---

## Section 3: Rank, Dimension และ Four Fundamental Subspaces  *(Strang 3.4–3.5)*

ในส่วนนี้เราจะเรียนรู้ "ขนาด" ของ Subspace ต่าง ๆ ซึ่งรวมกันเป็นภาพสมบูรณ์ที่ Strang เรียกว่า "The Big Picture of Linear Algebra"

### 3.1 Linear Independence และ Basis

**Linear Independence**: เซต $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ เป็น Linearly Independent ถ้า:
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \quad \Rightarrow \quad c_1 = c_2 = \cdots = c_k = 0$$

(ไม่สามารถเขียน vector ใดเป็น linear combination ของที่เหลือ)

**Basis**: เซต vectors ที่ (1) Linearly Independent และ (2) Span พื้นที่ทั้งหมด

**Dimension**: จำนวน vectors ใน Basis (ค่าเดียวกันสำหรับ basis ทุกชุดของ space เดียวกัน)

### 3.2 Rank และ Rank-Nullity Theorem

**Rank** $r = \text{rank}(\mathbf{A})$:
- จำนวน pivot columns ใน RREF
- $r = \dim(C(\mathbf{A})) = \dim(C(\mathbf{A}^T))$
- $r \leq \min(m, n)$

**Nullity** = $\dim(N(\mathbf{A})) =$ จำนวน free variables

**Rank-Nullity Theorem**:
$$\text{rank}(\mathbf{A}) + \text{nullity}(\mathbf{A}) = n \quad \text{(จำนวนคอลัมน์)}$$

**ตัวอย่าง**: จาก Section 2.2 — $\mathbf{A}$ ขนาด $3 \times 4$, rank = 2, nullity = 2  
ตรวจสอบ: $2 + 2 = 4 = n$ ✓

### 3.3 The Four Fundamental Subspaces (Big Picture ของ Strang)

แมทริกซ์ $\mathbf{A}$ ขนาด $m \times n$ มี rank $r$ → สร้าง subspace สำคัญ 4 ตัว:

```
                        A: R^n ─────→ R^m
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
        ใน R^n                                ใน R^m
      ┌──────┴──────┐                    ┌─────────┴────────┐
      │             │                    │                   │
  C(A^T)          N(A)               C(A)              N(A^T)
  Row space     Null space        Column space       Left null space
  dim = r        dim = n-r          dim = r            dim = m-r
      │             │                    │                   │
      └──── ⊥ ──────┘                    └──────── ⊥ ────────┘
  (orthogonal complements)           (orthogonal complements)
```

| Subspace | นิยาม | อยู่ใน | Dimension |
|----------|-------|-------|----------|
| Column Space $C(\mathbf{A})$ | span of columns | $\mathbb{R}^m$ | $r$ |
| Null Space $N(\mathbf{A})$ | $\{\mathbf{x}: \mathbf{Ax}=\mathbf{0}\}$ | $\mathbb{R}^n$ | $n-r$ |
| Row Space $C(\mathbf{A}^T)$ | span of rows | $\mathbb{R}^n$ | $r$ |
| Left Null Space $N(\mathbf{A}^T)$ | $\{\mathbf{y}: \mathbf{A}^T\mathbf{y}=\mathbf{0}\}$ | $\mathbb{R}^m$ | $m-r$ |

**ความสัมพันธ์ Orthogonal**:
- $C(\mathbf{A}) \perp N(\mathbf{A}^T)$ — เติมเต็ม $\mathbb{R}^m$ พอดี
- $C(\mathbf{A}^T) \perp N(\mathbf{A})$ — เติมเต็ม $\mathbb{R}^n$ พอดี

**ตัวอย่าง Concrete**:
$$\mathbf{A} = \begin{pmatrix}1&2\\3&6\end{pmatrix} \quad (m=2, n=2, r=1)$$

- $C(\mathbf{A})$: span$\{(1,3)\}$ — เส้นตรงใน $\mathbb{R}^2$, dim = 1
- $N(\mathbf{A})$: $x_1 + 2x_2 = 0$ → span$\{(-2,1)\}$, dim = 1
- $C(\mathbf{A}^T)$: span$\{(1,2)\}$ — เส้นตรงใน $\mathbb{R}^2$, dim = 1
- $N(\mathbf{A}^T)$: $\mathbf{A}^T\mathbf{y}=\mathbf{0}$ → span$\{(-3,1)\}$, dim = 1

ตรวจสอบ orthogonality: $(1,3) \cdot (-3,1) = -3+3 = 0$ ✓

```python
# ─── วิเคราะห์ Rank และ Subspaces ด้วย NumPy ──────────────────────────
# วัตถุประสงค์: แสดง rank, nullity และ Big Picture ของแมทริกซ์
import numpy as np

A = np.array([[1, 2, 2, 2],
              [2, 4, 6, 8],
              [3, 6, 8, 10]], dtype=float)

m, n = A.shape
r = np.linalg.matrix_rank(A)

print(f"Size: {m}×{n}")
print(f"Rank r = {r}")
print(f"Nullity (dim N(A)) = {n - r}")
print(f"dim N(A^T) = {m - r}")
print(f"\nRank-Nullity check: {r} + {n-r} = {r + (n-r)} = n = {n}")

# SVD เพื่อหา basis ของ subspaces (preview Week 4)
# วัตถุประสงค์: column vectors ของ U คือ basis ของ C(A), V คือ basis ของ row space
U, s, Vt = np.linalg.svd(A)
print(f"\nBasis of C(A) (first {r} columns of U):\n", U[:, :r].round(3))
print(f"Basis of C(A^T) (first {r} rows of Vt):\n", Vt[:r, :].round(3))
```

**DS Connection**: ใน Feature Engineering ถ้า Rank ของ Feature Matrix $\mathbf{X}$ น้อยกว่าจำนวน features ($r < n$) หมายความว่ามี features ที่ redundant — PCA (Week 4) แก้ปัญหานี้โดยหา basis ใหม่ที่ "เต็ม rank" สำหรับข้อมูล

---

## Case Study: Netflix Recommendation — Matrix Rank และ Latent Factors

**Scenario**  
Netflix มี Users 100 ล้านคน และ Movies 10,000 เรื่อง สร้าง Rating Matrix $\mathbf{R}$ ขนาด $100M \times 10K$ แต่ข้อมูลมีมากกว่า 99% ที่ว่างเปล่า (ผู้ใช้ดูหนังเพียงบางส่วน) ปัญหา: จะพยากรณ์คะแนนที่ผู้ใช้ยังไม่ได้ให้ได้อย่างไร?

**Data**  
สมมติ Rating Matrix $\mathbf{R}$ ย่อ (4 users × 5 movies):

| | Action1 | Action2 | Romance1 | Romance2 | Comedy1 |
|--|---------|---------|---------|---------|---------|
| User1 | 5 | 4 | 1 | 2 | 3 |
| User2 | 4 | 5 | 2 | 1 | 3 |
| User3 | 1 | 2 | 5 | 4 | 3 |
| User4 | 2 | 1 | 4 | 5 | 3 |

**Method**  
ถ้า $\mathbf{R}$ มี **Low Rank** (rank น้อยกว่า min(m,n)) หมายความว่าพฤติกรรมการดูหนังทั้งหมดอธิบายได้ด้วย **latent factors** จำนวนน้อย

```python
# ─── ตรวจสอบ Rank ของ Rating Matrix ──────────────────────────────────
# วัตถุประสงค์: แสดงว่า low-rank structure ใน data คือพื้นฐานของ Recommendation System
import numpy as np

R = np.array([[5, 4, 1, 2, 3],
              [4, 5, 2, 1, 3],
              [1, 2, 5, 4, 3],
              [2, 1, 4, 5, 3]], dtype=float)

rank = np.linalg.matrix_rank(R)
print(f"Rank of R = {rank} (out of min({R.shape[0]},{R.shape[1]}) = {min(R.shape)})")

# คำนวณ singular values เพื่อดู energy distribution
U, s, Vt = np.linalg.svd(R)
energy = s**2 / np.sum(s**2)
print(f"\nSingular values: {s.round(2)}")
print(f"Energy per component: {(energy*100).round(1)}%")
print(f"Top-2 components explain: {sum(energy[:2])*100:.1f}%")
```

**Result**  
Rank = 3 (ไม่ใช่ 4 เพราะ column 5 ทุกคนให้ 3 เท่ากัน)  
Singular values ≈ [13.5, 6.7, 1.0] — 2 ตัวแรกอธิบายได้ ~97% ของข้อมูล

**Insight**  
- Column Space ของ $\mathbf{R}$ มีมิติแค่ 3 แสดงว่า "รสนิยม" ของ user ทั้ง 4 คนอธิบายได้ด้วย **latent factors** 3 ตัว (เช่น: ชอบ Action, ชอบ Romance, Neutral)
- Matrix Factorization (SVD/Low-Rank Approximation) ที่จะเรียนใน Week 4 คืออัลกอริทึมพื้นฐานของ Netflix Prize ปี 2009

---

## สรุป (Summary)

| แนวคิด | นิยาม | Python Function |
|--------|-------|----------------|
| Column Space $C(\mathbf{A})$ | span of columns | `np.linalg.matrix_rank(A)` |
| Null Space $N(\mathbf{A})$ | $\mathbf{Ax}=\mathbf{0}$ solutions | RREF, free variables |
| Row Space $C(\mathbf{A}^T)$ | span of rows | transpose ก่อนแล้ว column space |
| Rank $r$ | dim of $C(\mathbf{A})$ = pivot count | `np.linalg.matrix_rank(A)` |
| Nullity | $n - r$ | $n - $ rank |
| Rank-Nullity | $r + (n-r) = n$ | — |
| Orthogonality | $C(\mathbf{A}) \perp N(\mathbf{A}^T)$ | dot product = 0 |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 2**: Matrix Operations และ Gaussian Elimination เป็นพื้นฐานของการหา RREF
- → **Week 3 (note 2)**: Column Space และ Orthogonality นำไปสู่ Projection และ Least Squares
- → **Week 4**: Rank บอกจำนวน Singular Values ที่ไม่เป็นศูนย์ — พื้นฐานของ SVD และ PCA
- → **Week 8**: Feature Matrix $\mathbf{X}$ ต้องมี full column rank ($r = p$) เพื่อหา OLS Estimate ได้
