# Slide Deck: Matrix Types and Operations
> Week 02 | CLO1 | Strang Ch.2 | 10 slides

---
## Slide 1 — Title
**Matrix: เครื่องมือแทน Transformation และ Dataset**  
Week 2 | CLO1 | LLo: ดำเนินการทางแมทริกซ์ได้

---
## Slide 2 — Week Overview
**Key Message**: Matrix คือ Vector ของ Vectors — dataset ทุกชนิดเก็บในรูป Matrix

สัปดาห์นี้เราจะขยายจาก Vector (สัปดาห์ที่แล้ว) ไปสู่ Matrix ซึ่งเป็นโครงสร้างข้อมูลหลักใน Machine Learning ทุกประเภท Dataset n samples × p features คือ Matrix **X** ∈ ℝⁿˣᵖ, weights ของ Neural Network คือ Matrix **W**, image คือ Matrix ของ pixel values เป้าหมายวันนี้คือให้นักศึกษาสร้างและดำเนินการ Matrix ด้วย NumPy ได้อย่างคล่องแคล่ว รวมถึงเข้าใจว่า Matrix multiplication แทน transformation อะไร

**สิ่งที่จะเรียนวันนี้**:
- Matrix types: zero, identity, diagonal, symmetric, triangular
- Operations: +, -, ×, transpose
- Properties: AB ≠ BA, (AB)ᵀ = BᵀAᵀ

---
## Slide 3 — Matrix Notation
**Key Message**: Aᵢⱼ คือ element แถว i คอลัมน์ j — index เริ่ม 1 ใน math, 0 ใน Python

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix} \in \mathbb{R}^{2 \times 3}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)    # (2, 3)  — 2 rows, 3 columns
print(A[0, 2])    # 3       — a₁₃ ใน math = A[0,2] ใน Python
print(A[:, 1])    # [2, 5]  — column 2 (index 1)
print(A[1, :])    # [4,5,6] — row 2 (index 1)
```

**ใน DS**: X ∈ ℝⁿˣᵖ → X[i, j] คือ feature j ของ sample i

---
## Slide 4 — Special Matrices
**Key Message**: Special matrices มีคุณสมบัติพิเศษที่ช่วยลด computation

| Matrix | คุณสมบัติ | ใช้ใน |
|--------|---------|------|
| Identity **I** | AI = IA = A | เป็น baseline transformation |
| Zero **0** | A + 0 = A | bias term ตอน init = 0 |
| Diagonal **D** | Dᵢⱼ = 0 ถ้า i ≠ j | scaling transformation |
| Symmetric **S** | S = Sᵀ | Covariance matrix, XᵀX |
| Triangular | L หรือ U | LU decomposition |

```python
I = np.eye(3)                    # Identity
D = np.diag([2, 3, 4])           # Diagonal
S = A.T @ A                      # Always symmetric
print(np.allclose(S, S.T))       # True
```

---
## Slide 5 — Matrix Multiplication
**Key Message**: (AB)ᵢⱼ = dot product ของ row i ของ A กับ column j ของ B

$$(\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} = \mathbf{a}_i^T \mathbf{b}_j$$

**ข้อกำหนด**: A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ → AB ∈ ℝᵐˣᵖ (inner dimensions ต้องเท่ากัน)

```python
A = np.array([[1, 2], [3, 4]])   # 2×2
B = np.array([[5, 6], [7, 8]])   # 2×2
print(A @ B)   # [[19, 22], [43, 50]]
# A[0,:] · B[:,0] = 1*5+2*7 = 19
# A[0,:] · B[:,1] = 1*6+2*8 = 22
```

[FIGURE: diagram แสดง row × column rule]

---
## Slide 6 — Non-Commutativity: AB ≠ BA
**Key Message**: Matrix multiplication ไม่ commutative — ลำดับสำคัญมากใน ML

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [2, 3]])

print('AB =\n', A @ B)
print('BA =\n', B @ A)
print('AB == BA?', np.allclose(A @ B, B @ A))  # False!
```

**ทำไมถึงสำคัญใน ML**:
- Normal Equations: **(XᵀX)** ≠ **(XXᵀ)** — ต้องเขียนถูกลำดับ
- Chain rule ของ Neural Network: dL/dW = (dL/dA)(dA/dW) — ลำดับสำคัญ
- Covariance: **XᵀX** (p×p) vs **XXᵀ** (n×n) คนละ matrix

---
## Slide 7 — Transpose Properties
**Key Message**: Transpose คือการ flip matrix — สำคัญมากในสูตร ML

$$(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$$
$$(\mathbf{A}^T)^T = \mathbf{A}$$
$$(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$$

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)

lhs = (A @ B).T
rhs = B.T @ A.T
print('(AB)ᵀ == BᵀAᵀ?', np.allclose(lhs, rhs))  # True
```

**ใน ML**: Normal Equation = (XᵀX)⁻¹Xᵀy — ต้องใช้ transpose ถูกต้อง

---
## Slide 8 — Matrix as Linear Transformation
**Key Message**: Matrix multiplication คือ geometric transformation ของ vector

[FIGURE: แสดง 2D vector ก่อนและหลัง transform ด้วย rotation matrix]

```python
# Rotation matrix: หมุน 90° counterclockwise
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v = np.array([1, 0])   # vector ชี้ขวา
v_rotated = R @ v
print('After 90° rotation:', v_rotated.round())  # [0, 1] ชี้ขึ้น
```

**ใน DS**: การ scale features = คูณด้วย diagonal matrix

---
## Slide 9 — Matrix ใน Data Science: X = Dataset
**Key Message**: Dataset คือ Matrix ที่ทุก row เป็น sample, ทุก column เป็น feature

```
X ∈ ℝⁿˣᵖ:
         age  income  edu_yrs  ...
sample 1: 25    50k      16
sample 2: 32    80k      18
sample 3: 28    60k      16
    ⋮
sample n: ...
```

**Common operations บน X**:
```python
X_centered = X - X.mean(axis=0)      # center features
cov = (X_centered.T @ X_centered) / (n-1)  # covariance matrix
gram = X @ X.T                        # n×n gram matrix (similarity)
```

---
## Slide 10 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- Matrix **A** ∈ ℝᵐˣⁿ: m rows, n columns, Aᵢⱼ คือ element
- Operations: A+B, cA, AB (@ ใน NumPy)
- AB ≠ BA (non-commutative), (AB)ᵀ = BᵀAᵀ
- Dataset = Matrix, transformation = Matrix multiplication

**สัปดาห์ต่อไป (ใน Lecture deck นี้)**: Gaussian Elimination — แก้ Ax = b
