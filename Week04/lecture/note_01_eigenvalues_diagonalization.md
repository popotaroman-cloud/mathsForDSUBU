# Note 1: Eigenvalues, Eigenvectors และ Diagonalization
> Week 4 | CLO1 | Strang Reference: Ch.6 (Eigenvalues & Eigenvectors)

---

## บทนำ

สัปดาห์นี้เราจะเรียนรู้ **Eigenvalues และ Eigenvectors** ซึ่งเป็นแนวคิดที่สำคัญที่สุดใน Linear Algebra สำหรับ Data Science ถ้าเราคิดว่า Matrix $\mathbf{A}$ เป็น "transformation" ที่เปลี่ยนทิศทางของ vectors ทั่วไป Eigenvectors คือ vectors พิเศษที่ **ไม่เปลี่ยนทิศทาง** เมื่อถูก $\mathbf{A}$ กระทำ — มีเพียงขนาดที่เปลี่ยนไปตาม Eigenvalue $\lambda$ เป้าหมายคือให้นักศึกษาสามารถหา Eigenvalues จาก Characteristic Equation, หา Eigenvectors จาก Null Space, และเข้าใจ Diagonalization $\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$ ได้ ในชีวิตจริง Eigenvalues ใช้ใน PageRank (Google), Spectral Clustering, PCA, Stability Analysis ของระบบ และการวิเคราะห์ Normal Modes ของโครงสร้าง — นี่คือสัปดาห์ที่ Linear Algebra "มีชีวิต" ขึ้นมาจริง ๆ

---

## Section 1: Eigenvalues และ Eigenvectors  *(Strang 6.1)*

ในส่วนนี้เราจะเรียนรู้นิยามและวิธีคำนวณ Eigenvalues/Eigenvectors เพื่อสร้างความเข้าใจพื้นฐานของ "ทิศทางพิเศษ" ที่แฝงอยู่ใน Matrix transformation ทุกตัว

### 1.1 นิยามและ Geometric Intuition

**นิยาม**: $\lambda$ เป็น Eigenvalue ของ $\mathbf{A}$ และ $\mathbf{x} \neq \mathbf{0}$ เป็น Eigenvector ถ้า:

$$\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$$

**Geometric Intuition**:
- ทั่วไป: $\mathbf{A}\mathbf{v}$ จะ **หมุน + ยืด** vector $\mathbf{v}$
- Eigenvector: $\mathbf{A}\mathbf{x}$ จะ **ยืด/หดเท่านั้น** ไม่หมุน ตามปัจจัย $\lambda$

| ค่า $\lambda$ | ความหมาย |
|-------------|---------|
| $\lambda > 1$ | ยืดออก |
| $0 < \lambda < 1$ | หดเข้า |
| $\lambda = 1$ | ไม่เปลี่ยน |
| $\lambda = 0$ | กลายเป็น $\mathbf{0}$ (singular!) |
| $\lambda < 0$ | กลับทิศ + scale |

### 1.2 Characteristic Equation

จาก $\mathbf{Ax} = \lambda\mathbf{x}$ เขียนใหม่:
$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{x} = \mathbf{0}$$

สมการนี้มีคำตอบ $\mathbf{x} \neq \mathbf{0}$ ก็ต่อเมื่อ $(\mathbf{A} - \lambda\mathbf{I})$ เป็น Singular:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0 \quad \text{(Characteristic Equation)}$$

ผลลัพธ์คือ **Characteristic Polynomial** ดีกรี $n$ ซึ่งมี roots = Eigenvalues ทั้งหมด

### 1.3 ตัวอย่าง 2×2 (step-by-step)

$$\mathbf{A} = \begin{pmatrix}3&1\\1&3\end{pmatrix}$$

**หา Eigenvalues**:
$$\det(\mathbf{A}-\lambda\mathbf{I}) = \det\begin{pmatrix}3-\lambda&1\\1&3-\lambda\end{pmatrix} = (3-\lambda)^2 - 1 = 0$$

$$\lambda^2 - 6\lambda + 8 = 0 \quad \Rightarrow \quad (\lambda-4)(\lambda-2) = 0$$

$$\lambda_1 = 4, \quad \lambda_2 = 2$$

**หา Eigenvector สำหรับ $\lambda_1 = 4$**:

$$(\mathbf{A} - 4\mathbf{I})\mathbf{x} = \begin{pmatrix}-1&1\\1&-1\end{pmatrix}\mathbf{x} = \mathbf{0}$$

RREF: $x_1 = x_2$ → $\mathbf{x}_1 = \begin{pmatrix}1\\1\end{pmatrix}$ (หรือ scalar ใด ๆ)

**หา Eigenvector สำหรับ $\lambda_2 = 2$**:

$$(\mathbf{A} - 2\mathbf{I})\mathbf{x} = \begin{pmatrix}1&1\\1&1\end{pmatrix}\mathbf{x} = \mathbf{0}$$

RREF: $x_1 = -x_2$ → $\mathbf{x}_2 = \begin{pmatrix}1\\-1\end{pmatrix}$

**ตรวจสอบ**:
$$\mathbf{A}\mathbf{x}_1 = \begin{pmatrix}3&1\\1&3\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}4\\4\end{pmatrix} = 4\begin{pmatrix}1\\1\end{pmatrix} = \lambda_1\mathbf{x}_1 \checkmark$$

### 1.4 ตัวอย่าง 3×3

$$\mathbf{A} = \begin{pmatrix}2&0&0\\0&3&0\\0&0&5\end{pmatrix} \quad \text{(Diagonal Matrix)}$$

Eigenvalues ของ Diagonal Matrix คือ entries บน diagonal ทันที:
$$\lambda_1 = 2, \quad \lambda_2 = 3, \quad \lambda_3 = 5$$

Eigenvectors คือ Standard Basis Vectors:
$$\mathbf{x}_1 = \mathbf{e}_1, \quad \mathbf{x}_2 = \mathbf{e}_2, \quad \mathbf{x}_3 = \mathbf{e}_3$$

### 1.5 คุณสมบัติสำคัญ

**Trace และ Determinant**:
$$\text{trace}(\mathbf{A}) = \sum_{i} a_{ii} = \sum_{i} \lambda_i \quad \text{(sum of eigenvalues)}$$
$$\det(\mathbf{A}) = \prod_{i} \lambda_i \quad \text{(product of eigenvalues)}$$

ตรวจสอบจากตัวอย่าง: trace = 6 = 4+2 ✓, det = 8 = 4×2 ✓

**Eigenvalues ของ $\mathbf{A}^k$**: ถ้า $\mathbf{Ax} = \lambda\mathbf{x}$ แล้ว $\mathbf{A}^k\mathbf{x} = \lambda^k\mathbf{x}$

```python
# ─── คำนวณ Eigenvalues และ Eigenvectors ด้วย NumPy ──────────────────────
# วัตถุประสงค์: แสดงวิธีที่ใช้จริงใน Data Science (eigh สำหรับ symmetric)
import numpy as np

# ─── กรณีทั่วไป (non-symmetric matrix) ──────────────────────────────
A = np.array([[3, 1],
              [1, 3]], dtype=float)

# np.linalg.eig ใช้สำหรับ general matrix
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# ตรวจสอบ: Av = λv
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = A @ v
    lv = lam * v
    print(f"\nλ={lam:.1f}: Av = {Av.round(4)}, λv = {lv.round(4)}")
    print(f"  Match: {np.allclose(Av, lv)}")

# ─── กรณี Symmetric Matrix (แนะนำสำหรับ Covariance Matrix) ─────────
# วัตถุประสงค์: np.linalg.eigh ให้ eigenvalues จริง เรียงจากน้อยไปมาก
# และ eigenvectors ตั้งฉากกัน (guaranteed สำหรับ symmetric matrix)
C = np.array([[4, 2, 0],
              [2, 3, 1],
              [0, 1, 2]], dtype=float)  # Symmetric

eigenvalues_sym, eigenvectors_sym = np.linalg.eigh(C)
print("\n[Symmetric] Eigenvalues:", eigenvalues_sym.round(4))
print("[Symmetric] Eigenvectors:\n", eigenvectors_sym.round(4))

# ตรวจสอบ orthogonality ของ eigenvectors ใน symmetric case
Q = eigenvectors_sym
print("\nQ^T Q (ต้องเป็น Identity):\n", (Q.T @ Q).round(8))
```

**DS Connection**: `np.linalg.eigh()` (symmetric) ให้ guaranteed real eigenvalues และ orthogonal eigenvectors — ใช้เสมอเมื่อทำงานกับ Covariance Matrix หรือ Gram Matrix $\mathbf{X}^T\mathbf{X}$

---

## Section 2: Diagonalization  *(Strang 6.2)*

ในส่วนนี้เราจะเรียนรู้ **Diagonalization** ซึ่งเป็นการเขียน Matrix ในรูป $\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$ เพื่อให้การคำนวณ powers ของ Matrix เร็วขึ้นมหาศาล และเพื่อเข้าใจ "โครงสร้างที่แท้จริง" ของ transformation

### 2.1 สูตรและเงื่อนไข

ถ้า $\mathbf{A}$ มี $n$ eigenvectors ที่ linearly independent $\mathbf{x}_1, \ldots, \mathbf{x}_n$:

$$\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$$

โดย:
- $\mathbf{P} = [\mathbf{x}_1 | \mathbf{x}_2 | \cdots | \mathbf{x}_n]$ — matrix ที่มี eigenvectors เป็น columns
- $\mathbf{D} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ — diagonal matrix ของ eigenvalues

**Derivation**: $\mathbf{AP} = \mathbf{P}\mathbf{D}$ เพราะ $\mathbf{A}[\mathbf{x}_1|\cdots|\mathbf{x}_n] = [\lambda_1\mathbf{x}_1|\cdots|\lambda_n\mathbf{x}_n] = \mathbf{P}\mathbf{D}$

### 2.2 ตัวอย่าง (ต่อจาก Section 1)

$$\mathbf{A} = \begin{pmatrix}3&1\\1&3\end{pmatrix}, \quad \lambda_1=4, \lambda_2=2, \quad \mathbf{x}_1=\begin{pmatrix}1\\1\end{pmatrix}, \mathbf{x}_2=\begin{pmatrix}1\\-1\end{pmatrix}$$

$$\mathbf{P} = \begin{pmatrix}1&1\\1&-1\end{pmatrix}, \quad \mathbf{D} = \begin{pmatrix}4&0\\0&2\end{pmatrix}$$

$$\mathbf{P}^{-1} = \frac{1}{-2}\begin{pmatrix}-1&-1\\-1&1\end{pmatrix} = \begin{pmatrix}1/2&1/2\\1/2&-1/2\end{pmatrix}$$

**ตรวจสอบ** $\mathbf{PDP}^{-1} = \mathbf{A}$:

$$\begin{pmatrix}1&1\\1&-1\end{pmatrix}\begin{pmatrix}4&0\\0&2\end{pmatrix}\begin{pmatrix}1/2&1/2\\1/2&-1/2\end{pmatrix}$$

$$= \begin{pmatrix}4&2\\4&-2\end{pmatrix}\begin{pmatrix}1/2&1/2\\1/2&-1/2\end{pmatrix} = \begin{pmatrix}2+1&2-1\\2-1&2+1\end{pmatrix} = \begin{pmatrix}3&1\\1&3\end{pmatrix} = \mathbf{A} \checkmark$$

### 2.3 Powers of Matrix: A^k = P D^k P^{-1}

$$\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1} \quad \text{โดย} \quad \mathbf{D}^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$$

**ทำไมถึงเร็ว**: คำนวณ $\lambda^k$ แค่ $n$ ตัว แทนที่จะคูณ Matrix $k$ ครั้ง!

**ตัวอย่าง** $\mathbf{A}^{10}$:
$$\mathbf{D}^{10} = \begin{pmatrix}4^{10}&0\\0&2^{10}\end{pmatrix} = \begin{pmatrix}1048576&0\\0&1024\end{pmatrix}$$

**DS Connection**: ใน **Markov Chains** (ที่ใช้ใน NLP, PageRank) เราสนใจ $\mathbf{A}^k$ เมื่อ $k \to \infty$ ถ้า $|\lambda_1| > |\lambda_2| > \cdots$ แล้ว $\mathbf{A}^k \approx \lambda_1^k \mathbf{x}_1 \mathbf{y}_1^T$ — Dominant Eigenvector! นี่คือ Power Method ที่ใช้ใน PageRank

---

## Section 3: Symmetric Matrices และ Positive Definite  *(Strang 6.4–6.5)*

ในส่วนนี้เราจะเรียนรู้คุณสมบัติพิเศษของ Symmetric Matrix ซึ่งสำคัญมากสำหรับ Data Science เพราะ Covariance Matrix และ Gram Matrix ล้วนเป็น Symmetric

### 3.1 Spectral Theorem

**Theorem (สำคัญมาก!)**: ถ้า $\mathbf{A} = \mathbf{A}^T$ (Symmetric) แล้ว:
1. Eigenvalues ทั้งหมดเป็น **จำนวนจริง** (real)
2. Eigenvectors ที่สอดคล้องกับ eigenvalues ต่างกัน **ตั้งฉากกัน** (orthogonal)
3. มี Orthogonal Diagonalization: $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ โดย $\mathbf{Q}^T = \mathbf{Q}^{-1}$

$$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i\mathbf{q}_i^T$$

(Spectral Decomposition — เขียน $\mathbf{A}$ เป็น sum ของ rank-1 matrices)

### 3.2 Positive Definite Matrix

**นิยาม**: $\mathbf{A}$ เป็น **Positive Definite** (PD) ถ้า $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ สำหรับทุก $\mathbf{x} \neq \mathbf{0}$

**เงื่อนไขที่เท่ากัน**:
- Eigenvalues ทั้งหมด $> 0$
- Pivot ทั้งหมด $> 0$ (จาก Gaussian Elimination)
- All leading determinants $> 0$

**Semi-Positive Definite**: $\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$ (eigenvalues $\geq 0$)

**ตัวอย่าง**:
$$\mathbf{A} = \begin{pmatrix}2&1\\1&3\end{pmatrix}: \quad \lambda_1,\lambda_2 = \frac{5 \pm \sqrt{5}}{2} > 0 \quad \Rightarrow \text{PD} ✓$$

$$\mathbf{B} = \begin{pmatrix}1&2\\2&4\end{pmatrix}: \quad \lambda = 5, 0 \quad \Rightarrow \text{PSD (not PD)} $$

### 3.3 Covariance Matrix

**นิยาม** (ใน Data Science): ถ้า $\mathbf{X}$ เป็น Centered Data Matrix (mean = 0):

$$\boldsymbol{\Sigma} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$$

**คุณสมบัติ**:
- $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$ (Symmetric) — เพราะ $(\mathbf{X}^T\mathbf{X})^T = \mathbf{X}^T\mathbf{X}$
- $\boldsymbol{\Sigma}$ เป็น Positive Semi-Definite — เพราะ $\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v} = \frac{1}{n-1}\|\mathbf{Xv}\|^2 \geq 0$
- Eigenvalues = Variances ในทิศทาง Principal Components (จะเรียนใน note 2)

```python
# ─── Eigendecomposition ของ Covariance Matrix ────────────────────────
# วัตถุประสงค์: แสดงว่า eigenvalues ของ Σ = variance ในแต่ละทิศทาง
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # 150 samples × 4 features

# Center data (ลบ mean)
# วัตถุประสงค์: ต้อง center ก่อนเสมอ เพราะ Covariance วัด variation รอบ mean
X_centered = X - X.mean(axis=0)

# คำนวณ Covariance Matrix
# วัตถุประสงค์: Σ_ij = covariance ระหว่าง feature i และ j
Sigma = (X_centered.T @ X_centered) / (len(X) - 1)
print("Covariance Matrix:\n", Sigma.round(3))

# Eigendecompose Σ ด้วย eigh (symmetric)
# วัตถุประสงค์: eigenvalues = variance, eigenvectors = principal directions
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
# eigh คืน eigenvalues เรียงจากน้อยไปมาก — ต้อง reverse
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues (variance per PC):", eigenvalues.round(4))
variance_explained = eigenvalues / eigenvalues.sum()
print("Variance explained:", (variance_explained * 100).round(1), "%")
print("Cumulative:", (np.cumsum(variance_explained) * 100).round(1), "%")
```

**DS Connection**: PCA คือการ Eigendecompose Covariance Matrix — Eigenvectors คือ Principal Components และ Eigenvalues คือ Variance ที่อธิบายได้แต่ละทิศทาง เราจะเรียนเต็มๆ ใน note 2

---

## Case Study: Vibration Analysis ด้วย Eigenvalues — Modal Analysis

**Scenario**  
วิศวกรโครงสร้างต้องออกแบบสะพานให้ทนต่อแผ่นดินไหว ต้องรู้ว่าสะพานจะ "สั่น" ที่ความถี่ใดบ้าง เพื่อหลีกเลี่ยง Resonance ปัญหานี้แปลงเป็น Eigenvalue Problem ได้

**Data**  
ระบบ Mass-Spring 2 degrees of freedom:

$$\mathbf{K} = \begin{pmatrix}2&-1\\-1&2\end{pmatrix} \text{ (Stiffness Matrix)}, \quad \mathbf{M} = \begin{pmatrix}1&0\\0&1\end{pmatrix} \text{ (Mass Matrix)}$$

**Method**  
Generalized Eigenvalue Problem: $\mathbf{K}\mathbf{v} = \omega^2\mathbf{M}\mathbf{v}$

สำหรับ $\mathbf{M} = \mathbf{I}$: เป็นแค่ $\mathbf{Kv} = \lambda\mathbf{v}$ โดย $\lambda = \omega^2$

```python
# ─── Vibration Analysis ด้วย Eigenvalues ───────────────────────────────
# วัตถุประสงค์: แสดง real-world application ของ eigenvalue ในงานวิศวกรรม
import numpy as np

# Stiffness matrix ของ 2-DOF system
K = np.array([[2, -1],
              [-1, 2]], dtype=float)

# หา eigenvalues = ω² (squared natural frequencies)
omega_sq, modes = np.linalg.eigh(K)
omega = np.sqrt(omega_sq)

print(f"Natural frequencies: ω₁ = {omega[0]:.4f}, ω₂ = {omega[1]:.4f} rad/s")
print(f"Mode shapes:\n  Mode 1: {modes[:, 0].round(4)}\n  Mode 2: {modes[:, 1].round(4)}")
```

**Result**  
$\omega_1 = 1.0$ rad/s (Mode ที่ 1: both masses sway together)  
$\omega_2 = \sqrt{3} \approx 1.732$ rad/s (Mode ที่ 2: masses sway in opposite directions)

**Insight**  
- Natural frequencies ของสะพานต้องไม่ตรงกับความถี่ของแผ่นดินไหว (2–10 Hz)
- Eigenvectors (Mode Shapes) บอกว่าสะพานจะสั่นรูปแบบไหน
- แนวคิดเดียวกันนี้ใช้ใน Spectral Clustering ใน ML: eigenvalues ของ Laplacian Matrix บอก "cluster structure" ของ Graph

---

## สรุป (Summary)

| แนวคิด | สูตร | Python |
|--------|------|--------|
| Eigenvalue equation | $\mathbf{Ax} = \lambda\mathbf{x}$ | — |
| Characteristic equation | $\det(\mathbf{A}-\lambda\mathbf{I})=0$ | `np.linalg.eig(A)` |
| Eigenvector | Null space ของ $(\mathbf{A}-\lambda\mathbf{I})$ | `np.linalg.eig(A)[1]` |
| Trace = $\sum\lambda_i$ | $\text{tr}(\mathbf{A}) = \sum a_{ii}$ | `np.trace(A)` |
| Det = $\prod\lambda_i$ | $\det(\mathbf{A}) = \prod\lambda_i$ | `np.linalg.det(A)` |
| Diagonalization | $\mathbf{A} = \mathbf{PDP}^{-1}$ | `np.linalg.eig` |
| Powers | $\mathbf{A}^k = \mathbf{PD}^k\mathbf{P}^{-1}$ | eigenvalues^k |
| Spectral Theorem | $\mathbf{A}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ (symmetric) | `np.linalg.eigh` |
| Positive Definite | all $\lambda_i > 0$ | check `eigenvalues` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 3**: Null Space ของ $(\mathbf{A}-\lambda\mathbf{I})$ เป็น Eigenvectors — ใช้ RREF ที่เรียนมา; Orthogonality ของ eigenvectors ใน symmetric case เชื่อมกับ Projection
- ← **Week 2**: Determinant ใช้ใน Characteristic Equation; $\det(\mathbf{A}) = \prod\lambda_i$ เชื่อม det กับ eigenvalues
- → **Week 4 (note 2)**: Eigendecomposition ของ Covariance Matrix คือ PCA; SVD คือ generalization ของ Eigendecomposition ไปยัง non-square matrix
- → **Week 14**: ใน Cross-Validation, Covariance Structure ของ data ส่งผลต่อ Bias-Variance — eigenvalues ของ $X^TX$ เป็นพื้นฐานของ Ridge Regression
