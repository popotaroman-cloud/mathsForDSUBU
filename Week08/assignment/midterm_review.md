# Midterm Review — ทบทวนก่อนสอบกลางภาค
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**ครอบคลุม**: CLO1 (Weeks 1–4) + CLO2 (Weeks 5–7)  
**สอบกลางภาค**: สัปดาห์ที่ 8 (30% ของคะแนนรวม)

---

## ภาพรวมสอบกลางภาค

สอบกลางภาคครอบคลุม **CLO1** (Linear Algebra, Weeks 1–4) และ **CLO2** (Statistical Learning Framework, Weeks 5–7) รูปแบบข้อสอบมีทั้ง multiple choice, short answer, และ problem solving ไม่มี open book แต่อนุญาตให้นำ formula sheet 1 หน้า A4 ได้

---

## Part A — CLO1: Linear Algebra (Weeks 1–4)

### Week 1 — Vector Basics
**Checklist ที่ต้องทำได้**:
- [ ] สร้าง vector ด้วย `np.array()` และตรวจสอบ shape, dtype, ndim
- [ ] คำนวณ magnitude: $\|\mathbf{v}\| = \sqrt{\sum v_i^2}$ = `np.linalg.norm(v)`
- [ ] คำนวณ unit vector: $\hat{v} = \mathbf{v}/\|\mathbf{v}\|$
- [ ] คำนวณ dot product: $\mathbf{u}\cdot\mathbf{v} = \sum u_i v_i$ = `np.dot(u, v)`
- [ ] คำนวณ cosine similarity: $\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$

**ตัวอย่างข้อสอบ**: กำหนด **u** = [3, 4, 0], **v** = [1, 0, 2] — หา ||u||, ||v||, **u**·**v**, cosine similarity

### Week 2 — Matrix Operations
**Checklist**:
- [ ] Matrix multiplication (AB): inner dimensions ต้องเท่ากัน
- [ ] (AB)ᵀ = BᵀAᵀ — verify ด้วย np.allclose
- [ ] แก้ Ax = b ด้วย `np.linalg.solve(A, b)`
- [ ] det(A): singular ↔ det = 0 ↔ no inverse
- [ ] LU decomposition: PA = LU (`scipy.linalg.lu`)

**ตัวอย่างข้อสอบ**: A = [[2,1],[1,3]] — หา det(A), A⁻¹, แก้ Ax = [5, 8]

### Week 3 — Subspaces และ Least Squares
**Checklist**:
- [ ] rank(A) = number of pivots = dim C(A)
- [ ] Rank-Nullity: rank + nullity = n (columns)
- [ ] Four Subspaces: C(A), N(A), C(Aᵀ), N(Aᵀ)
- [ ] Projection: P = A(AᵀA)⁻¹Aᵀ, P² = P, Pᵀ = P
- [ ] Least Squares: AᵀAx̂ = Aᵀb → x̂ = `np.linalg.lstsq(A, b)`

**ตัวอย่างข้อสอบ**: A มี rank 2, size 4×5 — dim ของแต่ละ subspace?

### Week 4 — Eigenvalues และ SVD/PCA
**Checklist**:
- [ ] Eigenvalue equation: Av = λv
- [ ] det(A − λI) = 0 → characteristic polynomial
- [ ] Diagonalization: A = PΛP⁻¹ (ถ้า n independent eigenvectors)
- [ ] SVD: A = UΣVᵀ — U, V orthogonal, Σ diagonal (singular values)
- [ ] PCA: principal components = eigenvectors ของ covariance matrix
- [ ] `np.linalg.eig(A)`, `np.linalg.svd(A)`, `sklearn PCA`

---

## Part B — CLO2: Statistical Learning Framework (Weeks 5–7)

### Week 5 — Statistical Learning Overview
**Checklist**:
- [ ] Supervised vs Unsupervised learning
- [ ] Regression (Y continuous) vs Classification (Y categorical)
- [ ] f(X) ≈ f̂(X): estimation ของ true function
- [ ] Reducible vs Irreducible error

### Week 6 — Bias-Variance Tradeoff
**Checklist**:
- [ ] Test MSE = Bias² + Variance + σ²
- [ ] High Bias = underfitting (model too simple)
- [ ] High Variance = overfitting (model too flexible)
- [ ] KNN: k เล็ก → low bias, high variance; k ใหญ่ → high bias, low variance
- [ ] Training MSE เสมอ ≤ Test MSE เมื่อ overfitting

### Week 7 — EDA และ Statistical Inference
**Checklist**:
- [ ] `.describe()`: mean, std, quartiles
- [ ] Pearson r: correlation [−1, 1]
- [ ] SE = s/√n, CLT: x̄ ~ N(μ, σ²/n)
- [ ] 95% CI: x̄ ± t* × SE
- [ ] t-test: t = (x̄ − μ₀)/SE, p-value = P(|T| ≥ |t|)
- [ ] p < 0.05 → reject H₀

---

## สูตรสำคัญที่ต้องจำ

| หัวข้อ | สูตร |
|--------|------|
| Cosine similarity | $\cos\theta = \mathbf{u}^T\mathbf{v} / (\|\mathbf{u}\|\|\mathbf{v}\|)$ |
| Least Squares | $\hat{x} = (A^TA)^{-1}A^Ty$ |
| Projection matrix | $P = A(A^TA)^{-1}A^T$ |
| SE ของ mean | $SE(\bar{X}) = s/\sqrt{n}$ |
| 95% CI | $\bar{X} \pm t^* \cdot SE$ |
| t-statistic | $t = (\bar{X} - \mu_0)/SE$ |
| Bias-Variance | $E[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \sigma^2$ |

---

## แนวข้อสอบตัวอย่าง

**ข้อ 1 (5 คะแนน)**: กำหนด matrix A = [[1,2,1],[2,4,3],[3,6,5]]
- หา rank(A) และ dim ของ N(A)
- หา basis ของ null space ด้วย SVD
- Verify ว่า Ax = 0 สำหรับ null vector

**ข้อ 2 (5 คะแนน)**: ข้อมูล 50 คะแนนสอบ: x̄ = 75, s = 10
- สร้าง 95% CI สำหรับ mean คะแนนของ population
- ทดสอบ H₀: μ = 70, H₁: μ > 70 ที่ α = 0.05

**ข้อ 3 (5 คะแนน)**: อธิบาย Bias-Variance tradeoff พร้อม diagram — ทำไม model ที่ fit training data ดีมากอาจไม่ดีบน test data?
