# Slide Deck: Diagonalization & Symmetric Matrices
> Week 4 | CLO1 | Strang Ch.6.2, 6.4–6.5 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 4**  
Diagonalization & Symmetric Matrices: เมื่อ Matrix มีโครงสร้างที่สวยงาม  
CLO1: นำ Spectral Theorem ประยุกต์ใช้กับ Covariance Matrix ใน PCA ได้

---

## Slide 2 — Week Overview (Deck 2)
**Key Message**: Diagonalization แยก matrix เป็น eigenstructure — ทำให้ compute Aᵏ และ Spectral Decomposition ได้ง่ายขึ้นมาก

Diagonalization คือการแยก matrix A ออกเป็น A = PDP⁻¹ ซึ่ง D เป็น diagonal matrix ของ eigenvalues ที่ทำให้การคำนวณหลายอย่างง่ายขึ้นมาก เช่น Aᵏ = PDᵏP⁻¹ ซึ่ง Dᵏ คำนวณได้ง่ายมากโดยยกแต่ละ diagonal element กำลัง k Symmetric matrix เป็น special case ที่สำคัญยิ่งเพราะ P กลายเป็น orthogonal matrix (P⁻¹ = Pᵀ) ทำให้ A = QΛQᵀ (Spectral Theorem) Covariance matrix Σ = (1/n)XᵀX เป็น symmetric positive semi-definite เสมอ และ Spectral Theorem คือ basis ของ PCA เป้าหมายของ deck นี้คือให้นักศึกษาทำ diagonalization ได้ อธิบาย Spectral Theorem ได้ และเชื่อมกับ covariance matrix ใน PCA ได้

**สิ่งที่จะเรียนรู้:**
- Diagonalization: A = PDP⁻¹
- เงื่อนไข: n independent eigenvectors
- Symmetric matrix: eigenvalues จริง, eigenvectors orthogonal
- Spectral Theorem: A = QΛQᵀ
- Positive Definite: λᵢ > 0 ↔ xᵀAx > 0

---

## Slide 3 — Diagonalization: A = PDP⁻¹
**Key Message**: ถ้า A มี n independent eigenvectors ก็ diagonalize ได้ — ทำให้ Aᵏ และ function of A คำนวณได้ง่าย

**การ Diagonalize:**
```
P = [x₁ | x₂ | ... | xₙ]   (eigenvectors เป็น columns)
D = diag(λ₁, λ₂, ..., λₙ)  (eigenvalues บน diagonal)

A = P D P⁻¹
```

**ตัวอย่าง:**
```
A = [[3, 1],    λ₁=3, x₁=[1,0]
     [0, 2]]    λ₂=2, x₂=[1,-1]/√2... (normalize)

P = [[1,  1],    D = [[3, 0],
     [0, -1]]         [0, 2]]

Verify: P @ D @ inv(P) = A ✓
```

**Conditions:**
- A ต้องมี n independent eigenvectors (diagonalizable)
- Matrix ที่ไม่ diagonalizable: repeated eigenvalues + GM < AM (defective matrix)
- Symmetric matrix: **เสมอ diagonalizable** (Spectral Theorem)

---

## Slide 4 — Powers of Matrices
**Key Message**: Aᵏ = PDᵏP⁻¹ ทำให้ matrix power คำนวณได้ใน O(n) เพราะ Dᵏ = diag(λ₁ᵏ, ..., λₙᵏ)

**ทำไมต้องใช้ Diagonalization:**
```
Aᵏ = (PDP⁻¹)(PDP⁻¹)...(PDP⁻¹)
   = P(DD...D)P⁻¹
   = P Dᵏ P⁻¹
```

**Dᵏ คำนวณได้ง่าย:**
```
Dᵏ = diag(λ₁ᵏ, λ₂ᵏ, ..., λₙᵏ)
```

**ตัวอย่าง — Fibonacci numbers:**
```
[[F(n+1)]] = [[1, 1]]ⁿ × [[1]]
[[F(n)  ]]   [[1, 0]]     [[0]]

ใช้ diagonalization → closed-form Fibonacci formula
λ = (1 ± √5) / 2  (golden ratio φ!)
```

**DS Application:**
- Markov chain: Pᵏ → steady state เมื่อ k→∞
- Convergence ขึ้นกับ |λ₂/λ₁| (spectral gap)

---

## Slide 5 — Symmetric Matrices: Spectral Theorem
**Key Message**: Symmetric matrix A = Aᵀ diagonalize เป็น A = QΛQᵀ เสมอ ด้วย orthogonal Q — นี่คือ mathematical foundation ของ PCA

**Spectral Theorem:**
```
ถ้า A = Aᵀ (symmetric), แล้ว:
A = Q Λ Qᵀ
```
- **Q**: orthogonal matrix (eigenvectors เป็น columns, QᵀQ = I)
- **Λ**: diagonal matrix ของ real eigenvalues
- เรียก Q⁻¹ = Qᵀ เพราะ Q orthogonal

**Properties ของ Symmetric Matrix:**
- Eigenvalues เป็น **real** เสมอ (แม้ A มี complex entries)
- Eigenvectors สำหรับ eigenvalues ต่างกัน **orthogonal** กัน
- สามารถ diagonalize ได้เสมอ (GM = AM สำหรับทุก λ)

**[FIGURE: แสดง A = Q × Λ × Qᵀ เป็นภาพ: rotate → stretch → rotate back]**

---

## Slide 6 — Positive Definite Matrices
**Key Message**: Positive Definite ⟺ xᵀAx > 0 ⟺ eigenvalues ทั้งหมด > 0 — สำคัญสำหรับ Covariance matrix และ Optimization

**นิยาม:**
```
A Positive Definite (PD):     xᵀAx > 0  สำหรับทุก x ≠ 0
                               ↔ eigenvalues ทั้งหมด > 0

A Positive Semi-Definite (PSD): xᵀAx ≥ 0
                               ↔ eigenvalues ทั้งหมด ≥ 0
```

**Tests สำหรับ PD:**
1. Eigenvalue test: λᵢ > 0 ทั้งหมด
2. Determinant test: leading minors > 0 ทั้งหมด
3. Cholesky test: A = LLᵀ มี solution

**ทำไม Covariance Matrix เป็น PSD:**
```
Σ = (1/n) Xᵀ X
xᵀΣx = (1/n) xᵀXᵀXx = (1/n) ||Xx||² ≥ 0  ✓
```

- Σ เป็น PD ถ้า X มี full column rank (features ไม่ linearly dependent)
- [FIGURE: level curves ของ xᵀAx — ellipse สำหรับ PD, parabolic cylinder สำหรับ PSD]

---

## Slide 7 — Spectral Decomposition
**Key Message**: A = Σᵢ λᵢ qᵢqᵢᵀ — matrix ทุกตัวเป็น weighted sum ของ rank-1 "components" ซึ่งตรงกับ PC ใน PCA

**Spectral Decomposition:**
```
A = Q Λ Qᵀ = λ₁ q₁q₁ᵀ + λ₂ q₂q₂ᵀ + ... + λₙ qₙqₙᵀ
```
- qᵢqᵢᵀ: rank-1 matrix ที่ project ลงทิศ qᵢ
- λᵢ: "weight" ของ component ที่ i

**เชื่อมกับ PCA:**
```
Covariance Σ = λ₁ q₁q₁ᵀ + λ₂ q₂q₂ᵀ + ...
             = PC1 component + PC2 component + ...
```
- เก็บเฉพาะ k components ที่ใหญ่ที่สุด → low-rank approximation ของ Σ
- Variance explained ≈ (λ₁ + ... + λₖ) / Σλᵢ

**[FIGURE: แสดง Σ = Σᵢ λᵢ qᵢqᵢᵀ เป็น visual decomposition]**

---

## Slide 8 — Python: Working with Symmetric Matrices
**Key Message**: `np.linalg.eigh` + sort descending + project คือ PCA from scratch ใน 5 บรรทัด

**PCA from Scratch (ทุกขั้นตอน):**
```python
# 1. Center data
X_c = X - X.mean(axis=0)

# 2. Covariance matrix
C = (1/(n-1)) * X_c.T @ X_c

# 3. Eigendecompose (eigh: sorted ascending, orthonormal)
vals, vecs = np.linalg.eigh(C)

# 4. Sort descending (PC1 = largest variance)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

# 5. Project onto top-k PCs
X_pca = X_c @ vecs[:, :k]
```

**Verify Spectral Theorem:**
```python
C_reconstructed = vecs @ np.diag(vals) @ vecs.T
np.allclose(C, C_reconstructed)  # True
```

---

## Slide 9 — Summary Table: Eigenvalue Properties
**Key Message**: ตาราง reference สรุป properties ของ matrix ต่างๆ — จำไว้ใช้ใน PCA และ ML

| Matrix type | Eigenvalues | Eigenvectors | Diagonalizable? | Python |
|------------|-------------|--------------|-----------------|--------|
| General (A) | อาจ complex | ไม่จำเป็น orthogonal | ถ้า n independent | `eig` |
| Symmetric (A=Aᵀ) | Real เสมอ | Orthogonal | เสมอ | `eigh` |
| PSD (xᵀAx≥0) | λᵢ ≥ 0 | Orthogonal | เสมอ | `eigh` |
| PD (xᵀAx>0) | λᵢ > 0 | Orthogonal | เสมอ | `eigh` |
| Covariance Σ | λᵢ ≥ 0 (PSD) | Orthogonal | เสมอ | `eigh` |

**Formula Reference:**
- Trace = Σλᵢ = Σσᵢ²/n (total variance)
- Det = Πλᵢ (volume scaling)
- Aᵏ = P Dᵏ P⁻¹
- Spectral: A = QΛQᵀ = Σ λᵢ qᵢqᵢᵀ

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Diagonalization**: A = PDP⁻¹ ทำให้ Aᵏ = PDᵏP⁻¹ คำนวณได้ง่าย
- **Spectral Theorem**: Symmetric A = QΛQᵀ เสมอ (Q orthogonal)
- **Positive Definite**: eigenvalues > 0 ↔ xᵀAx > 0
- **Covariance matrix**: symmetric PSD, eigenvalues = variance ต่อ direction
- **PCA foundation**: Σ = QΛQᵀ → eigenvectors = PCs, eigenvalues = variance explained

**สัปดาห์ถัดไป (Slide Deck 3):** Singular Value Decomposition (SVD) — A = UΣVᵀ สำหรับ matrix ทุกรูปร่าง ไม่ต้องเป็น square หรือ symmetric และ Eckart-Young theorem (best low-rank approximation)
