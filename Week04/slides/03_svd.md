# Slide Deck: Singular Value Decomposition (SVD)
> Week 4 | CLO1 | Strang Ch.7.1–7.2, 7.4 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 4**  
SVD: A = UΣVᵀ — The Most Important Factorization in Data Science  
CLO1: คำนวณ SVD และประยุกต์ใช้กับ low-rank approximation ได้

---

## Slide 2 — Week Overview (Deck 3)
**Key Message**: SVD ทำงานกับ matrix ทุกรูปร่าง ทำให้เป็น "universal tool" ที่ใช้ได้กับ data matrix จริงเสมอ — Eigendecomposition ทำได้เฉพาะ square matrix

SVD (Singular Value Decomposition) คือการแยก matrix A ใดๆ ออกเป็น A = UΣVᵀ ซึ่ง U และ V เป็น orthogonal matrices และ Σ เป็น diagonal matrix ของ singular values ที่เรียงจากมากไปน้อย SVD แตกต่างจาก eigendecomposition ตรงที่ทำงานกับ matrix ทุกรูปร่าง (m×n) ไม่ต้องเป็น square สิ่งนี้ทำให้ SVD เป็น tool ที่ใช้กับ data matrix จริงได้เสมอ เพราะ data มักมีรูปร่าง (n_samples × n_features) Eckart-Young theorem บอกว่า rank-k SVD approximation คือ best rank-k approximation ที่เป็นไปได้ในแง่ Frobenius norm ซึ่งเป็น theoretical foundation ของ image compression, Eigenfaces, Netflix Prize, และ PCA เป้าหมายคือให้นักศึกษาคำนวณ SVD ด้วย NumPy ได้ reconstruct matrix ได้ และเข้าใจ low-rank approximation

**สิ่งที่จะเรียนรู้:**
- SVD: A = UΣVᵀ — ความหมายของ U, Σ, V
- Geometric interpretation: rotate → stretch → rotate
- Full vs reduced SVD
- Low-rank approximation: A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ
- Eckart-Young theorem

---

## Slide 3 — SVD: A = UΣVᵀ
**Key Message**: SVD แยก transformation A เป็น 3 ขั้นตอนที่มีความหมาย: หมุน → ยืด → หมุน

**SVD Decomposition:**
```
A  =  U  ×  Σ  ×  Vᵀ
(m×n)  (m×m)  (m×n)  (n×n)
```

**ความหมายของแต่ละ component:**
- **U** (m×m): left singular vectors — orthonormal columns
  - u₁, u₂, ... เป็น eigenvectors ของ **AAᵀ**
- **Σ** (m×n): diagonal matrix ของ singular values **σ₁ ≥ σ₂ ≥ ... ≥ 0**
  - σᵢ = √λᵢ ของ AᵀA (และ AAᵀ)
- **Vᵀ** (n×n): right singular vectors transposed — orthonormal rows
  - v₁, v₂, ... เป็น eigenvectors ของ **AᵀA**

**[FIGURE: diagram แสดง A = U × Σ × Vᵀ พร้อมขนาดของแต่ละ matrix]**

---

## Slide 4 — Geometric Interpretation
**Key Message**: SVD แสดงว่า linear transformation ทุกอย่างสามารถมองเป็น rotate → stretch → rotate เสมอ

**3 ขั้นตอนของ transformation Ax:**
```
Ax = (U Σ Vᵀ) x
   = U (Σ (Vᵀx))

ขั้น 1: Vᵀx — หมุน input vector x ไปยัง coordinate ของ right SV
ขั้น 2: Σ(...)— ยืด/หดตาม singular values (scale แต่ละ dimension)
ขั้น 3: U(...)— หมุน output ไปยัง coordinate สุดท้าย
```

**[FIGURE: แสดง unit circle → สี่เหลี่ยม → ellipse — 3 ขั้นตอนการ transform ด้วย SVD]**

**ตัวอย่างตัวเลข:**
- σ₁ ใหญ่มาก: ทิศทาง u₁ ถูกยืดมาก (สำคัญ, information มาก)
- σᵣ ≈ 0: ทิศทาง uᵣ ถูกยุบเกือบเป็น 0 (noise, สามารถตัดทิ้งได้)

---

## Slide 5 — Full vs Reduced SVD
**Key Message**: ใช้ Reduced SVD ใน practice — ประหยัด memory โดยไม่สูญเสียข้อมูล

**Full SVD:**
```
A (m×n) = U (m×m) × Σ (m×n) × Vᵀ (n×n)
```
- Σ มีแค่ min(m,n) non-zero diagonal elements
- ส่วนที่เหลือของ U และ Vᵀ เป็น "wasted space"

**Reduced (Thin) SVD:**
```
r = rank(A) ≤ min(m,n)
A ≈ U_r (m×r) × Σ_r (r×r) × Vᵀ_r (r×n)
```

**Python:**
```python
# Full SVD: U(m×m), s(min(m,n),), Vt(n×n)
U, s, Vt = np.linalg.svd(A, full_matrices=True)

# Reduced SVD: U(m×r), s(r,), Vt(r×n)
U_r, s_r, Vt_r = np.linalg.svd(A, full_matrices=False)

# Reconstruct (full)
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, s)
A_back = U @ Sigma @ Vt
```

---

## Slide 6 — Singular Values & Rank
**Key Message**: Singular values บอก "ขนาด" ของ matrix ในแต่ละ dimension — เรียงจากมากไปน้อย ช่วยบอก effective rank

**ความสัมพันธ์ Singular Values กับ Eigenvalues:**
```
σᵢ = √λᵢ(AᵀA) = √λᵢ(AAᵀ)

ตัวอย่าง: A = [[1, 2, 3], ...]
AᵀA eigenvalues: λ₁=175.5, λ₂=0.5, λ₃≈0
Singular values: σ₁=13.25, σ₂=0.70, σ₃≈0
```

**Numerical Rank:**
- Rank ทาง theoretical: จำนวน non-zero singular values
- Rank ทาง numerical: จำนวน σᵢ ที่ > tolerance (เช่น σ₁ × 1e-10)
- Matrix "ใกล้ๆ" low-rank: σ₁ >> σ₂ >> ... >> σᵣ → compression ได้ดี

**[FIGURE: bar chart ของ singular values แสดง "elbow" — เหมือน Scree Plot]**

---

## Slide 7 — Low-rank Approximation & Eckart-Young
**Key Message**: A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ คือ best rank-k approximation ใน Frobenius norm (guaranteed optimal)

**Low-rank Approximation:**
```
A ≈ A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ
```
- เก็บเฉพาะ k singular triplets ที่ใหญ่สุด
- k=1: เก็บ "direction ที่สำคัญที่สุด" อย่างเดียว

**Eckart-Young Theorem:**
```
||A - A_k||_F = √(σ²_{k+1} + σ²_{k+2} + ... + σ²ᵣ)

ไม่มี rank-k matrix ไหนที่ approximate A ได้ดีกว่า A_k
(ใน Frobenius norm หรือ spectral norm)
```

**% Variance Explained:**
```
Var(k) = Σᵢ₌₁ᵏ σᵢ² / Σᵢ σᵢ² × 100%
```

---

## Slide 8 — Applications of SVD
**Key Message**: SVD อยู่เบื้องหลัง algorithm สำคัญที่สุดหลายตัวใน ML/DS — ทุกคนใช้ SVD อยู่แล้วโดยไม่รู้ตัว

**Image Compression:**
```
Image (m×n) → SVD → เก็บแค่ k components
Storage: m×n → k×(1+m+n) values
เช่น 100×100 → rank-10 → 100² = 10,000 → 10×201 = 2,010 values (5x)
```

**Netflix Prize / Recommender System:**
```
User-item matrix R (sparse) → SVD → R ≈ Uᵣ Σᵣ Vᵀᵣ
Predict missing ratings: R̂(user_i, item_j) = uᵢ · (Σ vⱼ)
```

**PCA (Slide Deck 4):**
```
X_centered = U Σ Vᵀ
Principal Components = columns of V
PCA scores = U Σ
```

**Text Analysis (LSA):**
```
Term-document matrix → SVD → semantic space
Documents ใกล้กันใน SVD space = เรื่องเดียวกัน
```

---

## Slide 9 — Python: np.linalg.svd คู่มือ
**Key Message**: `np.linalg.svd` คืน Vᵀ ไม่ใช่ V — ต้องระวังเวลา reconstruct

**ตัวอย่าง Complete:**
```python
import numpy as np

A = np.random.randn(50, 30)  # data matrix: 50 samples × 30 features

# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)
print(f'U: {U.shape}, s: {s.shape}, Vt: {Vt.shape}')
# U: (50,50), s: (30,), Vt: (30,30)

# Low-rank k=5 approximation
k = 5
A_k = sum(s[i] * np.outer(U[:, i], Vt[i, :]) for i in range(k))
# หรือ: U_r[:, :k] @ np.diag(s[:k]) @ Vt_r[:k, :]

# % variance explained
var_k = (s[:k]**2).sum() / (s**2).sum() * 100
print(f'k={k}: {var_k:.1f}% variance explained')

# Frobenius error
error = np.linalg.norm(A - A_k, 'fro')
print(f'||A - A_k||_F = {error:.3f}')
```

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **SVD**: A = UΣVᵀ — works for ANY matrix (m×n)
- **Geometric**: Vᵀ rotate → Σ stretch → U rotate
- **Full vs Reduced**: ใช้ `full_matrices=False` สำหรับ large data
- **Singular values**: σᵢ = √λᵢ(AᵀA), เรียง σ₁ ≥ σ₂ ≥ ... ≥ 0
- **Eckart-Young**: A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ คือ **best** rank-k approximation

| สิ่งที่ต้องการ | วิธี | Python |
|-------------|-----|--------|
| Full SVD | A = UΣVᵀ | `np.linalg.svd(A)` |
| Reduced SVD | A = UᵣΣᵣVᵀᵣ | `np.linalg.svd(A, full_matrices=False)` |
| Rank-k approx | Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ | ลูป หรือ slice + matmul |
| % Var explained | sum(s[:k]²)/sum(s²) | `(s[:k]**2).sum()/(s**2).sum()` |

**สัปดาห์ถัดไป (Slide Deck 4):** PCA — ใช้ SVD เพื่อหา Principal Components, Scree Plot, Dimensionality Reduction บน data จริง
