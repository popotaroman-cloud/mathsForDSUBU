# Slide Deck: Eigenvalues & Eigenvectors
> Week 4 | CLO1 | Strang Ch.6.1 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 4**  
Eigenvalues & Eigenvectors: เมื่อ Matrix ยืดข้อมูลโดยไม่เปลี่ยนทิศทาง  
CLO1: นำความรู้ Linear Algebra ประยุกต์ใช้กับ Data Science ได้

---

## Slide 2 — Week Overview
**Key Message**: Eigenvalue/Eigenvector เป็น "DNA" ของ matrix — บอกว่า transformation ทำอะไรกับข้อมูล

สัปดาห์นี้เราจะเรียนรู้ว่า Eigenvalues และ Eigenvectors คืออะไร และทำไมถึงสำคัญที่สุดใน Linear Algebra สำหรับ Data Science Eigenvector คือทิศทางพิเศษที่ matrix transformation ไม่เปลี่ยน เพียงแค่ยืดหรือหด และ Eigenvalue คือปัจจัยที่ยืด/หดนั้น แนวคิดนี้ใช้ใน PCA (หาทิศทางที่ข้อมูลกระจายมากที่สุด), Google PageRank (หาหน้าเว็บที่สำคัญที่สุด), differential equations (วิเคราะห์ stability), quantum mechanics, และ vibration analysis เป้าหมายของสัปดาห์นี้คือให้นักศึกษาคำนวณ eigenvalues/eigenvectors ได้ ทั้งด้วยมือและ NumPy อธิบายความหมายทางเรขาคณิตได้ และเชื่อมโยงกับ PCA ที่จะเรียนในสัปดาห์นี้ได้

**สิ่งที่จะเรียนรู้:**
- Eigenvalue equation: **Ax = λx**
- Characteristic polynomial: det(A - λI) = 0
- Trace = Σλᵢ, Det = Πλᵢ
- Geometric vs algebraic multiplicity
- [FIGURE: ภาพแสดง eigenvector ก่อนและหลัง transformation — ทิศทางไม่เปลี่ยน แต่ขนาดเปลี่ยน]

---

## Slide 3 — Eigenvalue Equation
**Key Message**: Ax = λx แปลว่า matrix A transform x โดยยืด/หดด้วยปัจจัย λ เท่านั้น ไม่เปลี่ยนทิศทาง

**นิยาม:**
```
Ax = λx
```
- **x** ≠ 0: eigenvector (ทิศทางที่ไม่เปลี่ยนเมื่อ A transform)
- **λ**: eigenvalue (ปัจจัยยืด/หด — อาจเป็นลบหรือ 0)
- λ > 1: ยืด, 0 < λ < 1: หด, λ < 0: พลิกทิศแล้วยืด/หด, λ = 0: ทำให้เป็น 0

**ตัวอย่าง Matrix 2×2:**
```
A = [[3, 1],    eigenvalues: λ₁ = 3, λ₂ = 2
     [0, 2]]    eigenvectors: x₁ = [1,0], x₂ = [1,1]/√2
```

- [FIGURE: แสดงใน 2D: vector ก่อน/หลัง Ax สำหรับทั้ง eigenvector และ non-eigenvector]
- Verify: A @ [1,0] = [3,0] = 3 × [1,0] ✓

---

## Slide 4 — Finding Eigenvalues: Characteristic Equation
**Key Message**: หา eigenvalues โดยแก้ det(A - λI) = 0 — polynomial ที่มีรากเป็น eigenvalues

**ขั้นตอน:**
```
Ax = λx
Ax - λx = 0
(A - λI)x = 0    ← x ≠ 0 ดังนั้น A-λI ต้องเป็น singular
det(A - λI) = 0  ← characteristic equation
```

**ตัวอย่าง:**
```
A = [[3, 1],
     [0, 2]]

det([[3-λ, 1 ],
     [0,   2-λ]]) = (3-λ)(2-λ) - (0)(1) = 0

λ² - 5λ + 6 = 0
(λ-3)(λ-2) = 0
→ λ₁ = 3,  λ₂ = 2
```

- Matrix n×n มี n eigenvalues (นับซ้ำตาม multiplicity)
- Trace(A) = λ₁ + λ₂ + … + λₙ = 3 + 2 = 5 ✓
- Det(A) = λ₁ × λ₂ × … × λₙ = 3 × 2 = 6 ✓

---

## Slide 5 — Finding Eigenvectors
**Key Message**: เมื่อได้ eigenvalue แต่ละค่าแล้ว หา eigenvector โดย solve (A - λI)x = 0 หา null space

**ขั้นตอนหา eigenvector:**
1. เลือก eigenvalue λ
2. สร้าง matrix (A - λI)
3. Solve (A - λI)x = 0 → หา null space
4. Normalize: ‖x‖ = 1

**ตัวอย่าง (ต่อ):**
```
สำหรับ λ₁ = 3:
A - 3I = [[0, 1], [0, -1]]
Row reduce: [[0, 1], [0, 0]] → x₂ = 0, x₁ = free
x₁ = [1, 0]

สำหรับ λ₂ = 2:
A - 2I = [[1, 1], [0, 0]]
Row reduce: [[1, 1], [0, 0]] → x₁ = -x₂, x₂ = free
x₂ = [1, -1]/√2
```

- **Python:** `vals, vecs = np.linalg.eig(A)` — columns ของ vecs คือ eigenvectors
- [EXAMPLE: แสดง Python code verify Ax = λx]

---

## Slide 6 — Geometric Interpretation
**Key Message**: Eigenvalue บอก "ความสำคัญ" ของแต่ละทิศทาง ทิศที่มี |λ| ใหญ่คือทิศที่ข้อมูลกระจายตัวมากที่สุด

**การมองแบบ Geometric:**
- Linear transformation A = stretch + rotate + stretch...
- Eigenvectors = แกนหลักของ transformation (invariant directions)
- Eigenvalues = scale factor ตามแต่ละแกนหลัก

**ตัวอย่างใน 2D:**
- [FIGURE: unit circle → transformed ellipse, eigenvectors แสดงเป็นแกน semi-major/semi-minor]
- Ellipse axes = eigenvectors ของ A
- Axis lengths = |eigenvalues| × 1

**DS Connection:**
- PCA: eigenvectors ของ covariance matrix = principal components
- ยิ่ง |λ| ใหญ่ → axis นั้น "สำคัญ" (ข้อมูลกระจายมาก) มากกว่า

---

## Slide 7 — Multiplicity & Special Cases
**Key Message**: Eigenvalue ซ้ำ (repeated) หรือ complex eigenvalue บอกว่า transformation มีพฤติกรรมพิเศษ

**Algebraic vs Geometric Multiplicity:**
- Algebraic multiplicity (AM): จำนวนครั้งที่ λ เป็นรากของ characteristic polynomial
- Geometric multiplicity (GM): dimension ของ null space ของ (A - λI) = จำนวน independent eigenvectors
- GM ≤ AM เสมอ, ถ้า GM = AM สำหรับทุก λ → matrix diagonalizable

**Cases สำคัญ:**
```
Symmetric matrix A = Aᵀ:
- Eigenvalues เป็น real เสมอ
- Eigenvectors orthogonal (ตั้งฉากกัน)
- ใช้ np.linalg.eigh แทน eig (เร็วกว่า, stable กว่า)

Covariance matrix Σ = (1/n) XᵀX:
- Symmetric positive semi-definite
- λᵢ ≥ 0 เสมอ
- λᵢ คือ variance ตาม PCᵢ
```

---

## Slide 8 — Python: np.linalg.eig vs np.linalg.eigh
**Key Message**: ใช้ `eigh` สำหรับ symmetric matrix — เร็วกว่า, ให้ eigenvalues จริง, sorted

**`np.linalg.eig(A)` — General Matrix:**
```python
vals, vecs = np.linalg.eig(A)
# vals: eigenvalues (อาจ complex)
# vecs: eigenvector matrix (columns = eigenvectors)
# ไม่ sorted, eigenvectors อาจไม่ orthogonal
```

**`np.linalg.eigh(A)` — Symmetric Matrix:**
```python
vals, vecs = np.linalg.eigh(A)
# vals: real eigenvalues sorted ASCENDING
# vecs: orthonormal eigenvectors
# เหมาะกับ covariance matrix ใน PCA
# หมายเหตุ: คืน ascending → ต้อง reverse เพื่อ PCA
```

**Rule of thumb:**
- Symmetric matrix (covariance, correlation) → **eigh**
- General matrix → **eig**
- Data matrix X (non-square) → **SVD** (ดู Slide 3.x)

---

## Slide 9 — Case Study: Google PageRank
**Key Message**: Eigenvalue ใหญ่สุดของ transition matrix คือ PageRank score — ลำดับความสำคัญของ web pages

**Scenario:**
Google ต้องการจัดอันดับ web pages ตาม "ความน่าเชื่อถือ" ซึ่งวัดจาก links ที่ได้รับจากหน้าอื่น

**Method — Power Iteration:**
```
P = transition matrix (สมมาตรจาก adjacency matrix)
r(t+1) = P @ r(t)  ← iterate จนลู่เข้า steady state
r* = eigenvector สำหรับ λ = 1 ของ P
```

**Result:**
- r* = PageRank vector — pages ที่มี r*(i) สูงขึ้นก่อน
- λ = 1 เสมอสำหรับ stochastic matrix (Perron-Frobenius theorem)
- Convergence rate = λ₂/λ₁ (อัตราส่วน eigenvalue สองตัวแรก)

**Insight:**
- PageRank = dominant eigenvector → เชื่อมกับ power method ใน numerical linear algebra
- ข้อมูลขนาดใหญ่: ใช้ sparse matrix + power iteration ไม่ใช่ full eigendecomposition

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Eigenvalue equation**: Ax = λx — matrix transform x โดยเปลี่ยนแค่ขนาด ไม่เปลี่ยนทิศทาง
- **Finding eigenvalues**: det(A - λI) = 0 → characteristic polynomial → roots = λ
- **Finding eigenvectors**: solve (A - λI)x = 0 สำหรับแต่ละ λ
- **Trace = Σλ, Det = Πλ** — eigenvalues encode ข้อมูลสำคัญของ matrix
- **eig vs eigh**: ใช้ eigh สำหรับ symmetric matrix (covariance) เสมอ

| เครื่องมือ | ใช้เมื่อ | Python |
|-----------|---------|--------|
| np.linalg.eig | General matrix | `vals, vecs = np.linalg.eig(A)` |
| np.linalg.eigh | Symmetric matrix | `vals, vecs = np.linalg.eigh(A)` |
| det(A-λI)=0 | ด้วยมือ (small matrix) | `np.linalg.det(A - lam*np.eye(n))` |

**สัปดาห์ถัดไป (Slide Deck 2):** Diagonalization A = PDP⁻¹ และ Symmetric Matrices — เชื่อมสู่ Spectral Theorem และ Covariance matrix ใน PCA
