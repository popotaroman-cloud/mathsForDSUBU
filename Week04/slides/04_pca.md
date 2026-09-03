# Slide Deck: Principal Component Analysis (PCA)
> Week 4 | CLO1 | Strang Ch.7.3 | 11 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 4**  
PCA: หาทิศทางที่ข้อมูลกระจายมากที่สุดเพื่อลด Dimension อย่างฉลาด  
CLO1: ทำ PCA from scratch และด้วย sklearn บน dataset จริงได้

---

## Slide 2 — Week Overview (Deck 4)
**Key Message**: PCA แก้ปัญหา "ข้อมูลมี dimension สูงเกินไป" โดยหาทิศทางที่สำคัญที่สุดและทิ้งส่วนที่เป็น noise

PCA (Principal Component Analysis) คือ algorithm ที่ใช้ SVD และ eigendecomposition เพื่อลด dimension ของข้อมูลโดยสูญเสีย information น้อยที่สุด แนวคิดหลักคือการหาทิศทางที่ข้อมูลกระจายตัวมากที่สุด (high variance directions) และ project ข้อมูลทั้งหมดไปยังทิศทางเหล่านั้น ทิศทางแรก (PC1) อธิบาย variance ได้มากที่สุด, PC2 อธิบาย variance ที่เหลือมากที่สุด (ต้องตั้งฉากกับ PC1) เป็นต้น PCA ถูกใช้ใน image compression (Eigenfaces), genomics (reduce 20,000 genes → 50 PCs), NLP, finance (factor model), และเป็น preprocessing ก่อนสร้าง ML model เป้าหมายของ deck นี้คือให้นักศึกษาทำ PCA ได้ทั้ง 2 method (eigendecomposition + SVD) อ่าน Scree Plot ได้ และ interpret principal components ได้

**สิ่งที่จะเรียนรู้:**
- PCA Goal: maximize variance along projected directions
- Method 1: Covariance eigendecomposition
- Method 2: SVD (preferred)
- Scree Plot และการเลือก k
- Interpretation: loadings, variance explained
- Limitations: linear only, interpretability

---

## Slide 3 — Why PCA? Dimensionality Problem
**Key Message**: High-dimensional data มีปัญหา "curse of dimensionality" — PCA ลด dimension โดยรักษา information ที่สำคัญไว้

**ปัญหาของ High Dimension:**
```
- 100 features: model ซับซ้อน, overfitting สูง
- 1000 features: เวลาสอน slow, visualization ไม่ได้
- Many features: features อาจ correlated → ซ้ำซ้อน
```

**PCA แก้ปัญหาอย่างไร:**
```
100 features  →  PCA  →  10 PCs
(ถ้า 10 PCs explain 95% variance)
```
- ลด overfitting
- Visualization ใน 2D/3D
- ตัดทิ้ง features ที่เป็น noise
- **ข้อสำคัญ**: PCA ไม่ตัดทิ้ง features — มัน combine features ให้เป็น PCs ใหม่

**[FIGURE: 2D scatter ของ data ที่มี correlation สูง → PC1 (main direction of variance), PC2 (orthogonal)]**

---

## Slide 4 — PCA Algorithm: Method 1 (Covariance Eigendecomposition)
**Key Message**: 5 ขั้นตอนของ PCA from scratch — center → covariance → eigendecompose → sort → project

**ขั้นตอน:**
```
1. Center data
   X_centered = X - mean(X, axis=0)
   [สำคัญ: ต้อง center ก่อนเสมอ]

2. Covariance matrix
   C = (1/(n-1)) × X_centered.T @ X_centered
   [C คือ (d×d) symmetric PSD matrix]

3. Eigendecompose C
   C = Q Λ Qᵀ  (ใช้ np.linalg.eigh)
   [eigenvalues = variance per direction]

4. Sort descending
   เรียง eigenvectors ตาม eigenvalue (มาก→น้อย)
   [PC1 = direction of max variance]

5. Project
   X_pca = X_centered @ Q_k
   [เลือก k columns แรกของ Q]
```

---

## Slide 5 — PCA Algorithm: Method 2 (SVD — Preferred)
**Key Message**: SVD ของ X_centered โดยตรง — เร็วกว่า, stable กว่า สำหรับ large dataset

**ทำไมใช้ SVD แทน Eigendecomposition:**
```
Eigendecomposition: ต้องคำนวณ XᵀX ก่อน → อาจ ill-conditioned
SVD: ทำงานกับ X โดยตรง → numerically stable กว่า
ผลเหมือนกัน: V จาก SVD = Q จาก eigendecomposition
```

**PCA via SVD:**
```
X_centered = U Σ Vᵀ  (SVD)

Principal Components = columns of V (= right singular vectors)
PCA Scores = U Σ = X_centered @ V  (= X_pca)
Eigenvalues = σᵢ² / (n-1)
```

**sklearn ใช้ SVD:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
X_pca = pca.fit_transform(X)
# pca.components_  = V.T (principal axes)
# pca.explained_variance_ratio_  = eigenvalues / total
```

---

## Slide 6 — Explained Variance & Scree Plot
**Key Message**: Scree Plot แสดง "elbow point" — จุดที่เพิ่ม PC อีกก็ไม่ได้ประโยชน์มากแล้ว

**Explained Variance Ratio:**
```
Var_i = λᵢ / Σλⱼ
Cumulative_k = Σᵢ₌₁ᵏ Var_i
```

**Scree Plot:**
- x-axis: PC number (1, 2, 3, ...)
- y-axis bar: individual explained variance %
- y-axis line: cumulative explained variance %
- Elbow = จุดที่ curve เริ่ม "flat"

**[FIGURE: Scree Plot ของ Iris data — bar chart + cumulative line + 95% threshold dashed]**

**Convention ในการเลือก k:**
- **95% rule**: เลือก k ที่ cumulative ≥ 95%
- **Elbow rule**: เลือก k ที่ "kink" ใน scree plot
- **Kaiser rule**: เลือก PC ที่มี eigenvalue > 1 (สำหรับ standardized data)

---

## Slide 7 — Interpreting Principal Components
**Key Message**: PC คือ linear combination ของ original features — loadings (weights) บอกว่า feature ไหนสำคัญ

**Loadings:**
```
PC1 = w₁₁×feature₁ + w₁₂×feature₂ + ... + w₁ₐ×featureₐ
```
- Loading = ค่าในแต่ละ column ของ eigenvector matrix Q
- |loading| ใหญ่ → feature นั้น "contribute" มากต่อ PC นั้น
- Python: `pca.components_` = loading matrix (k × d)

**ตัวอย่าง Iris:**
```
pca.components_[0] ≈ [0.52, -0.27, 0.58, 0.56]
PC1 = 0.52×sepal_len - 0.27×sepal_wid + 0.58×petal_len + 0.56×petal_wid
→ PC1 ≈ "overall flower size" (petal features dominate)

pca.components_[1] ≈ [0.37, 0.92, 0.02, 0.07]
PC2 = 0.37×sepal_len + 0.92×sepal_wid + ...
→ PC2 ≈ "sepal shape" (sepal width dominates)
```

---

## Slide 8 — PCA for Visualization
**Key Message**: 2D PCA scatter plot แสดงให้เห็น cluster structure ที่ hidden อยู่ใน high-dimensional data

**วิธีใช้ PCA สำหรับ Visualization:**
```python
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='Set1')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
```

**[FIGURE: 2D scatter ของ Iris data ใน PCA space — Setosa แยกชัด, Versicolor/Virginica overlap นิดหน่อย]**

**สิ่งที่สังเกตได้:**
- Clusters ใน PCA space = groups จริงในข้อมูล
- Overlap = classes ที่ยากต่อการ classify
- Outliers = จุดที่อยู่ห่างจาก cluster

---

## Slide 9 — PCA Applications & Limitations
**Key Message**: PCA ทรงพลังแต่มีข้อจำกัด — ใช้ได้กับ linear structure เท่านั้น และ interpretability ลดลง

**Applications:**
| Domain | Dataset | ใช้ PCA ทำอะไร |
|--------|---------|----------------|
| Image | Eigenfaces (AT&T) | 64×64 → 150 PCs (face recognition) |
| Genomics | RNA-seq | 20,000 genes → 50 PCs |
| Finance | Stock returns | 500 stocks → market factors |
| NLP | TF-IDF matrix | 50,000 words → topics (LSA) |
| ML Preprocessing | Any dataset | ลด overfitting, เพิ่มความเร็ว |

**Limitations:**
- **Linear only**: ไม่จับ nonlinear structure (ใช้ t-SNE, UMAP แทน)
- **Interpretability loss**: PC ไม่ใช่ original features
- **Scale sensitive**: ต้อง standardize ก่อนถ้า features มี scale ต่างกัน
- **Variance ≠ Importance**: feature ที่มี variance สูงอาจไม่ใช่ที่ predictive มากที่สุด

---

## Slide 10 — Case Study: Eigenfaces (Face Recognition)
**Key Message**: PCA บน face images สร้าง "eigenfaces" ซึ่งเป็นพื้นฐาน face recognition ก่อน deep learning

**Scenario:**  
Turk & Pentland (1991) สร้าง face recognition system โดยใช้ PCA บน AT&T face dataset (400 ภาพ, 40 คน, 10 ภาพ/คน)

**Data:**  
- 400 ภาพขนาด 64×64 grayscale = 400 × 4096 matrix
- Goal: compress + recognize ใน low-dimensional space

**Method:**  
1. Center: ลบ "mean face" ออกจากทุกภาพ
2. SVD: X = UΣVᵀ → เก็บ top-150 PCs
3. Eigenfaces = PC1, PC2, ... (visualize เป็นภาพ)
4. Recognize: project ภาพใหม่ → nearest neighbor ใน PC space

**Result:**  
150 PCs อธิบาย variance ได้ 95% (จาก 4096 dimensions)  
Accuracy ≈ 96% ด้วย 150-NN ใน PC space

**Insight:**  
- ใบหน้าทุกใบหน้า ≈ weighted sum ของ eigenfaces
- Eigenfaces capture: lighting, expression, identity, glasses, ...
- นี่คือ PCA ทำงานจริง ก่อนจะมี deep learning

---

## Slide 11 — Summary & Preview
**สิ่งที่เรียนรู้ใน Week 4**

| เรื่อง | สูตรหลัก | Python |
|--------|----------|--------|
| Eigenvalues | Ax = λx, det(A-λI)=0 | `np.linalg.eig/eigh` |
| Diagonalization | A = PDP⁻¹ | ใช้ eig แล้วสร้าง P, D |
| Spectral Theorem | A = QΛQᵀ (symmetric) | `np.linalg.eigh` |
| SVD | A = UΣVᵀ (any shape) | `np.linalg.svd` |
| PCA | X_centered = UΣVᵀ → project | `sklearn.decomposition.PCA` |
| Explained variance | λᵢ/Σλ, cumsum | `pca.explained_variance_ratio_` |

**PCA Pipeline ใน 5 บรรทัด:**
```python
pca = PCA(n_components=k)        # สร้าง PCA object
X_pca = pca.fit_transform(X)     # center + project
ratio = pca.explained_variance_ratio_  # variance per PC
components = pca.components_     # loading matrix (k × d)
```

**สัปดาห์ถัดไป (Week 5):** Statistical Learning Framework — จาก Linear Algebra สู่ Machine Learning: f(x), prediction, model assessment, supervised vs unsupervised (ISLP Ch.1–2)
