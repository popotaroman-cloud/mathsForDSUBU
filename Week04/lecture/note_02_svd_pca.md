# Note 2: Singular Value Decomposition (SVD) และ PCA
> Week 4 | CLO1 | Strang Reference: Ch.7 (SVD) + Ch.7.3 (PCA)

---

## บทนำ

บันทึกนี้คือจุดสูงสุดของ Linear Algebra สำหรับ Data Science: **SVD** และ **PCA** SVD เป็น Factorization ที่ทรงพลังที่สุดใน Linear Algebra เพราะใช้ได้กับ Matrix ทุกรูปแบบ (ไม่ต้องเป็น square หรือ symmetric) และ PCA คือการประยุกต์ SVD เพื่อลดมิติของข้อมูลโดยรักษา "ข้อมูลสำคัญ" ไว้ให้มากที่สุด เป้าหมายคือให้นักศึกษา Implement PCA ได้ทั้งจาก Covariance Eigendecomposition และ SVD, อ่าน Scree Plot ได้, และเลือกจำนวน Components ที่เหมาะสม ในชีวิตจริง SVD/PCA ใช้ใน Face Recognition, Image Compression, NLP (LSA), Genomics และ Recommender Systems — เราจะเห็นทั้งหมดนี้ในบันทึกเดียว

---

## Section 1: Singular Value Decomposition (SVD)  *(Strang 7.1–7.2)*

ในส่วนนี้เราจะเรียนรู้ว่า SVD คืออะไร และทำไมถึงเป็น "Swiss Army Knife" ของ Linear Algebra โดยทำให้มองเห็น geometry ของ Matrix transformation อย่างสมบูรณ์

### 1.1 SVD Theorem

**Theorem**: Matrix $\mathbf{A}$ ขนาด $m \times n$ ใด ๆ (ไม่จำเป็นต้อง square) สามารถ factorize ได้เป็น:

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

| Matrix | ขนาด | ความหมาย |
|--------|------|---------|
| $\mathbf{U}$ | $m \times m$ | Left Singular Vectors — orthonormal basis ของ $\mathbb{R}^m$ |
| $\boldsymbol{\Sigma}$ | $m \times n$ | Diagonal: Singular Values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $\mathbf{V}^T$ | $n \times n$ | Right Singular Vectors — orthonormal basis ของ $\mathbb{R}^n$ |

**Geometric Interpretation** ($\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$):
1. $\mathbf{V}^T$: หมุน/สะท้อน input space
2. $\boldsymbol{\Sigma}$: ยืด/หดตาม axes
3. $\mathbf{U}$: หมุน/สะท้อน output space

**ความสัมพันธ์กับ Eigenvalues**:
$$\sigma_i = \sqrt{\lambda_i(\mathbf{A}^T\mathbf{A})} \quad \text{(Singular Values = square root ของ eigenvalues ของ } \mathbf{A}^T\mathbf{A})$$

- Columns ของ $\mathbf{V}$ = Eigenvectors ของ $\mathbf{A}^T\mathbf{A}$
- Columns ของ $\mathbf{U}$ = Eigenvectors ของ $\mathbf{A}\mathbf{A}^T$

### 1.2 Thin (Reduced) SVD

สำหรับ $m \times n$ ที่มี rank $r$:

$$\mathbf{A} = \mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^T$$

โดย $\mathbf{U}_r$ ขนาด $m \times r$, $\boldsymbol{\Sigma}_r$ ขนาด $r \times r$, $\mathbf{V}_r^T$ ขนาด $r \times n$

**ตัวอย่าง** $2 \times 2$:
$$\mathbf{A} = \begin{pmatrix}3&0\\4&5\end{pmatrix}$$

$\mathbf{A}^T\mathbf{A} = \begin{pmatrix}25&20\\20&25\end{pmatrix}$

Eigenvalues ของ $\mathbf{A}^T\mathbf{A}$: $\lambda_1 = 45, \lambda_2 = 5$

Singular values: $\sigma_1 = \sqrt{45} = 3\sqrt{5} \approx 6.708$, $\sigma_2 = \sqrt{5} \approx 2.236$

### 1.3 Low-Rank Approximation (Eckart-Young Theorem)

**Theorem**: Rank-$k$ approximation ที่ดีที่สุดของ $\mathbf{A}$ (ใน Frobenius norm) คือ:

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i\mathbf{v}_i^T = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^T$$

**Error**: $\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2$

**Application — Image Compression**:

```python
# ─── Image Compression ด้วย SVD ────────────────────────────────────────
# วัตถุประสงค์: แสดง Eckart-Young theorem ด้วย low-rank approximation ของภาพ
import numpy as np
import matplotlib.pyplot as plt

# สร้าง synthetic grayscale image (pattern คล้ายใบหน้า)
# วัตถุประสงค์: ใช้ synthetic data เพื่อหลีกเลี่ยง dependency กับ dataset จริง
np.random.seed(42)
n = 64
X_img, Y_img = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
img = np.exp(-(X_img**2 + Y_img**2)/2) + 0.3*np.random.randn(n, n)
img = (img - img.min()) / (img.max() - img.min())  # normalize 0-1

# SVD ของ image matrix
# วัตถุประสงค์: แยก image เป็น rank-1 components เรียงตาม importance
U, s, Vt = np.linalg.svd(img)
print(f"Image size: {img.shape}, Total singular values: {len(s)}")
print(f"Top-5 singular values: {s[:5].round(2)}")

# Low-rank reconstruction: เก็บแค่ top-k components
# วัตถุประสงค์: ดู trade-off ระหว่าง compression ratio กับ quality
for k in [1, 5, 10, 20]:
    # Reconstruct ด้วยแค่ k components
    img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    # คำนวณ compression ratio และ error
    compression = k * (n + n + 1) / (n * n)
    error = np.linalg.norm(img - img_k, 'fro') / np.linalg.norm(img, 'fro')
    energy = (s[:k]**2).sum() / (s**2).sum()

    print(f"k={k:2d}: compression={compression*100:.1f}%, "
          f"energy={energy*100:.1f}%, relative_error={error:.4f}")
```

**DS Connection**: Rank-$k$ SVD คือพื้นฐานของ **Latent Semantic Analysis (LSA)** ใน NLP ซึ่งใช้หา "topics" แฝงในชุดเอกสาร ด้วยการ SVD ของ Term-Document Matrix

---

## Section 2: Principal Component Analysis (PCA)  *(Strang 7.3)*

ในส่วนนี้เราจะเรียน PCA อย่างละเอียด ทั้งสองวิธี (Covariance Eigendecomposition และ SVD) เพื่อให้สามารถ Implement ได้จาก scratch และเชื่อมกับทฤษฎีที่เรียนมา

### 2.1 เป้าหมายของ PCA

**ปัญหา**: Data Matrix $\mathbf{X}$ ขนาด $n \times p$ (n samples, p features) อาจมี $p = 100, 1000$ หรือมากกว่า แต่ข้อมูลจริงมักอยู่ใน **low-dimensional structure** — PCA ค้นหา "ทิศทางของ variance สูงสุด" เพื่อ represent ข้อมูลใน $k < p$ มิติ โดยสูญเสียข้อมูลน้อยที่สุด

### 2.2 PCA วิธีที่ 1: Covariance Eigendecomposition

**ขั้นตอน**:

**Step 1**: Center data
$$\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^T \quad \text{(ลบ mean ทุก column)}$$

**Step 2**: Covariance Matrix
$$\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} \quad \text{(}p \times p\text{ matrix)}$$

**Step 3**: Eigendecomposition
$$\mathbf{C} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T \quad (\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0)$$

**Step 4**: Principal Components = Eigenvectors
$$\text{PC}_i = \mathbf{q}_i \quad \text{(column } i \text{ ของ } \mathbf{Q})$$

**Step 5**: Project data
$$\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{Q}_k \quad (n \times k \text{ scores matrix, } k < p)$$

**Variance Explained**:
$$\text{Var explained by PC}_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j} \times 100\%$$

### 2.3 PCA วิธีที่ 2: SVD (แนะนำ)

**ขั้นตอน**:

**Step 1**: Center data → $\tilde{\mathbf{X}}$

**Step 2**: SVD ของ Centered Data
$$\tilde{\mathbf{X}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

**Relationship กับ Covariance**:
$$\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \frac{1}{n-1}\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T$$

- Principal Components = **Columns ของ V** (Right Singular Vectors)
- Singular Values $\sigma_i = \sqrt{(n-1)\lambda_i}$
- Scores = $\mathbf{U}\boldsymbol{\Sigma}$ (หรือ $\tilde{\mathbf{X}}\mathbf{V}_k$)

**ทำไม SVD ดีกว่า**: หลีกเลี่ยงการคำนวณ $\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$ โดยตรง ซึ่งอาจ ill-conditioned เมื่อ $p >> n$

### 2.4 ตัวอย่าง PCA บน Iris Dataset (step-by-step)

```python
# ─── PCA from Scratch บน Iris Dataset ────────────────────────────────
# วัตถุประสงค์: แสดง full pipeline ของ PCA ตั้งแต่ raw data จนถึง visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.astype(float)          # 150 × 4
y = iris.target                       # class labels (0, 1, 2)
feature_names = iris.feature_names
class_names = iris.target_names

# Step 1: Center data ────────────────────────────────────────────────
# วัตถุประสงค์: PCA วัด variance รอบ mean ดังนั้นต้อง center ก่อนเสมอ
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Step 2: SVD ───────────────────────────────────────────────────────
# วัตถุประสงค์: ใช้ SVD แทน Covariance eigendecomposition เพื่อ numerical stability
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
# full_matrices=False ให้ Thin SVD: U(150×4), s(4,), Vt(4×4)

# Step 3: Principal Components = rows ของ Vt ──────────────────────
# วัตถุประสงค์: แต่ละ row ของ Vt คือทิศทาง PC ใน original feature space
V = Vt.T  # columns ของ V = principal directions (4×4)
print("Principal Components (columns):\n", V.round(4))

# Step 4: Variance Explained ─────────────────────────────────────────
# วัตถุประสงค์: singular values² ∝ variance ใน PC direction
variance = s**2 / (len(X) - 1)       # eigenvalues ของ Covariance Matrix
var_ratio = variance / variance.sum()

print("\nVariance Explained:")
for i, (v, r) in enumerate(zip(variance, var_ratio)):
    print(f"  PC{i+1}: eigenvalue={v:.4f}, explained={r*100:.1f}%, "
          f"cumulative={np.cumsum(var_ratio)[i]*100:.1f}%")

# Step 5: Project data ────────────────────────────────────────────────
# วัตถุประสงค์: transform data จาก 4D → 2D โดยใช้ 2 PCs แรก
Z = X_centered @ V[:, :2]             # 150 × 2 scores
print(f"\nProjected data shape: {Z.shape}")

# Plot: Scree Plot ──────────────────────────────────────────────────
# วัตถุประสงค์: visualize variance explained เพื่อเลือกจำนวน PCs ที่เหมาะสม
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, 5), var_ratio * 100, color='steelblue')
axes[0].plot(range(1, 5), np.cumsum(var_ratio) * 100, 'r-o', label='Cumulative')
axes[0].axhline(y=95, color='orange', linestyle='--', label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('Scree Plot')
axes[0].legend()

# Plot: 2D PCA Scatter ─────────────────────────────────────────────
# วัตถุประสงค์: visualize separation ระหว่าง classes ใน PC space
colors = ['red', 'green', 'blue']
for class_idx, (color, name) in enumerate(zip(colors, class_names)):
    mask = y == class_idx
    axes[1].scatter(Z[mask, 0], Z[mask, 1], c=color, label=name, alpha=0.7)
axes[1].set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
axes[1].set_title('Iris PCA 2D Projection')
axes[1].legend()

plt.tight_layout()
plt.savefig("iris_pca.png", dpi=100, bbox_inches='tight')
print("\nPlot saved!")

# เปรียบเทียบกับ sklearn PCA ──────────────────────────────────────────
# วัตถุประสงค์: verify ว่า from-scratch ได้ผลเดียวกับ library
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Z_sklearn = pca.fit_transform(X)

# ผลลัพธ์อาจต่างเครื่องหมาย (sign ambiguity ของ eigenvectors)
print("\nvariance_ratio sklearn:", (pca.explained_variance_ratio_ * 100).round(2))
print("variance_ratio scratch:", (var_ratio[:2] * 100).round(2))
```

**Output ที่คาดหวัง**:
- PC1 อธิบาย ~72.9% ของ variance
- PC2 อธิบาย ~22.9% — รวม 2 ตัวแรกได้ ~95.8%
- ใน scatter plot เห็น Setosa แยกออกชัดเจน; Versicolor และ Virginica ซ้อนกันบางส่วน

---

## Section 3: การเลือกจำนวน Components และ Limitations

ในส่วนนี้เราจะเรียนรู้ **criteria** สำหรับเลือก $k$ และข้อจำกัดของ PCA เพื่อให้นำไปใช้ได้อย่างถูกต้องในงานจริง

### 3.1 วิธีเลือก k

| วิธี | เกณฑ์ | ใช้เมื่อ |
|------|-------|---------|
| Cumulative Variance | เก็บจนรวมได้ ≥ 95% (หรือ 99%) | งานทั่วไป |
| Elbow Method | ดู Scree Plot หาจุดหักข้อศอก | EDA, visualization |
| Kaiser Criterion | เก็บ PC ที่ $\lambda_i > 1$ | Factor Analysis |
| Domain Knowledge | กำหนดตาม business logic | Production systems |

### 3.2 PCA Limitations

1. **Linear Only**: PCA หา linear combinations — ถ้าข้อมูลมี non-linear structure ใช้ Kernel PCA หรือ t-SNE
2. **Interpretability Loss**: PC ใหม่เป็น linear combination ของ features ทั้งหมด — ตีความยาก
3. **Scale Sensitive**: ถ้า features มี scale ต่างกันมาก features ที่มี scale ใหญ่จะ dominate → ต้อง Standardize ก่อน
4. **Unsupervised**: PCA ไม่รู้ class labels — ทิศทาง variance สูงสุดอาจไม่ใช่ทิศทางที่แยก class ดีที่สุด (ใช้ LDA แทน)

---

## Case Study: Eigenfaces — Face Recognition ด้วย PCA

**Scenario**  
AT&T Laboratories ต้องการระบบ Face Recognition ที่ทำงานบน hardware ช้าในปี 1991 ปัญหา: ภาพแต่ละภาพมีขนาด 92×112 pixels = 10,304 dimensions — สูงเกินไปที่จะ classify โดยตรง

**Data**  
40 บุคคล × 10 รูป = 400 รูป ขนาด 92×112 pixels grayscale  
Data Matrix $\mathbf{X}$ ขนาด $400 \times 10304$

**Method**

```python
# ─── Eigenfaces Demo (Simplified) ──────────────────────────────────────
# วัตถุประสงค์: แสดงหลักการ Eigenfaces — PCA บน Face Dataset
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# โหลด Olivetti Faces (400 faces × 4096 pixels, 40 subjects)
faces = fetch_olivetti_faces()
X_faces = faces.data      # 400 × 4096
print(f"Original: {X_faces.shape[0]} faces, {X_faces.shape[1]} dimensions")

# PCA ลดเหลือ 50 components
# วัตถุประสงค์: 50 Eigenfaces อธิบาย variance ส่วนใหญ่
pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X_faces)

var_explained = pca.explained_variance_ratio_.sum()
compression = X_reduced.size / X_faces.size

print(f"Reduced: {X_reduced.shape[0]} faces, {X_reduced.shape[1]} PCs")
print(f"Variance explained: {var_explained*100:.1f}%")
print(f"Compression ratio: {compression*100:.1f}% of original size")

# Reconstruct faces
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = np.linalg.norm(X_faces - X_reconstructed) / np.linalg.norm(X_faces)
print(f"Reconstruction error: {reconstruction_error*100:.2f}%")
```

**Result**  
50 Eigenfaces อธิบาย ~90% ของ variance ด้วยข้อมูลเพียง 1.2% ของขนาดต้นฉบับ (50/4096)

**Insight**  
- "Eigenfaces" คือ Principal Directions ของ Face Space — แต่ละ Eigenface เป็น Weighted Average ของ "facial features" ต่าง ๆ
- Classification ทำใน 50D ด้วย nearest neighbor แทนที่ 4096D — เร็วกว่า 80× โดยแทบไม่สูญเสีย accuracy
- วิธีนี้เป็นต้นแบบของ Deep Face Recognition systems ทั้งหมดในปัจจุบัน

---

## สรุป (Summary)

| แนวคิด | สูตร | Python |
|--------|------|--------|
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | `np.linalg.svd(A)` |
| Singular values | $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^T\mathbf{A})}$ | `np.linalg.svd(A)[1]` |
| Low-rank approx | $\mathbf{A}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^T$ | slice U, s, Vt |
| PCA Step 1 | Center: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$ | `X - X.mean(0)` |
| PCA Step 2 | SVD: $\tilde{\mathbf{X}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | `np.linalg.svd(X_c)` |
| PCA Components | Columns ของ $\mathbf{V}$ | `Vt.T` |
| Variance Explained | $\sigma_i^2 / \sum\sigma_j^2$ | `s**2 / s**2.sum()` |
| Project data | $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}_k$ | `X_c @ V[:, :k]` |
| sklearn | `PCA(n_components=k).fit_transform(X)` | `sklearn.decomposition.PCA` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 4 (note 1)**: Eigendecomposition ของ Covariance Matrix คือ PCA Method 1; SVD คือ generalization ของ Eigendecomposition
- ← **Week 3**: Orthogonality ของ U, V ใน SVD เชื่อมกับ Gram-Schmidt และ QR; Rank ของ $\mathbf{A}$ = จำนวน non-zero singular values
- → **Week 5**: เริ่ม ISLP — PCA เป็น preprocessing step ที่ใช้บ่อยก่อน Statistical Learning models
- → **Week 9**: Principal Components Regression (PCR) ใช้ PCA + Regression เป็น pipeline เดียว
- → **Week 12**: LDA (Linear Discriminant Analysis) คือ "supervised PCA" ที่รู้ class labels
