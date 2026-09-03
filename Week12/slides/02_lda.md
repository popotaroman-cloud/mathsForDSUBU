# Slide Deck 2: Linear Discriminant Analysis (LDA)
> Week 12 | 10 slides | CLO3

---

## Slide 1 — Title

**Linear Discriminant Analysis (LDA)**  
ISLP Ch.4.4.1–4.4.2 | Week 12 | CLO3

---

## Slide 2 — LDA: Core Idea

**Key Message**: LDA หา linear combination ของ features ที่แยก class ได้ดีที่สุด โดย assume Gaussian shared covariance

**Setup:**
- K classes, prior πₖ = Pr(Y=k)
- Within each class: X|Y=k ~ N(μₖ, **Σ**) — **Σ เดียวกันทุก class**
- Goal: classify x ไปยัง class k ที่ทำให้ Pr(Y=k|X=x) สูงสุด

**Discriminant Function δₖ(x):**
$$\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log(\pi_k)$$

**Classify**: assign x ไปยัง k ที่ δₖ(x) ใหญ่ที่สุด

**[FIGURE: 2D scatter, 2 class means μ₁, μ₂, shared Σ (ellipses same shape), decision boundary เส้นตรง ตั้งฉากกับ μ₁μ₂]**

---

## Slide 3 — LDA for p=1: Derivation

**Key Message**: เมื่อมีเพียง 1 feature LDA ให้ decision boundary ที่ midpoint ของ class means (เมื่อ π₁=π₂)

**p = 1 case:**
$$\delta_k(x) = \frac{x \mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$$

**Decision boundary (K=2):**  
Set δ₁(x) = δ₂(x) → solve for x:

$$x^* = \frac{\mu_1 + \mu_2}{2} - \frac{\sigma^2}{\mu_1 - \mu_2}\log\left(\frac{\pi_1}{\pi_2}\right)$$

**เมื่อ π₁ = π₂:** 
$$x^* = \frac{\mu_1 + \mu_2}{2}$$
→ boundary อยู่ที่ midpoint ของ class means

**[FIGURE: 1D number line, 2 Gaussian curves N(μ₁,σ²) และ N(μ₂,σ²), x* อยู่กลาง]**

---

## Slide 4 — LDA for p>1

**Key Message**: LDA ขยายสู่ multiple features ด้วย Σ (covariance matrix) แทน σ²

**Multivariate case:**
$$X|Y=k \sim \mathcal{N}(\mu_k, \Sigma)$$

**Decision boundary: linear ใน x**  
เพราะ term xᵀΣ⁻¹μₖ linear → boundary เป็น hyperplane

**Parameter Estimation จาก Training Data:**

| Parameter | Estimate | ความหมาย |
|-----------|---------|---------|
| μ̂ₖ | x̄ₖ (sample mean ของ class k) | mean vector |
| Σ̂ | pooled sample covariance | shared covariance |
| π̂ₖ | nₖ/n | prior |

**Connection to Week 4:** Σ เป็น covariance matrix → LDA ใช้ Σ⁻¹ ซึ่งต้องการ matrix inversion!

**[FIGURE: 2D scatter, 2 Gaussian ellipses (same shape different centers), linear boundary]**

---

## Slide 5 — Python: Fit LDA

**Key Message**: sklearn ทำ estimation Σ̂, μ̂ₖ, π̂ₖ อัตโนมัติ เพียง 2 บรรทัด

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Fit LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# ดู parameters ที่ estimate ได้
print('Class priors:', lda.priors_)      # π̂ₖ
print('Class means:', lda.means_)        # μ̂ₖ
print('Covariance:', lda.covariance_)    # Σ̂ (pooled)

# Predict
y_pred = lda.predict(X_test)
y_proba = lda.predict_proba(X_test)[:,1]  # posterior Pr(Y=1|X)

# LDA Projection (dimensionality reduction)
X_lda = lda.transform(X_train)  # ← project ลง K-1 dimensions
```

**Output จาก Default dataset:**
```
Class priors: [0.967 0.033]   ← 96.7% No Default, 3.3% Default
Class means:
  Class 0: [balance=803, income=34k, student=0.29]
  Class 1: [balance=1748, income=32k, student=0.27]
```

---

## Slide 6 — LDA Confusion Matrix + Threshold

**Key Message**: Default threshold = 0.5 มักไม่เหมาะกับ imbalanced data — ปรับ threshold ตาม cost

**LDA Confusion Matrix (Default dataset, threshold=0.5):**

|  | Pred: No Default | Pred: Default |
|--|-----------------|--------------|
| **True: No Default** | 9627 (TN) | 44 (FP) |
| **True: Default** | 252 (FN) | 77 (TP) |

**Accuracy = (9627+77)/10000 = 97.0%**  
**Recall = 77/(77+252) = 23.4%** ← ต่ำมาก!

**แก้: ลด threshold จาก 0.5 → 0.2**
```python
y_proba = lda.predict_proba(X_test)[:,1]
y_pred_low = (y_proba >= 0.2).astype(int)
```

**[FIGURE: 2 confusion matrices side by side: threshold=0.5 vs threshold=0.2]**

---

## Slide 7 — LDA as Dimensionality Reduction

**Key Message**: LDA ไม่ใช่แค่ classifier — ยังใช้ลด dimension ได้เป็น K-1 dimensions

**LDA Projection:**
- Project data จาก p dimensions → **K−1 dimensions**
- K=2 classes → 1D projection (แนวที่แยก class ได้ดีที่สุด)
- K=3 classes → 2D projection (plot ได้!)

```python
# 3-class example (Wine dataset)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
X_2d = lda.transform(X_train)  # shape: (n, 2)

# Scatter plot
plt.scatter(X_2d[:,0], X_2d[:,1], c=y_train, cmap='Set1')
plt.xlabel('LD1'); plt.ylabel('LD2')
```

**LDA vs PCA:**
| | LDA | PCA |
|-|-----|-----|
| Supervised? | ✓ | ✗ |
| Objective | Maximize class separation | Maximize variance |
| Boundary | Linear | — |

**[FIGURE: LDA projection ของ Wine dataset → 3 well-separated clusters ใน 2D]**

---

## Slide 8 — LDA on Smarket Dataset

**Key Message**: LDA ไม่ได้ดีเสมอ — ถ้า data ไม่ meet assumptions ผลอาจไม่ดีกว่า Logistic

**Smarket: predict Up/Down ของ S&P500**
- Features: Lag1–Lag5 (% return วันก่อน), Volume
- Train: 2001–2004, Test: 2005

**ผลลัพธ์:**

| Method | Test Accuracy (2005) |
|--------|---------------------|
| Logistic Regression | 55.6% |
| **LDA** | **56.0%** |

**สังเกต:** ทั้ง 2 method ให้ accuracy ใกล้ 50% มาก  
→ ตลาดหุ้นมี **random walk** คุณสมบัติ  
→ Lag returns ไม่ predictive มากพอ

**[FIGURE: Smarket scatter: Lag1 vs Lag2 colored by Direction — no clear separation]**

---

## Slide 9 — LDA Assumptions Check

**Key Message**: ตรวจ assumption ก่อนใช้ LDA — ถ้า assumption ผิด อาจต้องใช้ QDA หรือ LR แทน

**Assumption 1: Gaussian Distribution ของ X ใน class k**
- ตรวจด้วย: histogram ของแต่ละ feature แยกตาม class
- ผลสำหรับ balance (Default): roughly Gaussian ✓
- ผลสำหรับ Lag1 (Smarket): roughly Gaussian ✓

**Assumption 2: Shared Covariance Matrix Σ**
- ตรวจด้วย: Box's M test (formal) หรือ plot covariance ellipses
- ถ้า Σ₁ ≠ Σ₂ อย่างชัดเจน → ใช้ QDA แทน

**Rule of Thumb:**
- LDA เมื่อ: p เล็ก, n เล็ก, classes roughly Gaussian, shared Σ
- QDA เมื่อ: n ใหญ่, covariances ต่างกันชัดเจน

**[FIGURE: 2 panels: LDA ellipses (same size/shape) vs QDA ellipses (different)]**

---

## Slide 10 — Summary

**Key Message**: LDA เป็น powerful linear classifier ที่ตีความได้และ reduce dimension ได้

**LDA สรุป:**
- Discriminant function: δₖ(x) = xᵀΣ⁻¹μₖ − ½μₖᵀΣ⁻¹μₖ + log(πₖ)
- Estimate: μ̂ₖ (class mean), Σ̂ (pooled covariance), π̂ₖ = nₖ/n
- Decision boundary: **linear** (hyperplane)
- ลด dimension ได้สูงสุด K−1 ด้วย `lda.transform()`

**เหมาะสำหรับ:**
- n เล็ก, K ≥ 2
- Classes well-separated
- ต้องการ interpretable linear boundary

**Python:** `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`

**สัปดาห์ต่อไป:** QDA ผ่อนคลาย assumption shared Σ → quadratic boundary
