# Slide Deck 3: QDA and Naive Bayes
> Week 12 | 9 slides | CLO3

---

## Slide 1 — Title

**Quadratic Discriminant Analysis + Naive Bayes**  
ISLP Ch.4.4.3–4.4.4 | Week 12 | CLO3

---

## Slide 2 — From LDA to QDA

**Key Message**: QDA ผ่อนคลาย assumption shared Σ → แต่ละ class มี covariance ของตัวเอง

**LDA assumption (restrictive):**
$$X|Y=k \sim \mathcal{N}(\mu_k, \mathbf{\Sigma}) \quad \text{(Σ เหมือนกันทุก class)}$$

**QDA assumption (flexible):**
$$X|Y=k \sim \mathcal{N}(\mu_k, \mathbf{\Sigma_k}) \quad \text{(Σₖ ต่างกันแต่ละ class)}$$

**QDA Discriminant Function:**
$$\delta_k(x) = -\frac{1}{2}x^\top \Sigma_k^{-1} x + x^\top \Sigma_k^{-1}\mu_k - \frac{1}{2}\mu_k^\top \Sigma_k^{-1}\mu_k - \frac{1}{2}\log|\Sigma_k| + \log(\pi_k)$$

**สังเกต:** มี term xᵀΣₖ⁻¹x → **quadratic** ใน x → boundary โค้ง

**[FIGURE: side by side — LDA: linear boundary; QDA: curved/elliptical boundary]**

---

## Slide 3 — LDA vs QDA: Bias-Variance Trade-off

**Key Message**: LDA = less flexible (low variance) vs QDA = more flexible (low bias) — เลือกตาม n

**Parameters:**

| | LDA | QDA |
|-|-----|-----|
| Σ per class | 1 shared Σ | K separate Σₖ |
| Parameters (p=10, K=2) | 10×10 = 100 | 2 × 100 = 200 |
| Boundary | Linear | Quadratic |
| Bias | Higher | Lower |
| Variance | Lower | Higher |

**Rule:**

| สถานการณ์ | แนะนำ | เพราะ |
|-----------|-------|-------|
| n เล็ก | **LDA** | variance ต่ำ ไม่ overfit |
| n ใหญ่ | **QDA** | bias ต่ำ estimate Σₖ ได้แม่น |
| Σₖ ≈ เท่ากัน | LDA | shared assumption เป็นจริง |
| Σₖ ≠ กันมาก | **QDA** | ใช้ flexibility ได้เต็มที่ |

**[FIGURE: bias-variance axes: LDA อยู่มุมบน-ซ้าย (high bias, low variance), QDA อยู่มุมล่าง-ขวา]**

---

## Slide 4 — Python: QDA

**Key Message**: QDA ใช้ง่ายเหมือน LDA — เปลี่ยนแค่ class name

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Fit QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Predict
y_pred_qda = qda.predict(X_test)
y_proba_qda = qda.predict_proba(X_test)[:,1]

# ดู per-class covariance matrices
print('Class 0 covariance shape:', qda.covariance_[0].shape)  # (p, p)
print('Class 1 covariance shape:', qda.covariance_[1].shape)  # (p, p)
```

**ข้อสังเกต:**
- `qda.covariance_` → list ของ Σₖ แต่ละ class
- `qda.means_` → μ̂ₖ เหมือน LDA
- QDA ไม่มี `transform()` (ไม่สามารถ reduce dimension แบบ linear ได้)

**Warning:** QDA อาจ error ถ้า n class เล็กเกินไป (singular Σₖ) → ใช้ `reg_param` แก้ได้

---

## Slide 5 — Naive Bayes: Independence Assumption

**Key Message**: Naive Bayes assume features independent กันใน class — "naive" แต่ใช้ได้จริง

**Naive Independence Assumption:**
$$\Pr(X|Y=k) = \prod_{j=1}^{p} f_{kj}(x_j)$$

→ แต่ละ feature Xⱼ independent กับ Xⱼ' เมื่อกำหนด Y=k

**ทำไม "Naive"?**
- ใน reality features มักมี correlation
- แต่ independence assumption ลด parameters ได้มาก
- ทำให้ estimate แม่นขึ้นเมื่อ n เล็ก หรือ p ใหญ่

**Classifier:**
$$\hat{k} = \arg\max_k \left[\log \pi_k + \sum_{j=1}^{p} \log f_{kj}(x_j)\right]$$

**[FIGURE: Naive Bayes model: Y → X₁, Y → X₂, Y → X₃ (ลูกศรออกจาก Y ไป Xⱼ ทุกตัว, ไม่มีลูกศรระหว่าง Xⱼ)]**

---

## Slide 6 — Naive Bayes Variants

**Key Message**: เลือก NB variant ตามประเภทของ feature

| Variant | fₖⱼ(xⱼ) | ใช้กับ | Python |
|---------|---------|-------|--------|
| **Gaussian NB** | N(μₖⱼ, σ²ₖⱼ) | continuous features | `GaussianNB` |
| **Bernoulli NB** | Bernoulli(pₖⱼ) | binary features (0/1) | `BernoulliNB` |
| **Multinomial NB** | Multinomial | word counts in text | `MultinomialNB` |
| **Complement NB** | Modified multinomial | imbalanced text | `ComplementNB` |

**Gaussian NB (สัปดาห์นี้):**
$$f_{kj}(x_j) = \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left(-\frac{(x_j-\mu_{kj})^2}{2\sigma_{kj}^2}\right)$$

Estimate: μ̂ₖⱼ = mean ของ feature j ใน class k, σ̂²ₖⱼ = variance

**[EXAMPLE: Email spam — Bernoulli NB: word present/absent; topic model — Multinomial NB: word count]**

---

## Slide 7 — Python: Gaussian Naive Bayes

**Key Message**: GaussianNB ต้องการ estimate แค่ mean + variance ต่อ feature ต่อ class

```python
from sklearn.naive_bayes import GaussianNB

# Fit Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ดู parameters
print('Class means (θ):', gnb.theta_)    # μ̂ₖⱼ — shape (K, p)
print('Class vars  (σ²):', gnb.var_)     # σ̂²ₖⱼ — shape (K, p)
print('Class priors:', gnb.class_prior_) # π̂ₖ

# Predict
y_pred_nb = gnb.predict(X_test)
y_proba_nb = gnb.predict_proba(X_test)

# Class probabilities ทุก class
print('Posterior Pr(Y=k|x) for first 3 test samples:')
print(y_proba_nb[:3].round(4))
```

**ข้อสังเกต:**
- GaussianNB ไม่ต้อง standardize features (estimate σ²ₖⱼ แยกกันอยู่แล้ว)
- แต่ KNN ต้อง standardize!

---

## Slide 8 — NB in Practice: Text Classification

**Key Message**: Naive Bayes โด่งดังใน NLP — spam filter ยุคแรก ใช้ NB ทั้งนั้น

**Email Spam Detection (Classic Application)**

**Setup:**
- Features: TF-IDF ของ words ใน email (sparse binary/count matrix)
- Y = 0 (ham) / 1 (spam)
- p = vocabulary size (ปกติ 10,000–100,000 words!)

**ทำไม NB เหมาะกับ text?**
1. p ใหญ่มาก (10K features) → NB estimate น้อย parameters กว่า LDA มาก
2. Sparse features → independence assumption ทำให้คำนวณเร็ว
3. Interpretable: log Pr(word|spam) บอกว่า word ไหน indicative

**Example:**
```
log P("free"|spam) = -2.3  → "free" พบบ่อยใน spam
log P("meeting"|ham) = -3.1 → "meeting" พบบ่อยใน ham
```

**[FIGURE: word cloud — words สีแดงสำหรับ spam, สีน้ำเงินสำหรับ ham]**

---

## Slide 9 — Summary + Comparison

**Key Message**: QDA flexible กว่า LDA; NB ใช้ได้เมื่อ p ใหญ่หรือ features เป็น categorical

**QDA:**
- Assumption: Gaussian per-class Σₖ
- Boundary: Quadratic
- ดีกว่า LDA เมื่อ n ใหญ่ หรือ Σₖ ต่างกันชัดเจน
- Python: `QuadraticDiscriminantAnalysis()`

**Naive Bayes:**
- Assumption: Features independent กันใน class
- Boundary: Flexible (ขึ้นกับ distribution ที่เลือก)
- ดีกว่าเมื่อ: p ใหญ่, text classification, fast training
- Python: `GaussianNB()`, `MultinomialNB()`, `BernoulliNB()`

**LDA vs QDA vs NB:**

| | LDA | QDA | NB |
|-|-----|-----|----|
| Assumption | Σ shared | Σₖ per-class | Independent |
| n เล็ก | ✓✓ | ✗ | ✓ |
| p ใหญ่ | ✓ | ✗ | ✓✓ |
| Non-linear data | ✗ | ✓ | partial |
