# Slide Deck 4: KNN Classifier + Method Comparison
> Week 12 | 10 slides | CLO3

---

## Slide 1 — Title

**KNN Classifier + Comparison of Classification Methods**  
ISLP Ch.2.2.3 + Ch.4.5 + Ch.4.7.6 | Week 12 | CLO3

---

## Slide 2 — K-Nearest Neighbors: The Idea

**Key Message**: KNN ไม่ assume distribution ใดๆ — ใช้ความใกล้เคียงในพื้นที่ feature space

**KNN Classifier:**

> Classify x₀ โดยดู K neighbors ที่ใกล้ที่สุดใน training data แล้ว vote majority class

**Formal definition:**
$$\Pr(Y=j|X=x_0) = \frac{1}{K}\sum_{i \in \mathcal{N}_0} \mathbb{I}(y_i = j)$$

- 𝒩₀ = set of K neighbors ที่ใกล้ x₀ ที่สุด
- Classify ไปยัง j ที่ Pr สูงสุด (majority vote)

**[FIGURE: 2D scatter, x₀ อยู่ตรงกลาง, K=5 neighbors วงกลม, 3 แดง 2 น้ำเงิน → classify as red]**

---

## Slide 3 — K Effect: Bias-Variance Trade-off

**Key Message**: K เล็ก = overfit, K ใหญ่ = underfit — ต้องเลือก K ที่เหมาะสม

**K=1 (Overfit):**
- Boundary รอบ training points ทุกจุด
- Training error = 0 แต่ test error สูง
- High variance, Low bias

**Large K (Underfit):**
- Boundary เรียบมาก → smooth prediction
- Miss local patterns
- Low variance, High bias

**[FIGURE: 3 panels: K=1 (irregular jagged boundary), K=10 (smooth), K=50 (nearly linear)]**

**ดู K ที่ดีจากกราฟ:**
```python
k_values = [1, 3, 5, 7, 10, 15, 20]
test_errors = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    test_errors.append(1 - knn.score(X_test, y_test))
plt.plot(k_values, test_errors, 'o-')  # หา K ที่ error ต่ำสุด
```

---

## Slide 4 — KNN: Distance + Scaling

**Key Message**: KNN sensitive มากกับ feature scale — ต้อง standardize เสมอ

**Distance metric (default: Euclidean):**
$$d(x_0, x_i) = \sqrt{\sum_{j=1}^{p}(x_{0j} - x_{ij})^2}$$

**ปัญหาถ้าไม่ scale:**
- Balance ($0–3000) vs Student (0/1) → balance dominates distance
- Income ($10K–100K) ทำให้ feature อื่นไร้ความหมาย

**Always standardize for KNN:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)   # ← fit บน train เท่านั้น!

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc, y_train)
```

**[FIGURE: before/after scaling — KNN boundary เปลี่ยนไปมากหลัง scale]**

---

## Slide 5 — Python: KNeighborsClassifier

**Key Message**: sklearn KNN ใช้ง่าย — กำหนดแค่ n_neighbors และ metric

```python
from sklearn.neighbors import KNeighborsClassifier

# K=5, Euclidean distance (default)
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean',  # หรือ 'manhattan', 'minkowski'
    weights='uniform'    # หรือ 'distance' (ให้น้ำหนักตาม inverse distance)
)
knn.fit(X_train_sc, y_train)

# Predict
y_pred = knn.predict(X_test_sc)

# Probability (majority vote → smooth version ด้วย K)
y_proba = knn.predict_proba(X_test_sc)[:,1]

# KNN ไม่มี coef_ หรือ feature_importances_ — black box!
```

**ข้อจำกัดของ KNN:**
- Slow ใน prediction เมื่อ n ใหญ่ (ต้อง compute distance ทุก training point)
- ไม่ interpretable — ไม่รู้ว่า feature ไหนสำคัญ
- Curse of dimensionality: เมื่อ p ใหญ่ distance ไม่มีความหมาย

---

## Slide 6 — Choosing K via Cross-Validation

**Key Message**: CV เป็น principled way ในการหา optimal K (preview Week 14)

```python
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_train_sc, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())

# หา optimal K
best_k = k_range[np.argmax(cv_scores)]
print(f'Optimal K = {best_k}, CV F1 = {max(cv_scores):.4f}')

# Plot
plt.plot(list(k_range), cv_scores, 'o-')
plt.xlabel('K'); plt.ylabel('CV F1 Score')
plt.axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
```

**Rule of thumb ถ้าไม่ใช้ CV:** K ≈ √n แต่ CV ให้ผลแม่นกว่าเสมอ

**[FIGURE: CV score vs K curve — peak ที่ K optimal, drop ทั้งสองข้าง]**

---

## Slide 7 — Comparison: All 5 Methods

**Key Message**: ไม่มี best classifier — เลือกตาม data, n, boundary complexity

**Comparison on Default Dataset (test set):**

| Method | Accuracy | Recall | AUC | Notes |
|--------|----------|--------|-----|-------|
| Logistic Regression | 0.974 | 0.62 | 0.949 | baseline, interpretable |
| LDA | 0.974 | 0.63 | 0.948 | stable, linear |
| QDA | 0.971 | 0.61 | 0.941 | slight overfit |
| Naive Bayes | 0.963 | 0.75 | 0.935 | high recall, lower precision |
| KNN (K=5) | 0.951 | 0.58 | 0.891 | no scaling used here |

**หมายเหตุ:** KNN ต้องการ standardize features ก่อน — ผลข้างต้นไม่ได้ scale

**[FIGURE: grouped bar chart — แต่ละ method เป็นกลุ่ม, bar ย่อย = Accuracy/Recall/AUC]**

---

## Slide 8 — When to Use Each Method

**Key Message**: เลือก classifier ตาม n, p, boundary type, interpretability requirement

| Method | ใช้เมื่อ | ระวัง |
|--------|---------|-------|
| **Logistic Regression** | n ใหญ่, linear boundary, interpretability สำคัญ | ไม่ดีเมื่อ classes well-separated มาก |
| **LDA** | n เล็กหรือ classes well-separated, need projection | ต้อง check Gaussian + shared Σ |
| **QDA** | n ใหญ่ (>10×p²), Σₖ ต่างกัน | overfit เมื่อ n เล็ก |
| **Naive Bayes** | p ใหญ่, text/sparse data, fast training | features มี correlation สูง |
| **KNN** | boundary ซับซ้อน, n ใหญ่ | ต้อง scale, slow prediction, no interpretation |

**Practical Guidance:**
1. เริ่มด้วย Logistic Regression เสมอ (baseline)
2. ลอง LDA ถ้า classes well-separated
3. ลอง QDA ถ้า LDA ให้ผลไม่ดีและ n ใหญ่พอ
4. ลอง NB ถ้า p ใหญ่หรือ categorical features
5. ลอง KNN ถ้า boundary ซับซ้อนและ n ≥ 1000

---

## Slide 9 — ROC Curve: Model-Agnostic Comparison

**Key Message**: ROC Curve + AUC เปรียบเทียบ classifier โดยไม่ขึ้นกับ threshold

**ROC (Receiver Operating Characteristic) Curve:**
- X-axis: FPR = FP/(FP+TN) = 1 − Specificity
- Y-axis: TPR = TP/(TP+FN) = Recall/Sensitivity
- แต่ละจุดบน curve = threshold ค่าหนึ่ง

**AUC (Area Under Curve):**
- AUC = 1.0: perfect classifier
- AUC = 0.5: random classifier (diagonal)
- AUC > 0.8: ดี | AUC > 0.9: ดีมาก

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.plot(fpr, tpr, label=f'LDA (AUC={auc:.3f})')
```

**[FIGURE: ROC curves ของทุก 5 methods ใน plot เดียว — Logistic และ LDA สูงสุด]**

---

## Slide 10 — Summary + Full Comparison Table

**Key Message**: ทุก method ที่เรียนมาเป็นส่วนหนึ่งของ Classification toolbox ที่ Data Scientist ต้องรู้

**Classification Methods ใน ISLP Ch.4:**

| Method | Type | Boundary | Python | Best for |
|--------|------|----------|--------|---------|
| Logistic Regression | Discriminative | Linear | `LogisticRegression` | General baseline |
| LDA | Generative | Linear | `LinearDiscriminantAnalysis` | n เล็ก, well-separated |
| QDA | Generative | Quadratic | `QuadraticDiscriminantAnalysis` | n ใหญ่, Σ ต่างกัน |
| Naive Bayes | Generative | Flexible | `GaussianNB` | p ใหญ่, text |
| KNN | Non-parametric | Non-linear | `KNeighborsClassifier` | Complex boundary |

**Key Takeaways:**
1. ไม่มี "best" classifier ในทุกสถานการณ์
2. เริ่มด้วย simple methods ก่อนเสมอ
3. ตรวจ assumption ก่อนเลือก — Gaussian? shared Σ? independence?
4. ใช้ ROC + AUC เปรียบเทียบ classifier โดยไม่ขึ้นกับ threshold
5. Standardize features ก่อน KNN เสมอ!

**สัปดาห์ต่อไป (Week 13):** Resampling Methods — Cross-Validation, Bootstrap สำหรับ model selection
