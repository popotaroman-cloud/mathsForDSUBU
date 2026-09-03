# Slide Deck 4: Full Classification Pipeline
> Week 13 | 10 slides | CLO3

---

## Slide 1 — Title

**Full Classification Pipeline: Best Practices**  
ISLP Ch.4.7 Lab | Week 13 | CLO3

---

## Slide 2 — What is a Pipeline?

**Key Message**: Pipeline = ลำดับขั้นตอนที่ reproducible ตั้งแต่ raw data จนถึง production model

**Classification Pipeline ประกอบด้วย:**

```
Raw Data
    ↓
1. Data Loading + EDA
    ↓
2. Preprocessing (clean, encode, scale)
    ↓
3. Train/Test Split
    ↓
4. Fit Multiple Classifiers
    ↓
5. Evaluate (CM, metrics, ROC/AUC)
    ↓
6. Select Best Model + Justify
    ↓
7. Threshold Tuning (business need)
    ↓
Production Model
```

**ทำไม Pipeline สำคัญ?**
- Reproducible: ทุกคนได้ผลเดียวกัน
- Scalable: เพิ่ม classifier ใหม่ได้ง่าย
- Auditable: อธิบายทุกขั้นตอนได้

---

## Slide 3 — Step 1–2: Load + EDA

**Key Message**: EDA ก่อน fit model เสมอ — ข้อมูลบอกว่า classifier ไหนน่าจะใช้ได้

**สิ่งที่ต้องตรวจสอบใน EDA:**

| สิ่งที่ตรวจ | ทำอย่างไร | บอกอะไร |
|-----------|---------|---------|
| Class balance | `y.value_counts()` | imbalanced → adjust threshold/class_weight |
| Feature distributions | `df.hist()` | Gaussian? → LDA อาจเหมาะ |
| Correlations | `df.corr()` | High correlation → NB อาจแย่ |
| Missing values | `df.isnull().sum()` | ต้องจัดการก่อน fit |
| Outliers | boxplot, scatter | อาจส่งผลต่อ LDA/KNN |

**[FIGURE: EDA dashboard — 4 plots: class distribution bar, histogram grid, correlation heatmap, boxplot vs class]**

```python
# Quick EDA
print(df.shape, df.isnull().sum())
print(y.value_counts(normalize=True))  # class balance
df.hist(figsize=(12,8)); plt.show()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

---

## Slide 4 — Step 3: Train/Test Split

**Key Message**: Split ก่อน preprocess เสมอ — ป้องกัน data leakage

**วิธี Split:**

| Case | วิธี | Code |
|------|------|------|
| IID data | Random split | `train_test_split(X, y, test_size=0.2)` |
| Time series | Time-based | `X[year <= 2004]` |
| Imbalanced | Stratified | `train_test_split(..., stratify=y)` |

**CRITICAL — Data Leakage:**
```python
# ❌ WRONG: fit scaler บน ALL data ก่อน split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← ใช้ข้อมูล test ใน fit!
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ CORRECT: split ก่อน แล้ว fit scaler บน TRAIN เท่านั้น
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform
X_test_sc  = scaler.transform(X_test)        # transform only!
```

**[FIGURE: diagram showing correct vs incorrect data flow, X mark on wrong path, checkmark on correct path]**

---

## Slide 5 — Step 4: Preprocessing

**Key Message**: Preprocessing ที่ถูกต้องส่งผลโดยตรงต่อ model performance

**Feature Scaling:**
- **KNN**: ต้องการ scale (Euclidean distance)
- **Logistic, LDA**: benefit จาก scale
- **Tree-based (จะเรียนทีหลัง), NB**: ไม่ต้องการ scale

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: zero mean, unit variance — ใช้บ่อยที่สุด
scaler = StandardScaler()

# MinMaxScaler: scale ไป [0,1] — ใช้เมื่อ range สำคัญ
scaler = MinMaxScaler()
```

**Encoding Categorical Variables:**
```python
# One-Hot Encoding (สำหรับ nominal)
df = pd.get_dummies(df, columns=['category_col'], drop_first=True)

# Ordinal Encoding (สำหรับ ordinal เช่น low/medium/high)
from sklearn.preprocessing import OrdinalEncoder
```

**Handling Missing Values:**
```python
df.fillna(df.median(), inplace=True)  # simple imputation
# หรือใช้ sklearn.impute.SimpleImputer
```

---

## Slide 6 — Step 5: Fit Multiple Classifiers

**Key Message**: Fit ทุก classifier ใน loop เดียว — code สะอาด reproducible

```python
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Dictionary ของ classifiers ที่จะ compare
classifiers = {
    'Logistic': LogisticRegression(max_iter=1000),
    'LDA'     : LinearDiscriminantAnalysis(),
    'QDA'     : QuadraticDiscriminantAnalysis(),
    'NB'      : GaussianNB(),
    'KNN(5)'  : KNeighborsClassifier(n_neighbors=5)
}

# Fit ทุก model และเก็บ results
results = []
for name, clf in classifiers.items():
    clf.fit(X_train_sc, y_train)
    y_pred  = clf.predict(X_test_sc)
    y_proba = clf.predict_proba(X_test_sc)[:,1]
    results.append({
        'Method'    : name,
        'Accuracy'  : accuracy_score(y_test, y_pred),
        'F1'        : f1_score(y_test, y_pred, zero_division=0),
        'AUC'       : roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))
```

---

## Slide 7 — Step 6: Evaluate

**Key Message**: ใช้หลาย metrics เสมอ — ไม่มี single metric ที่บอกทุกอย่าง

**Evaluation Toolkit:**

| Metric | ใช้เมื่อ | Python |
|--------|---------|--------|
| **Accuracy** | Balanced classes | `accuracy_score()` |
| **Precision** | FP cost สูง | `precision_score()` |
| **Recall** | FN cost สูง | `recall_score()` |
| **F1** | Balanced FP/FN cost | `f1_score()` |
| **AUC** | Threshold-free comparison | `roc_auc_score()` |
| **CM** | Visual diagnosis | `ConfusionMatrixDisplay` |

**สำหรับ Imbalanced classes:**
- Accuracy อาจ misleading (97% accuracy ถ้า predict majority เสมอ)
- ใช้ F1, Recall, AUC แทน

**[FIGURE: comparison table 5 classifiers บน stock market data]**

---

## Slide 8 — Step 7: Select Best Model

**Key Message**: เลือก best model ต้อง justify โดยใช้ metrics ที่สอดคล้องกับ business objective

**Framework การเลือก:**

1. **Define objective ก่อน:**
   - Maximize recall (medical screening): ลด FN
   - Maximize precision (spam filter): ลด FP
   - Balanced: ใช้ F1 หรือ AUC

2. **เลือก metric ที่ตรงกับ objective:**
   - Medical: Recall หลัก
   - Finance fraud: Recall หลัก
   - Email spam: Precision + Recall (F1)
   - Ranking/Scoring: AUC

3. **Consider non-metric factors:**
   - Interpretability: LDA/LR อธิบายได้ง่ายกว่า KNN
   - Training time: NB เร็วที่สุด
   - Deployment: KNN ช้าใน prediction

**[FIGURE: decision tree — FN costly? → maximize recall, FP costly? → maximize precision, balanced? → AUC]**

---

## Slide 9 — Step 7: Threshold Tuning

**Key Message**: Threshold = lever ที่ปรับ trade-off Precision/Recall ตาม business need

**Threshold Tuning Workflow:**
```python
# หลังเลือก best model แล้ว
best_proba = best_clf.predict_proba(X_test)[:,1]

thresholds = np.arange(0.1, 0.9, 0.05)
for t in thresholds:
    pred_t = (best_proba >= t).astype(int)
    p = precision_score(y_test, pred_t, zero_division=0)
    r = recall_score(y_test, pred_t, zero_division=0)
    f = f1_score(y_test, pred_t, zero_division=0)
    print(f't={t:.2f}: P={p:.3f}, R={r:.3f}, F1={f:.3f}')

# หา threshold ที่ตรงกับ requirement
# เช่น: ต้องการ Recall ≥ 0.80
```

**Business Question Template:**
> "ถ้า FN cost = X× FP cost → ต้องการ Recall ≥ ? → เลือก threshold = ?"

**[FIGURE: precision-recall trade-off plot — เส้น Precision (down-trend) และ Recall (up-trend) as threshold decreases, จุด recommended threshold]**

---

## Slide 10 — Summary + Full Pipeline Checklist

**Key Message**: Full Pipeline = ลำดับขั้นตอนที่ต้องทำครบทุกขั้น ไม่ข้าม

**Full Classification Pipeline Checklist:**

- [ ] 1. Load data + check shape, dtypes, missing
- [ ] 2. EDA: class balance, distributions, correlations
- [ ] 3. Train/test split (stratify ถ้า imbalanced)
- [ ] 4. Preprocess: encode, scale (fit on train only!)
- [ ] 5. Fit ≥ 3 classifiers
- [ ] 6. Evaluate: CM, Accuracy, F1, AUC
- [ ] 7. Plot ROC Curves ทุก classifier
- [ ] 8. Select best model + justify
- [ ] 9. Tune threshold ตาม business objective
- [ ] 10. Report findings ใน plain language

**สิ่งที่เรียนรู้ทั้ง Week 11–13:**
- Week 11: Logistic Regression (discriminative)
- Week 12: LDA, QDA, NB, KNN (generative + non-parametric)
- Week 13: GLM, ROC/AUC, Full Pipeline

**สัปดาห์ต่อไป (Week 14):** Cross-Validation + Bootstrap: หา optimal hyperparameter อย่าง principled
