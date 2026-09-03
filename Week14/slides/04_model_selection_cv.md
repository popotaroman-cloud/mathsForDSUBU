# Slide Deck 4: Model Selection with Cross-Validation
> Week 14 | 10 slides | CLO4

---

## Slide 1 — Title

**Model Selection with Cross-Validation**  
ISLP Ch.5.1.3–5.1.5 | Week 14 | CLO4

---

## Slide 2 — The Model Selection Problem

**Key Message**: Model selection = เลือก complexity ที่สมดุล bias และ variance

**What is Model Selection?**
- เลือก polynomial degree ที่เหมาะสม (1, 2, 3, ...?)
- เลือก K สำหรับ KNN (1, 3, 5, 10, ...?)
- เลือก classifier (Logistic? LDA? KNN?)
- เลือก hyperparameters (C ใน Logistic? λ ใน Ridge?)

**Without CV:** เดา หรือใช้ training error → bias สูง

**With CV:** Principled estimation ของ test error → reliable

**Full Pipeline:**
```
1. Data → Train/Test Split (hold out test set!)
2. [Train set only] → k-Fold CV for each candidate model
3. Compare CV errors → select best model
4. Refit selected model on full train set
5. Report performance on test set (once!)
```

**[FIGURE: flowchart ของ full pipeline — test set ถูก lock จนขั้นตอนสุดท้าย]**

---

## Slide 3 — Choosing Polynomial Degree

**Key Message**: CV curve แสดง U-shape — ชัดเจนว่า degree ไหน optimal

**Auto Dataset Example (ISLP 5.1):**

```python
degrees = range(1, 11)
cv_mse  = []
for d in degrees:
    pipe = Pipeline([('poly', PolynomialFeatures(d)),
                     ('lr', LinearRegression())])
    score = cross_val_score(pipe, X, y,
                           cv=KFold(10, shuffle=True, random_state=1),
                           scoring='neg_mean_squared_error')
    cv_mse.append(-score.mean())
```

**[FIGURE: CV MSE vs polynomial degree — เส้นลงจาก degree 1, ต่ำสุดที่ degree 2, เพิ่มขึ้นหลัง degree 4+]**

**สังเกต:**
- Degree 1: high bias (underfitting)
- Degree 2–3: sweet spot
- Degree 7+: high variance (overfitting)

**สรุป:** degree = 2 ให้ minimum CV MSE สอดคล้องกับ true relationship ที่เป็น quadratic

---

## Slide 4 — Choosing K in KNN

**Key Message**: CV curve สำหรับ KNN: K เล็ก = overfit, K ใหญ่ = underfit

**Example:**
```python
k_range = range(1, 31)
cv_error = []
for k in k_range:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsClassifier(k))])
    score = cross_val_score(pipe, X_train, y_train,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring='f1')
    cv_error.append(score.mean())

best_k = k_range[np.argmax(cv_error)]
print(f'Optimal K = {best_k}')
```

**Pattern:**
- K=1: CV F1 ต่ำ (overfit to training noise)
- K ≈ 5–15: CV F1 สูงสุด (optimal zone)
- K=30+: CV F1 ลด (underfit, too smooth)

**[FIGURE: CV F1 vs K — peak ที่ K optimal, สมมาตรลง 2 ข้าง]**

---

## Slide 5 — Comparing Multiple Classifiers

**Key Message**: ใช้ CV AUC เปรียบเทียบ classifier ต่างชนิดได้ยุติธรรม

**Grid ของ candidates:**
```python
candidates = {
    'Logistic': LogisticRegression(max_iter=1000),
    'LDA'     : LinearDiscriminantAnalysis(),
    'QDA'     : QuadraticDiscriminantAnalysis(),
    'NB'      : GaussianNB(),
    'KNN(10)' : KNeighborsClassifier(n_neighbors=10)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, clf in candidates.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=skf, scoring='roc_auc')
    cv_results[name] = {'mean': scores.mean(), 'se': scores.std()/np.sqrt(5)}

# Print table
for name, r in cv_results.items():
    print(f'{name}: AUC = {r["mean"]:.4f} ± {r["se"]:.4f}')
```

**[FIGURE: bar chart ของ CV AUC ± 1 SE สำหรับแต่ละ classifier]**

---

## Slide 6 — One Standard Error Rule (Details)

**Key Message**: One-SE rule เลือก simplest model ที่ performance ไม่ต่างกันอย่างมีนัยสำคัญ

**Formal Definition:**

$$\text{Select}: \arg\min_{m: \text{CV}(m) \leq \text{CV}^* + \text{SE}^*} \text{Complexity}(m)$$

where CV* = minimum CV error, SE* = SE at minimum

**Python Implementation:**
```python
cv_means = np.array([...])  # CV MSE ของแต่ละ candidate
cv_ses   = np.array([...])  # SE ของแต่ละ candidate

# Best model
best_idx = np.argmin(cv_means)
threshold = cv_means[best_idx] + cv_ses[best_idx]

# One-SE Rule: simplest model ≤ threshold
# สำหรับ polynomial: เลือก smallest degree
# สำหรับ KNN: เลือก largest K (simpler = smoother = larger K)
```

**[TABLE: degree vs CV MSE ± SE — highlight best และ one-SE choice]**

**เมื่อใช้ One-SE Rule:**
- ต้องการ interpretability สูง: เลือก simpler model
- Production: simpler model stable กว่า
- ไม่ต้องการ: เลือก minimum CV เลย

---

## Slide 7 — Wrong Way vs Right Way

**Key Message**: Test set ต้องถูก lock จนกว่าจะเลือก final model เสร็จสิ้น

**❌ WRONG — Test Set Contamination:**
```
1. Split data → train/test
2. For each model: evaluate on TEST → pick best
3. Report test error
→ Test error optimistic! (เห็น test ระหว่าง selection)
```

**✅ CORRECT — Proper Pipeline:**
```
1. Split data → train (80%) + test (20%)  ← LOCK test set!
2. [train only] k-Fold CV → select best model
3. [train only] Refit selected model on full train
4. [once] Evaluate final model on test → report test error
```

**Why does it matter?**
- ถ้า evaluate test ระหว่าง selection → เราได้เลือก model ที่ fit test data โดยบังเอิญ
- → reported test error ต่ำกว่าจริง (optimistic bias)
- → model ทำงานแย่กว่าที่คาดใน production

**[FIGURE: timeline diagram — train period (CV + selection), then single test evaluation]**

---

## Slide 8 — Full Pipeline Code

**Key Message**: Full pipeline 10 บรรทัด — reproducible, readable, correct

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Step 1: Lock test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: CV Model Selection (train only)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
print(f'CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std()/np.sqrt(5):.4f}')

# Step 3: Refit on full train
pipe.fit(X_train, y_train)

# Step 4: Report test (once)
test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
print(f'Test AUC: {test_auc:.4f}')
```

---

## Slide 9 — CV for Time Series

**Key Message**: Time series ต้องใช้ time-based split ไม่ใช่ random k-Fold

**ปัญหาของ random k-Fold กับ time series:**
- Data: stock prices Jan–Dec 2024
- Random fold 3 อาจ include Nov ใน train → ใช้ "future" เพื่อ predict "past" (look-ahead bias)

**Time Series CV:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_va = X[train_idx], X[test_idx]
    # train_idx เสมอ "ก่อน" test_idx ใน time!
```

**[FIGURE: TimeSeriesSplit diagram — folds grow progressively, always train before test in time]**

**ใช้เมื่อ:**
- Stock prices, energy demand, weather prediction
- Any data ที่มี temporal ordering สำคัญ

---

## Slide 10 — Summary: Complete Model Selection

**Key Message**: CV-based model selection = standard practice ที่ต้องใช้ในทุก ML project

**ขั้นตอนสรุป:**

```
Data
  ↓ Step 1: Split (lock test set)
Train / Test
  ↓ Step 2: CV Grid (train only)
k-Fold CV เปรียบเทียบ candidates
  ↓ Step 3: Select
best model (+ One-SE Rule)
  ↓ Step 4: Refit
model fit บน full train
  ↓ Step 5: Evaluate (once)
final test performance
```

**Key Principles:**
1. ไม่ใช้ test set ระหว่าง model selection
2. ใช้ Pipeline ป้องกัน data leakage ใน CV
3. ใช้ Stratified k-Fold สำหรับ imbalanced classification
4. Report SE พร้อมกับ CV mean
5. พิจารณา One-SE Rule สำหรับ interpretability

**Connection to Final Project (Week 15):**
- นักศึกษาจะใช้ CV เลือก model ใน final project
- → ต้องทำตาม pipeline ข้างต้นอย่างครบถ้วน

**สัปดาห์ต่อไป (Week 15):** Final Project Presentation
