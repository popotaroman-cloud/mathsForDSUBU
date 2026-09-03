# Slide Deck 1: Why Resampling?
> Week 14 | 9 slides | CLO4

---

## Slide 1 — Title

**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล — Week 14**  
Resampling Methods: Cross-Validation + Bootstrap  
CLO4: ใช้ Cross-Validation และ Bootstrap ประเมินและเลือก model ที่เหมาะสมได้

---

## Slide 2 — Week Overview

**Key Message**: Resampling แปลง model selection จาก "เดา" เป็น "วิทยาศาสตร์"

เราเรียนมา 13 สัปดาห์และสะสม models มากมาย: Linear Regression, Logistic Regression, LDA, QDA, Naive Bayes, KNN แต่คำถามสำคัญที่สุดคือ **"จะรู้ได้อย่างไรว่า model ไหนดีที่สุด?"** Training Error ต่ำไม่ตอบคำถามนี้ได้ เพราะ model ที่ซับซ้อนมากจะมี Training Error = 0 แต่ fail บน new data สัปดาห์นี้เราเรียน **Cross-Validation** สำหรับ model selection และ **Bootstrap** สำหรับ uncertainty estimation เครื่องมือเหล่านี้เป็น foundation ของ AutoML, hyperparameter tuning และ model comparison ใน production ทุกระบบ ML ที่ professional ใช้ CV เสมอ

- **Validation Set**: simple แต่ high variance
- **LOOCV**: low bias แต่ computationally expensive
- **k-Fold CV**: practical trade-off (k=5 หรือ 10)
- **Bootstrap**: SE estimation สำหรับ statistic ใดๆ
- **One-SE Rule**: เลือก simpler model ที่ยังดีพอ

---

## Slide 3 — The Fundamental Problem

**Key Message**: Training Error ต่ำไม่ได้แปลว่า Test Error ต่ำ — นี่คือ core problem ของ ML

**[FIGURE: bias-variance U-curve จาก Week 6]**

- **Training Error** ลดเรื่อยๆ เมื่อ model ซับซ้อนขึ้น
- **Test Error** มีรูปร่าง U-shape — มี optimal complexity

**ปัญหาในทางปฏิบัติ:**
- ถ้าเรามี test set ใช้ test set ประเมิน → แต่ test set ควรใช้ **ครั้งเดียวตอนสุดท้าย**
- ถ้าใช้ test set เพื่อ select model → ไม่ใช่ test set อีกต่อไป (bias)
- วิธีแก้: **จำลอง test set จาก training data** = Resampling

**Connection to Week 6:**
- Week 6: อธิบาย problem ของ Bias-Variance
- Week 14: ให้ tools แก้ปัญหา

---

## Slide 4 — What is Resampling?

**Key Message**: Resampling = สุ่มข้อมูลซ้ำซ้อนจาก training data เพื่อ estimate test performance

**Resampling Idea:**
1. มีเพียง training data
2. แบ่งส่วนหนึ่งของ training data เป็น "pseudo test set"
3. Train บน ส่วนที่เหลือ, evaluate บน pseudo test
4. ทำซ้ำหลายครั้ง → average error = estimated test error

**Two main approaches:**

| Method | จุดประสงค์ | ประสิทธิภาพ |
|--------|-----------|-----------|
| **Cross-Validation** | Estimate Test Error | O(k × n) fit operations |
| **Bootstrap** | Estimate SE of any statistic | O(B × n) fit operations |

**[FIGURE: conceptual diagram — training data pool, CV pulls validation subsets, Bootstrap resamples with replacement]**

---

## Slide 5 — Validation Set: The Simplest Approach

**Key Message**: Split ครั้งเดียว = ง่ายที่สุด แต่มี variance สูงมาก

**Validation Set Approach:**
- Split: 50/50 หรือ 70/30 randomly
- Train บน training half
- Evaluate บน validation half

**ข้อดี:** Simple, fast, easy to understand

**ข้อเสีย:** **High Variance!**
- ผลลัพธ์ขึ้นกับ random split มาก
- ถ้า outliers อยู่ใน train: test MSE สูงกว่าจริง
- ถ้า outliers อยู่ใน test: test MSE ต่ำกว่าจริง

**[FIGURE: 10 different random splits of same data → 10 different test MSE values, spread wide]**

**สรุป:** ใช้ validation set เป็น quick estimate แต่ไม่ reliable พอสำหรับ model selection จริงๆ

---

## Slide 6 — Why K=5 or K=10?

**Key Message**: k=5 หรือ 10 เป็น practical sweet spot ระหว่าง bias และ variance

**Bias-Variance ของ CV:**

| k | Train size | Bias | Variance |
|---|-----------|------|---------|
| 2 (50/50) | 50% | High | Medium |
| 5 | 80% | Moderate | Low |
| **10** | **90%** | **Low** | **Low** | ← recommended |
| n (LOOCV) | n-1 ≈ 100% | Very Low | High |

**Empirical finding (Hastie et al.):**
- k=10 fold CV ให้ test error estimate ที่ดีที่สุดในทางปฏิบัติ
- k=5 ก็ดีพอสำหรับ n ใหญ่
- LOOCV: ดีสำหรับ n เล็ก (<100) แต่ expensive สำหรับ n ใหญ่

**Python:**
```python
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_mse = -scores.mean()
cv_se  = scores.std() / np.sqrt(10)
```

---

## Slide 7 — Stratified k-Fold for Classification

**Key Message**: สำหรับ classification ต้องใช้ Stratified k-Fold ป้องกัน fold ที่ไม่ balanced

**ปัญหาของ regular KFold สำหรับ imbalanced data:**
- Default dataset: 96.7% No Default, 3.3% Default
- Random fold อาจได้ fold ที่ไม่มี "Default=Yes" เลย
- → model ไม่ได้เรียน positive class ใน fold นั้น → test error ผิดพลาด

**Stratified k-Fold:**
- แต่ละ fold มี class distribution ≈ เหมือน original data
- 5-Fold: แต่ละ fold มี Default=Yes ≈ 3.3%

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    clf, X_train, y_train,
    cv=skf,
    scoring='f1'           # ← ใช้ F1 ไม่ใช่ accuracy สำหรับ imbalanced!
)
```

**[FIGURE: Regular KFold fold composition vs Stratified KFold — เห็นว่า Stratified มี class distribution เท่าๆ กันทุก fold]**

---

## Slide 8 — CV in Practice: Pipeline

**Key Message**: ใช้ Pipeline เพื่อป้องกัน data leakage ใน CV

**CRITICAL: Data Leakage ใน CV**
```python
# ❌ WRONG: scaler fit บน ALL data ก่อน CV
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
scores = cross_val_score(model, X_scaled, y, cv=5)
# ↑ scaler เห็น validation data ทำให้ estimate optimistic

# ✅ CORRECT: ใช้ Pipeline ให้ scaler fit ใน loop อัตโนมัติ
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])
scores = cross_val_score(pipe, X, y, cv=5)
# ↑ Pipeline fit scaler บน train fold เท่านั้นในแต่ละ iteration
```

**Pipeline ป้องกัน leakage เพราะ:**
- `scaler.fit_transform()` เรียกเฉพาะบน training fold ใน CV loop
- `scaler.transform()` เรียกบน validation fold

**[FIGURE: CV loop diagram — แต่ละ iteration, Pipeline fit scaler + model บน K-1 folds]**

---

## Slide 9 — Summary: Why Resampling?

**Key Message**: Resampling แก้ปัญหา test error estimation — ไม่มีเหตุผลไม่ใช้ CV

**Problem Summary:**
- Training Error misleading: model ซับซ้อน → Train Error ↓ แต่ Test Error ↑
- Single validation split: high variance, unreliable

**Solution — Cross-Validation:**
- k-Fold CV: estimate test error reliably
- Stratified k-Fold: สำหรับ imbalanced classification
- LOOCV: low bias, สำหรับ n เล็ก
- Always use Pipeline to prevent data leakage

**Connection:**
- ← Week 6: Bias-Variance (motivation)
- ↑ Week 14: CV + Bootstrap (solution)
- → Week 14: k-Fold CV implementation + Bootstrap
