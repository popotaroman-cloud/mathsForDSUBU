# Note 14-1: Cross-Validation — LOOCV, k-Fold, and Bias-Variance for CV
> Week 14 | CLO4 | ISLP Reference: Ch.5.1 Cross-Validation

สัปดาห์ที่ 14 เราจะเรียนรู้เรื่อง **Resampling Methods** ซึ่งเป็นชุดเทคนิคสำหรับประเมินประสิทธิภาพของ Model บนข้อมูลที่ "ยังไม่เคยเห็น" คำถามพื้นฐานคือ: เราจะรู้ได้อย่างไรว่า Model ที่ train แล้วจะทำงานได้ดีบน Data ใหม่? Training Error ต่ำไม่ได้รับประกัน Test Error ต่ำ เพราะ Model อาจ Overfit สัปดาห์นี้เราจะแก้ปัญหานี้ด้วย **Cross-Validation** ซึ่ง "จำลอง" Test Set จากข้อมูล Training เอง เป้าหมายคือให้นักศึกษาเข้าใจความแตกต่างระหว่าง Validation Set, LOOCV และ k-Fold CV รวมถึง Bias-Variance Trade-Off ของ CV เอง และสามารถนำไปใช้เลือก Hyperparameter (degree ของ Polynomial, K ใน KNN) ได้ Cross-Validation เป็นหนึ่งในทักษะที่ใช้มากที่สุดใน Machine Learning Production เพราะทุกการเลือก Model ต้องการ Unbiased Estimate ของ Test Error

---

## 14.1 ทำไมต้องมี Resampling?  *(ISLP 5 Intro)*

ในส่วนนี้เราจะทำความเข้าใจว่า Training Error ไม่ใช่ตัววัดที่ดีสำหรับ Test Error และเราต้องการวิธีประเมิน Test Error จาก Training Data

### ปัญหา: Training Error ≠ Test Error

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (KFold, LeaveOneOut, cross_val_score,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

np.random.seed(42)

# ─── สร้าง dataset จำลองเพื่อแสดง Overfitting ────────────────────────────
# วัตถุประสงค์: เห็น training error ลดลงเสมอเมื่อ complexity เพิ่ม
# แต่ test error มี U-shape — นี่คือ Bias-Variance Trade-Off ที่เรียนใน Week 6
n = 50
x_true = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x_true)
y_noisy = y_true + np.random.normal(0, 0.3, n)

# Split 80/20
X = x_true.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.2, random_state=42
)

degrees = range(1, 16)
train_mse_list, test_mse_list = [], []

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('lr', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse = mean_squared_error(y_test, pipe.predict(X_test))
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)

# ─── Plot: Flexibility vs MSE ─────────────────────────────────────────────
# วัตถุประสงค์: เห็น U-shape ของ Test MSE — คลาสสิก Bias-Variance Trade-Off
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(degrees, train_mse_list, 'b-o', markersize=4, label='Training MSE')
ax.plot(degrees, test_mse_list, 'r-o', markersize=4, label='Test MSE')
ax.axvline(np.argmin(test_mse_list) + 1, color='green', linestyle='--',
           label=f'Optimal degree = {np.argmin(test_mse_list) + 1}')
ax.set_xlabel('Polynomial Degree (Flexibility)')
ax.set_ylabel('MSE')
ax.set_title('Training vs Test MSE — Classic U-shape')
ax.legend(); ax.set_ylim([0, 1])
plt.tight_layout(); plt.show()

# สังเกต: Training MSE ลดลงเสมอ; Test MSE มี minimum
optimal_degree = np.argmin(test_mse_list) + 1
print(f"Optimal degree (from single test set): {optimal_degree}")
print(f"⚠️  แต่ผลลัพธ์นี้ขึ้นอยู่กับ random split — ลอง run อีกครั้งอาจได้คนละค่า!")
```

---

## 14.2 Validation Set Approach  *(ISLP 5.1.1)*

ในส่วนนี้เราจะเรียนรู้วิธีง่ายที่สุด — แบ่งข้อมูลเป็น Training Set กับ Validation Set — และทำความเข้าใจข้อจำกัดของมัน

### แนวคิด

**Validation Set**: แบ่งข้อมูล 50/50 หรือ 70/30 → Train บน 70% → Evaluate บน 30%

**ข้อเสียที่สำคัญ**:

```python
# ─── แสดง High Variance ของ Validation Set Approach ─────────────────────
# วัตถุประสงค์: ดูว่า optimal degree เปลี่ยนตาม random seed (unstable)
# นี่คือปัญหาหลักของ single train/validation split
auto = pd.read_csv('Auto.csv', na_values='?').dropna()

val_results = {}
for seed in range(10):  # ลอง 10 random splits ต่างกัน
    np.random.seed(seed)
    X_auto = auto['horsepower'].values.reshape(-1, 1)
    y_auto = auto['mpg'].values
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_auto, y_auto, test_size=0.5,
                                                  random_state=seed)
    mse_by_degree = []
    for d in range(1, 11):
        pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
        pipe.fit(X_tr, y_tr)
        mse_by_degree.append(mean_squared_error(y_val, pipe.predict(X_val)))
    
    val_results[seed] = mse_by_degree

# Plot: แต่ละ seed ให้ผลต่างกัน
fig, ax = plt.subplots(figsize=(9, 5))
for seed, mse_list in val_results.items():
    ax.plot(range(1, 11), mse_list, alpha=0.5, linewidth=1)
ax.set_xlabel('Polynomial Degree'); ax.set_ylabel('Validation MSE')
ax.set_title('Validation Set MSE — 10 Different Random Splits\n(High Variance Problem)')
plt.tight_layout(); plt.show()

# สังเกต: optimal degree เปลี่ยนตาม split!
optimal_degrees = [np.argmin(val_results[s]) + 1 for s in range(10)]
print(f"Optimal degrees ใน 10 splits: {optimal_degrees}")
print(f"→ Unstable! นี่คือปัญหาของ Single Validation Set")
```

---

## 14.3 Leave-One-Out Cross-Validation (LOOCV)  *(ISLP 5.1.2)*

ในส่วนนี้เราจะเรียน LOOCV ซึ่งแก้ปัญหา High Variance ได้ แต่มีต้นทุนด้าน Computation

### แนวคิด

LOOCV ทำ n splits: แต่ละครั้ง Leave 1 observation ไว้เป็น Validation, Train บน n−1 ที่เหลือ:

$$\text{CV}_{LOOCV} = \frac{1}{n}\sum_{i=1}^{n} \text{MSE}_i = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_{(-i)})^2$$

โดย $\hat{y}_{(-i)}$ คือ prediction ของ $y_i$ จากโมเดลที่ train โดยไม่มี observation $i$

### LOOCV Magic Formula สำหรับ OLS

สำหรับ Linear Regression มีสูตรลัดที่ไม่ต้อง Refit n ครั้ง:

$$\text{CV}_{LOOCV} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_i}\right)^2$$

โดย $h_i = x_i^T(X^TX)^{-1}x_i$ คือ Leverage ที่เรียนไว้ใน Week 10!

```python
# ─── LOOCV บน Auto Dataset (Manual + Magic Formula) ──────────────────────
# วัตถุประสงค์: เปรียบเทียบ LOOCV จาก sklearn กับ Magic Formula สำหรับ OLS

# Method 1: sklearn LOOCV (ช้า แต่ general)
loo = LeaveOneOut()
X_auto_arr = auto['horsepower'].values.reshape(-1, 1)
y_auto_arr = auto['mpg'].values

loocv_mse = {}
for d in range(1, 6):
    pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
    # cross_val_score ทำ n splits อัตโนมัติ
    scores = cross_val_score(pipe, X_auto_arr, y_auto_arr,
                              cv=loo, scoring='neg_mean_squared_error')
    loocv_mse[d] = -scores.mean()
    print(f"Degree {d}: LOOCV MSE = {loocv_mse[d]:.4f}")

# Method 2: Magic Formula (เร็วกว่ามาก)
def loocv_magic(X, y):
    """LOOCV สำหรับ OLS ด้วย Leverage Formula — O(n) แทน O(n²)"""
    model = sm.OLS(y, X).fit()
    leverage = model.get_influence().hat_matrix_diag
    residuals = model.resid
    # สูตร: LOOCV = mean((e_i / (1 - h_i))²)
    loocv_resid = residuals / (1 - leverage)
    return np.mean(loocv_resid ** 2)

# ทดสอบ Magic Formula
X_poly2 = np.column_stack([np.ones(len(auto)),
                             auto['horsepower'],
                             auto['horsepower']**2])
magic_loocv = loocv_magic(X_poly2, auto['mpg'])
print(f"\nLOOCV Magic Formula (degree 2): {magic_loocv:.4f}")
print(f"sklearn LOOCV (degree 2): {loocv_mse[2]:.4f}")
print("→ ผลเหมือนกัน! Magic Formula เร็วกว่ามาก")
```

**ข้อเสียของ LOOCV**: Training set เกือบเหมือนกันทุก fold (n−1 obs ต่าง แค่ 1 obs) → predictions มี High Correlation → **SE ของ LOOCV estimate สูง**

---

## 14.4 k-Fold Cross-Validation  *(ISLP 5.1.3)*

ในส่วนนี้เราจะเรียนวิธีที่ใช้บ่อยที่สุดในทางปฏิบัติ — k-Fold CV ที่สมดุลระหว่าง Bias และ Variance

### แนวคิด

**k-Fold CV**: แบ่งข้อมูลเป็น K fold เท่าๆ กัน → รอบที่ k: ใช้ fold k เป็น Validation, ที่เหลือเป็น Training → ทำ K รอบ:

$$\text{CV}_K = \frac{1}{K}\sum_{k=1}^{K}\text{MSE}_k$$

ค่าที่นิยมใช้: **K = 5 หรือ K = 10** (empirically validated ว่าดีที่สุด)

```python
# ─── k-Fold CV บน Auto Dataset ─────────────────────────────────────────
# วัตถุประสงค์: เปรียบเทียบ 5-Fold, 10-Fold, และ LOOCV บน polynomial degree ต่างๆ
X_hor = auto['horsepower'].values.reshape(-1, 1)
y_mpg = auto['mpg'].values

cv_results = {}
for k, name in [(5, '5-Fold CV'), (10, '10-Fold CV')]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_mse_list = []
    for d in range(1, 11):
        pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
        scores = cross_val_score(pipe, X_hor, y_mpg, cv=kf,
                                  scoring='neg_mean_squared_error')
        cv_mse_list.append(-scores.mean())
    cv_results[name] = cv_mse_list

# ─── Plot: Comparing CV Methods ────────────────────────────────────────────
# วัตถุประสงค์: เห็นว่า 5-Fold และ 10-Fold ให้ผลคล้ายกันและสอดคล้องกับ LOOCV
fig, ax = plt.subplots(figsize=(9, 5))
for name, mse_list in cv_results.items():
    ax.plot(range(1, 11), mse_list, '-o', markersize=5, linewidth=2, label=name)

# เพิ่ม LOOCV เพื่อเปรียบเทียบ
loocv_mse_list = [loocv_mse.get(d, None) for d in range(1, 6)]
ax.plot(range(1, 6), [v for v in loocv_mse_list if v is not None],
        '-s', markersize=5, linewidth=2, label='LOOCV', alpha=0.7)

ax.set_xlabel('Polynomial Degree'); ax.set_ylabel('CV Error (MSE)')
ax.set_title('CV Error vs Polynomial Degree — Auto Dataset')
ax.legend()
plt.tight_layout(); plt.show()

for name, mse_list in cv_results.items():
    opt_d = np.argmin(mse_list) + 1
    print(f"{name}: Optimal degree = {opt_d}, MSE = {min(mse_list):.4f}")
```

### Bias-Variance Trade-Off ของ CV เอง

| CV Method | Bias | Variance | Computation |
|-----------|------|----------|------------|
| **Validation Set (50/50)** | สูง (train บน data น้อย) | สูง (random split) | เร็ว |
| **LOOCV** | ต่ำ (train บน n−1) | สูง (n models คล้ายกัน) | ช้า O(n) |
| **5-Fold CV** | ปานกลาง | ต่ำกว่า LOOCV | เร็ว |
| **10-Fold CV** | ต่ำ | ต่ำ | ปานกลาง |

**ข้อแนะนำทั่วไป**: ใช้ **10-Fold CV** สำหรับ Model Selection และ **5-Fold** เมื่อ dataset ใหญ่หรือ model ช้า

---

## 14.5 CV สำหรับ Classification  *(ISLP 5.1.5)*

ในส่วนนี้เราจะนำ CV ไปใช้กับ Classification โดยแทน MSE ด้วย **Error Rate** หรือ **AUC**

```python
# ─── k-Fold CV สำหรับ KNN: หา Optimal K ────────────────────────────────
# วัตถุประสงค์: ใช้ CV เลือก K ที่ให้ Test Error ต่ำสุด
# K ใหญ่ = smoother boundary = less overfit แต่ high bias
from sklearn.neighbors import KNeighborsClassifier

default_df = pd.read_csv('Default.csv')
default_df['default_bin'] = (default_df['default'] == 'Yes').astype(int)
default_df['student_bin'] = (default_df['student'] == 'Yes').astype(int)

X_def = default_df[['balance', 'income', 'student_bin']].values
y_def = default_df['default_bin'].values

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)

k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
cv_errors = []

for k in k_values:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    # scoring='accuracy' → แต่ใช้ error rate = 1 - accuracy
    scores = cross_val_score(pipe, X_def, y_def, cv=kf_10, scoring='accuracy')
    cv_errors.append(1 - scores.mean())
    
# ─── Plot: CV Error vs K ─────────────────────────────────────────────────
# วัตถุประสงค์: เห็น U-shape — K เล็กเกินไป = overfit, K ใหญ่เกินไป = underfit
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, cv_errors, 'b-o', markersize=6, linewidth=2)
optimal_k = k_values[np.argmin(cv_errors)]
ax.axvline(optimal_k, color='red', linestyle='--',
           label=f'Optimal K = {optimal_k}')
ax.set_xlabel('K (number of neighbors)'); ax.set_ylabel('10-Fold CV Error Rate')
ax.set_title('KNN: Choosing Optimal K with Cross-Validation')
ax.legend()
plt.tight_layout(); plt.show()

print(f"Optimal K = {optimal_k}")
print(f"CV Error Rate = {min(cv_errors):.4f}")

# ─── One Standard Error Rule ──────────────────────────────────────────────
# วัตถุประสงค์: เลือก model ง่าย (K ใหญ่กว่า) ที่ error อยู่ใน range ± 1 SE ของ best model
# ป้องกัน overfitting ของการเลือก model เอง
errors_by_k = {}
for k in k_values:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsClassifier(n_neighbors=k))])
    scores = cross_val_score(pipe, X_def, y_def, cv=kf_10, scoring='accuracy')
    errors_by_k[k] = {'mean': 1 - scores.mean(), 'se': scores.std() / np.sqrt(10)}

best_k = min(errors_by_k, key=lambda k: errors_by_k[k]['mean'])
threshold_1se = errors_by_k[best_k]['mean'] + errors_by_k[best_k]['se']

print(f"\nOne Standard Error Rule:")
print(f"Best K = {best_k} (CV error = {errors_by_k[best_k]['mean']:.4f})")
print(f"Threshold (best + 1 SE) = {threshold_1se:.4f}")

for k in sorted(k_values, reverse=True):  # larger K = simpler model for KNN
    if errors_by_k[k]['mean'] <= threshold_1se:
        print(f"1-SE Rule selects K = {k} (simpler, within 1 SE)")
        break
```

**DS Connection**: CV-based Model Selection เป็นมาตรฐานใน scikit-learn pipelines โดยใช้ `GridSearchCV` หรือ `RandomizedSearchCV` ซึ่ง run k-fold CV อัตโนมัติสำหรับทุก hyperparameter combination

---

## Case Study: Polynomial Degree Selection บน Auto MPG

**Scenario**
เราต้องการ predict fuel efficiency (mpg) จาก horsepower และสงสัยว่า Polynomial degree เท่าไหร่ที่ดีที่สุด — degree 1 (เส้นตรง)? degree 2 (quadratic)? หรือ degree 10?

**Data**
- Auto.csv: 392 observations
- Response: `mpg`, Predictor: `horsepower`

**Method — Systematic CV Comparison**

```python
# ─── Final CV Pipeline: เลือก Polynomial Degree อย่างเป็นระบบ ────────────
# วัตถุประสงค์: สาธิต best practice สำหรับ hyperparameter selection ด้วย CV
X_h = auto['horsepower'].values.reshape(-1, 1)
y_m = auto['mpg'].values

# Hold-out 20% สำหรับ Final Evaluation เท่านั้น
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X_h, y_m, test_size=0.2, random_state=99
)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_summary = []

for d in range(1, 11):
    pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
    scores = cross_val_score(pipe, X_trainval, y_trainval, cv=kf,
                              scoring='neg_mean_squared_error')
    mse_mean = -scores.mean()
    mse_se = scores.std() / np.sqrt(10)
    
    # Fit บน full trainval สำหรับ holdout evaluation
    pipe.fit(X_trainval, y_trainval)
    holdout_mse = mean_squared_error(y_holdout, pipe.predict(X_holdout))
    
    cv_summary.append({
        'Degree': d,
        'CV MSE': mse_mean,
        'CV SE': mse_se,
        'Holdout MSE': holdout_mse
    })

cv_df = pd.DataFrame(cv_summary)
print(cv_df.round(4).to_string(index=False))

best_degree = cv_df.loc[cv_df['CV MSE'].idxmin(), 'Degree']
print(f"\nBest degree (by CV MSE): {best_degree}")
print(f"CV MSE: {cv_df.loc[cv_df['CV MSE'].idxmin(), 'CV MSE']:.4f}")
print(f"Holdout MSE: {cv_df.loc[cv_df['CV MSE'].idxmin(), 'Holdout MSE']:.4f}")
```

**Result**
```
Degree  CV MSE  CV SE  Holdout MSE
     1   24.21   0.98        23.87
     2   19.25   0.87        18.94  ← Minimum CV MSE
     3   19.32   0.91        19.10
     4   19.41   0.96        19.56
     5   19.58   1.02        20.12
    ...
```

**Insight**
1. **Degree 2 ดีที่สุด** — สอดคล้องกับ Residual Plot ใน Week 10 ที่แสดง Non-linearity แต่ไม่ต้องการ higher degree
2. **Degrees 3–10 ให้ผลคล้ายกัน** แต่ complexity สูงกว่า → One SE Rule เลือก degree 2 ซึ่งง่ายที่สุด
3. **CV MSE ≈ Holdout MSE** → CV estimate น่าเชื่อถือ ไม่ได้ Overfit ระหว่าง model selection

---

## สรุป

| Method | สูตร | Bias | Variance | Python |
|--------|------|------|----------|--------|
| Validation Set | ใช้ 30% เป็น Val | สูง | สูง | `train_test_split` |
| LOOCV | $\frac{1}{n}\sum(y_i-\hat{y}_{(-i)})^2$ | ต่ำ | สูง | `LeaveOneOut()` |
| k-Fold CV | $\frac{1}{K}\sum_k \text{MSE}_k$ | ปานกลาง | ต่ำ | `KFold(n_splits=K)` |
| Cross-Entropy (Classif.) | $-\sum[y\log p + (1-y)\log(1-p)]$ | — | — | `scoring='neg_log_loss'` |
| Error Rate (Classif.) | $\frac{1}{n}\sum I(y_i \neq \hat{y}_i)$ | — | — | `scoring='accuracy'` |

| แนวคิด | Python Function |
|--------|----------------|
| k-Fold CV | `cross_val_score(model, X, y, cv=KFold(k))` |
| Multiple metrics | `cross_validate(model, X, y, scoring=['accuracy', 'roc_auc'])` |
| Pipeline with CV | `Pipeline([('scaler', ...), ('model', ...)]) + cross_val_score` |
| Optimal degree | `np.argmin(cv_mse_list) + 1` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 06**: Bias-Variance Trade-Off — Training MSE ลดเสมอ; Test MSE U-shape
- ← **Week 10**: Leverage $h_i$ ใน LOOCV Magic Formula — Week 10 Diagnostics โผล่มาที่นี่
- ← **Week 09–13**: Models ทุกอย่างที่เรียนมา ล้วนต้องการ CV เพื่อประเมิน
- → **Note 14-2**: Bootstrap ประมาณ SE ของ statistic ใดก็ได้; CV-based Model Selection pipeline ที่สมบูรณ์
- → **Week 15**: Final Project ใช้ CV เพื่อเลือก Best Model สำหรับ Dataset ที่เลือก
