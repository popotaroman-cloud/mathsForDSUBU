# Slide Deck 2: LOOCV and k-Fold Cross-Validation
> Week 14 | 9 slides | CLO4

---

## Slide 1 — Title

**LOOCV + k-Fold Cross-Validation**  
ISLP Ch.5.1 | Week 14 | CLO4

---

## Slide 2 — Leave-One-Out CV (LOOCV)

**Key Message**: LOOCV ใช้ทุก observation เป็น validation ทีละ 1 — low bias, deterministic

**Algorithm:**
1. For i = 1, 2, ..., n:
   - Train model บน {(x₁,y₁), ..., (xᵢ₋₁,yᵢ₋₁), (xᵢ₊₁,yᵢ₊₁), ..., (xₙ,yₙ)} (n−1 points)
   - Evaluate on (xᵢ, yᵢ): compute MSEᵢ = (yᵢ − ŷᵢ)²
2. LOOCV Error = (1/n) Σᵢ MSEᵢ

**Properties:**
- **Deterministic**: ไม่มี random split → same result ทุกครั้ง
- **Low Bias**: train on n−1 ≈ n points → เกือบเหมือน full dataset
- **High Variance**: n models แต่ละตัว trained on nearly same data → highly correlated
- **Expensive**: n model fits (n อาจ = 10,000+)

**[FIGURE: n=8 example — 8 iterations, แต่ละ row highlight observation ที่เป็น validation (●) rest เป็น train (○)]**

---

## Slide 3 — LOOCV Magic Formula for OLS

**Key Message**: สำหรับ OLS ไม่ต้องทำ n fits! มีสูตรปิดที่คำนวณได้ใน O(n) เดียว

**Magic Formula:**
$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_i}\right)^2$$

- ŷᵢ = prediction จาก model ที่ fit บน **all n** observations
- hᵢ = **leverage** ของ observation i (จาก hat matrix H = X(XᵀX)⁻¹Xᵀ)
- hᵢ ∈ [1/n, 1], hᵢ สูง = observation มีอิทธิพลมาก

**Interpretation:**
- เมื่อ hᵢ สูง → (1−hᵢ) เล็ก → MSEᵢ ใหญ่ขึ้น → high-leverage observation ส่งผลมากกว่า
- สอดคล้องกับ Week 10 (Diagnostics): leverage เชื่อมกับ LOOCV!

**Python:**
```python
from statsmodels.regression.linear_model import OLS
model = OLS(y, sm.add_constant(X)).fit()
influence = model.get_influence()
h = influence.hat_matrix_diag  # leverage values
loocv_mse = np.mean(((y - model.fittedvalues) / (1 - h))**2)
```

---

## Slide 4 — k-Fold Cross-Validation

**Key Message**: k-Fold แบ่งข้อมูลเป็น K กลุ่ม แล้ว rotate validation group — practical สุด

**Algorithm:**
1. แบ่ง data เป็น K folds ขนาด n/K เท่าๆ กัน
2. For k = 1, 2, ..., K:
   - Train: folds ทั้งหมดยกเว้น fold k
   - Validate: fold k → MSEₖ
3. k-Fold CV = (1/K) Σₖ MSEₖ

**สำหรับ Classification:**
$$\text{CV}_{K} = \frac{1}{K}\sum_{k=1}^{K}\text{Error Rate}_k = \frac{1}{K}\sum_{k=1}^K \frac{\sum_{i \in F_k} \mathbb{I}(y_i \neq \hat{y}_i)}{|F_k|}$$

**[FIGURE: k=5 diagram — 5 rows, แต่ละ row: 4 blue (train) + 1 orange (validate) rotating]**

---

## Slide 5 — CV Curve: Model Selection

**Key Message**: CV curve บอกว่า complexity ไหน minimize test error

**Example: Polynomial Degree Selection**

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

degrees = range(1, 11)
cv_mse = []
for deg in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('lr', LinearRegression())
    ])
    scores = cross_val_score(pipe, X, y, cv=KFold(10, shuffle=True, random_state=42),
                             scoring='neg_mean_squared_error')
    cv_mse.append(-scores.mean())

plt.plot(degrees, cv_mse, 'o-')
plt.xlabel('Degree'); plt.ylabel('CV MSE')
```

**[FIGURE: U-shaped CV MSE curve: MSE สูงที่ degree 1 (underfitting), ลดต่ำที่ degree 2–3 (just right), เพิ่มอีกที่ degree 8+ (overfitting)]**

---

## Slide 6 — One Standard Error Rule

**Key Message**: เลือก simpler model ที่ยังอยู่ใน statistical noise ของ best model

**Motivation:**
- Best CV score อาจแตกต่างกันนิดเดียวระหว่าง degree 2 และ degree 3
- ถ้าความแตกต่างไม่ significant → เลือก degree 2 (simpler, more interpretable)

**One-SE Rule:**
$$\text{Select smallest model with } \text{CV}(m) \leq \min_j \text{CV}(j) + \text{SE}_j$$

**Algorithm:**
```python
cv_mean = np.array(cv_means)   # CV MSE ของแต่ละ degree
cv_se   = np.array(cv_ses)     # SE ของแต่ละ degree

best_idx = np.argmin(cv_mean)
threshold = cv_mean[best_idx] + cv_se[best_idx]  # best + 1 SE

# One-SE rule: smallest model ที่ CV ≤ threshold
one_se_idx = next(i for i, m in enumerate(cv_mean) if m <= threshold)
```

**[FIGURE: CV MSE plot with error bars + horizontal red dashed line at best+1SE + arrows showing 'best' and 'one-SE rule' selections]**

---

## Slide 7 — LOOCV for Classification

**Key Message**: LOOCV และ k-Fold ใช้กับ classification ได้เช่นกัน — แค่เปลี่ยน metric

**Classification CV Error:**
$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(y_i \neq \hat{y}_i)$$

**Python:**
```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
loocv_error = 1 - cross_val_score(
    clf, X, y, cv=loo,
    scoring='accuracy'
).mean()
print(f'LOOCV Error Rate: {loocv_error:.4f}')

# สำหรับ imbalanced: ใช้ F1 หรือ AUC
loocv_f1 = cross_val_score(clf, X, y, cv=loo, scoring='f1').mean()
```

**เมื่อใช้ LOOCV กับ classification:**
- ดีสำหรับ n เล็ก (<200)
- n ใหญ่ → ใช้ Stratified 10-Fold แทน (เร็วกว่ามาก)
- LOOCV ไม่สามารถ stratify ได้ตาม definition

---

## Slide 8 — CV vs Training Error: Visualization

**Key Message**: CV curve ตรงกับ Bias-Variance U-shape — เห็นชัดกว่า training curve

**[FIGURE: 2 plots side-by-side]**

**Left — Without CV:**
- Training MSE: ลดเรื่อยๆ เมื่อ degree เพิ่ม → misleading
- ไม่รู้ว่าจะ overfit ตรงไหน

**Right — With CV:**
- CV MSE: U-shape ชัดเจน
- Minimum อยู่ที่ degree 2–3
- Degree 7+: CV MSE สูงขึ้น = overfitting visible!

**ข้อสังเกต:**
- Training MSE ≤ CV MSE เสมอ (เพราะ model fit บน training data)
- Gap ระหว่าง Training MSE กับ CV MSE = **degree of overfitting**
- Large gap → overfit → ต้องใช้ simpler model หรือ regularization

---

## Slide 9 — Summary: LOOCV vs k-Fold

**Key Message**: k=10 fold เป็น default ที่ดีที่สุดสำหรับ most situations

**Comparison:**

| | LOOCV | 5-Fold | 10-Fold |
|-|-------|--------|---------|
| Bias | Lowest | Moderate | Low |
| Variance | High | Low | Low |
| Cost | O(n²) fit | O(5n) fit | O(10n) fit |
| Deterministic | ✓ | ✗ (random) | ✗ (random) |
| Use when | n < 100 | n large | Default choice |

**เลือก:**
- **n ≤ 50**: LOOCV
- **n = 100–1000**: 10-Fold CV
- **n > 1000**: 5-Fold CV (เร็วกว่า, เพียงพอ)
- **Imbalanced**: Stratified k-Fold เสมอ
- **Time series**: Time-based split (ไม่ใช้ random)

**Python Quick Reference:**
```python
KFold(n_splits=10, shuffle=True, random_state=42)           # regression
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # classification
LeaveOneOut()                                                 # small n
```
