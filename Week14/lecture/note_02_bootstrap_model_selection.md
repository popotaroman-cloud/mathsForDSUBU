# Note 14-2: Bootstrap & CV-Based Model Selection Pipeline
> Week 14 | CLO4 | ISLP Reference: Ch.5.2 Bootstrap + Ch.5.1.3–5.1.5

Note ที่ 2 ของสัปดาห์นี้เรียนรู้ **Bootstrap** ซึ่งเป็น Resampling Method ที่ทรงพลังที่สุดสำหรับ Estimating Standard Errors ของ Statistic ที่ซับซ้อน ไม่ว่าจะเป็น Coefficient ของ Regression, Correlation, หรือแม้แต่ Portfolio Weights Bootstrap ทำงานโดยการ Sampling with Replacement จากข้อมูลเดิม เพื่อจำลองว่า "ถ้าเราได้ Sample ใหม่จาก Population" ผลจะเป็นอย่างไร จากนั้นเราจะนำ CV มาประกอบเป็น **Full Model Selection Pipeline** ที่สมบูรณ์ ซึ่งเป็น Workflow ที่ Data Scientist ใช้จริงในการตัดสินใจเลือก Model เป้าหมายคือให้นักศึกษาสามารถประมาณ SE ด้วย Bootstrap และสร้าง Systematic CV Pipeline สำหรับเปรียบเทียบ Models ได้

---

## 14.6 The Bootstrap  *(ISLP 5.2)*

ในส่วนนี้เราจะเรียนรู้ Bootstrap ซึ่งเป็นวิธี "Universal" ที่ประเมิน SE ของ statistic ใดๆ ได้โดยไม่ต้องมีสูตร Analytical

### แนวคิดหลัก

**ปัญหา**: SE ของ β̂ ใน Linear Regression มีสูตร $\text{SE}(\hat{\beta}) = \hat{\sigma}\sqrt{(X^TX)^{-1}_{jj}}$ แต่ สำหรับ statistic ที่ซับซ้อน (เช่น Median, Quantile, Portfolio Weights) ไม่มีสูตร Closed-form

**Bootstrap Solution**:
1. จาก dataset ขนาด n สุ่ม n observations **with replacement** → Bootstrap Sample $Z^{*b}$
2. คำนวณ statistic ที่สนใจจาก $Z^{*b}$ → ได้ $\hat{\alpha}^{*b}$
3. ทำซ้ำ B ครั้ง (B = 1000 ทั่วไป)
4. SE ≈ Standard Deviation ของ B Bootstrap estimates

$$\widehat{\text{SE}}_B(\hat{\alpha}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\hat{\alpha}^{*b} - \bar{\hat{\alpha}}^*\right)^2}$$

### Worked Example: Investment Portfolio

ISLP ใช้ตัวอย่าง: ลงทุนใน Asset X และ Y โดยสัดส่วน α ใน X และ 1-α ใน Y เพื่อ **minimize variance ของ return**:

$$\alpha = \frac{\sigma_Y^2 - \sigma_{XY}}{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}}$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ─── Bootstrap สำหรับ Portfolio Optimization ─────────────────────────────
# วัตถุประสงค์: ประมาณ SE ของ optimal α โดยไม่มีสูตร Analytical
# แสดงว่า Bootstrap ทำงานได้แม้กับ complex statistics

# สร้างข้อมูล Return จำลอง
n = 100
returns_X = np.random.normal(0.01, 0.05, n)
returns_Y = np.random.normal(0.01, 0.03, n) + 0.5 * returns_X  # X,Y correlate
returns = pd.DataFrame({'X': returns_X, 'Y': returns_Y})

def optimal_alpha(X, Y):
    """คำนวณ α ที่ minimize portfolio variance"""
    sigma_X2 = np.var(X, ddof=1)
    sigma_Y2 = np.var(Y, ddof=1)
    sigma_XY = np.cov(X, Y, ddof=1)[0, 1]
    return (sigma_Y2 - sigma_XY) / (sigma_X2 + sigma_Y2 - 2 * sigma_XY)

# Original estimate
alpha_hat = optimal_alpha(returns['X'], returns['Y'])
print(f"Original α̂ = {alpha_hat:.4f}")

# ─── Bootstrap: 1000 iterations ───────────────────────────────────────────
# วัตถุประสงค์: จำลอง sampling distribution ของ α̂ ด้วย resampling
B = 1000
bootstrap_alphas = []

for _ in range(B):
    # Resample with replacement
    boot_sample = returns.sample(n=n, replace=True, random_state=None)
    alpha_b = optimal_alpha(boot_sample['X'], boot_sample['Y'])
    bootstrap_alphas.append(alpha_b)

bootstrap_alphas = np.array(bootstrap_alphas)

# Bootstrap SE
boot_se = np.std(bootstrap_alphas, ddof=1)
# Bootstrap CI (percentile method)
ci_lower, ci_upper = np.percentile(bootstrap_alphas, [2.5, 97.5])

print(f"Bootstrap SE(α̂) = {boot_se:.4f}")
print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ─── Visualize Bootstrap Distribution ────────────────────────────────────
# วัตถุประสงค์: เห็น sampling distribution ที่ Bootstrap จำลองขึ้น
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(bootstrap_alphas, bins=40, density=True, alpha=0.7, color='steelblue',
        label='Bootstrap distribution')
ax.axvline(alpha_hat, color='red', linewidth=2, label=f'α̂ = {alpha_hat:.3f}')
ax.axvline(ci_lower, color='orange', linestyle='--', label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
ax.axvline(ci_upper, color='orange', linestyle='--')
ax.set_xlabel('Bootstrap α̂*'); ax.set_ylabel('Density')
ax.set_title(f'Bootstrap Distribution of α̂ (B={B})\nSE = {boot_se:.4f}')
ax.legend()
plt.tight_layout(); plt.show()
```

### Bootstrap vs Analytical SE สำหรับ Regression

```python
# ─── Bootstrap SE เปรียบเทียบกับ statsmodels SE ──────────────────────────
# วัตถุประสงค์: ตรวจสอบว่า Bootstrap SE สอดคล้องกับ Analytical SE ใน OLS
import statsmodels.api as sm

auto = pd.read_csv('Auto.csv', na_values='?').dropna()
X_bs = sm.add_constant(auto['horsepower'])
y_bs = auto['mpg']

# Analytical SE
model_sm = sm.OLS(y_bs, X_bs).fit()
print("Analytical SE (statsmodels):")
print(model_sm.bse.round(6))

# ─── Bootstrap SE ─────────────────────────────────────────────────────────
# วัตถุประสงค์: แสดงว่า Bootstrap ได้ SE ใกล้เคียงกันโดยไม่ใช้สูตร
B = 1000
boot_coefs = []
data_combined = pd.concat([X_bs, y_bs], axis=1)

for _ in range(B):
    boot_df = data_combined.sample(n=len(data_combined), replace=True)
    X_b = boot_df[['const', 'horsepower']]
    y_b = boot_df['mpg']
    coefs = sm.OLS(y_b, X_b).fit().params
    boot_coefs.append(coefs.values)

boot_coefs = np.array(boot_coefs)
boot_se_df = pd.Series(boot_coefs.std(axis=0, ddof=1), 
                        index=['const', 'horsepower'])
print("\nBootstrap SE:")
print(boot_se_df.round(6))
print(f"\nRatio Bootstrap SE / Analytical SE:")
print((boot_se_df / model_sm.bse).round(4))
# ค่า ≈ 1 แสดงว่า Bootstrap ทำงานถูกต้อง
```

### ทำไม Bootstrap SE อาจต่างจาก Analytical SE?

Bootstrap SE ต่างจาก statsmodels SE ได้เมื่อ:
1. OLS Assumptions ไม่ครบ (Heteroscedasticity, Correlation) — Bootstrap มักแม่นยำกว่า
2. Sample Size เล็ก — Bootstrap CI อาจ accurate กว่า Normal approximation
3. Statistic ซับซ้อน — Bootstrap ทำได้ Analytical ไม่ได้

**DS Connection**: Bootstrap ใช้กันมากใน:
- **A/B Testing**: CI ของ metric ที่ไม่ Normal (conversion rate, revenue)
- **Finance**: Portfolio optimization, Risk estimation (VaR)
- **ML**: Bagging และ Random Forest ใช้ Bootstrap sampling เป็นพื้นฐาน

---

## 14.7 Full Model Selection Pipeline  *(Best Practice)*

ในส่วนนี้เราจะรวมทุกอย่างเป็น Systematic Pipeline สำหรับการเลือก Model ที่ดีที่สุด

### Pipeline สมบูรณ์แบบ

```python
# ─── Full Model Selection Pipeline ──────────────────────────────────────
# วัตถุประสงค์: สาธิต Production-grade workflow สำหรับ Model Selection
# ขั้นตอน: Hold-out split → Inner CV → Grid Search → Evaluate on Hold-out

from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─── 1. โหลดและ Preprocess ───────────────────────────────────────────────
# วัตถุประสงค์: เตรียม features ให้พร้อมก่อน model selection
auto = pd.read_csv('Auto.csv', na_values='?').dropna()
X_full = auto[['horsepower', 'weight', 'cylinders', 'year']].values
y_full = auto['mpg'].values

# ─── 2. Hold-Out Split (20% สำหรับ Final Test เท่านั้น) ──────────────────
# วัตถุประสงค์: แยก hold-out set ออกก่อน — ห้ามใช้ระหว่าง model selection
from sklearn.model_selection import train_test_split
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
print(f"Train+Val: {len(X_trainval)}, Final Test: {len(X_test_final)}")

# ─── 3. Grid Search + Inner 10-Fold CV ────────────────────────────────────
# วัตถุประสงค์: ค้นหา hyperparameter ที่ดีที่สุดด้วย CV โดยไม่แตะ test set
param_grid = {'poly__degree': [1, 2, 3, 4, 5]}

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('lr', LinearRegression())
])

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=10,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)
grid_search.fit(X_trainval, y_trainval)

# ─── 4. ผลลัพธ์ Grid Search ────────────────────────────────────────────
# วัตถุประสงค์: สรุป CV MSE ทุก hyperparameter เพื่อ comparison
cv_results_df = pd.DataFrame({
    'Degree': [1, 2, 3, 4, 5],
    'Train MSE': -grid_search.cv_results_['mean_train_score'],
    'CV MSE': -grid_search.cv_results_['mean_test_score'],
    'CV SE': grid_search.cv_results_['std_test_score']
}).round(4)

print("\nGrid Search Results:")
print(cv_results_df.to_string(index=False))

best_degree = grid_search.best_params_['poly__degree']
best_cv_mse = -grid_search.best_score_
print(f"\nBest degree = {best_degree}, CV MSE = {best_cv_mse:.4f}")

# ─── 5. Evaluate on Hold-out Test Set ────────────────────────────────────
# วัตถุประสงค์: ประเมิน FINAL performance บน data ที่ไม่เคยใช้เลย
best_model = grid_search.best_estimator_
final_mse = mean_squared_error(y_test_final, best_model.predict(X_test_final))
print(f"Final Test MSE = {final_mse:.4f}")
print(f"Final Test RMSE = {np.sqrt(final_mse):.4f} mpg")

# ─── 6. CV Curve Visualization ────────────────────────────────────────────
# วัตถุประสงค์: เห็น Trade-off ระหว่าง complexity และ CV error
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(cv_results_df['Degree'], cv_results_df['CV MSE'],
            yerr=cv_results_df['CV SE'], fmt='-o', capsize=5,
            color='blue', label='CV MSE ± SE')
ax.plot(cv_results_df['Degree'], cv_results_df['Train MSE'], 
        '--s', color='red', label='Train MSE')
ax.axvline(best_degree, color='green', linestyle='--',
           label=f'Best degree = {best_degree}')
ax.set_xlabel('Polynomial Degree'); ax.set_ylabel('MSE')
ax.set_title('Model Selection: CV MSE vs Polynomial Degree\n(with One-SE Rule guidance)')
ax.legend()
plt.tight_layout(); plt.show()
```

### Nested Cross-Validation

เมื่อต้องการ **Unbiased Estimate ของ Test Error** หลัง Model Selection:

```python
# ─── Nested CV: Outer CV สำหรับ Test Error, Inner CV สำหรับ Model Selection ─
# วัตถุประสงค์: ป้องกัน optimism bias — ถ้าใช้ CV เลือก model แล้ว evaluate ด้วย CV เดิม
# ผลจะ optimistically biased Nested CV แก้ปัญหานี้

outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = []

for train_idx, test_idx in outer_kf.split(X_trainval):
    X_outer_train, X_outer_test = X_trainval[train_idx], X_trainval[test_idx]
    y_outer_train, y_outer_test = y_trainval[train_idx], y_trainval[test_idx]
    
    # Inner CV: เลือก best degree
    inner_grid = GridSearchCV(pipe, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    inner_grid.fit(X_outer_train, y_outer_train)
    
    # Evaluate on outer fold
    outer_score = mean_squared_error(y_outer_test,
                                      inner_grid.predict(X_outer_test))
    nested_scores.append(outer_score)

print(f"\nNested CV MSE: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
print(f"Simple CV MSE: {best_cv_mse:.4f}")
print("(Nested CV = more honest estimate of generalization error)")
```

---

## สรุป: Bootstrap + Full Pipeline

| แนวคิด | สูตร/Method | Python |
|--------|------------|--------|
| Bootstrap SE | $\sqrt{\frac{1}{B-1}\sum(\hat{\alpha}^{*b} - \bar{\hat{\alpha}}^*)^2}$ | `resample(data, replace=True)` |
| Bootstrap CI | Percentile ของ Bootstrap distribution | `np.percentile(boot_stats, [2.5, 97.5])` |
| Grid Search + CV | ค้นหา best hyperparameter | `GridSearchCV(pipe, param_grid, cv=K)` |
| Nested CV | Outer CV evaluate, Inner CV select | 2-level loop |
| One-SE Rule | Best CV + 1 SE → simpler model | `best_mean + best_se` |

### Model Selection Workflow (ใช้ใน Final Project)

```
1. Load + EDA → เข้าใจ data
2. Hold-out split → 20% สำหรับ final test (ห้ามแตะ)
3. Preprocessing pipeline → StandardScaler, PolynomialFeatures, encoding
4. GridSearchCV + 10-Fold CV → ค้นหา best hyperparameter
5. เลือก model → best CV score หรือ One-SE Rule
6. Retrain บน X_trainval ด้วย best params
7. Final evaluation บน X_test_final
8. Report: CV MSE ± SE + Final Test MSE + interpretation
```

---

## เชื่อมกับสัปดาห์อื่น

- ← **Note 14-1**: k-Fold CV เป็น "Inner" CV ใน Pipeline นี้
- ← **Week 06**: Bias-Variance เป็น motivation ว่าทำไมต้องมี CV — training error ≠ test error
- ← **Week 10**: Bootstrap ใน ML → **Bagging** (Bootstrap Aggregating) และ **Random Forest** (เรียนต่อใน Advanced ML)
- → **Week 15**: นักศึกษาใช้ Pipeline นี้ใน Final Project เพื่อเลือก Best Model สำหรับ Dataset ที่เลือก
