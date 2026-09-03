# Slide Deck: Multicollinearity และ VIF
> Week 10 | CLO3 | ISLP Ch.3.3.3 | 9 slides

---
## Slide 1 — Title
**Multicollinearity: ตรวจและแก้ปัญหา Correlated Predictors**  
Week 10 | CLO3 | ISLP 3.3.3 (Problem 6)  
LLo: ตรวจ multicollinearity ด้วย VIF และเสนอแนวทางแก้ไขได้

---
## Slide 2 — Multicollinearity คืออะไร?
**Key Message**: multicollinearity เกิดเมื่อ predictors correlated กัน — ทำให้ SE สูงและ interpretation ยาก

**นิยาม**: สองหรือมากกว่าสอง predictors มี correlation สูงระหว่างกัน

**ตัวอย่างจริง**:
- Age และ CreditLimit ใน credit dataset (correlation ≈ 0.99!)
- TV และ Radio ใน Advertising (correlation ≈ 0.35 — ยังพอรับได้)
- BMI และ Weight (เกือบ perfect collinear ถ้ามี Height ด้วย)

**ผล**:
- SE(β̂ⱼ) สูงมาก → t-statistic เล็ก → p-value ใหญ่
- β̂ⱼ ไม่แน่นอน — ถ้าเก็บ sample ใหม่อาจเปลี่ยนมาก
- ตีความ β̂ⱼ ได้ยาก เพราะ X₁ และ X₂ เปลี่ยนพร้อมกัน

---
## Slide 3 — ผลของ Collinearity ต่อ SE
**Key Message**: ยิ่ง X₁ กับ X₂ correlated มาก SE(β̂) ยิ่งพอง

```python
# ─── Simulate collinearity effect ─────────────────────────
# วัตถุประสงค์: แสดงว่า SE พุ่งขึ้นเมื่อ correlation ระหว่าง X₁ X₂ สูง
import numpy as np, pandas as pd, statsmodels.formula.api as smf

n = 100
correlations = [0.0, 0.5, 0.9, 0.99]
results = []

for r in correlations:
    X1 = np.random.normal(0, 1, n)
    X2 = r * X1 + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
    y  = 1 + 2*X1 + 3*X2 + np.random.normal(0, 1, n)
    df_sim = pd.DataFrame({'y':y, 'X1':X1, 'X2':X2})
    m = smf.ols('y ~ X1 + X2', data=df_sim).fit()
    results.append({'corr(X1,X2)': r, 'SE(β̂_X1)': round(m.bse['X1'],3), 'SE(β̂_X2)': round(m.bse['X2'],3)})

print(pd.DataFrame(results).to_string(index=False))
# corr=0.0 → SE≈0.1; corr=0.99 → SE >> 1 !
```

---
## Slide 4 — ตรวจ: Correlation Matrix
**Key Message**: correlation matrix เป็นขั้นตอนแรกของ collinearity check — ดูคู่ที่มี |r| > 0.8

```python
# ─── Correlation matrix ────────────────────────────────────
# วัตถุประสงค์: ดูก่อนว่ามี pair ใดที่ correlated สูง (|r| > 0.8)
import seaborn as sns

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True)
plt.title('Correlation Matrix')
plt.show()
```

**ข้อจำกัด**: correlation matrix ดูเฉพาะ pairwise correlation  
ไม่ catch **multicollinearity**: X₃ = X₁ + X₂ (X₃ correlated กับทั้งคู่แต่ r กับแต่ละตัวอาจไม่สูง)

---
## Slide 5 — VIF: Variance Inflation Factor
**Key Message**: VIF วัด "inflation" ของ SE จาก multicollinearity — VIF > 5–10 → ปัญหา

**สูตร VIF**:
$$VIF(\hat{\beta}_j) = \frac{1}{1 - R^2_{X_j|X_{-j}}}$$

- R²_{Xⱼ|X₋ⱼ} = R² จาก regression Xⱼ ~ predictors อื่น ๆ ทั้งหมด
- VIF = 1: ไม่มี collinearity
- VIF = 5: SE พองขึ้น √5 ≈ 2.24 เท่า
- VIF = 10: SE พองขึ้น √10 ≈ 3.16 เท่า

**เกณฑ์**:
| VIF | ความรุนแรง |
|-----|-----------|
| 1–2 | ปกติ |
| 2–5 | ระวัง |
| 5–10 | ปัญหาปานกลาง |
| > 10 | ปัญหาร้ายแรง |

---
## Slide 6 — Python: คำนวณ VIF
**Key Message**: `variance_inflation_factor` จาก statsmodels คำนวณ VIF ทุก predictor ในคำสั่งเดียว

```python
# ─── คำนวณ VIF ──────────────────────────────────────────────
# วัตถุประสงค์: ตรวจ multicollinearity โดย VIF > 5 = ปัญหา
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd, numpy as np

# ต้องสร้าง feature matrix รวม intercept
X_mat = sm.add_constant(df[['TV', 'Radio', 'Newspaper']])

vif_df = pd.DataFrame({
    'Feature': X_mat.columns,
    'VIF': [variance_inflation_factor(X_mat.values, i)
            for i in range(X_mat.shape[1])]
})
print(vif_df.round(2))

# Advertising result: TV≈2, Radio≈2, Newspaper≈2 — OK
# Credit (Limit+Age): VIF >> 100 — problem!
```

---
## Slide 7 — ตัวอย่าง: High Collinearity ใน Credit Data
**Key Message**: Limit และ Age มี VIF สูงมาก → SE พอง → t-statistic เล็ก → อาจ miss true effects

**Credit dataset** — Predictors: Limit, Rating, Age, Cards, Education

```
Feature    VIF
Rating     160.67   ← ปัญหาร้ายแรง
Limit      160.49   ← ปัญหาร้ายแรง (Limit ≈ Rating × constant)
Age          1.01   ← ปกติ
Cards        1.03   ← ปกติ
Education    1.01   ← ปกติ
```

**เหตุผล**: Credit Limit ≈ f(Credit Rating) → corr ≈ 0.995 → VIF >> 100

**ผล**: SE(β̂_Limit), SE(β̂_Rating) สูงมาก → t เล็ก → อาจสรุปผิดว่าทั้งคู่ไม่มีผล

---
## Slide 8 — วิธีแก้ Multicollinearity
**Key Message**: 3 วิธีหลัก — ลบตัวหนึ่ง, combine (PCA), หรือใช้ regularization

**วิธีที่ 1: ลบ predictor ที่ collinear ออก**
- ถ้า X₁ กับ X₂ correlated มาก → เลือกเก็บตัวที่มี domain relevance มากกว่า
- ข้อดี: ง่าย | ข้อเสีย: สูญเสีย information

**วิธีที่ 2: Combine ด้วย PCA (Week 4)**
- แทนที่ [X₁, X₂] ด้วย principal component Z₁ = 0.7X₁ + 0.7X₂
- ข้อดี: ไม่สูญเสีย information | ข้อเสีย: interpret ยาก

**วิธีที่ 3: Regularization (Ridge Regression)**
- เพิ่ม penalty term → β̂ stable แม้ predictors correlated
- ข้อดี: ใช้ได้ทุกกรณี | ข้อเสีย: biased estimates

```python
# วิธีที่ 2: แก้ด้วย PCA + regression (PCR)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

pcr = make_pipeline(PCA(n_components=2), LinearRegression())
pcr.fit(df[['Limit','Rating','Age']], df['Balance'])
```

---
## Slide 9 — Summary
**Key Message**: VIF > 5–10 เป็น alarm signal — ต้องแก้ก่อนเชื่อ t-statistic ของ predictors นั้น

**สิ่งที่เรียนรู้วันนี้**:
- **Multicollinearity**: predictors correlated → SE ↑ → t ↓ → อาจ miss real effects
- **VIF** = 1/(1−R²_{Xⱼ}) — ยิ่งสูงยิ่งปัญหา
- **ตรวจด้วย**: correlation matrix (pairwise) + VIF (multivariate)
- **แก้ด้วย**: ลบตัวหนึ่ง, PCA, หรือ regularization

**Week 10 Summary** (ทุก Slides):
```
Extensions:
  - Dummy variables: categorical → 0/1
  - Interaction: X₁X₂ → synergy/antagonism
  - Polynomial: X² → non-linear relationship

Diagnostics:
  - Residuals vs Fitted → non-linearity, heterosc.
  - QQ-plot → normality
  - Leverage / Cook's D → influential points
  - VIF → multicollinearity
```

**สัปดาห์ต่อไป — Week 11**: Logistic Regression — จาก regression สู่ classification
