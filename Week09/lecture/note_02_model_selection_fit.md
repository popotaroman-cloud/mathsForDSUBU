# Note 2: Variable Selection, Adjusted R² และ Confidence vs Prediction Intervals
> Week 9 | CLO3 | ISLP Reference: Ch.3.2.2

---

## บทนำ

บันทึกนี้ตอบคำถามสำคัญสองข้อที่เกิดขึ้นเสมอเมื่อทำ Multiple Regression จริง: (1) **"ควรใส่ predictor กี่ตัวและตัวไหนบ้าง?"** และ (2) **"เมื่อ predict ค่าใหม่ ช่วงความไม่แน่นอนของ prediction คือเท่าไหร่?"** เป้าหมายคือให้นักศึกษาเข้าใจว่า R² เพิ่มขึ้นเสมอเมื่อเพิ่ม predictor จึงต้องใช้ Adjusted R² ที่ penalize สำหรับ model complexity, รู้จัก Variable Selection methods, และแยกความแตกต่างระหว่าง Confidence Interval (สำหรับ mean response) กับ Prediction Interval (สำหรับ individual response) ซึ่งต่างกันอย่างมีนัยสำคัญ ทักษะเหล่านี้เป็นพื้นฐานของ Model Selection ที่จะเรียนอย่างเป็นระบบใน Week 14 (Cross-Validation)

---

## Section 1: R² vs Adjusted R²  *(ISLP 3.2.2 Q3)*

ในส่วนนี้เราจะเรียนรู้ว่าทำไม R² ธรรมดาถึงไม่เหมาะสำหรับเปรียบเทียบ models ที่มี predictors จำนวนต่างกัน และ Adjusted R² แก้ปัญหานี้อย่างไร

### 1.1 ปัญหาของ R² เมื่อเพิ่ม Predictors

**สูตร R²**:
$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{TSS} - \text{RSS}}{\text{TSS}}$$

**คุณสมบัติที่เป็นปัญหา**: R² จะ **เพิ่มขึ้นเสมอหรือคงที่** เมื่อเพิ่ม predictor — แม้ว่า predictor นั้นจะไม่มีความสัมพันธ์กับ $Y$ เลย!

**เหตุผล**: RSS ลดลงได้เสมอ (หรือเท่าเดิม) เมื่อมี parameter เพิ่มเติม เพราะ optimization มี degree of freedom มากขึ้น

### 1.2 Adjusted R²

$$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$$

**คุณสมบัติ**:
- เพิ่ม **penalty** สำหรับการเพิ่ม $p$ ผ่าน denominator $(n-p-1)$
- ถ้า predictor ใหม่ improve fit จริง → $R^2_{\text{adj}}$ เพิ่ม
- ถ้า predictor ใหม่ไม่ช่วย → $R^2_{\text{adj}}$ ลด

**สูตรเทียบกับ R²**:
$$R^2_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-p-1}$$

เมื่อ $p$ เพิ่ม → $\frac{n-1}{n-p-1}$ เพิ่ม → penalty เพิ่ม

### 1.3 ตาราง Comparison: 3 กรณี

```python
# ─── เปรียบเทียบ R² vs Adjusted R² ──────────────────────────────────────
# วัตถุประสงค์: แสดงว่า R² ไม่ลดลงแม้เพิ่ม predictor ที่ไม่มีประโยชน์
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 200

TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
Newspaper = np.random.uniform(0.3, 114.0, n)
# Random noise features — ไม่มีความสัมพันธ์กับ Sales จริง ๆ
Noise1 = np.random.randn(n)
Noise2 = np.random.randn(n)
Sales = 2.9 + 0.046*TV + 0.189*Radio - 0.001*Newspaper + np.random.normal(0, 1.5, n)

# กรณีที่ 1: TV เท่านั้น (SLR)
X1 = sm.add_constant(TV)
m1 = sm.OLS(Sales, X1).fit()

# กรณีที่ 2: TV + Radio + Newspaper (MLR ที่ดี)
X2 = sm.add_constant(np.column_stack([TV, Radio, Newspaper]))
m2 = sm.OLS(Sales, X2).fit()

# กรณีที่ 3: TV + Radio + Newspaper + Noise1 + Noise2 (เพิ่ม garbage features)
X3 = sm.add_constant(np.column_stack([TV, Radio, Newspaper, Noise1, Noise2]))
m3 = sm.OLS(Sales, X3).fit()

# แสดงผลเปรียบเทียบ
print(f"{'Model':<35} {'p':>4} {'R²':>8} {'Adj R²':>10} {'AIC':>10}")
print("-" * 70)
for name, m, p in [
    ("Case 1: TV only", m1, 1),
    ("Case 2: TV + Radio + Newspaper", m2, 3),
    ("Case 3: + 2 Noise features", m3, 5)
]:
    print(f"{name:<35} {p:>4} {m.rsquared:>8.4f} {m.rsquared_adj:>10.4f} {m.aic:>10.2f}")
```

**ผลที่คาดหวัง** (ตัวอย่าง):

| Model | p | R² | Adj R² | AIC |
|-------|---|-----|--------|-----|
| TV only | 1 | 0.612 | 0.610 | 887.4 |
| TV + Radio + Newspaper | 3 | 0.897 | 0.896 | 586.3 |
| + 2 Noise features | 5 | 0.897 | 0.894 | 589.0 |

**สังเกต**:
- Case 3: R² ไม่ลดลงเลย (≈ Case 2) แต่ Adj R² **ลดลง** เพราะ noise ไม่ได้ improve fit จริง
- AIC ของ Case 3 ใหญ่กว่า Case 2 → Case 2 ดีกว่า

---

## Section 2: Variable Selection Methods  *(ISLP 3.2.2 Q2)*

ในส่วนนี้เราจะเรียนรู้วิธี **เลือก predictor** ที่เหมาะสมเมื่อมีตัวแปรหลายสิบตัวและต้องการ model ที่ parsimonious (เรียบง่ายแต่มีประสิทธิภาพ)

### 2.1 Best Subset Selection

ลอง **ทุก combination** ของ predictors ($2^p$ models) แล้วเลือกที่ดีที่สุด  
**ข้อดี**: ครบถ้วนที่สุด  
**ข้อเสีย**: $p=40$ → $2^{40} \approx 10^{12}$ models — คำนวณไม่ได้!

### 2.2 Forward Stepwise Selection

1. เริ่มจาก null model ($p=0$)
2. ทีละขั้น: เพิ่ม predictor ที่ improve fit มากที่สุด ($\Delta R^2_{\text{adj}}$ หรือ $\Delta \text{AIC}$)
3. หยุดเมื่อไม่มี predictor ไหนช่วยอีก

**Complexity**: $O(p^2)$ — เร็วกว่า Best Subset มาก

### 2.3 Backward Stepwise Selection

1. เริ่มจาก full model (ทุก predictor)
2. ทีละขั้น: ลบ predictor ที่มี p-value สูงสุด (least useful)
3. หยุดเมื่อทุก predictor ที่เหลือ significant ($p < 0.05$)

**ข้อดี**: ง่าย, ใช้ p-value ที่เข้าใจง่าย

### 2.4 Model Selection Criteria

| Criterion | สูตร | เลือกอย่างไร |
|-----------|------|------------|
| Adjusted R² | $1-(1-R^2)\frac{n-1}{n-p-1}$ | maximize |
| AIC | $n\ln(\text{RSS}/n) + 2(p+1)$ | minimize |
| BIC | $n\ln(\text{RSS}/n) + \ln(n)(p+1)$ | minimize |
| Mallow's Cp | $\frac{\text{RSS}+2p\hat{\sigma}^2}{\hat{\sigma}^2}$ | minimize |

**BIC penalizes complexity มากกว่า AIC** (เพราะ $\ln(n) > 2$ เมื่อ $n > 7$) → BIC ให้ model เล็กกว่า

```python
# ─── Forward Stepwise Selection ──────────────────────────────────────────
# วัตถุประสงค์: implement forward selection อย่างง่ายด้วย Adjusted R²
import numpy as np
import statsmodels.api as sm
from itertools import combinations

def forward_selection(X_df, y, criterion='adj_r2'):
    """
    Forward stepwise selection
    เลือก predictor ทีละตัวที่ improve criterion มากที่สุด
    """
    features_available = list(X_df.columns)
    features_selected = []
    best_criterion = -np.inf if criterion == 'adj_r2' else np.inf

    while features_available:
        best_feature = None
        current_best = best_criterion

        for f in features_available:
            # ลอง add feature f เข้า model
            cols = features_selected + [f]
            X_try = sm.add_constant(X_df[cols])
            m = sm.OLS(y, X_try).fit()

            # วัด criterion
            val = m.rsquared_adj if criterion == 'adj_r2' else -m.aic
            if val > current_best:
                current_best = val
                best_feature = f

        # ถ้าไม่มีการปรับปรุง หยุด
        if best_feature is None or current_best <= best_criterion:
            break

        features_selected.append(best_feature)
        features_available.remove(best_feature)
        best_criterion = current_best
        adj_r2 = current_best if criterion == 'adj_r2' else None
        print(f"Add '{best_feature}' → Adj R² = {best_criterion:.4f}")

    return features_selected

# ใช้ forward selection กับ Advertising data
import pandas as pd
np.random.seed(42)
n = 200
TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
Newspaper = np.random.uniform(0.3, 114.0, n)
Noise = np.random.randn(n)
Sales = 2.9 + 0.046*TV + 0.189*Radio - 0.001*Newspaper + np.random.normal(0, 1.5, n)

X_df = pd.DataFrame({'TV': TV, 'Radio': Radio, 'Newspaper': Newspaper, 'Noise': Noise})
selected = forward_selection(X_df, Sales)
print(f"\nSelected features: {selected}")
```

---

## Section 3: Confidence Interval vs Prediction Interval  *(ISLP 3.2.2 Q4)*

ในส่วนนี้เราจะเรียนรู้ความแตกต่างระหว่าง **Confidence Interval** และ **Prediction Interval** ซึ่งเป็นแนวคิดที่สำคัญมากสำหรับการสื่อสารผลต่อ stakeholders

### 3.1 Confidence Interval สำหรับ Mean Response

**ถามว่า**: ค่าเฉลี่ยของ Sales สำหรับ **ตลาดทั้งหมดที่มี** TV=100, Radio=20 คือเท่าไหร่?

$$\text{CI สำหรับ } \mathbb{E}[Y|X=x_0]: \quad \hat{f}(x_0) \pm t^* \cdot \hat{\sigma}\sqrt{\mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0}$$

**ความไม่แน่นอนมาจาก**: ความไม่แน่นอนในการ estimate $\boldsymbol{\beta}$

### 3.2 Prediction Interval สำหรับ Individual Response

**ถามว่า**: ตลาด **ตัวใหม่ตัวเดียว** ที่มี TV=100, Radio=20 จะมี Sales เท่าไหร่?

$$\text{PI สำหรับ } Y_{new}|X=x_0: \quad \hat{f}(x_0) \pm t^* \cdot \hat{\sigma}\sqrt{1 + \mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0}$$

**ความไม่แน่นอนมาจาก**: ความไม่แน่นอนใน $\boldsymbol{\beta}$ **บวก** irreducible error $\varepsilon$

### 3.3 เปรียบเทียบ CI และ PI

| | CI (Mean Response) | PI (Individual) |
|-|------------------|----------------|
| ตอบคำถาม | average ของ population | individual new obs |
| ความกว้าง | แคบกว่า | กว้างกว่าเสมอ |
| Extra term | $\mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0$ | เพิ่ม "+1" ข้างหน้า |
| ใช้เมื่อ | policy decisions | individual prediction |

**ตัวอย่าง**: คาด Sales สำหรับ TV=100, Radio=20:
- CI: (14.5, 16.2) — "ค่าเฉลี่ยของตลาดแบบนี้อยู่ในช่วงนี้"
- PI: (11.0, 19.7) — "ตลาดตัวนี้จะมี Sales อยู่ในช่วงนี้ (กว้างกว่ามาก เพราะมี noise)"

```python
# ─── CI vs PI สำหรับ MLR Predictions ───────────────────────────────────
# วัตถุประสงค์: แสดงความแตกต่างระหว่าง CI (mean) และ PI (individual)
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 200
TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
Sales = 2.9 + 0.046*TV + 0.189*Radio + np.random.normal(0, 1.5, n)

X = sm.add_constant(np.column_stack([TV, Radio]))
model = sm.OLS(Sales, X).fit()

# New observation: TV=100, Radio=20
x_new = np.array([[1, 100, 20]])  # shape (1, 3) with intercept

# CI สำหรับ mean response
# วัตถุประสงค์: ความไม่แน่นอนจาก estimating β เท่านั้น
pred_ci = model.get_prediction(x_new).summary_frame(alpha=0.05)
print("Prediction for TV=100, Radio=20:")
print(f"  Point estimate: {pred_ci['mean'].values[0]:.3f}")
print(f"  95% CI (mean): ({pred_ci['mean_ci_lower'].values[0]:.3f}, "
      f"{pred_ci['mean_ci_upper'].values[0]:.3f})")
print(f"  95% PI (individual): ({pred_ci['obs_ci_lower'].values[0]:.3f}, "
      f"{pred_ci['obs_ci_upper'].values[0]:.3f})")
print(f"\n  CI width: {pred_ci['mean_ci_upper'].values[0] - pred_ci['mean_ci_lower'].values[0]:.3f}")
print(f"  PI width: {pred_ci['obs_ci_upper'].values[0] - pred_ci['obs_ci_lower'].values[0]:.3f}")
```

---

## Case Study: Housing Price Prediction — CI vs PI in Practice

**Scenario**  
ธนาคารต้องการอนุมัติสินเชื่อบ้าน แต่ต้องการ estimate ว่า "บ้านหลังนี้ควรราคาเท่าไหร่" เพื่อกำหนดวงเงินกู้

**คำถาม**: ควรใช้ CI หรือ PI?

```python
# ─── Bank Loan Scenario ──────────────────────────────────────────────────
# วัตถุประสงค์: แสดง practical implication ของ CI vs PI
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 100
area = np.random.uniform(50, 300, n)
rooms = np.random.randint(1, 6, n).astype(float)
price = 1000 + 15*area + 80*rooms + np.random.normal(0, 200, n)

X = sm.add_constant(np.column_stack([area, rooms]))
model = sm.OLS(price, X).fit()

# บ้านใหม่: พื้นที่ 120 ตร.ม., 3 ห้องนอน
x_new = np.array([[1, 120, 3]])
pred = model.get_prediction(x_new).summary_frame(alpha=0.05)

point_est = pred['mean'].values[0]
ci_lower, ci_upper = pred['mean_ci_lower'].values[0], pred['mean_ci_upper'].values[0]
pi_lower, pi_upper = pred['obs_ci_lower'].values[0], pred['obs_ci_upper'].values[0]

print(f"บ้าน: 120 ตร.ม., 3 ห้องนอน")
print(f"ราคาประมาณ: {point_est:.0f} บาท")
print(f"95% CI สำหรับ mean price: {ci_lower:.0f} – {ci_upper:.0f} บาท")
print(f"95% PI สำหรับ actual price: {pi_lower:.0f} – {pi_upper:.0f} บาท")
print(f"\nธนาคารควรใช้ PI เพราะกำลัง appraise บ้านหลังเดียว (individual)")
print(f"ถ้าใช้ CI จะ underestimate uncertainty และอนุมัติวงเงินสูงเกินไป!")
```

**Insight**  
- CI ตอบ: "บ้าน type นี้ average เท่าไหร่" → ใช้สำหรับ policy/market analysis
- PI ตอบ: "บ้านหลังนี้จะขายได้เท่าไหร่" → ใช้สำหรับ individual appraisal
- ธนาคารต้องใช้ PI เพราะ underwriting เป็น individual decision ไม่ใช่ population level

---

## สรุป (Summary)

| แนวคิด | สูตร | Python |
|--------|------|--------|
| R² | $1-\text{RSS}/\text{TSS}$ | `model.rsquared` |
| Adj R² | $1-(1-R^2)\frac{n-1}{n-p-1}$ | `model.rsquared_adj` |
| AIC | $n\ln(\text{RSS}/n) + 2(p+1)$ | `model.aic` |
| BIC | $n\ln(\text{RSS}/n) + \ln(n)(p+1)$ | `model.bic` |
| CI width | proportional to $\sqrt{\mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0}$ | `get_prediction().summary_frame()` |
| PI width | larger by $\hat{\sigma}^2$ (irreducible error) | `obs_ci_lower`, `obs_ci_upper` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 9 (note 1)**: F-statistic บอก "มี predictor useful หรือเปล่า" → note นี้ตอบ "predictor ไหน useful"
- ← **Week 7**: AIC/BIC ใช้ likelihood framework จาก statistical inference; CI ใช้ SE และ t-distribution
- → **Week 10**: Regression Diagnostics — ตรวจสอบว่า MLR assumptions ถูกละเมิดหรือไม่
- → **Week 14**: Cross-Validation เป็นวิธีที่ optimal ที่สุดสำหรับ model selection แทน Adj R²/AIC
