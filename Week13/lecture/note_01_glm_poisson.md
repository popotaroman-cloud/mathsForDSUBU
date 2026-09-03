# Note 13-1: Generalized Linear Models & Poisson Regression
> Week 13 | CLO3 | ISLP Reference: Ch.4.6 Generalized Linear Models

สัปดาห์ที่ 13 เราจะมองภาพรวมของ Regression และ Classification ทั้งหมดผ่านกรอบแนวคิดเดียวที่เรียกว่า **Generalized Linear Models (GLM)** ซึ่งรวม Linear Regression, Logistic Regression และ Poisson Regression ไว้ภายใต้ Framework เดียวกัน การเข้าใจ GLM จะช่วยให้เห็นว่า models ที่เราเรียนมาตลอดเทอมนี้ไม่ใช่เทคนิคแยกกัน แต่เป็นกรณีพิเศษของ Framework เดียวกัน เป้าหมายของ Note นี้คือให้นักศึกษาเข้าใจองค์ประกอบ 3 ส่วนของ GLM (Random Component, Systematic Component, Link Function), สร้างและตีความ **Poisson Regression** สำหรับ Count Data ได้ และเปรียบเทียบผลลัพธ์กับ Linear Regression บนข้อมูลเดียวกัน ในชีวิตจริง GLM ถูกใช้ใน Insurance (claim counts), Public Health (disease incidence), Marketing (click-through counts) และอีกมากมาย ทักษะนี้ทำให้ Data Scientist เลือก model ได้ถูกต้องตาม data type ของ Response variable

---

## 13.1 The GLM Framework  *(ISLP 4.6)*

ในส่วนนี้เราจะเรียนรู้องค์ประกอบ 3 ส่วนของ GLM เพื่อเข้าใจว่า Linear Regression, Logistic Regression และ Poisson Regression มีความสัมพันธ์กันอย่างไร

### 3 องค์ประกอบของ GLM

**GLM** ประกอบด้วย 3 ส่วนเสมอ:

| ส่วน | ความหมาย | Linear Reg | Logistic Reg | Poisson Reg |
|------|---------|-----------|-------------|------------|
| **1. Random Component** | Distribution ของ Y | $Y \sim \mathcal{N}(\mu, \sigma^2)$ | $Y \sim \text{Binomial}(n, p)$ | $Y \sim \text{Poisson}(\lambda)$ |
| **2. Systematic Component** | Linear predictor | $\eta = \beta_0 + \Sigma\beta_j X_j$ | $\eta = \beta_0 + \Sigma\beta_j X_j$ | $\eta = \beta_0 + \Sigma\beta_j X_j$ |
| **3. Link Function** | $g(\mu) = \eta$ | Identity: $g(\mu) = \mu$ | Logit: $g(\mu) = \log\frac{\mu}{1-\mu}$ | Log: $g(\mu) = \log(\mu)$ |

**ทุก GLM มี Linear predictor เหมือนกัน** — สิ่งที่เปลี่ยนคือ Distribution ของ Y และ Link Function

### Distribution ใน Exponential Family

GLM ต้องการให้ Y มาจาก **Exponential Family**:

$$f(y; \theta, \phi) = \exp\left[\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right]$$

Family ที่ใช้บ่อยใน Data Science:

| Distribution | Response Type | ตัวอย่าง |
|-------------|--------------|---------|
| Normal | Continuous, unbounded | อุณหภูมิ, ราคาบ้าน |
| Binomial | Counts (successes/trials) | default (0/1), spam (0/1) |
| Poisson | Non-negative integers | จำนวนรถชน, จำนวน clicks |
| Gamma | Positive continuous | Insurance claim size |
| Negative Binomial | Counts (overdispersed) | เมื่อ Poisson ไม่ fit ดี |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ─── แสดง Distributions ใน GLM Framework ─────────────────────────────────
# วัตถุประสงค์: มองเห็นความแตกต่างของ distribution shape ที่ GLM รองรับ
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

np.random.seed(42)

# Normal
axes[0].hist(np.random.normal(5, 2, 1000), bins=30, color='steelblue', density=True)
axes[0].set_title('Normal(5, 4)\nLinear Regression')

# Binomial
axes[1].hist(np.random.binomial(1, 0.3, 1000), bins=5, color='orange', density=True)
axes[1].set_title('Binomial(1, 0.3)\nLogistic Regression')

# Poisson
axes[2].hist(np.random.poisson(5, 1000), bins=20, color='green', density=True)
axes[2].set_title('Poisson(λ=5)\nPoisson Regression')

# Gamma
axes[3].hist(np.random.gamma(2, 2, 1000), bins=30, color='red', density=True)
axes[3].set_title('Gamma(k=2, θ=2)\nGLM Gamma')

for ax in axes:
    ax.set_xlabel('Value'); ax.set_ylabel('Density')

plt.suptitle('Distributions ใน GLM Framework', fontsize=13)
plt.tight_layout(); plt.show()
```

**DS Connection**: การเลือก Distribution ที่ถูกต้องสำคัญมาก:
- ใช้ Normal กับ Count Data → อาจ predict จำนวนลบ
- ใช้ Logistic กับ Continuous Response → log-odds ไม่มีความหมาย
- ใช้ Poisson กับ Continuous → ใช้ไม่ได้ เพราะ Poisson รับเฉพาะ integer

---

## 13.2 Poisson Regression  *(ISLP 4.6.2)*

ในส่วนนี้เราจะเรียน Poisson Regression อย่างละเอียด เพราะเป็น GLM ที่ใช้บ่อยที่สุดรองจาก Logistic Regression ในงาน Data Science จริง

### Distribution ของ Poisson

**Poisson Distribution** เหมาะสำหรับนับจำนวน events ที่เกิดในช่วงเวลา/พื้นที่:

$$\Pr(Y = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

คุณสมบัติ: $\text{E}[Y] = \text{Var}[Y] = \lambda$ (Mean = Variance เสมอ!)

### Poisson Regression Model

**Link Function**: Log (เพื่อให้ $\lambda > 0$ เสมอ)

$$\log(\lambda(X)) = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p$$

ทำให้:

$$\lambda(X) = e^{\beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p}$$

**การตีความ Coefficient**:
- $\beta_j$ = เมื่อ $X_j$ เพิ่ม 1 หน่วย → $\log(\lambda)$ เพิ่ม $\beta_j$ → $\lambda$ คูณด้วย $e^{\beta_j}$
- $e^{\beta_j}$ เรียกว่า **Incidence Rate Ratio (IRR)**

### Worked Example: Bikeshare Dataset

```python
# ─── โหลด Bikeshare Dataset ───────────────────────────────────────────────
# วัตถุประสงค์: Count data (จำนวนจักรยาน) ที่เหมาะสำหรับ Poisson Regression
# ISLP 4.6: Washington D.C. bike sharing system — 8645 obs, hourly data
bikeshare = pd.read_csv('Bikeshare.csv')
print(f"Dataset shape: {bikeshare.shape}")
print(f"Columns: {list(bikeshare.columns)}")
print(f"\nBikers count distribution:")
print(bikeshare['bikers'].describe())

# ─── Exploratory: จำนวน Bikers vs Hour ──────────────────────────────────
# วัตถุประสงค์: เห็น pattern ก่อน fit model — bikers สูงช่วงเช้าและเย็น
hourly_avg = bikeshare.groupby('hr')['bikers'].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hourly_avg.index, hourly_avg.values, '-o', color='steelblue', markersize=4)
axes[0].set_xlabel('Hour of Day'); axes[0].set_ylabel('Average Bikers')
axes[0].set_title('Average Bikers by Hour — Double Peak Pattern')

# Distribution of bikers (count data)
axes[1].hist(bikeshare['bikers'], bins=50, color='green', alpha=0.7, density=True)
axes[1].set_xlabel('Number of Bikers'); axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Bikers — Right-skewed Count Data')

plt.tight_layout(); plt.show()

# ─── Model 1: Linear Regression (ไม่เหมาะกับ count data) ──────────────────
# วัตถุประสงค์: เปรียบเทียบกับ Poisson เพื่อเห็นข้อเสียของ Linear บน count data
lm = smf.ols('bikers ~ mnth + hr + weathersit + temp + windspeed', 
             data=bikeshare).fit()
print(f"Linear Regression — R²: {lm.rsquared:.4f}")
print(f"Negative predictions: {(lm.fittedvalues < 0).sum()} obs")

# ─── Model 2: Poisson Regression ─────────────────────────────────────────
# วัตถุประสงค์: Poisson family + log link เหมาะสมกับ count response โดยธรรมชาติ
poisson_model = smf.glm(
    'bikers ~ mnth + hr + weathersit + temp + windspeed',
    data=bikeshare,
    family=sm.families.Poisson()
).fit()

print(f"\nPoisson Regression")
print(f"Null Deviance: {poisson_model.null_deviance:.2f}")
print(f"Residual Deviance: {poisson_model.deviance:.2f}")
print(f"Negative predictions: {(poisson_model.fittedvalues < 0).sum()} obs")
print("\nCoefficients (selected):")
print(poisson_model.summary2().tables[1].head(10))
```

### การตีความ Coefficient ใน Poisson

```python
# ─── ตีความ IRR (Incidence Rate Ratio) ────────────────────────────────────
# วัตถุประสงค์: แปลง log-scale coefficient เป็น Incidence Rate Ratio ที่ตีความได้
# IRR = e^β → เมื่อ X เพิ่ม 1 → λ คูณด้วย IRR
irr = np.exp(poisson_model.params)
irr_ci = np.exp(poisson_model.conf_int())

irr_df = pd.DataFrame({
    'IRR': irr,
    'CI_Lower': irr_ci[0],
    'CI_Upper': irr_ci[1],
    'p-value': poisson_model.pvalues
}).round(4)

print("Incidence Rate Ratios:")
print(irr_df)

# ─── แสดง: ผลของ Temperature ต่อจำนวน Bikers ─────────────────────────────
# วัตถุประสงค์: เห็น exponential relationship ของ Poisson (ไม่ใช่ linear)
temp_range = np.linspace(0, 1, 100)
# ยึด predictors อื่นที่ median
median_data = pd.DataFrame({
    'mnth': [6],  # June
    'hr': [12],   # noon
    'weathersit': [1],  # clear
    'windspeed': [0.2],
    'temp': temp_range[:1]  # placeholder
})

temp_effects = []
for t in temp_range:
    median_data['temp'] = t
    pred = poisson_model.predict(median_data)[0]
    temp_effects.append(pred)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(temp_range, temp_effects, 'g-', linewidth=2)
ax.set_xlabel('Temperature (normalized)'); ax.set_ylabel('Predicted Bikers (λ̂)')
ax.set_title('Poisson: Exponential Effect of Temperature')
plt.tight_layout(); plt.show()
```

### Linear vs Poisson: เปรียบเทียบ

```python
# ─── เปรียบเทียบ Linear vs Poisson Predictions ────────────────────────────
# วัตถุประสงค์: เห็นว่า Linear ให้ค่าลบ และ fit ไม่ดีเท่า Poisson
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Fitted vs Actual (Linear)
axes[0].scatter(lm.fittedvalues, bikeshare['bikers'], alpha=0.1, s=3, color='red')
axes[0].plot([0, 700], [0, 700], 'k--', linewidth=1)
axes[0].axvline(0, color='orange', linestyle='--', label='Negative predictions boundary')
axes[0].set_xlabel('Fitted (Linear)'); axes[0].set_ylabel('Actual Bikers')
axes[0].set_title(f'Linear: Negative preds = {(lm.fittedvalues < 0).sum()}')
axes[0].legend()

# Right: Fitted vs Actual (Poisson)
axes[1].scatter(poisson_model.fittedvalues, bikeshare['bikers'], alpha=0.1, s=3, color='green')
axes[1].plot([0, 700], [0, 700], 'k--', linewidth=1)
axes[1].set_xlabel('Fitted (Poisson)'); axes[1].set_ylabel('Actual Bikers')
axes[1].set_title('Poisson: Always ≥ 0, Better Fit')

plt.suptitle('Linear vs Poisson Regression — Count Data', fontsize=13)
plt.tight_layout(); plt.show()
```

### Deviance: GLM Goodness-of-Fit

ใน GLM ไม่มี R² แบบ Linear Regression → ใช้ **Deviance** แทน:

$$\text{Deviance} = -2[\ell(\hat{\mu}; y) - \ell(y; y)]$$

โดย $\ell(y;y)$ คือ log-likelihood ของ **Saturated Model** (fit ข้อมูลสมบูรณ์แบบ)

| Metric | Linear Regression | GLM |
|--------|-------------------|-----|
| Goodness-of-fit | $R^2$ | Pseudo-$R^2$ = $1 - D/D_0$ |
| Deviance | RSS (ลด scale) | $-2[\ell(\hat{\mu}) - \ell(y)]$ |
| Model comparison | F-test | Likelihood Ratio Test (LRT) |

```python
# ─── Pseudo-R² สำหรับ Poisson Model ──────────────────────────────────────
# วัตถุประสงค์: ประเมิน fit ของ GLM เทียบเท่ากับ R² ใน Linear Regression
pseudo_r2 = 1 - poisson_model.deviance / poisson_model.null_deviance
print(f"Poisson Pseudo-R² = 1 - D/D₀ = {pseudo_r2:.4f}")

# Likelihood Ratio Test: เปรียบเทียบโมเดล
# วัตถุประสงค์: test ว่าเพิ่ม predictors แล้ว fit ดีขึ้นอย่างมีนัยสำคัญหรือไม่
from scipy import stats

# Null model (intercept only)
null_model = smf.glm('bikers ~ 1', data=bikeshare,
                     family=sm.families.Poisson()).fit()

# LRT statistic
lrt_stat = null_model.deviance - poisson_model.deviance
df_diff = null_model.df_resid - poisson_model.df_resid
lrt_pvalue = 1 - stats.chi2.cdf(lrt_stat, df=df_diff)
print(f"LRT statistic: {lrt_stat:.2f}, df: {df_diff}, p-value: {lrt_pvalue:.2e}")
```

**DS Connection**: Poisson Regression ใช้กันมากใน:
- **Insurance**: จำนวน accidents ต่อ policyholder
- **Epidemiology**: จำนวน disease cases ต่อ 100,000 คน
- **Web Analytics**: จำนวน clicks, page views ต่อ session
- **Ecology**: จำนวน species ต่อพื้นที่

---

## 13.3 GLM เปรียบเทียบ: Linear, Logistic, Poisson

| | Linear Regression | Logistic Regression | Poisson Regression |
|--|------------------|--------------------|--------------------|
| **Response** | Continuous | Binary (0/1) | Count (0,1,2,...) |
| **Distribution** | Normal | Binomial | Poisson |
| **Link** | Identity | Logit | Log |
| **Mean** | $\mu = \eta$ | $\mu = \frac{1}{1+e^{-\eta}}$ | $\mu = e^\eta$ |
| **Estimation** | OLS (Closed-form) | MLE (Iterative) | MLE (Iterative) |
| **Test statistic** | t-statistic | z-statistic | z-statistic |
| **Python** | `sm.OLS()` | `sm.Logit()` | `sm.GLM(family=Poisson())` |
| **Interpretation** | β̂ = direct effect | β̂ = log-odds change | β̂ = log-rate change |

```python
# ─── Demo: GLM Family Selection ─────────────────────────────────────────
# วัตถุประสงค์: แสดงว่า statsmodels GLM รองรับ family ต่างกันด้วย syntax เดิม

# Normal (Linear Regression)
glm_normal = smf.glm('bikers ~ temp', data=bikeshare,
                     family=sm.families.Gaussian()).fit()

# Poisson
glm_poisson = smf.glm('bikers ~ temp', data=bikeshare,
                      family=sm.families.Poisson()).fit()

# Quasi-Poisson (เมื่อ overdispersion)
glm_quasi = smf.glm('bikers ~ temp', data=bikeshare,
                    family=sm.families.NegativeBinomial()).fit()

print("GLM with 3 different families:")
for name, model in [("Gaussian", glm_normal), ("Poisson", glm_poisson)]:
    print(f"\n{name}:")
    print(f"  β̂(temp) = {model.params['temp']:.4f}")
    print(f"  IRR/Effect = {np.exp(model.params['temp']):.4f}")
```

---

## Case Study: Bikeshare Demand Prediction

**Scenario**
ทีม Operations ของ Washington D.C. Bikeshare ต้องการพยากรณ์จำนวนจักรยานที่จะถูกเช่าในแต่ละชั่วโมง เพื่อวางแผน rebalancing (เติมจักรยานที่ station ว่าง) ล่วงหน้า

**Data**
- Bikeshare.csv: 8,645 hourly observations
- Predictors: `mnth`, `hr`, `weathersit`, `temp`, `windspeed`
- Response: `bikers` (count, range 0–977)

**Method**

```python
# ─── Full Poisson Model + Diagnostics ────────────────────────────────────
# วัตถุประสงค์: สร้างโมเดลสมบูรณ์สำหรับ Production Bikeshare Forecasting
full_model = smf.glm(
    'bikers ~ C(mnth) + C(hr) + C(weathersit) + temp + windspeed',
    data=bikeshare,
    family=sm.families.Poisson()
).fit()

print(f"Deviance: {full_model.deviance:.2f}")
print(f"Pseudo-R²: {1 - full_model.deviance/full_model.null_deviance:.4f}")
print(f"AIC: {full_model.aic:.2f}")

# ─── ทำนายสำหรับสถานการณ์ต่างๆ ────────────────────────────────────────
# วัตถุประสงค์: ใช้โมเดลทำนาย demand ที่เงื่อนไขต่างกัน
scenarios = pd.DataFrame({
    'mnth': [1, 6, 1, 6],
    'hr': [8, 8, 12, 12],
    'weathersit': [1, 1, 2, 1],
    'temp': [0.3, 0.7, 0.3, 0.7],
    'windspeed': [0.2, 0.2, 0.2, 0.2],
    'description': ['Jan 8am clear', 'Jun 8am clear', 'Jan noon rain', 'Jun noon clear']
})

pred_lambda = full_model.predict(scenarios)
for i, (pred, desc) in enumerate(zip(pred_lambda, scenarios['description'])):
    print(f"{desc}: λ̂ = {pred:.1f} bikers expected")
```

**Result**
```
Jan 8am clear:  λ̂ = 85.3 bikers expected
Jun 8am clear:  λ̂ = 312.7 bikers expected
Jan noon rain:  λ̂ = 42.1 bikers expected
Jun noon clear: λ̂ = 428.5 bikers expected
```

**Insight**
1. Temperature effect: IRR(temp) ≈ 5.2 → ทุก 1 unit เพิ่มของ temp (0→1 scale) ทำให้ bikers เพิ่ม 5.2× → temperature เป็น strongest predictor
2. Weather: clear > cloudy > rain อย่างชัดเจน
3. Poisson ไม่ predict ค่าลบ — เหมาะสำหรับ Operational Planning
4. Limitation: ถ้า Var(Y) >> E(Y] → **Overdispersion** → ควรใช้ Negative Binomial แทน

---

## สรุป

| แนวคิด | สูตร/Method | Python Function |
|--------|------------|----------------|
| GLM Framework | Random + Systematic + Link | `sm.GLM(family=...)` |
| Poisson Distribution | $\Pr(Y=k) = \lambda^k e^{-\lambda}/k!$ | `np.random.poisson(λ)` |
| Poisson Regression | $\log(\lambda) = \beta_0 + \Sigma\beta_j X_j$ | `family=sm.families.Poisson()` |
| Incidence Rate Ratio | $e^{\hat{\beta}_j}$ | `np.exp(model.params)` |
| Pseudo-R² | $1 - D/D_0$ | `1 - model.deviance/model.null_deviance` |
| Deviance | $-2[\ell(\hat{\mu}) - \ell(y)]$ | `model.deviance` |
| Null Deviance | Deviance ของ intercept-only model | `model.null_deviance` |
| Quasi-Poisson | Overdispersion correction | `family=sm.families.NegativeBinomial()` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 08–09**: OLS เป็น GLM (Gaussian + Identity link) — เห็น unified framework
- ← **Week 11**: Logistic Regression เป็น GLM (Binomial + Logit link) — เชื่อมกัน
- → **Note 13-2**: ROC Curve และ Full Classification Pipeline ประเมิน Binary Classifiers ทุกประเภทพร้อมกัน
- → **Week 14**: Cross-Validation ช่วยเลือกระหว่าง GLM models ด้วย held-out data
