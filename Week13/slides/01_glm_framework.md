# Slide Deck 1: Generalized Linear Models Framework
> Week 13 | 9 slides | CLO3

---

## Slide 1 — Title

**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล — Week 13**  
Generalized Linear Models, ROC Curve & Full Classification Pipeline  
CLO3: ดำเนิน full classification pipeline บนชุดข้อมูลจริงได้

---

## Slide 2 — Week Overview

**Key Message**: GLM เป็น unified framework ที่รวม Linear, Logistic และ Poisson Regression ไว้ในที่เดียว

สัปดาห์นี้เราจะ connect dots จาก weeks ที่ผ่านมา Linear Regression (Week 8–10) และ Logistic Regression (Week 11) ล้วนเป็นสมาชิกของ family เดียวกันที่เรียกว่า **Generalized Linear Models (GLM)** ความเข้าใจ GLM framework ทำให้เราสามารถขยายความรู้ไปสู่ response Y ชนิดอื่นได้ เช่น count data (Poisson), rate data (Gamma), survival time ทักษะที่จะเรียนรู้วันนี้: (1) GLM framework และ link functions, (2) Poisson Regression สำหรับ count data, (3) ROC Curve + AUC สำหรับ model comparison, (4) Full classification pipeline ที่ professional ใน industry Data Science จริงๆ เราจะใช้ทักษะเหล่านี้กับ Bikeshare dataset (count data) และ Default/Smarket datasets (classification)

- **GLM**: 3 components — random + systematic + link
- **Poisson Regression**: for count data Y ∈ {0,1,2,...}
- **ROC + AUC**: threshold-free classifier comparison
- **Full Pipeline**: end-to-end best practices

---

## Slide 3 — The Linear Model Family

**Key Message**: Linear Regression เป็น special case ของ GLM ที่ใหญ่กว่า

**ปัญหาของ Linear Regression:**
- Y ~ N(μ, σ²) → ทำนายค่า continuous ได้ดี
- แต่ถ้า Y เป็น binary (0/1) → predict probability ออกนอก [0,1]
- ถ้า Y เป็น count (0,1,2,...) → predict ค่าลบได้

**GLM แก้ปัญหานี้โดย:**
1. ยอมให้ Y มี distribution ใดก็ได้ใน **exponential family**
2. เพิ่ม **link function** g(μ) แทนที่จะ model μ ตรงๆ

**[FIGURE: tree diagram — GLM ที่ root, แตกเป็น Normal (→ Linear Reg), Binomial (→ Logistic Reg), Poisson (→ Poisson Reg), Gamma (→ Gamma Reg)]**

---

## Slide 4 — GLM: 3 Components

**Key Message**: GLM = Random + Systematic + Link — เปลี่ยน component ใดก็ได้ตามต้องการ

**Component 1: Random Component**
$$Y_i \sim \text{Exponential Family}(\theta_i)$$
กำหนดว่า Y มี distribution อะไร

**Component 2: Systematic Component**
$$\eta_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip}$$
linear predictor — เหมือน Linear Regression

**Component 3: Link Function**
$$g(\mu_i) = \eta_i \quad \text{where } \mu_i = E[Y_i]$$
เชื่อม mean ของ Y กับ linear predictor

**[TABLE: ตาราง GLM models]**

| Model | Y ~ | Link g(μ) | g⁻¹(η) |
|-------|-----|----------|--------|
| Linear Reg | Normal | Identity: μ | η |
| Logistic Reg | Binomial | Logit: log(μ/(1-μ)) | 1/(1+e⁻η) |
| Poisson Reg | Poisson | Log: log(μ) | eη |

---

## Slide 5 — MLE for GLM

**Key Message**: GLM ทุกตัว fit ด้วย Maximum Likelihood — ไม่ใช่ Least Squares

**ทำไมไม่ใช้ Least Squares?**
- RSS เหมาะกับ Normal distribution เท่านั้น
- Poisson, Binomial มี likelihood function ต่างกัน

**GLM ใช้ Iteratively Reweighted Least Squares (IRLS):**
- Algorithm ที่ solve MLE ของ GLM ซ้ำๆ จนลู่เข้า
- statsmodels handle ให้อัตโนมัติ

**Goodness-of-fit: Deviance**
$$D = 2[\ell(\hat{\mu}_{saturated}) - \ell(\hat{\mu}_{model})]$$
แทนที่ RSS ใน GLM — ยิ่งน้อยยิ่งดี

**Python:**
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit Logistic GLM (เหมือนกับ smf.logit)
logistic_glm = smf.glm(
    'y ~ x1 + x2',
    data=df,
    family=sm.families.Binomial()  # logit link
).fit()

# Fit Poisson GLM
poisson_glm = smf.glm(
    'count ~ x1 + x2',
    data=df,
    family=sm.families.Poisson()   # log link
).fit()
```

---

## Slide 6 — Exponential Family Distributions

**Key Message**: Exponential family มี distributions หลายตัว — เลือกตาม Y ที่เรามี

**[TABLE]**

| Distribution | Y คือ | Python family | ใช้เมื่อ |
|-------------|-------|--------------|---------|
| Normal(μ,σ²) | continuous, ∈ ℝ | `Gaussian()` | regression ทั่วไป |
| Binomial(n,p) | binary/proportion | `Binomial()` | classification |
| Poisson(λ) | count, ∈ {0,1,2,...} | `Poisson()` | event counts |
| Gamma(α,β) | positive continuous | `Gamma()` | insurance claims, duration |
| Negative Binomial | overdispersed count | `NegativeBinomial()` | count with overdispersion |

**เลือก distribution จาก Y ของ dataset:**
- จำนวนครั้งที่ customer โทร → Poisson
- เวลาที่ machine ทำงานก่อน breakdown → Gamma
- Binary outcome → Binomial

**[FIGURE: 4 plots ของ distributions ต่างๆ — Normal bell, Binomial bar, Poisson bar (right-skew), Gamma (right-skew)]**

---

## Slide 7 — Why GLM Matters in Data Science

**Key Message**: GLM เป็น foundation ของ statistical modeling — รู้จัก GLM = understand ML ลึกขึ้น

**Real-World Applications:**

| Application | Response Y | GLM Model |
|------------|-----------|-----------|
| Email spam | binary (spam/not) | Logistic Regression |
| Hospital visits | count per year | Poisson Regression |
| Insurance premium | continuous positive | Gamma Regression |
| Click-through rate | proportion 0–1 | Binomial (Beta) |
| Time to failure | positive continuous | Gamma/Weibull |

**GLM เป็น foundation ของ:**
- Generalized Additive Models (GAM): สำหรับ non-linear effects
- Mixed Effects Models: สำหรับ hierarchical data
- Neural Networks: final layer เป็น GLM output layer

**[FIGURE: diagram showing GLM → GAM → Neural Network as progression of complexity]**

---

## Slide 8 — Deviance + Residuals in GLM

**Key Message**: GLM ใช้ Deviance แทน RSS — ตีความคล้ายกัน

**Pearson Residual (สำหรับ Poisson):**
$$r_i^P = \frac{y_i - \hat{\lambda}_i}{\sqrt{\hat{\lambda}_i}}$$

**Deviance Residual:**
$$r_i^D = \text{sign}(y_i - \hat{\mu}_i)\sqrt{d_i}$$
→ plot แบบ same diagnostic as Linear Regression

```python
# GLM Diagnostics
influence = poisson_glm.get_influence()
pearson_res = influence.resid_pearson

# Plot Pearson residuals vs fitted
plt.scatter(poisson_glm.fittedvalues, pearson_res, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values (λ̂)'); plt.ylabel('Pearson Residuals')
```

**Overdispersion (Poisson):**
- ถ้า Var(Y) > Mean(Y) → overdispersed → ใช้ Quasi-Poisson หรือ Negative Binomial แทน

---

## Slide 9 — Summary: GLM Framework

**Key Message**: GLM เปลี่ยนแค่ family + link → ใช้ได้กับ response Y ทุกชนิด

**GLM สรุป:**
- 3 components: Random (distribution) + Systematic (linear predictor) + Link (g)
- Fit ด้วย MLE (IRLS algorithm)
- Evaluation: Deviance แทน RSS

**Python pattern:**
```python
smf.glm(formula, data=df, family=sm.families.XXX()).fit()
```

| XXX | ใช้กับ |
|-----|-------|
| `Gaussian()` | continuous Y |
| `Binomial()` | binary Y |
| `Poisson()` | count Y |
| `Gamma()` | positive continuous Y |

**Connection:**
- ← Week 8–10: Linear Regression = GLM with Gaussian + Identity link
- ← Week 11: Logistic Regression = GLM with Binomial + Logit link
- → Week 13: Poisson Regression = GLM with Poisson + Log link
