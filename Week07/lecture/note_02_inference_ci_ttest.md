# Note 2: Standard Error, Confidence Intervals และ Hypothesis Testing
> Week 7 | CLO2 | ISLP Reference: Ch.3.1.2 (Statistical Inference)

---

## บทนำ

บันทึกนี้คือสะพานที่เชื่อม EDA กับ Statistical Learning: **Confidence Intervals (CI)** และ **Hypothesis Testing** ซึ่งจะถูกใช้ใน ISLP Chapter 3 ทุกหน้า เมื่อ fit Linear Regression และดู p-value ของ $\hat{\beta}$ นั่นคือสิ่งที่เรียนในสัปดาห์นี้ เป้าหมายคือให้นักศึกษาเข้าใจ Central Limit Theorem และ Standard Error, สร้าง CI ได้, ทำ t-test และอ่าน p-value ได้ถูกต้อง และสำคัญที่สุด: รู้ว่า "p < 0.05" หมายความว่าอะไร**จริง ๆ** (ไม่ใช่แค่กด button แล้วอ่านผล) ใน Data Science ทุก regression output ที่เห็น p-value, SE, confidence interval — ทั้งหมดมาจากทฤษฎีในบันทึกนี้

---

## Section 1: Sampling Distributions และ Central Limit Theorem

ในส่วนนี้เราจะเรียนรู้ว่า **Sampling Distribution** คืออะไร และทำไม **Central Limit Theorem (CLT)** ถึงเป็นเหตุผลที่ Statistical Inference ใช้งานได้ในชีวิตจริง

### 1.1 Population vs Sample

**Population**: ทุก data point ที่เราสนใจ (มักใหญ่หรือ infinite)  
**Sample**: subset ที่เราวัดได้จริง (ขนาด $n$)

เราต้องการ **inference** จาก sample → population:
- $\bar{x}$ (sample mean) เป็น estimator ของ $\mu$ (population mean)
- $s^2$ (sample variance) เป็น estimator ของ $\sigma^2$ (population variance)

### 1.2 Sampling Distribution ของ $\bar{X}$

ถ้าเราเก็บ samples หลาย ๆ ครั้ง (ขนาด $n$ แต่ละครั้ง) $\bar{x}$ จาก samples ต่าง ๆ จะกระจายตัวเป็น **Sampling Distribution**

**Central Limit Theorem (CLT)**:  
สำหรับ sample ขนาด $n$ จาก population ใด ๆ (ที่มี $\mu$ และ $\sigma^2$ จำกัด):

$$\bar{X} \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{เมื่อ } n \to \infty$$

**ในทางปฏิบัติ**: $n \geq 30$ มักพอเพียง

### 1.3 Standard Error (SE)

$$\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

ถ้าไม่รู้ $\sigma$ ใช้ $s$ แทน:
$$\hat{\text{SE}}(\bar{X}) = \frac{s}{\sqrt{n}}$$

**ความหมาย**: SE วัด "ความแม่นยำ" ของ $\bar{x}$ ในฐานะ estimator ของ $\mu$
- $n$ ใหญ่ขึ้น → SE เล็กลง → estimate แม่นยำขึ้น
- $\sigma$ ใหญ่ขึ้น → SE ใหญ่ขึ้น → estimate ไม่แม่นยำ

**ตัวอย่าง**: วัด salary 30 คน: $\bar{x} = 45,000$, $s = 8,000$  
$$\text{SE} = \frac{8000}{\sqrt{30}} = \frac{8000}{5.477} \approx 1461 \text{ บาท}$$

**DS Connection**: ใน regression output, `std err` หรือ `SE` ของ $\hat{\beta}$ คือ Standard Error ของ coefficient estimator เป็นพื้นฐานของ t-test และ CI ใน Week 8

---

## Section 2: Confidence Intervals  *(ISLP 3.1.2)*

ในส่วนนี้เราจะเรียนรู้ **Confidence Interval** ซึ่งเป็นช่วงที่บอกว่า parameter ที่แท้จริงน่าจะอยู่ที่ไหน — และตีความได้ถูกต้องอย่างไร

### 2.1 สูตร CI สำหรับ Population Mean

**ถ้ารู้ $\sigma$** (กรณีทฤษฎี, ใช้ Normal distribution):
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

โดย $z_{0.025} = 1.96$ สำหรับ 95% CI

**ถ้าไม่รู้ $\sigma$** (กรณีจริง, ใช้ t-distribution):
$$\bar{x} \pm t^*_{n-1, \alpha/2} \cdot \frac{s}{\sqrt{n}}$$

โดย $t^*$ คือ critical value จาก t-distribution ที่ degrees of freedom = $n-1$

**สำหรับ $n = 30$, 95% CI**: $t^*_{29, 0.025} \approx 2.045$ (ใกล้กับ 1.96)

### 2.2 การตีความ CI (Critical!)

**ที่ถูกต้อง**:
> "ถ้าเราเก็บ sample ซ้ำหลายครั้งและสร้าง CI ทุกครั้ง 95% ของ intervals เหล่านั้นจะครอบคลุม true $\mu$"

**ที่ผิด** (ห้ามพูด!):
> ❌ "ความน่าจะเป็น 95% ที่ $\mu$ อยู่ใน interval นี้" — $\mu$ เป็นค่าคงที่ ไม่ใช่ random variable!

### 2.3 ตัวอย่าง step-by-step

**Data**: age ของผู้ใช้แอป 25 คน: mean = 32.4, std = 6.8

**95% CI**:
1. $t^*_{24, 0.025} = 2.064$ (จาก t-table หรือ scipy)
2. $\text{SE} = 6.8/\sqrt{25} = 6.8/5 = 1.36$
3. Margin of Error = $2.064 \times 1.36 = 2.81$
4. CI = $[32.4 - 2.81,\ 32.4 + 2.81] = [29.59,\ 35.21]$

ตีความ: ด้วยความเชื่อมั่น 95% อายุเฉลี่ยของผู้ใช้ทั้งหมดอยู่ระหว่าง 29.6 ถึง 35.2 ปี

```python
# ─── Confidence Interval ด้วย SciPy ──────────────────────────────────
# วัตถุประสงค์: คำนวณ CI สำหรับ population mean จาก sample
import numpy as np
from scipy import stats

np.random.seed(42)

# สมมติ sample อายุผู้ใช้แอป
ages = np.random.normal(32, 7, 25)  # true mean=32, std=7, n=25

n = len(ages)
mean = ages.mean()
se = ages.std(ddof=1) / np.sqrt(n)

# 95% CI ด้วย t-distribution (σ ไม่ทราบ)
# วัตถุประสงค์: t.interval ให้ CI ที่ถูกต้องเมื่อ n เล็กและ σ ไม่ทราบ
alpha = 0.05
ci_95 = stats.t.interval(1 - alpha, df=n-1, loc=mean, scale=se)
print(f"Sample mean: {mean:.2f}")
print(f"SE: {se:.2f}")
print(f"95% CI: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")

# เปรียบเทียบ CI ที่ confidence levels ต่างกัน
# วัตถุประสงค์: เห็นว่า CI กว้างขึ้นเมื่อ confidence level สูงขึ้น
for conf in [0.90, 0.95, 0.99]:
    ci = stats.t.interval(conf, df=n-1, loc=mean, scale=se)
    print(f"{conf*100:.0f}% CI: ({ci[0]:.2f}, {ci[1]:.2f}), width = {ci[1]-ci[0]:.2f}")

# แสดงผลของ sample size ต่อ CI width
# วัตถุประสงค์: เห็น √n ใน denominator ทำงานอย่างไร
print("\nEffect of sample size (same mean & std, 95% CI):")
for n_size in [10, 25, 50, 100, 500]:
    sample = np.random.normal(32, 7, n_size)
    ci = stats.t.interval(0.95, df=n_size-1, loc=sample.mean(),
                          scale=sample.std(ddof=1)/np.sqrt(n_size))
    print(f"  n={n_size:4d}: ({ci[0]:.2f}, {ci[1]:.2f}), width = {ci[1]-ci[0]:.2f}")
```

---

## Section 3: Hypothesis Testing  *(ISLP 3.1.2)*

ในส่วนนี้เราจะเรียนรู้ **Hypothesis Testing** ซึ่งเป็น formal framework ในการตัดสินใจทางสถิติ และเป็นพื้นฐานของการแปลผล regression ใน ISLP ทั้งหมด

### 3.1 Framework

1. **$H_0$** (Null Hypothesis): สมมติฐานที่ "ไม่มีผล" เช่น $\beta_1 = 0$ (TV ไม่มีผลต่อ Sales)
2. **$H_1$** (Alternative Hypothesis): สิ่งที่เราต้องการ test เช่น $\beta_1 \neq 0$
3. **Significance Level** $\alpha$: threshold ที่ยอมรับ (มักใช้ 0.05)
4. **Test Statistic**: statistics ที่ compute จาก data
5. **p-value**: probability ของ observing data ที่ extreme กว่าหรือเท่ากับที่เห็น **ถ้า $H_0$ เป็นจริง**
6. **Decision**: reject $H_0$ ถ้า p-value $< \alpha$

### 3.2 t-test

**One-Sample t-test**: ทดสอบว่า $\mu = \mu_0$

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \sim t_{n-1} \text{ (ถ้า } H_0 \text{ เป็นจริง)}$$

**ตัวอย่าง step-by-step**:

H₀: คะแนนเฉลี่ย = 70, H₁: $\neq 70$, $\alpha = 0.05$  
Data: n=25, $\bar{x}=74$, $s=10$

$$t = \frac{74 - 70}{10/\sqrt{25}} = \frac{4}{2} = 2.0$$

Critical value: $t_{24, 0.025} = 2.064$  
$|t| = 2.0 < 2.064$ → **Fail to reject $H_0$**

(หรือคำนวณ p-value ≈ 0.057 > 0.05 → Fail to reject)

### 3.3 p-value: ตีความให้ถูกต้อง

**p-value คือ**: ถ้า $H_0$ เป็นจริง ความน่าจะเป็นที่จะเห็น test statistic ที่มีค่าเท่านี้หรือ extreme กว่า

**p-value ไม่ใช่**:
- ❌ ความน่าจะเป็นที่ $H_0$ เป็นจริง
- ❌ บอกว่า effect มี practical significance
- ❌ แปลว่า "ผล significant = สำคัญในชีวิตจริง"

**Common Mistakes**:
- p = 0.049 vs p = 0.051 — ต่างกันอย่างมีนัยสำคัญจริง ๆ หรือ?
- ตัวอย่าง: n=10,000, ค่าเฉลี่ยต่างกัน 0.001 บาท → p < 0.001 (significant!) แต่ไม่มีความหมายในชีวิตจริง

### 3.4 Connection กับ Regression

ใน output ของ `statsmodels`:
```
           coef    std err          t      P>|t|    [0.025    0.975]
Intercept  7.032      0.458     15.36      0.000     6.130     7.935
TV         0.048      0.003     17.67      0.000     0.043     0.054
```

- `coef` = $\hat{\beta}$
- `std err` = SE($\hat{\beta}$) = $\hat{\sigma}\sqrt{(X^TX)^{-1}_{jj}}$
- `t` = $\hat{\beta}$ / SE (test $H_0: \beta = 0$)
- `P>|t|` = p-value สำหรับ two-sided test
- `[0.025 0.975]` = 95% CI สำหรับ $\beta$

```python
# ─── t-test และ hypothesis testing ด้วย SciPy ─────────────────────────
# วัตถุประสงค์: แสดง complete hypothesis testing workflow
import numpy as np
from scipy import stats

np.random.seed(42)

# ─── One-sample t-test ─────────────────────────────────────────────────
# H₀: mean salary = 40,000 บาท
salary_sample = np.random.normal(43000, 8000, 30)
t_stat, p_value = stats.ttest_1samp(salary_sample, popmean=40000)
print(f"One-sample t-test:")
print(f"  t = {t_stat:.4f}, p-value = {p_value:.4f}")
print(f"  Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# ─── Two-sample t-test ────────────────────────────────────────────────
# H₀: salary ชาย = salary หญิง
# วัตถุประสงค์: เปรียบเทียบค่าเฉลี่ยสอง groups อิสระ
male_salary = np.random.normal(46000, 9000, 50)
female_salary = np.random.normal(43000, 8000, 50)

t_2, p_2 = stats.ttest_ind(male_salary, female_salary, equal_var=False)
ci_diff = stats.t.interval(0.95, df=98,
                            loc=male_salary.mean() - female_salary.mean(),
                            scale=np.sqrt(male_salary.var()/50 + female_salary.var()/50))
print(f"\nTwo-sample t-test (salary by gender):")
print(f"  Male mean: {male_salary.mean():.0f}, Female mean: {female_salary.mean():.0f}")
print(f"  t = {t_2:.4f}, p-value = {p_2:.4f}")
print(f"  95% CI for difference: ({ci_diff[0]:.0f}, {ci_diff[1]:.0f})")
print(f"  Decision: {'Reject H₀' if p_2 < 0.05 else 'Fail to reject H₀'}")

# ─── Effect size: Cohen's d ───────────────────────────────────────────
# วัตถุประสงค์: p-value บอก "significant?" แต่ effect size บอก "มากแค่ไหน?"
pooled_std = np.sqrt((male_salary.var() + female_salary.var()) / 2)
cohens_d = (male_salary.mean() - female_salary.mean()) / pooled_std
print(f"\nCohen's d = {cohens_d:.3f} (small=0.2, medium=0.5, large=0.8)")
```

---

## Case Study: A/B Test — วัดผลการเปลี่ยน UI ด้วย t-test

**Scenario**  
บริษัท E-commerce เปลี่ยนสีปุ่ม "Buy Now" จากน้ำเงินเป็นแดง และต้องการทดสอบว่าเพิ่ม conversion rate จริงหรือไม่

**Data**:
- Control (น้ำเงิน): 1,000 users, 100 conversions = 10%
- Treatment (แดง): 1,000 users, 120 conversions = 12%

**Method — Two-proportion z-test**:

```python
# ─── A/B Test Analysis ──────────────────────────────────────────────────
# วัตถุประสงค์: ทดสอบว่า conversion rate ต่างกันอย่างมีนัยสำคัญหรือไม่
from scipy import stats
import numpy as np

n_ctrl = 1000; x_ctrl = 100   # control group
n_trt = 1000;  x_trt = 120    # treatment group

p_ctrl = x_ctrl / n_ctrl      # 0.10
p_trt = x_trt / n_trt         # 0.12
p_pool = (x_ctrl + x_trt) / (n_ctrl + n_trt)  # pooled proportion

# Z-test สำหรับ two proportions
# วัตถุประสงค์: ทดสอบ H₀: p_ctrl = p_trt
z = (p_trt - p_ctrl) / np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_trt))
p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided

print(f"Control rate: {p_ctrl*100:.1f}%")
print(f"Treatment rate: {p_trt*100:.1f}%")
print(f"Absolute lift: {(p_trt - p_ctrl)*100:.1f} percentage points")
print(f"Relative lift: {(p_trt/p_ctrl - 1)*100:.1f}%")
print(f"Z = {z:.4f}, p-value = {p_value:.4f}")
print(f"Decision: {'Significant change!' if p_value < 0.05 else 'No significant change'}")
```

**Result**: Z = 1.45, p-value = 0.147 → **ไม่ significant** แม้ว่า conversion จะเพิ่มจาก 10% → 12%

**Insight**  
- effect ดูเหมือนใหญ่ (2 percentage points) แต่ p-value บอกว่าอาจเกิดจาก chance
- ต้อง collect data เพิ่ม หรือทำ Power Analysis ก่อน design experiment เพื่อให้มี enough statistical power
- ใน Tech industry นี้คือขั้นตอนปกติของ A/B testing ใน Google, Meta, Netflix ทุกสัปดาห์

---

## สรุป (Summary)

| แนวคิด | สูตร | Python |
|--------|------|--------|
| SE ของ $\bar{X}$ | $\sigma/\sqrt{n}$ หรือ $s/\sqrt{n}$ | manual หรือ `sem(x)` |
| 95% CI (σ ไม่ทราบ) | $\bar{x} \pm t^* \cdot s/\sqrt{n}$ | `scipy.stats.t.interval(0.95,...)` |
| One-sample t | $t = (\bar{x}-\mu_0)/(s/\sqrt{n})$ | `scipy.stats.ttest_1samp` |
| Two-sample t | $t = (\bar{x}_1-\bar{x}_2)/\text{SE}_{\text{diff}}$ | `scipy.stats.ttest_ind` |
| p-value | $\Pr(\|T\| \geq \|t\| \mid H_0)$ | output ของ ttest |
| Decision | Reject $H_0$ ถ้า p < α | `if p_value < 0.05` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 7 (note 1)**: EDA ให้ descriptive statistics → บันทึกนี้ขยายสู่ inference
- ← **Week 6**: Bayes Theorem → Frequentist (สัปดาห์นี้) กับ Bayesian inference เป็น 2 frameworks ที่ต่างกัน
- → **Week 8**: ใน SLR output: SE(β̂₁), t-statistic, p-value, 95% CI ของ β̂ล้วนมาจากทฤษฎีในบันทึกนี้
- → **Week 9**: F-statistic ใน MLR คือ multi-variable generalization ของ t-test (test ว่า model ใด ๆ useful หรือไม่)
