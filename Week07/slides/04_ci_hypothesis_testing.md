# Slide Deck: Confidence Intervals and Hypothesis Testing
> Week 07 | CLO2 | ISLP 3.1.2 | 10 slides

---
## Slide 1 — Title
**Confidence Intervals และ Hypothesis Testing**  
Week 7 | CLO2 | Bridge โดยตรงสู่ regression inference ใน Week 8–9

---
## Slide 2 — Confidence Interval คืออะไร?
**Key Message**: CI คือ range ที่เชื่อว่า true parameter อยู่ข้างใน — ไม่ใช่ probability ที่ parameter อยู่ใน

**นิยาม 95% CI**: ถ้าเก็บ sample ซ้ำ 100 ครั้งและสร้าง CI ทุกครั้ง จะมี 95 intervals ที่ครอบ true μ

$$\bar{X} \pm t^* \cdot \frac{s}{\sqrt{n}}$$

[FIGURE: 20 CI intervals — 19 ครอบ true μ, 1 ไม่ครอบ]

**Misconception ที่พบบ่อย**:
- ❌ "95% probability ที่ μ อยู่ใน CI" → ผิด! μ คงที่ ไม่มี probability
- ✓ "95% ของ CI แบบนี้จะครอบ μ" → ถูก

---
## Slide 3 — คำนวณ 95% CI สำหรับ Mean
**Key Message**: 4 ขั้นตอน: คำนวณ x̄, s, SE, แล้ว × t*

```python
from scipy import stats

sales = df['Sales'].values
n = len(sales)
x_bar = sales.mean()
s     = sales.std(ddof=1)
se    = s / np.sqrt(n)

# t* สำหรับ 95% CI (two-tailed)
t_star = stats.t.ppf(0.975, df=n-1)  # 1-α/2 = 0.975
ci = (x_bar - t_star*se, x_bar + t_star*se)
print(f'95% CI for μ_sales: ({ci[0]:.3f}, {ci[1]:.3f})')

# หรือใช้ scipy โดยตรง
ci2 = stats.t.interval(0.95, df=n-1, loc=x_bar, scale=se)
```

**CI width ควบคุมโดย**: n ↑ → width ↓ | confidence ↑ → width ↑

---
## Slide 4 — CI ใน Regression (สำคัญมาก!)
**Key Message**: CI ของ β̂₁ ใช้สูตรเดียวกัน — แค่เปลี่ยน estimate และ SE

**CI สำหรับ β̂₁ (Week 8)**:
$$\hat{\beta}_1 \pm 2 \cdot SE(\hat{\beta}_1)$$

**จาก statsmodels output (Week 8)**:
```
               coef    std err    t      P>|t|  [0.025  0.975]
Intercept     7.032      0.458   15.36  0.000   6.130   7.935
TV            0.048      0.003   17.67  0.000   0.042   0.053
```

- coef = β̂₁ = 0.048
- std err = SE(β̂₁) = 0.003
- [0.025, 0.975] = **95% CI** = (0.042, 0.053)
- เพราะ 0 ∉ CI → β₁ ≠ 0 → TV มีผลต่อ Sales

---
## Slide 5 — Hypothesis Testing Framework
**Key Message**: Hypothesis test ตอบ "ข้อมูลที่เห็นสอดคล้องกับ H₀ ได้แค่ไหน?"

**5 ขั้นตอน**:
1. ตั้ง H₀ (null) และ H₁ (alternative)
2. เลือก significance level α (มักใช้ 0.05)
3. คำนวณ test statistic (t, χ², F)
4. คำนวณ p-value
5. ตัดสินใจ: reject H₀ ถ้า p < α

**t-statistic**:
$$t = \frac{\text{estimate} - H_0 \text{ value}}{SE}$$

---
## Slide 6 — p-value: ความหมายที่ถูกต้อง
**Key Message**: p-value ≠ probability ที่ H₀ จริง — เป็น probability ของ data ภายใต้ H₀

**นิยาม**: P(|T| ≥ |t_obs| | H₀ เป็นจริง)

[FIGURE: t-distribution สองหาง แสดง critical region และ p-value area]

```python
# One-sample t-test: H₀: μ = 14
t_stat, p_val = stats.ttest_1samp(sales, popmean=14)
print(f't = {t_stat:.4f}, p = {p_val:.4f}')
# ถ้า p < 0.05 → reject H₀
```

**Misconceptions**:
- ❌ p = 0.03 ≠ "3% chance H₀ is true"
- ✓ p = 0.03: ถ้า H₀ จริง โอกาสได้ t ≥ t_obs นี้ = 3%

---
## Slide 7 — One-Sample, Two-Sample, Paired t-test
**Key Message**: เลือก t-test ให้ถูกตาม design ของ study

| Test | เมื่อใช้ | Python |
|------|---------|--------|
| One-sample | เปรียบ x̄ กับ μ₀ ที่กำหนด | `ttest_1samp(data, popmean)` |
| Two-sample (independent) | เปรียบ 2 กลุ่มแยกกัน | `ttest_ind(grp1, grp2)` |
| Paired | เปรียบ before-after บนคนเดิม | `ttest_rel(before, after)` |

```python
# Two-sample: H₀: mean_high_tv == mean_low_tv
t, p = stats.ttest_ind(high_tv_sales, low_tv_sales)

# Paired: H₀: treatment ไม่มีผล
t, p = stats.ttest_rel(before, after)
```

---
## Slide 8 — Chi-square Test ของความเป็นอิสระ
**Key Message**: Chi-square test วัดว่า 2 categorical variables เกี่ยวข้องกันหรือไม่

H₀: Variable A และ Variable B เป็นอิสระต่อกัน (ไม่มีความสัมพันธ์)

```python
# Contingency table: TV_level vs Sales_level
contingency = pd.crosstab(df['TV_level'], df['Sales_level'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f'χ² = {chi2:.3f}, p = {p:.4f}, dof = {dof}')
# p < 0.05 → reject H₀ → TV_level และ Sales_level สัมพันธ์กัน
```

[FIGURE: contingency table + expected frequencies]

---
## Slide 9 — Connection to Regression Inference (Bridge)
**Key Message**: t-test ใน regression คือ one-sample t-test ที่ H₀: β = 0

**Regression t-test (Week 8)**:
$$t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)} \sim t(n-2)$$

```
coef    std err    t      P>|t|
0.048   0.003     17.67   0.000
```

- t = 0.048 / 0.003 = 17.67 >> 2 → reject H₀
- p ≈ 0 << 0.05 → TV มีผลต่อ Sales อย่างมีนัยสำคัญ

**นี่คือ logic เดียวกับ one-sample t-test ที่เพิ่งเรียน!**

---
## Slide 10 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- **CI**: x̄ ± t* × SE — ยิ่ง n มาก ยิ่งแคบ
- **p-value**: P(|T| ≥ |t_obs| | H₀) — ยิ่งต่ำ ยิ่ง strong evidence against H₀
- **t-test**: one-sample, two-sample, paired — เลือกตาม study design
- **Chi-square**: categorical independence test

**Bridge to Week 8**:
```
SE → CI สำหรับ β̂₁ → t-test: H₀: β₁ = 0 → p-value
→ "Is TV a significant predictor of Sales?"
```

**สัปดาห์ต่อไป — Week 8**: Simple Linear Regression + Full Inference
