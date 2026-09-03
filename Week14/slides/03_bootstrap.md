# Slide Deck 3: The Bootstrap
> Week 14 | 8 slides | CLO4

---

## Slide 1 — Title

**The Bootstrap: Resampling for Uncertainty Quantification**  
ISLP Ch.5.2 | Week 14 | CLO4

---

## Slide 2 — The Core Problem

**Key Message**: Bootstrap แก้ปัญหาการประเมิน SE ของ statistic ที่ไม่มีสูตรปิด

**Problem:**
- SE(β̂₁) มีสูตรปิด: SE(β̂₁) = σ/√Σ(xᵢ−x̄)²
- SE(median) → ไม่มีสูตรง่าย
- SE(correlation) → complex formula
- SE(max eigenvalue) → ไม่มีสูตร
- SE(portfolio weight α) → ไม่มีสูตร

**Bootstrap Solution:**
> **Simulate the sampling distribution** ของ statistic โดยไม่ต้องรู้ underlying distribution

**Key insight:**
- ถ้าเราเก็บ sample ใหม่จาก population ได้ → estimate SE ได้
- แต่ population ไม่มี → **ใช้ sample ที่มีแทน** (sampling with replacement)

**[FIGURE: 3-tier diagram — Population → Sample → Bootstrap Samples, ลูกศรชี้จาก sample ลงมา bootstrap]**

---

## Slide 3 — Bootstrap Algorithm

**Key Message**: Bootstrap = สุ่ม with replacement จาก data ทำซ้ำ B ครั้ง แล้วดู distribution ของ statistic

**Algorithm:**
1. มี original data: Z = {z₁, z₂, ..., zₙ}
2. สร้าง B bootstrap samples: Z*₁, Z*₂, ..., Z*_B
   - แต่ละ Z*ᵦ: sample n observations **with replacement** จาก Z
3. คำนวณ statistic T บน แต่ละ Z*ᵦ: T*₁, T*₂, ..., T*_B
4. Bootstrap SE = std({T*ᵦ})

**ทำไม "with replacement"?**
- ถ้า sample without replacement → ได้ original data เสมอ
- With replacement → simulation ของ "sampling from population"

**[FIGURE: n=5 example: original [1,2,3,4,5], bootstrap 1 = [1,1,3,4,5], bootstrap 2 = [2,3,3,4,5], ...]**

```python
n = len(data)
B = 1000
boot_estimates = np.zeros(B)
for b in range(B):
    idx = np.random.choice(n, n, replace=True)  # ← with replacement!
    boot_sample = data[idx]
    boot_estimates[b] = compute_statistic(boot_sample)
se_boot = boot_estimates.std()
```

---

## Slide 4 — Bootstrap SE vs Analytical SE

**Key Message**: Bootstrap SE ≈ Analytical SE — validation ว่า Bootstrap ถูกต้อง

**Example: SE(β̂₁) ใน SLR**

Analytical formula:
$$\text{SE}(\hat{\beta}_1) = \frac{\hat{\sigma}}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}}$$

Bootstrap (B=1000):
$$\text{SE}_B(\hat{\beta}_1) = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (\hat{\beta}_{1,b}^* - \bar{\hat{\beta}}_1^*)^2}$$

**ผลลัพธ์ (ตัวอย่าง, n=100):**

| Method | SE(β̂₁) |
|--------|--------|
| Analytical | 0.0453 |
| Bootstrap (B=100) | 0.0467 |
| Bootstrap (B=1000) | 0.0451 |

**สังเกต:** Bootstrap ลู่เข้าสู่ analytical SE เมื่อ B ใหญ่ขึ้น

**[FIGURE: histogram ของ B=1000 bootstrap β̂₁ values — bell-shaped, centered at point estimate]**

---

## Slide 5 — Bootstrap Confidence Intervals

**Key Message**: Bootstrap CI ไม่ต้องสมมติ normality — งานได้กับ distribution ใดๆ

**Method 1: Percentile Bootstrap CI**
$$[\hat{\theta}^*_{(B \cdot \alpha/2)}, \hat{\theta}^*_{(B \cdot (1-\alpha/2))}]$$

**95% CI:**
```python
ci_low  = np.percentile(boot_estimates, 2.5)
ci_high = np.percentile(boot_estimates, 97.5)
```

**Method 2: Normal Approximation**
$$\hat{\theta} \pm 1.96 \cdot \text{SE}_{boot}$$

**เมื่อ method ต่างกัน:**
- ถ้า distribution symmetric: ทั้ง 2 methods ≈ เหมือนกัน
- ถ้า distribution skewed: Percentile Bootstrap ดีกว่า

**[FIGURE: histogram ของ bootstrap estimates + 2 CI methods — ต่างกันเมื่อ distribution ไม่ symmetric]**

---

## Slide 6 — Bootstrap for Portfolio Optimization (ISLP Example)

**Key Message**: Bootstrap ใช้ประเมิน SE ของ optimal portfolio weight ที่ไม่มีสูตร SE ง่ายๆ

**Problem (ISLP 5.2):**
มี 2 assets X และ Y ต้องการหา α ที่ minimize variance ของ portfolio:

$$\alpha^* = \frac{\sigma_Y^2 - \sigma_{XY}}{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}}$$

แต่ σ²ₓ, σ²ᵧ, σ_{XY} ไม่ทราบ → estimate จาก data → α̂ มี variability

**Bootstrap SE ของ α̂:**
```python
def alpha_hat(returns):
    X, Y = returns[:,0], returns[:,1]
    var_X = np.var(X); var_Y = np.var(Y)
    cov = np.cov(X, Y)[0,1]
    return (var_Y - cov) / (var_X + var_Y - 2*cov)

B = 1000
boot_alpha = [alpha_hat(data[np.random.choice(n, n, replace=True)]) for _ in range(B)]
print(f'SE(α̂) = {np.std(boot_alpha):.4f}')
```

**[FIGURE: histogram ของ bootstrap α̂ — centered at sample estimate, spread shows uncertainty]**

---

## Slide 7 — Bootstrap for Model Validation

**Key Message**: Bootstrap สามารถ estimate test error ได้เช่นกัน — ทางเลือกนอกจาก CV

**.632 Bootstrap Estimator:**
$$\hat{\text{Err}}^{.632} = 0.368 \cdot \overline{\text{err}} + 0.632 \cdot \hat{\text{Err}}^{1}$$

- $\overline{\text{err}}$ = training error
- $\hat{\text{Err}}^{1}$ = bootstrap OOB error

**Out-of-Bag (OOB) Error:**
- ใน bootstrap sample ขนาด n, ประมาณ **36.8%** ของ observations ไม่ถูก sample
- OOB observations = "natural test set" สำหรับแต่ละ bootstrap

**[FIGURE: bootstrap sample diagram — 63.2% in-bag (dark), 36.8% OOB (light) → OOB เป็น free test set]**

**Note:** Random Forest ใช้ OOB error เป็น built-in validation (จะเรียนในวิชา ML)

---

## Slide 8 — Summary: Bootstrap

**Key Message**: Bootstrap = universal tool สำหรับ uncertainty quantification

**Bootstrap สรุป:**
- Algorithm: sample with replacement → compute statistic → repeat B times
- SE: std ของ B bootstrap estimates
- 95% CI: percentile method = np.percentile(boot, [2.5, 97.5])
- Works for **any statistic** — median, correlation, eigenvalue, model coefficient

**When to use Bootstrap:**
1. ไม่มีสูตร SE ของ statistic ที่สนใจ
2. Sample distribution ไม่ Normal → analytical CI ผิด
3. ต้องการ CI ของ complex function ของ parameters
4. Validate analytical SE (sanity check)

**Rule of thumb:**
- B = 100: rough estimate
- B = 1000: reliable SE
- B = 10000: precise CI tail probabilities

**Python:**
```python
boot_results = [stat(data[np.random.choice(n, n, replace=True)]) for _ in range(1000)]
se_boot = np.std(boot_results)
ci = np.percentile(boot_results, [2.5, 97.5])
```
