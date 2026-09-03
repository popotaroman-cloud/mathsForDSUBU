# Slide Deck: Sampling Distributions and Standard Error
> Week 07 | CLO2 | Math Bridge to ISLP Ch.3 | 9 slides

---
## Slide 1 — Title
**Sampling Distributions และ Standard Error**  
Week 7 | CLO2 | Bridge: CLT → SE → CI → t-test → Regression Inference

---
## Slide 2 — Population vs Sample
**Key Message**: เราเห็นแค่ sample แต่ต้องการสรุปเกี่ยวกับ population

[FIGURE: diagram แสดง population (ทุกคน) → random sampling → sample → inference → population]

| | Population | Sample |
|--|-----------|--------|
| Size | N (มักไม่รู้) | n (รู้) |
| Mean | μ (parameter) | x̄ (statistic) |
| SD | σ (parameter) | s (statistic) |
| Proportion | p | p̂ |

**เป้าหมาย**: ใช้ statistic ของ sample (x̄, s) ประมาณ parameter ของ population (μ, σ)

---
## Slide 3 — Sampling Distribution ของ x̄
**Key Message**: x̄ เองมี distribution — ถ้า sample หลาย ๆ ครั้ง x̄ จะกระจายรอบ μ

**ทดลอง**: sample 100 ค่าจาก population 10,000 ครั้ง → plot x̄ ทั้ง 10,000 ค่า

[FIGURE: histogram ของ population (skewed) vs histogram ของ 1000 sample means (normal)]

**ผลลัพธ์**: distribution ของ x̄ เป็น normal ไม่ว่า population จะมี distribution อะไร!
→ **Central Limit Theorem**

---
## Slide 4 — Central Limit Theorem (CLT)
**Key Message**: CLT คือ foundation ของ statistical inference ทั้งหมด

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

**เงื่อนไข**: n ≥ 30 (rule of thumb สำหรับ most distributions)

```python
# Simulation CLT
population = np.random.exponential(scale=2, size=10000)  # skewed!
sample_means = [np.random.choice(population, 50).mean() for _ in range(1000)]

plt.hist(population, bins=50, alpha=0.5, label='Population')
plt.hist(sample_means, bins=50, alpha=0.5, label='Sample means (n=50)')
plt.legend()
```

---
## Slide 5 — Standard Error (SE)
**Key Message**: SE วัดว่า x̄ ของเรา "แม่น" แค่ไหนในการประมาณ μ — ยิ่ง n มาก SE ยิ่งเล็ก

$$SE(\bar{X}) = \frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}}$$

```python
n = len(df['Sales'])
s = df['Sales'].std(ddof=1)
se = s / np.sqrt(n)

print(f'n = {n}')
print(f's = {s:.4f}')
print(f'SE = {se:.4f}')
# SE คือ SD ของ sampling distribution ของ x̄
```

**Insight**: ถ้า n เพิ่ม 4 เท่า → SE ลด 2 เท่า (√n ใน denominator)

---
## Slide 6 — SE ใน Regression (Preview Week 8)
**Key Message**: SE ของ β̂₁ ใช้ logic เดียวกับ SE ของ x̄ — นี่คือ bridge สำคัญ

**SE ของ x̄**: $SE(\bar{X}) = \frac{s}{\sqrt{n}}$

**SE ของ β̂₁ (regression)**: $SE(\hat{\beta}_1) = \frac{\hat{\sigma}}{\sqrt{\sum(x_i - \bar{x})^2}}$

**ทั้งสองอย่าง**:
- ยิ่ง σ (noise) มาก → SE มาก → ประมาณได้แม่นน้อยลง
- ยิ่ง n มาก → SE น้อย → ประมาณได้แม่นขึ้น

---
## Slide 7 — t-distribution vs Normal
**Key Message**: เมื่อ σ ไม่รู้ (ซึ่งปกติเสมอ) → ใช้ t-distribution แทน Normal

[FIGURE: t-distribution degrees of freedom 1, 5, 30, infinity (=normal)]

- **n ≥ 30**: t ≈ Normal → ผลต่างน้อยมาก
- **n < 30**: t มีหาง "หนักกว่า" → CI กว้างขึ้น (เพราะ uncertainty มากขึ้น)
- **df = n − 1** สำหรับ one-sample t-test
- **df = n − 2** สำหรับ regression (slope)

```python
# t critical value สำหรับ 95% CI
n = 25
t_star = stats.t.ppf(0.975, df=n-1)   # n small → t > 1.96
z_star = stats.norm.ppf(0.975)          # = 1.96
print(f't*(n=25) = {t_star:.3f} > z* = {z_star:.3f}')
```

---
## Slide 8 — Law of Large Numbers
**Key Message**: ยิ่ง n มาก x̄ → μ — ทำให้ estimation แม่นขึ้นเรื่อย ๆ

```python
# Simulation: x̄ converges to μ = 5 as n increases
mu = 5
for n in [5, 10, 50, 100, 500, 1000]:
    sample = np.random.normal(mu, 2, n)
    x_bar = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)
    print(f'n={n:5d}: x̄={x_bar:.3f}, SE={se:.4f}')
```

[FIGURE: line plot แสดง x̄ convergence เมื่อ n เพิ่มขึ้น]

---
## Slide 9 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- Sample statistics (x̄, s) ประมาณ population parameters (μ, σ)
- **CLT**: x̄ ~ N(μ, σ²/n) สำหรับ n ≥ 30
- **SE** = s/√n = accuracy ของ x̄ ในการประมาณ μ
- t-distribution ใช้เมื่อ σ ไม่รู้ (แทบทุกครั้ง)

**Connection**:
```
SE → CI (± t* × SE) → t-test (t = estimate/SE) → p-value
```

**สัปดาห์ต่อไป — ในชุดนี้**: CI และ Hypothesis Testing
