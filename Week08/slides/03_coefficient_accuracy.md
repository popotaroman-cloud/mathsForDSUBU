# Slide Deck: Coefficient Accuracy — SE, CI, t-test
> Week 08 | CLO3 | ISLP Ch.3.1.2 | 10 slides

---
## Slide 1 — Title
**Assessing Accuracy of Coefficients: SE, CI, t-statistic, p-value**  
Week 8 | CLO3 | ISLP 3.1.2  
LLo: ประเมินความแม่นยำของ β̂ ด้วย SE, CI, t-test และอ่าน statsmodels summary ได้

---
## Slide 2 — ทำไมต้องประเมินความแม่นยำ?
**Key Message**: β̂ เป็น estimate จาก sample ที่เลือกมา — ถ้าเลือก sample ใหม่ β̂ จะต่างออกไป

[FIGURE: simulation — 10 sample → 10 fitted lines ที่แตกต่างกันเล็กน้อย รอบ true line]

**ปัญหา**: เรามี data 200 ตลาด → ได้ β̂₁ = 0.0475  
- ถ้าเก็บ data ชุดใหม่อีก 200 ตลาด → β̂₁ อาจเป็น 0.044 หรือ 0.051
- ค่าจริง β₁ (true population slope) คืออะไร?

**เราต้องการ**:
1. **SE**: β̂₁ ของเรา "ไกล" จาก β₁ จริงเฉลี่ยแค่ไหน?
2. **CI**: range ที่เชื่อได้ว่า β₁ จริงอยู่ใน?
3. **t-test**: β₁ = 0 (no relationship) ได้ไหม?

---
## Slide 3 — Standard Error ของ β̂
**Key Message**: SE คือ "ความผิดพลาดเฉลี่ย" ของ estimator — SE น้อย = β̂ เชื่อถือได้มากกว่า

**สูตร SE**:
$$SE(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum(x_i - \bar{x})^2}$$
$$SE(\hat{\beta}_0)^2 = \sigma^2\left[\frac{1}{n} + \frac{\bar{x}^2}{\sum(x_i - \bar{x})^2}\right]$$

**σ² ประมาณด้วย RSE²** = RSS/(n−2)

**ปัจจัยที่ทำให้ SE น้อย (= β̂ แม่นยำ)**:
| ปัจจัย | ผล |
|--------|-----|
| n มาก | SE ↓ (ข้อมูลมาก → ประมาณได้แม่นขึ้น) |
| Var(X) สูง | SE(β̂₁) ↓ (X กระจายมาก → slope ชัดเจนขึ้น) |
| σ² ต่ำ | SE ↓ (errors เล็ก → signal ชัด) |

---
## Slide 4 — 95% Confidence Interval สำหรับ β₁
**Key Message**: CI ของ β̂₁ บอกว่า "ถ้า 0 ไม่อยู่ใน CI → TV มีผลจริง"

**สูตร CI (เหมือน Week 7 CI สำหรับ mean)**:
$$\hat{\beta}_1 \pm t^* \cdot SE(\hat{\beta}_1) \approx \hat{\beta}_1 \pm 2 \cdot SE(\hat{\beta}_1)$$

**จาก statsmodels (TV → Sales)**:
```
               coef    std err    t      P>|t|  [0.025  0.975]
Intercept     7.032      0.458   15.36  0.000   6.130   7.935
TV            0.048      0.003   17.67  0.000   0.042   0.053
```

**อ่านผล**:
- β̂₁ = 0.048, SE = 0.003
- 95% CI = (0.042, 0.053)
- **0 ∉ CI** → reject H₀: β₁ = 0 → TV มีผลต่อ Sales

**Bridge Week 7**: นี่คือสูตรเดิม x̄ ± t* × SE แค่เปลี่ยน estimate จาก x̄ เป็น β̂₁

---
## Slide 5 — t-statistic และ p-value
**Key Message**: t-test ใน regression คือ one-sample t-test ที่ H₀: β₁ = 0

**Hypothesis Test**:
- H₀: β₁ = 0 (TV ไม่มีผลต่อ Sales)
- H₁: β₁ ≠ 0 (TV มีผลต่อ Sales)

**t-statistic**:
$$t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)} = \frac{0.0475}{0.00269} \approx 17.67$$

**p-value**: P(|T| ≥ 17.67 | β₁ = 0) ≈ 2×10⁻⁴² ≈ 0

[FIGURE: t-distribution(df=198) — shaded area ≈ 0 ที่ |t| ≥ 17.67]

**ตัดสินใจ**: p ≈ 0 << 0.05 → **Reject H₀** → TV มีผลต่อ Sales อย่างมีนัยสำคัญ

---
## Slide 6 — อ่าน statsmodels Summary: ทุกบรรทัดหมายความว่าอะไร
**Key Message**: summary table บอกทุกอย่างที่ต้องการ — อ่านเป็นก็จบ

```
OLS Regression Results
=========================================================
Dep. Variable:          Sales   R-squared:         0.612
No. Observations:         200   Adj. R-squared:    0.610
Df Residuals:             198   F-statistic:       312.1
                                Prob (F-statistic): 1.47e-42
=========================================================
              coef    std err          t      P>|t|
---------------------------------------------------------
Intercept    7.0326      0.4578      15.36      0.000
TV           0.0475      0.0027      17.67      0.000
=========================================================
                   [0.025      0.975]
---------------------------------------------------------
Intercept    6.1295      7.9356
TV           0.0422      0.0528
=========================================================
```

| Column | ความหมาย |
|--------|---------|
| `coef` | β̂ (estimate) |
| `std err` | SE(β̂) |
| `t` | β̂ / SE |
| `P>|t|` | p-value (two-tailed) |
| `[0.025, 0.975]` | 95% CI |

---
## Slide 7 — Verify ด้วยมือ: t = coef / std err
**Key Message**: ตรวจสอบได้เสมอว่า t = β̂ / SE และ CI = β̂ ± 2×SE

```python
# ─── อ่านค่าจาก summary ─────────────────────────────────────
# วัตถุประสงค์: ดึงค่า coef, SE, t, p-value, CI ออกมา verify
params = model.params      # β̂₀, β̂₁
bse    = model.bse         # SE ของแต่ละ coef
tvalues= model.tvalues     # t-statistic
pvalues= model.pvalues     # p-values
conf_int = model.conf_int()  # 95% CI

# ─── verify ด้วยมือ ─────────────────────────────────────────
# วัตถุประสงค์: แสดงว่า t = coef/SE จริง ๆ
beta1_hat = params['TV']
se_beta1  = bse['TV']
t_manual  = beta1_hat / se_beta1
print(f'β̂₁ = {beta1_hat:.4f}')
print(f'SE  = {se_beta1:.4f}')
print(f't   = β̂₁/SE = {t_manual:.4f}')  # ≈ 17.67
print(f'95% CI ≈ ({beta1_hat - 2*se_beta1:.4f}, {beta1_hat + 2*se_beta1:.4f})')
```

**ผลลัพธ์**: t ≈ 17.67, CI ≈ (0.042, 0.053) — ตรงกับ summary

---
## Slide 8 — ตีความ: β̂₁ มีนัยสำคัญจริงหรือ?
**Key Message**: significant ทางสถิติ ≠ significant ทางธุรกิจ — ต้องดูขนาดผลด้วย

**Statistical Significance** (p < 0.05):
- TV: p ≈ 0 → significant ✓
- ข้อสรุป: "มีหลักฐานเพียงพอว่า TV มีผลต่อ Sales"

**Practical Significance** (ขนาดของผล):
- β̂₁ = 0.0475 หน่วย/พัน$
- ทุก $1,000 TV → 47.5 หน่วย (หน่วยละ $10 → รายได้ $475)
- ถ้า ROI > cost ของงบโฆษณา → worth investing

**⚠ ระวัง**:
- n ใหญ่มาก → p เล็กได้แม้ β̂₁ เล็กมาก ๆ (เช่น β̂₁ = 0.0001 แต่ p < 0.05)
- ต้องดู CI ด้วยว่า effect size มีนัยสำคัญในทางปฏิบัติไหม

---
## Slide 9 — Case Study: TV vs Radio — ใครมีหลักฐานชัดกว่า?
**Key Message**: เปรียบ t-statistic ระหว่าง TV และ Radio เพื่อดูว่าตัวใด "significant กว่า"

| | TV → Sales | Radio → Sales |
|---|---|---|
| β̂₁ | 0.0475 | 0.2025 |
| SE | 0.0027 | 0.0204 |
| t-stat | **17.67** | **9.92** |
| p-value | **≈ 0** | **≈ 0** |
| 95% CI | (0.042, 0.053) | (0.162, 0.243) |

**การตีความ**:
- ทั้งสองมีนัยสำคัญ (p ≈ 0 ทั้งคู่)
- Radio มี β̂₁ ใหญ่กว่า ($1,000 Radio → 202.5 หน่วย vs TV → 47.5 หน่วย)
- แต่ R² ต่างกัน — ต้องดู Slide 4 (R² และ RSE) เพื่อตัดสินว่า fit ดีกว่า

---
## Slide 10 — Summary
**Key Message**: SE, CI, t-test คือเครื่องมือตอบคำถาม "β̂ น่าเชื่อถือแค่ไหน?"

**สิ่งที่เรียนรู้วันนี้**:
- **SE(β̂₁)**: วัด sampling variability ของ slope — SE น้อย → น่าเชื่อถือมากกว่า
- **95% CI**: β̂₁ ± 2·SE — ถ้า 0 ∉ CI → predictor significant
- **t-statistic**: β̂₁/SE → ยิ่งสูง ยิ่ง evidence ต่อต้าน H₀: β₁ = 0
- **p-value**: P(|T| ≥ |t| | β₁=0) → ยิ่งน้อย ยิ่ง significant

| Concept | สูตร | Python |
|---------|------|--------|
| SE(β̂₁) | √(σ̂²/Σ(xᵢ−x̄)²) | `model.bse['TV']` |
| 95% CI | β̂₁ ± 2×SE | `model.conf_int()` |
| t-stat | β̂₁/SE | `model.tvalues['TV']` |
| p-value | P(|T|≥|t|) | `model.pvalues['TV']` |

**Bridge from Week 7**: one-sample t-test กับ SE(x̄) → regression t-test กับ SE(β̂) — logic เดียวกัน!

**สัปดาห์ต่อไป — Slide 4**: R² และ RSE — "โมเดล fit ข้อมูลดีแค่ไหน?"
