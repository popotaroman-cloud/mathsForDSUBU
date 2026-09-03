# Slide Deck 2: Poisson Regression
> Week 13 | 9 slides | CLO3

---

## Slide 1 — Title

**Poisson Regression: Count Data Modeling**  
ISLP Ch.4.6.2 | Week 13 | CLO3

---

## Slide 2 — Count Data in the Real World

**Key Message**: Count data มีลักษณะพิเศษที่ Linear Regression handle ได้ไม่ดี

**ตัวอย่าง Count Data:**
- จำนวนจักรยานที่เช่าต่อชั่วโมง
- จำนวนการ default ของลูกค้าต่อเดือน
- จำนวนอุบัติเหตุบนถนนต่อวัน
- จำนวนครั้งที่ customer โทรหา call center

**ลักษณะของ Count Data:**
- ค่าเป็น **non-negative integers**: 0, 1, 2, 3, ...
- Distribution มักเป็น **right-skewed**
- **Mean ≈ Variance** (property ของ Poisson distribution)

**ปัญหาถ้าใช้ Linear Regression:**
- อาจ predict ค่าลบ (เช่น -2 อุบัติเหตุ)
- Residuals ไม่ Normal (skewed)
- Variance ไม่ constant (heteroscedasticity)

**[FIGURE: histogram ของ count data (right-skewed), Normal distribution overlay ทับ → ไม่ match]**

---

## Slide 3 — Poisson Distribution

**Key Message**: Poisson distribution เหมาะกับ count data — มี parameter เดียว λ

**Poisson(λ):**
$$P(Y = k) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

**Properties:**
- Mean = Variance = **λ**
- λ > 0 เสมอ
- PDF เป็น right-skewed เมื่อ λ เล็ก, ใกล้ Normal เมื่อ λ ใหญ่

**[FIGURE: 4 Poisson distributions: λ=1, 3, 5, 10 — ยิ่ง λ ใหญ่ยิ่งใกล้ Normal]**

**ตัวอย่าง:**
- ร้านค้าได้ลูกค้าเฉลี่ย λ=5 คนต่อชั่วโมง
- P(Y=0) = e⁻⁵ = 0.0067 (โอกาสที่ไม่มีลูกค้าเลย)
- P(Y=10) = 0.036

---

## Slide 4 — Poisson Regression Model

**Key Message**: Poisson Regression ใช้ log link ทำให้ λ เป็น positive เสมอ

**Model:**
$$\log(\lambda_i) = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip}$$

→ เทียบเท่า:
$$\lambda_i = e^{\beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip}} = e^{\beta_0} \cdot e^{\beta_1 X_{1}} \cdots e^{\beta_p X_{p}}$$

**ทำไม log link?**
- log กำหนดให้ λ > 0 เสมอ (เพราะ e^η > 0 ทุกค่าของ η)
- เปลี่ยน multiplicative relationship เป็น additive ใน log scale

**Interpretation:**
- β₁ → เพิ่ม X₁ 1 หน่วย → log(λ) เพิ่ม β₁
- **Incidence Rate Ratio (IRR)** = e^β₁
  - เพิ่ม X₁ 1 หน่วย → λ คูณด้วย e^β₁

**[FIGURE: left: scatter count vs X; right: log(count) vs X → relationship linear]**

---

## Slide 5 — Interpreting Coefficients

**Key Message**: ตีความ Poisson coefficient ผ่าน IRR = e^β เสมอ

**ตัวอย่าง: Bikeshare Model**
$$\log(\lambda) = 3.5 + 0.5 \cdot \text{hour\_norm} - 1.2 \cdot \text{rain} + 0.05 \cdot \text{temp}$$

**IRR Table:**

| Predictor | β̂ | IRR = e^β̂ | ความหมาย |
|-----------|---|----------|---------|
| rain | -1.2 | 0.30 | ฝนตก → bike rental ลด 70% |
| temp | +0.05 | 1.05 | temp เพิ่ม 1°C → rental เพิ่ม 5% |
| hour_norm | +0.5 | 1.65 | ช่วงกลางวัน → rental สูงกว่า 65% |

**IRR interpretation:**
- IRR > 1: predictor เพิ่ม → λ เพิ่ม (positive effect)
- IRR < 1: predictor เพิ่ม → λ ลด (negative effect)
- IRR = 1: ไม่มีผล

**[FIGURE: bar chart ของ IRR ทุกตัวแปร — rain bar ต่ำกว่า 1, ตัวอื่นสูงกว่า 1]**

---

## Slide 6 — Python: Fit Poisson Regression

**Key Message**: statsmodels GLM Poisson ใช้ง่าย — เหมือน Logistic Regression แต่เปลี่ยน family

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Fit Poisson Regression
poisson_model = smf.glm(
    formula = 'count ~ hour_normalized + temp + rain + weekday',
    data    = bike_df,
    family  = sm.families.Poisson()   # log link (default)
).fit()

# Summary
print(poisson_model.summary())

# Compute IRR = e^coefficient
import numpy as np
irr = np.exp(poisson_model.params)
print('IRR:', irr.round(4))

# Predict new data
new_obs = pd.DataFrame({'hour_normalized': [0], 'temp': [25],
                         'rain': [0], 'weekday': [1]})
lambda_hat = poisson_model.predict(new_obs)  # predicted count
print(f'Predicted count: {lambda_hat.values[0]:.1f}')
```

**Key outputs ใน summary:**
- `coef`: β̂ (log scale)
- `z`: test statistic (z ไม่ใช่ t)
- `P>|z|`: p-value
- Null deviance vs Residual deviance: ยิ่ง residual deviance ต่ำ ยิ่งดี

---

## Slide 7 — Poisson vs Linear: Side-by-Side

**Key Message**: Poisson หลีกเลี่ยง negative prediction และ fit count data ได้ดีกว่า

**[FIGURE: 2 plots side-by-side]**

**Plot 1 — Linear Regression on count data:**
- Prediction line → extend เป็น negative ที่ extreme values
- Confidence interval wide เท่ากัน (constant variance)
- RMSE = X.X (higher)

**Plot 2 — Poisson Regression on count data:**
- Prediction curve → always positive
- Prediction interval wider เมื่อ λ สูง (heteroscedastic ตาม Poisson)
- RMSE = X.X (lower)

**Real-world consequence:**
- Linear model predict bike rentals = -15 ในคืน rain หนัก → nonsense
- Poisson model predict bike rentals = 2 → สมเหตุสมผล

---

## Slide 8 — Offset in Poisson Regression

**Key Message**: Offset ใช้เมื่อ exposure time ต่างกัน — แปลง count เป็น rate

**ปัญหา:** บางสถานีมีเวลา operation นานกว่า → มี count สูงกว่า ไม่ใช่เพราะ popular กว่า

**Offset:**
$$\log(\lambda_i) = \beta_0 + \beta_1 X_i + \text{offset}(\log(t_i))$$

→ model **rate** = count/t แทน count

```python
import numpy as np

# เพิ่ม log(exposure) เป็น offset
df['log_exposure'] = np.log(df['exposure_hours'])

poisson_rate_model = smf.glm(
    formula = 'count ~ x1 + x2 + offset(log_exposure)',
    data    = df,
    family  = sm.families.Poisson()
).fit()
```

**ใช้เมื่อ:**
- ป้าย billboard ที่มี traffic ต่างกัน → model rate clicks/traffic
- อาชญากรรมในเมืองที่มีประชากรต่างกัน → model per capita

---

## Slide 9 — Summary: Poisson Regression

**Key Message**: Poisson Regression = GLM สำหรับ count data ที่ guarantee non-negative predictions

**Poisson Regression สรุป:**
- Y ~ Poisson(λ): count data, non-negative integer
- Link: log(λ) = Xᵀβ
- Estimate β ด้วย MLE
- Interpret β ผ่าน **IRR = e^β**

**Use cases:**
| Use Case | Y | สิ่งที่ model |
|----------|---|-------------|
| Bikeshare | rentals/hour | λ = mean rentals |
| Call center | calls/day | λ = expected call volume |
| Epidemiology | disease cases | λ = incidence rate |
| Traffic | accidents/year | λ = mean accidents |

**เตือน:**
- Poisson assumes **Var(Y) = Mean(Y)**
- ถ้า Var >> Mean → overdispersion → ใช้ Quasi-Poisson หรือ Negative Binomial
- ตรวจด้วย Pearson statistic ≈ n-p
