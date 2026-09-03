# Slide Deck: R² และ RSE — Model Fit Assessment
> Week 08 | CLO3 | ISLP Ch.3.1.3 | 10 slides

---
## Slide 1 — Title
**Assessing Model Fit: R² และ RSE**  
Week 8 | CLO3 | ISLP 3.1.3  
LLo: คำนวณและตีความ R² และ RSE เพื่อวัดว่าโมเดล fit ข้อมูลได้ดีแค่ไหน

---
## Slide 2 — สองคำถามเกี่ยวกับ Model Fit
**Key Message**: coefficient significant แต่ model อาจ fit ไม่ดี — ต้องประเมิน fit แยกต่างหาก

**คำถามที่ 1** (Slide 3): "β₁ = 0 ไหม?" → ตอบด้วย t-test / p-value ✓ (เรียนแล้ว)

**คำถามที่ 2** (Slide นี้): "โมเดล fit ข้อมูลดีแค่ไหน?"
- Model อาจ significant แต่ R² = 0.05 → อธิบายแค่ 5% ของ variance
- สองตัวชี้วัดหลัก:

| ตัวชี้วัด | วัดอะไร | หน่วย |
|-----------|---------|-------|
| **RSE** | average error (absolute) | หน่วยเดียวกับ Y |
| **R²** | fraction of variance explained | ไม่มีหน่วย [0,1] |

---
## Slide 3 — RSE: Residual Standard Error
**Key Message**: RSE คือ "ค่าผิดพลาดเฉลี่ย" ของโมเดล — บอกว่า prediction ผิดโดยเฉลี่ยเท่าไร

**สูตร**:
$$RSE = \sqrt{\frac{RSS}{n - 2}} = \sqrt{\frac{\sum(y_i - \hat{y}_i)^2}{n - 2}}$$

- n−2: degrees of freedom (ประมาณ 2 parameters: β₀ และ β₁)
- RSE ≈ σ̂: estimate ของ true σ (standard deviation ของ ε)

**ตัวอย่าง (Advertising, TV → Sales)**:
$$RSE = \sqrt{\frac{2102.53}{200 - 2}} = \sqrt{10.62} \approx 3.26 \text{ (พันหน่วย)}$$

**ตีความ**: โมเดล predict Sales ผิดโดยเฉลี่ย ±3,260 หน่วยต่อตลาด  
ค่าเฉลี่ย Sales = 14,022 หน่วย → Error ≈ 23% — ยังมีนัยสำคัญ

---
## Slide 4 — TSS, RSS, และ R²
**Key Message**: R² = 1 − RSS/TSS วัด "สัดส่วนของ variance ที่โมเดลอธิบายได้"

**Building Blocks**:

$$TSS = \sum(y_i - \bar{y})^2 \quad \text{(Total Sum of Squares)}$$
$$RSS = \sum(y_i - \hat{y}_i)^2 \quad \text{(Residual Sum of Squares)}$$
$$\text{ESS} = TSS - RSS \quad \text{(Explained Sum of Squares)}$$

**R²**:
$$R^2 = 1 - \frac{RSS}{TSS} = \frac{ESS}{TSS}$$

[FIGURE: bar diagram: TSS แบ่งเป็น RSS (unexplained) + ESS (explained)]

- R² = 0: model ไม่ช่วยอธิบาย Y เลย (แย่เท่ากับใช้ mean)
- R² = 1: model อธิบาย Y ได้ทั้งหมด (ไม่มี residual)

---
## Slide 5 — ตีความ R²: ตัวอย่างจริง
**Key Message**: R² = 0.612 หมายความว่า TV อธิบาย 61.2% ของ variance ใน Sales

**Advertising: Sales ~ TV**:

$$R^2 = 1 - \frac{2102.53}{5417.15} = 0.612$$

**ตีความ**:
- **61.2%** ของความแปรปรวนใน Sales อธิบายได้ด้วย TV budget
- **38.8%** อธิบายไม่ได้ — อาจมาจาก Radio, Newspaper, ฤดูกาล, ราคา ฯลฯ

**เปรียบเทียบ**:
| Model | R² | ตีความ |
|-------|----|-------|
| Sales ~ TV | 0.612 | TV อธิบายได้ 61.2% |
| Sales ~ Radio | 0.332 | Radio อธิบายได้ 33.2% |
| Sales ~ Newspaper | 0.052 | Newspaper อธิบายได้ 5.2% |

**Insight**: TV อธิบาย Sales ได้ดีที่สุดในบรรดา single predictor

---
## Slide 6 — R² = r² ใน SLR
**Key Message**: ใน SLR เท่านั้น R² = (Pearson correlation)² — ใน MLR ไม่เป็นแบบนี้

**SLR: R² = r²**:
$$R^2 = r_{XY}^2 = \left(\frac{\widehat{Cov}(X,Y)}{\hat{\sigma}_X \hat{\sigma}_Y}\right)^2$$

```python
# ─── verify R² = r² ──────────────────────────────────────────
# วัตถุประสงค์: แสดงว่าใน SLR r² = R² เป็นสมบัติพิเศษของ bivariate regression
r = df['TV'].corr(df['Sales'])
r_sq = r**2
print(f'Pearson r = {r:.4f}')
print(f'r²        = {r_sq:.4f}')   # ≈ 0.612

R2_model = model.rsquared
print(f'R² (model)= {R2_model:.4f}')  # ≈ 0.612 — ตรงกัน!
```

**⚠ ข้อสำคัญ**: ใน MLR (Week 9) R² ≠ r² เพราะมีหลาย predictors

---
## Slide 7 — Python: คำนวณ RSE และ R² ด้วยมือ
**Key Message**: คำนวณเองเพื่อ verify statsmodels — เข้าใจที่มาของตัวเลข

```python
import numpy as np

# ─── คำนวณ RSE และ R² ด้วยมือ ──────────────────────────────
# วัตถุประสงค์: verify ผลจาก statsmodels ด้วยสูตรเพื่อเข้าใจที่มา
y     = df['Sales'].values
y_hat = model.fittedvalues.values
y_bar = y.mean()
n     = len(y)

# RSS: sum of squared residuals
RSS = ((y - y_hat)**2).sum()

# TSS: total sum of squares (variability รอบ mean)
TSS = ((y - y_bar)**2).sum()

# RSE: estimate ของ sigma (n-2 เพราะประมาณ 2 parameters)
RSE = np.sqrt(RSS / (n - 2))

# R²: fraction of variance explained
R2  = 1 - RSS / TSS

print(f'RSS = {RSS:.2f}')          # ≈ 2102.53
print(f'TSS = {TSS:.2f}')          # ≈ 5417.15
print(f'RSE = {RSE:.4f}')          # ≈ 3.2589
print(f'R²  = {R2:.4f}')           # ≈ 0.6119
print(f'Check: model.rsquared = {model.rsquared:.4f}')  # ตรงกัน
```

---
## Slide 8 — RSE vs R²: ใช้ตัวไหนเมื่อไร?
**Key Message**: RSE บอกขนาดความผิดพลาดในหน่วยจริง, R² บอกสัดส่วน — ต้องดูทั้งสอง

| | RSE | R² |
|--|-----|-----|
| **วัดอะไร** | average prediction error | proportion of variance explained |
| **หน่วย** | หน่วยเดียวกับ Y | ไม่มีหน่วย [0,1] |
| **ยิ่งดี** | RSE ต่ำ | R² สูง |
| **ใช้เมื่อ** | ต้องการรู้ว่าผิดเฉลี่ยเท่าไร | ต้องการเปรียบเทียบโมเดล |
| **ข้อจำกัด** | ขึ้นกับหน่วยของ Y | ไม่บอกว่าผิดเยอะแค่ไหนใน absolute |

**ตัวอย่าง**:
- RSE = 3.26: predict Sales ผิดเฉลี่ย ±3,260 หน่วย — ต้องตัดสินใจเองว่า "มาก" หรือ "น้อย"
- R² = 0.612: TV อธิบาย 61.2% — เปรียบเทียบกับ Radio (33.2%) ได้เลย

---
## Slide 9 — Case Study: เลือก Predictor ดีที่สุดสำหรับ Sales
**Key Message**: R² และ RSE ช่วยตัดสินว่า predictor ตัวใด fit Sales ได้ดีที่สุด

**เปรียบเทียบ 3 SLR models**:

| Predictor | β̂₁ | SE | t | p | R² | RSE |
|-----------|-----|-----|---|---|-----|-----|
| TV | 0.0475 | 0.0027 | 17.67 | ≈0 | **0.612** | **3.26** |
| Radio | 0.2025 | 0.0204 | 9.92 | ≈0 | 0.332 | 4.28 |
| Newspaper | 0.0547 | 0.0166 | 3.30 | 0.001 | 0.052 | 5.09 |

**สรุป**:
1. **TV** fit ดีที่สุด (R² สูงสุด, RSE ต่ำสุด)
2. ทั้ง 3 significant แต่ Newspaper มี R² ต่ำมาก (5.2%)
3. ยังมี variance ที่อธิบายไม่ได้ → MLR Week 9 จะรวมทั้ง 3

---
## Slide 10 — Summary
**Key Message**: R² และ RSE ควรดูคู่กัน — significant predictor ≠ good model fit

**สิ่งที่เรียนรู้วันนี้**:
- **RSE** = √(RSS/(n−2)): average prediction error ในหน่วยของ Y
- **TSS**: total variability ใน Y รอบ mean
- **R²** = 1 − RSS/TSS: proportion of variance explained [0,1]
- **R² = r²** ใน SLR เท่านั้น
- **สองตัวต้องดูคู่กัน**: significant (p < 0.05) + good fit (R² สูง, RSE ต่ำ)

| Concept | สูตร | Python |
|---------|------|--------|
| RSS | Σ(yᵢ−ŷᵢ)² | `((y-yhat)**2).sum()` |
| TSS | Σ(yᵢ−ȳ)² | `((y-y.mean())**2).sum()` |
| RSE | √(RSS/(n−2)) | `np.sqrt(RSS/(n-2))` หรือ `model.mse_resid**0.5` |
| R² | 1 − RSS/TSS | `model.rsquared` |

**Week Summary** (Slides 1–4):
```
SLR: Y = β₀ + β₁X + ε
  → Least Squares → β̂₀, β̂₁
  → SE, CI, t-test → "β₁ ≠ 0?"
  → R², RSE → "Model fit ดีแค่ไหน?"
```

**สัปดาห์ต่อไป — Week 9**: Multiple Linear Regression — เพิ่ม predictor เป็น p ตัว
