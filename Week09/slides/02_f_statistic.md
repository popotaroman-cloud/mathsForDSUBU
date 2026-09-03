# Slide Deck: F-statistic — Testing All Predictors Jointly
> Week 09 | CLO3 | ISLP Ch.3.2.2 | 9 slides

---
## Slide 1 — Title
**F-statistic: ทดสอบว่า Predictor ทั้งหมดมีความสัมพันธ์กับ Y หรือไม่**  
Week 9 | CLO3 | ISLP 3.2.2  
LLo: อธิบาย F-statistic และความต่างจาก t-statistic ได้

---
## Slide 2 — ทำไมต้องมี F-test?
**Key Message**: ถ้า run t-test ทีละตัว 100 ตัว จะ false positive อย่างน้อย 5 ตัว — F-test แก้ปัญหานี้

**ปัญหา Multiple Testing**:
- สมมติ p = 100 predictors, ทุกตัวไม่มีผลจริง (H₀ ทั้งหมดเป็นจริง)
- ถ้า test ทีละตัวที่ α = 0.05 → expected 5 false rejections!
- p < 0.05 สำหรับ individual predictor ≠ model มีประโยชน์

**F-test แก้ปัญหา**:
- Test ทุก predictor **พร้อมกัน** ใน single test
- H₀: β₁ = β₂ = … = βₚ = 0 (ไม่มีตัวใดมีผลเลย)
- H₁: อย่างน้อยหนึ่ง βⱼ ≠ 0

---
## Slide 3 — F-statistic: สูตรและความหมาย
**Key Message**: F = MSR/MSE — ratio ของ variance explained ต่อ residual variance

$$F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)} = \frac{MSR}{MSE}$$

| ส่วน | ชื่อ | ความหมาย |
|------|------|---------|
| (TSS−RSS)/p | Mean Regression SS | variance ที่ model อธิบายต่อ parameter |
| RSS/(n−p−1) | Mean Squared Error | variance ที่ model อธิบายไม่ได้ |
| F | F-statistic | สัดส่วนระหว่างสองส่วน |

**ถ้า H₀ จริง** (predictors ทั้งหมดไม่มีผล):
- TSS ≈ RSS → F ≈ 1

**ถ้า H₀ ผิด** (มี predictor อย่างน้อยหนึ่งตัวมีผล):
- TSS >> RSS → F >> 1 → reject H₀

---
## Slide 4 — F-distribution และ p-value
**Key Message**: F ใหญ่ → p เล็ก → evidence ว่า predictors มีความสัมพันธ์กับ Y

[FIGURE: F-distribution(df1=3, df2=196) — shaded tail area = p-value]

**Advertising: TV + Radio + Newspaper → Sales**:
$$F = \frac{(5417.15 - 556.83)/3}{556.83/196} = \frac{1620.1}{2.841} = 570.3$$

**p-value** = P(F ≥ 570.3 | H₀) ≈ 1.58 × 10⁻⁹⁶ ≈ **0**

**ตัดสินใจ**: p ≈ 0 << 0.05 → **Reject H₀** → อย่างน้อยหนึ่ง predictor มีผลต่อ Sales

---
## Slide 5 — F-test vs t-test: ความต่างที่สำคัญ
**Key Message**: F-test = "overall significance", t-test = "individual significance" — ต้องดูทั้งสอง

| | F-test | t-test |
|--|--------|--------|
| **ทดสอบ** | β₁ = β₂ = … = βₚ = 0 (ทั้งหมด) | βⱼ = 0 (ทีละตัว) |
| **ถาม** | "Model มีประโยชน์ไหม?" | "Predictor ตัวนี้จำเป็นไหม?" |
| **Statistic** | F = MSR/MSE | t = β̂ⱼ/SE |
| **df** | F(p, n−p−1) | t(n−p−1) |
| **ใช้เมื่อ** | ตรวจ model ก่อน | หลัง F significant แล้ว |

**กฎปฏิบัติ**:
1. Check F-test ก่อน — ถ้า p_F > 0.05 → stop (model ไม่มีประโยชน์)
2. ถ้า F significant → ดู t-test ของแต่ละตัวเพื่อหาว่าตัวไหน contributes

---
## Slide 6 — กรณีที่ F และ t ขัดแย้งกัน
**Key Message**: F significant แต่ t ทุกตัวไม่ significant เป็นไปได้เมื่อ predictors correlated

**Case 1**: F significant, t บางตัว significant → ปกติ

**Case 2**: F significant แต่ t ทุกตัว NOT significant → multicollinearity สูง
- Predictors correlated กันมาก → SE ทุกตัวสูง → t ทุกตัวไม่ significant
- แต่รวมกันแล้วยังอธิบาย Y ได้ → F significant

**Case 3**: F NOT significant แต่ t บางตัว significant → multiple testing issue หรือ model misspecification

```python
# ─── ดู F-statistic และ t-statistics ──────────────────────
# วัตถุประสงค์: compare F-test (overall) กับ t-test (individual)
print(f'F-statistic: {model.fvalue:.2f}')
print(f'F p-value:   {model.f_pvalue:.2e}')
print('\nt-statistics:')
print(model.tvalues)
print('\nt p-values:')
print(model.pvalues)
```

---
## Slide 7 — Python: คำนวณ F ด้วยมือ
**Key Message**: verify F = MSR/MSE ด้วยสูตรตรง ๆ

```python
# ─── คำนวณ F-statistic ด้วยมือ ──────────────────────────────
# วัตถุประสงค์: ให้เข้าใจว่า F มาจากไหน
y     = df['Sales'].values
y_hat = model.fittedvalues.values
y_bar = y.mean()
n, p  = len(y), 3  # 3 predictors

# TSS: total variability
TSS = ((y - y_bar)**2).sum()
# RSS: residual variability
RSS = ((y - y_hat)**2).sum()
# MSR, MSE
MSR = (TSS - RSS) / p
MSE = RSS / (n - p - 1)
F   = MSR / MSE

print(f'TSS = {TSS:.2f}')
print(f'RSS = {RSS:.2f}')
print(f'MSR = {MSR:.2f}')
print(f'MSE = {MSE:.2f}')
print(f'F   = {F:.2f}')          # ≈ 570.3
print(f'Check: {model.fvalue:.2f}')  # ตรงกัน
```

---
## Slide 8 — กรณีพิเศษ: p = 1 (SLR)
**Key Message**: ใน SLR, F = t² — F-test และ t-test เทียบเท่ากันสมบูรณ์

**SLR: Sales ~ TV**:

$$F = t^2 = 17.67^2 = 312.2$$

```python
# ─── SLR: verify F = t² ──────────────────────────────────
# วัตถุประสงค์: แสดงความสัมพันธ์ F = t² เมื่อ p = 1
model_slr = smf.ols('Sales ~ TV', data=df).fit()
t_tv = model_slr.tvalues['TV']
F_slr = model_slr.fvalue
print(f't = {t_tv:.4f}')
print(f't² = {t_tv**2:.4f}')
print(f'F  = {F_slr:.4f}')  # ตรงกัน
```

**Insight**: F-test ใน SLR redundant กับ t-test — แต่ใน MLR ต่างกัน

---
## Slide 9 — Summary
**Key Message**: ตรวจ F-test ก่อนเสมอ — ถ้า F p-value > 0.05 ไม่ต้องดู t-test

**สิ่งที่เรียนรู้วันนี้**:
- **F-test**: H₀: β₁=…=βₚ=0 — ทดสอบ model โดยรวม
- **F = MSR/MSE**: ยิ่งสูง ยิ่ง evidence ว่า predictors มีผล
- **F vs t**: F = overall, t = individual — ต้องดูทั้งสอง
- **ใน SLR**: F = t² — เทียบเท่าสมบูรณ์

| Concept | สูตร | Python |
|---------|------|--------|
| F-statistic | (TSS−RSS)/p ÷ RSS/(n−p−1) | `model.fvalue` |
| F p-value | P(F ≥ f_obs) | `model.f_pvalue` |
| t-statistic | β̂ⱼ/SE(β̂ⱼ) | `model.tvalues` |

**สัปดาห์ต่อไป — Slide 3**: Variable Selection — "ควรใส่ predictor ตัวใดใน model?"
