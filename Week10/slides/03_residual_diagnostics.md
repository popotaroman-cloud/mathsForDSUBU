# Slide Deck: Regression Diagnostics — Residual Analysis
> Week 10 | CLO3 | ISLP Ch.3.3.3 | 10 slides

---
## Slide 1 — Title
**Regression Diagnostics: ตรวจหา Potential Problems**  
Week 10 | CLO3 | ISLP 3.3.3  
LLo: ใช้ diagnostic plots ตรวจ non-linearity, heteroscedasticity, outliers และ high leverage ได้

---
## Slide 2 — ทำไมต้องทำ Diagnostics?
**Key Message**: p-value น้อยและ R² สูง ≠ model ถูกต้อง — ต้องตรวจ residuals เสมอ

**6 Potential Problems** ที่ต้องตรวจทุกครั้ง:

| # | ปัญหา | ผล |
|---|--------|-----|
| 1 | Non-linearity | predictions ผิดระบบ |
| 2 | Correlated errors | SE ผิด → inference ผิด |
| 3 | Heteroscedasticity | SE ผิด → p-value ผิด |
| 4 | Outliers | RSS สูงเกินจริง |
| 5 | High leverage | β̂ ถูก pull |
| 6 | Multicollinearity | SE สูง → power ต่ำ |

**Rule**: plot residuals ก่อนเชื่อผล regression เสมอ

---
## Slide 3 — Diagnostic Plot 1: Residuals vs Fitted
**Key Message**: pattern ใน residual plot → assumptions ผิด; สุ่มรอบ 0 → ดี

[FIGURE: 2 residual plots คู่กัน:
- ซ้าย (ดี): สุ่ม รอบ y=0 ไม่มี pattern
- ขวา (แย่): U-shape → non-linearity]

**อ่านผล**:
- **สุ่มรอบ 0**: linearity assumption OK
- **U-shape หรือ curve**: non-linearity → พิจารณา transformation หรือ polynomial
- **Fan shape**: heteroscedasticity → variance เพิ่มตาม fitted value
- **Systematic pattern**: อาจขาด predictor สำคัญ

```python
# ─── Residual vs Fitted plot ────────────────────────────────
# วัตถุประสงค์: ตรวจ non-linearity และ heteroscedasticity
fitted  = model.fittedvalues
resid   = model.resid

plt.scatter(fitted, resid, alpha=0.5)
plt.axhline(0, color='r', lw=1.5)
plt.xlabel('Fitted values'); plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
```

---
## Slide 4 — Diagnostic Plot 2: QQ-Plot (Normality)
**Key Message**: QQ-plot ตรวจว่า residuals มี normal distribution — สำคัญสำหรับ t-test, CI

[FIGURE: 2 QQ-plots:
- ซ้าย: points บน diagonal → normality OK
- ขวา: heavy tails → normality violated]

```python
# ─── QQ-plot ──────────────────────────────────────────────
# วัตถุประสงค์: ตรวจ normality assumption ของ residuals
from scipy import stats
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5,5))
stats.probplot(model.resid, dist='norm', plot=ax)
ax.set_title('Normal Q-Q Plot')
plt.show()
```

**อ่านผล**:
- Points บน diagonal: normality OK ✓
- Heavy tails (S-shape): distribution มี outliers
- Skewed: residuals skewed ไปด้านใดด้านหนึ่ง

---
## Slide 5 — Diagnostic Plot 3: Scale-Location (Homoscedasticity)
**Key Message**: heteroscedasticity ทำให้ SE และ p-value ผิด — ต้อง fix ก่อนเชื่อ inference

[FIGURE: Scale-Location plot:
- ซ้าย (ดี): √|residual| กระจายสม่ำเสมอตาม fitted
- ขวา (แย่): trend ขึ้น → fan shape]

```python
# ─── Scale-Location plot ───────────────────────────────────
# วัตถุประสงค์: ตรวจ homoscedasticity (variance คงที่ตลอด)
import numpy as np

sqrt_abs_resid = np.sqrt(np.abs(model.resid))
plt.scatter(model.fittedvalues, sqrt_abs_resid, alpha=0.5)
plt.xlabel('Fitted values')
plt.ylabel('√|Standardized Residuals|')
plt.title('Scale-Location (Spread-Location)')
# เส้น smoother ที่ควรแบนราบถ้า homoscedastic
plt.show()
```

**Fix heteroscedasticity**: log(Y) หรือ √Y transformation ก่อน fit

---
## Slide 6 — Outliers: Studentized Residuals
**Key Message**: studentized residual |rᵢ| > 3 → outlier candidate ที่ต้องตรวจสอบ

**Ordinary residual**: eᵢ = yᵢ − ŷᵢ — ขึ้นกับหน่วยของ Y

**Studentized residual** (standardized):
$$r_i = \frac{e_i}{SE(e_i)}$$

- มี t-distribution(df=n−p−2)
- |rᵢ| > 3: outlier (ตกนอก 99.7% interval)

```python
# ─── Studentized residuals ─────────────────────────────────
# วัตถุประสงค์: identify outliers ที่มี unusual Y value
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(model)
stud_resid = influence.resid_studentized_external

plt.scatter(range(len(stud_resid)), stud_resid, alpha=0.6)
plt.axhline( 3, color='r', lw=1.5, linestyle='--', label='|r|=3')
plt.axhline(-3, color='r', lw=1.5, linestyle='--')
plt.ylabel('Studentized Residuals')
outlier_idx = np.where(np.abs(stud_resid) > 3)[0]
print(f'Outliers at indices: {outlier_idx}')
```

---
## Slide 7 — High Leverage Points
**Key Message**: high leverage คือ point ที่ X ผิดปกติ — pull β̂ ได้แม้ Y ปกติ

**Leverage statistic**:
$$h_i = x_i^T(X^TX)^{-1}x_i$$

- hᵢ ∈ [1/n, 1]
- เฉลี่ย = (p+1)/n
- hᵢ >> (p+1)/n → high leverage

```python
# ─── Leverage plot (Cook's Distance) ────────────────────────
# วัตถุประสงค์: identify high leverage points ที่ส่งผลต่อ β̂ มาก
leverage    = influence.hat_matrix_diag
cooks_d     = influence.cooks_distance[0]
avg_leverage = (sm_model.df_model + 1) / len(df)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].scatter(range(len(leverage)), leverage, alpha=0.6)
axes[0].axhline(2 * avg_leverage, color='r', lw=1.5, linestyle='--', label=f'2×avg={2*avg_leverage:.3f}')
axes[0].set_title('Leverage'); axes[0].legend()

axes[1].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
axes[1].axhline(4/len(df), color='r', lw=1.5, linestyle='--', label="4/n")
axes[1].set_title("Cook's Distance"); axes[1].legend()
plt.tight_layout(); plt.show()
```

---
## Slide 8 — Outlier vs High Leverage: ความต่าง
**Key Message**: outlier ≠ high leverage — ต้องแยกแยะเพื่อตัดสินใจว่าจะจัดการอย่างไร

[FIGURE: 4 cases:
1. ปกติ: ไม่ใช่ทั้งคู่
2. High leverage แต่ on regression line: ไม่เป็นปัญหา
3. High leverage + outlier: อันตรายมาก — pull β̂ มาก
4. Low leverage outlier: เพิ่ม RSS แต่ไม่ pull β̂ มาก]

| Type | X unusual? | Y unusual? | ผล |
|------|-----------|-----------|-----|
| Outlier only | ✗ | ✓ | RSS ↑ แต่ β̂ ไม่เปลี่ยนมาก |
| High leverage only | ✓ | ✗ | ไม่เป็นปัญหา |
| Both | ✓ | ✓ | **อันตราย**: β̂ ถูก pull ออกไป |

**Cook's Distance** > 4/n → influential point ที่ต้องตรวจสอบ

---
## Slide 9 — 4 Diagnostic Plots ใน One Figure
**Key Message**: ดู 4 plots พร้อมกันเสมอ — เหมือน R's `plot(lm())`

```python
# ─── 4-panel diagnostic plot ────────────────────────────────
# วัตถุประสงค์: overview ทุก diagnostic ในภาพเดียว
from statsmodels.graphics.gofplots import ProbPlot

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Residuals vs Fitted
axes[0,0].scatter(model.fittedvalues, model.resid, alpha=0.5)
axes[0,0].axhline(0, color='r'); axes[0,0].set_title('Residuals vs Fitted')

# 2. QQ-plot
ProbPlot(model.resid).qqplot(line='s', ax=axes[0,1])
axes[0,1].set_title('Normal Q-Q')

# 3. Scale-Location
axes[1,0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.5)
axes[1,0].set_title('Scale-Location')

# 4. Leverage vs Cook's Distance
leverage = OLSInfluence(model).hat_matrix_diag
cooks = OLSInfluence(model).cooks_distance[0]
axes[1,1].scatter(leverage, cooks, alpha=0.5)
axes[1,1].set_title("Leverage vs Cook's Distance")

plt.tight_layout(); plt.show()
```

---
## Slide 10 — Summary
**Key Message**: diagnostic plots เป็น "health check" ของ model — ทำก่อนเชื่อผลเสมอ

**สิ่งที่เรียนรู้วันนี้**:
- **Residuals vs Fitted**: ตรวจ non-linearity, heteroscedasticity
- **QQ-plot**: ตรวจ normality ของ residuals
- **Scale-Location**: ตรวจ homoscedasticity (variance คงที่)
- **Leverage / Cook's D**: ตรวจ influential observations

| Plot | ตรวจอะไร | อ่านอย่างไร |
|------|---------|-----------|
| Residuals vs Fitted | non-linearity, heterosc. | ควรสุ่มรอบ 0 |
| QQ-plot | normality | ควรอยู่บน diagonal |
| Scale-Location | homoscedasticity | ควรแบนราบ |
| Leverage/Cook's | influential points | hᵢ>>avg หรือ Cook's>4/n |

**สัปดาห์ต่อไป — Slide 4**: Multicollinearity และ VIF
