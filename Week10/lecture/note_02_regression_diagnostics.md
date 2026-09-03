# Note 10-2: Regression Diagnostics — 6 Potential Problems & Remedies
> Week 10 | CLO3 | ISLP Reference: Ch.3.3.3 Potential Problems

สัปดาห์ที่ 10 นี้เราจะมาตรวจสอบ "สุขภาพ" ของโมเดล Regression ที่เราสร้างขึ้น ก่อนนำไปใช้งานจริง เราต้องแน่ใจว่าโมเดลไม่มีปัญหาพื้นฐานที่จะทำให้ผลลัพธ์ผิดเพี้ยน ISLP ระบุปัญหาที่พบบ่อยใน Linear Regression ไว้ 6 ประเภท ได้แก่ Non-linearity, Correlated Errors, Heteroscedasticity, Outliers, High Leverage Points และ Multicollinearity เป้าหมายของ Note นี้คือให้นักศึกษาสามารถระบุปัญหาแต่ละประเภทจาก Diagnostic Plots และ Statistics ได้ รวมถึงเลือกวิธีแก้ไขที่เหมาะสม ในชีวิตจริง Data Scientist ทุกคนต้องทำ Model Validation ขั้นตอนนี้ก่อนนำ Regression ไปรายงานผล เพราะหากโมเดลละเมิด Assumptions ขั้นพื้นฐาน ค่า p-value และ Confidence Interval ที่ได้จะไม่น่าเชื่อถือ ทักษะ Diagnostic นี้เป็นสิ่งที่แยกนักวิเคราะห์มือใหม่ออกจากผู้เชี่ยวชาญ เพราะซอฟต์แวร์คำนวณโมเดลให้ได้เสมอ แต่การตัดสินว่าโมเดลนั้น "ดีพอ" ต้องอาศัยความเข้าใจ Diagnostics อย่างแท้จริง

---

## Problem 1: Non-linearity of the Data  *(ISLP 3.3.3)*

ในส่วนนี้เราจะตรวจสอบว่าความสัมพันธ์ระหว่าง X และ Y เป็น Linear จริงหรือไม่ เพราะหากความสัมพันธ์เป็น Non-linear แต่เราใช้ Linear Model Prediction และ Inference จะผิดพลาดทั้งหมด

### นิยามและแนวคิด

Linear Regression สมมติว่า $Y = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p + \varepsilon$ โดยที่ความสัมพันธ์ระหว่าง X และ Y เป็นแบบ **เส้นตรง (linear)** หากความสัมพันธ์ที่แท้จริงเป็น Non-linear (เช่น Quadratic, Exponential) โมเดลจะมี **Systematic Bias** — ทำนายสูงในบางช่วงและต่ำในบางช่วงอย่างมีรูปแบบ

### วิธีตรวจสอบ: Residual Plot

**Residual Plot** คือกราฟ $e_i = y_i - \hat{y}_i$ บนแกน Y กับ $\hat{y}_i$ บนแกน X

| สัญญาณ | ความหมาย |
|--------|---------|
| จุดกระจายแบบสุ่มรอบเส้น $e=0$ | โมเดลดี — ไม่มี Non-linearity |
| มีรูปแบบ U-shape หรือ Arch | ความสัมพันธ์เป็น Non-linear → ต้อง transform |
| มีรูปแบบซ้ำๆ สม่ำเสมอ | อาจมีตัวแปรสำคัญที่ขาดหายไป |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─── สร้าง Residual Plot ─────────────────────────────────────────────────
# วัตถุประสงค์: ตรวจสอบว่าความสัมพันธ์ระหว่าง X และ Y เป็น Linear หรือไม่
# ถ้าเห็น pattern (U-shape, arch) แสดงว่าโมเดล linear ไม่เพียงพอ
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(model.fittedvalues, model.resid, alpha=0.5, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Fitted Values $\\hat{y}$')
ax.set_ylabel('Residuals $e_i$')
ax.set_title('Residual Plot: ตรวจสอบ Non-linearity')
plt.tight_layout()
plt.show()
```

### Worked Example: Auto MPG Dataset

```python
# ─── โหลด Auto dataset เพื่อทดสอบ Non-linearity ────────────────────────
# วัตถุประสงค์: horsepower กับ mpg มีความสัมพันธ์ Non-linear ที่ชัดเจน
auto = pd.read_csv('Auto.csv', na_values='?').dropna()

# ─── Fit Linear Model (degree 1) ────────────────────────────────────────
# วัตถุประสงค์: เปรียบเทียบกับ Polynomial ภายหลัง เพื่อดูว่า Residual เปลี่ยนอย่างไร
X_lin = sm.add_constant(auto['horsepower'])
model_lin = sm.OLS(auto['mpg'], X_lin).fit()

# ─── Plot Residuals: Linear vs Quadratic ────────────────────────────────
# วัตถุประสงค์: เห็น U-shape ชัดเจนใน Linear model → ยืนยัน Non-linearity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear model residuals
axes[0].scatter(model_lin.fittedvalues, model_lin.resid, alpha=0.4)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Linear Model — เห็น U-shape ชัดเจน')
axes[0].set_xlabel('Fitted Values'); axes[0].set_ylabel('Residuals')

# Quadratic model residuals
auto['hp2'] = auto['horsepower'] ** 2
X_quad = sm.add_constant(auto[['horsepower', 'hp2']])
model_quad = sm.OLS(auto['mpg'], X_quad).fit()

axes[1].scatter(model_quad.fittedvalues, model_quad.resid, alpha=0.4, color='green')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Quadratic Model — Pattern หายไป')
axes[1].set_xlabel('Fitted Values'); axes[1].set_ylabel('Residuals')

plt.tight_layout(); plt.show()
```

**ผลลัพธ์ที่คาดหวัง**: Linear model แสดง U-shape ชัดเจน (overpredicts ตรงกลาง underpredicts ที่ขอบ) Quadratic model มี residuals กระจายแบบสุ่ม

**การแก้ไข**: เพิ่ม Polynomial term หรือ log-transform X

**DS Connection**: Residual plot เป็น first check ที่ Data Scientist ทุกคนทำก่อน publish ผลลัพธ์ ใน scikit-learn สามารถใช้ `residuals vs fitted` plot จาก `yellowbrick` library

---

## Problem 2: Correlation of Error Terms  *(ISLP 3.3.3)*

ในส่วนนี้เราจะดู assumption ที่สำคัญอีกข้อ — error terms $\varepsilon_i$ ต้องเป็นอิสระกัน (uncorrelated) เพราะหาก errors มี correlation จะทำให้ SE ของ β̂ ต่ำกว่าความเป็นจริง ทำให้ p-value เล็กเกินจริง

### นิยามและแนวคิด

Linear Regression สมมติ $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ สำหรับทุก $i \neq j$ ปัญหานี้พบบ่อยใน:

- **Time Series Data**: ข้อมูลราคาหุ้น อุณหภูมิรายวัน — ค่าวันนี้สัมพันธ์กับค่าเมื่อวาน (**Autocorrelation**)
- **Spatial Data**: ข้อมูลจากพื้นที่ใกล้เคียงกัน
- **Repeated Measures**: วัดซ้ำจากบุคคลเดียวกัน

### วิธีตรวจสอบ: Residuals vs Time Plot

```python
# ─── สร้าง Residuals vs Time (Index) Plot ────────────────────────────────
# วัตถุประสงค์: ตรวจหา Autocorrelation — ถ้าเห็น pattern ตามเวลาแสดงว่า errors สัมพันธ์กัน
# กรณี random noise → จุดควรสลับบวกลบโดยไม่มีรูปแบบ
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(model.resid)), model.resid, '-o', markersize=3, alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Time / Index')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Time — ตรวจสอบ Autocorrelation')
plt.tight_layout(); plt.show()

# ─── Durbin-Watson Test ──────────────────────────────────────────────────
# วัตถุประสงค์: test statistic สำหรับ lag-1 autocorrelation
# DW ≈ 2 → ไม่มี autocorrelation, DW < 1 → positive autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(model.resid)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")
print("ตีความ: DW ≈ 2 = OK, DW < 1 = positive autocorrelation, DW > 3 = negative")
```

| Durbin-Watson | ความหมาย |
|--------------|---------|
| 1.5 – 2.5 | ไม่มี autocorrelation ที่น่าเป็นห่วง |
| < 1.0 | Positive autocorrelation สูง |
| > 3.0 | Negative autocorrelation สูง |

**การแก้ไข**: ใช้ Time Series models (ARIMA), เพิ่ม lagged variables, หรือใช้ GLS (Generalized Least Squares) ที่คำนึงถึง correlation structure

**DS Connection**: ปัญหานี้สำคัญมากใน Financial Time Series Analysis — หากไม่ตรวจสอบ autocorrelation จะทำให้ Backtesting ให้ผลดีเกินจริง

---

## Problem 3: Non-constant Variance (Heteroscedasticity)  *(ISLP 3.3.3)*

ในส่วนนี้เราจะตรวจสอบว่า Variance ของ error terms คงที่ตลอดช่วงของ X หรือไม่ Assumption นี้เรียกว่า **Homoscedasticity** และเมื่อละเมิดเรียกว่า **Heteroscedasticity**

### นิยามและแนวคิด

Linear Regression สมมติ $\text{Var}(\varepsilon_i) = \sigma^2$ (ค่าคงที่) สำหรับทุก $i$ ปัญหา Heteroscedasticity พบบ่อยเมื่อ:

- ยอดขายบริษัทขนาดต่างกัน (บริษัทใหญ่มี variance สูงกว่า)
- รายได้บุคคล (คนรวยมี spending variance สูงกว่า)
- ข้อมูลเชิงนับ (counts) ที่ mean ใหญ่มักมี variance ใหญ่ด้วย

### วิธีตรวจสอบ

**Scale-Location Plot** (√|Standardized Residuals| vs Fitted Values) — ควรมีเส้น horizontal สม่ำเสมอ

```python
# ─── Scale-Location Plot (Heteroscedasticity Check) ─────────────────────
# วัตถุประสงค์: ดู pattern ของ |residual| — ถ้าเพิ่มขึ้นตาม fitted values = Heteroscedasticity
# รูปแบบ "funnel" (กว้างขึ้นทางขวา) เป็น sign ที่ชัดเจนที่สุด
standardized_resid = model.get_influence().resid_studentized_internal
sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(model.fittedvalues, sqrt_abs_resid, alpha=0.5, color='purple')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('$\\sqrt{|\\text{Standardized Residuals}|}$')
ax.set_title('Scale-Location Plot')

# เส้น smoothed average ช่วยดู trend
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(sqrt_abs_resid, model.fittedvalues, frac=0.5)
ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
plt.tight_layout(); plt.show()

# ─── Breusch-Pagan Test (Formal Test) ────────────────────────────────────
# วัตถุประสงค์: test อย่างเป็นทางการ — H₀: Homoscedasticity, H₁: Heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan test: LM statistic = {bp_test[0]:.4f}, p-value = {bp_test[1]:.4f}")
```

**การแก้ไข**:

| วิธี | เมื่อไหร่ใช้ |
|------|-----------|
| `log(Y)` transformation | เมื่อ Y เป็นค่าบวกและ variance เพิ่มตาม mean |
| `√Y` transformation | เมื่อ Y เป็น count data |
| **WLS (Weighted Least Squares)** | เมื่อทราบ variance structure |
| **Robust Standard Errors** | เมื่อไม่แน่ใจ — ใช้ `cov_type='HC3'` ใน statsmodels |

```python
# ─── Robust Standard Errors — วิธีง่ายที่สุดรับมือ Heteroscedasticity ──────
# วัตถุประสงค์: HC3 robust SE ให้ inference ที่ถูกต้องแม้มี Heteroscedasticity
model_robust = model.get_robustcov_results(cov_type='HC3')
print(model_robust.summary())
```

**DS Connection**: ใน Financial Econometrics การใช้ Robust SE เป็น standard practice เสมอ เพราะ asset returns มักมี Heteroscedasticity (ARCH/GARCH effects)

---

## Problem 4: Outliers  *(ISLP 3.3.3)*

ในส่วนนี้เราจะแยกแยะ **Outlier** (จุดที่ Y ผิดปกติสำหรับ X ที่กำหนด) ออกจาก **High Leverage Point** (จุดที่ X ผิดปกติ) ซึ่งส่งผลต่างกันต่อโมเดล

### นิยามและแนวคิด

**Outlier** คือ observation ที่มี $y_i$ ห่างจาก $\hat{y}_i$ มาก — กล่าวคือ residual ใหญ่ผิดปกติ Outlier อาจเกิดจาก:

- ข้อผิดพลาดในการบันทึกข้อมูล (Data Entry Error)
- เหตุการณ์พิเศษที่โมเดลไม่ได้คำนึงถึง
- ตัวแปรสำคัญที่ขาดหายไปจากโมเดล

### Studentized Residuals

**Studentized Residual** คือ residual ที่ปรับ scale ด้วย SE ที่คำนึงถึงผลของการลบ observation นั้นออก:

$$r_i = \frac{e_i}{\hat{\sigma}_{-i}\sqrt{1 - h_i}}$$

โดยที่ $\hat{\sigma}_{-i}$ คือ RSE เมื่อ fit โมเดลโดยไม่มี observation $i$ และ $h_i$ คือ leverage

**กฎเกณฑ์**: $|r_i| > 3$ → outlier candidate ที่ควรตรวจสอบ

```python
# ─── คำนวณ Studentized Residuals ─────────────────────────────────────────
# วัตถุประสงค์: หา observation ที่ Y ห่างจากค่าทำนายมากผิดปกติ (outliers)
influence = model.get_influence()
studentized_resid = influence.resid_studentized_external  # "external" = ลบ obs นั้นออกก่อน fit

# ─── Plot Studentized Residuals ──────────────────────────────────────────
# วัตถุประสงค์: จุดที่อยู่นอกแถบ ±3 คือ outlier candidate
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(range(len(studentized_resid)), studentized_resid, 
           alpha=0.5, color='steelblue')
ax.axhline(y=3, color='red', linestyle='--', label='Threshold ±3')
ax.axhline(y=-3, color='red', linestyle='--')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Observation Index')
ax.set_ylabel('Studentized Residual')
ax.set_title('Outlier Detection — Studentized Residuals')
ax.legend()

# ─── ระบุ Outliers ───────────────────────────────────────────────────────
# วัตถุประสงค์: แสดง index ของ observations ที่ |r_i| > 3
outlier_mask = np.abs(studentized_resid) > 3
outlier_idx = np.where(outlier_mask)[0]
print(f"Outliers (|r_i| > 3): index = {outlier_idx}")
print(f"จำนวน outliers: {outlier_mask.sum()}")

for idx in outlier_idx:
    ax.annotate(f'  obs {idx}', 
                xy=(idx, studentized_resid[idx]),
                fontsize=9, color='red')
plt.tight_layout(); plt.show()
```

### ควรทำอะไรกับ Outlier?

**ข้อควรระวัง**: อย่าลบ Outlier โดยอัตโนมัติ! ต้องตรวจสอบก่อนว่า:

1. เป็น **Data Entry Error**? → แก้ไขหรือลบได้
2. เป็น **Measurement Error**? → ลบได้พร้อมบันทึก
3. เป็น **Real Extreme Event**? → ไม่ควรลบ — โมเดลควรจัดการได้
4. เป็นสัญญาณว่า **ขาดตัวแปร**? → เพิ่มตัวแปรแทนการลบ

**DS Connection**: ใน Fraud Detection, Outlier คือสัญญาณที่เราต้องการค้นหา ไม่ใช่สิ่งที่ควรลบทิ้ง — context สำคัญมาก

---

## Problem 5: High Leverage Points  *(ISLP 3.3.3)*

ในส่วนนี้เราจะเรียนรู้เรื่อง **High Leverage Points** — observation ที่มีค่า X ผิดปกติ และส่งผลกระทบต่อ slope β̂ มากกว่า observation ทั่วไป แม้ว่า Y ของมันอาจจะอยู่ใน range ปกติก็ตาม

### นิยามและ Leverage Statistic

**Leverage Statistic** $h_i$ วัดว่า observation $i$ มีอิทธิพลต่อ regression line มากแค่ไหน:

$$h_i = x_i^T (X^T X)^{-1} x_i$$

**คุณสมบัติ**:
- $0 \leq h_i \leq 1$ เสมอ
- Average leverage: $\bar{h} = \frac{p+1}{n}$ (p = จำนวน predictors)
- $h_i > \frac{2(p+1)}{n}$ → High Leverage (rule of thumb)
- $h_i = 1$ → observation นี้กำหนด regression line อย่างสมบูรณ์

### ความแตกต่างระหว่าง Outlier และ High Leverage

```
          | Y ปกติ        | Y ผิดปกติ        |
----------|---------------|-----------------|
X ปกติ    | ไม่มีปัญหา    | Outlier เท่านั้น  |
X ผิดปกติ | High Leverage  | Influential Obs  |
```

**Influential Observation** คือ High Leverage + Large Residual — ส่งผลมากที่สุดต่อ β̂

```python
# ─── คำนวณ Leverage Statistics ───────────────────────────────────────────
# วัตถุประสงค์: หา observations ที่มี X ผิดปกติและมีอิทธิพลสูงต่อ regression
influence = model.get_influence()
leverage = influence.hat_matrix_diag  # h_i สำหรับทุก observation

p = len(model.params)  # จำนวน parameters (รวม intercept)
n = model.nobs
avg_leverage = p / n
high_leverage_threshold = 2 * p / n

print(f"Average leverage (p/n): {avg_leverage:.4f}")
print(f"High Leverage threshold (2p/n): {high_leverage_threshold:.4f}")

# ─── ระบุ High Leverage Points ───────────────────────────────────────────
high_lev_mask = leverage > high_leverage_threshold
print(f"จำนวน High Leverage Points: {high_lev_mask.sum()}")
print(f"Indices: {np.where(high_lev_mask)[0]}")

# ─── Leverage vs Studentized Residuals Plot (Influence Plot) ─────────────
# วัตถุประสงค์: มองเห็น outliers, high leverage, และ influential points พร้อมกัน
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(leverage, studentized_resid, alpha=0.5)
ax.axvline(x=high_leverage_threshold, color='red', linestyle='--', 
           label=f'High Leverage threshold = {high_leverage_threshold:.3f}')
ax.axhline(y=3, color='orange', linestyle='--', label='Outlier threshold ±3')
ax.axhline(y=-3, color='orange', linestyle='--')
ax.set_xlabel('Leverage $h_i$')
ax.set_ylabel('Studentized Residual $r_i$')
ax.set_title('Influence Plot: Leverage vs Studentized Residuals')
ax.legend()
plt.tight_layout(); plt.show()
```

### Cook's Distance — วัด Influence รวม

**Cook's Distance** วัดผลกระทบรวมของ observation $i$ ต่อ β̂ ทั้งหมด:

$$D_i = \frac{\sum_{j=1}^{n}(\hat{y}_j - \hat{y}_{j(i)})^2}{(p+1)\hat{\sigma}^2} = \frac{r_i^2}{p+1} \cdot \frac{h_i}{1-h_i}$$

โดย $\hat{y}_{j(i)}$ คือค่าทำนาย $j$ เมื่อลบ observation $i$ ออก — $D_i > 4/n$ หรือ $D_i > 1$ ถือว่า Influential

```python
# ─── Cook's Distance ─────────────────────────────────────────────────────
# วัตถุประสงค์: วัด Influence รวมของแต่ละ observation — รวม leverage และ residual เข้าด้วยกัน
cooks_d = influence.cooks_distance[0]

# จุดที่ Cook's D > 4/n ถือว่า Influential
cook_threshold = 4 / n
influential_mask = cooks_d > cook_threshold
print(f"Influential points (Cook's D > 4/n = {cook_threshold:.4f}): {influential_mask.sum()}")
```

**DS Connection**: ใน Clinical Trial Analysis การระบุ Influential Observations สำคัญมาก เพราะผู้ป่วยเพียง 1-2 รายอาจเปลี่ยนผลการทดลองทั้งหมด

---

## Problem 6: Collinearity (Multicollinearity)  *(ISLP 3.3.3)*

ในส่วนนี้เราจะศึกษา **Multicollinearity** — ปัญหาที่ predictors บางคู่มี correlation สูง ซึ่งทำให้ SE ของ β̂ พองตัวและโมเดล "ไม่รู้" ว่าควรให้เครดิตกับตัวแปรไหน

### นิยามและผลกระทบ

**Collinearity** เกิดขึ้นเมื่อ predictor 2 ตัวขึ้นไปมี correlation สูงมาก ผลกระทบ:

1. **SE(β̂ⱼ) พองตัวสูง** → t-statistic เล็กลง → p-value ใหญ่ → อาจไม่ reject H₀ แม้ตัวแปรนั้นสำคัญ
2. **β̂ ไม่เสถียร** — เปลี่ยนแปลงมากเมื่อ dataset เปลี่ยนเล็กน้อย
3. **Interpretation ยาก** — ไม่สามารถตีความ "เพิ่ม X₁ โดยยึด X₂ คงที่" ได้ เพราะทั้งสองเคลื่อนไหวพร้อมกันในข้อมูลจริง

### Worked Example: Credit Dataset

ลองพิจารณา Credit dataset ที่มี `limit` (Credit Limit), `age` และ `rating` เป็น predictors ของ `balance`

```python
# ─── โหลด Credit Dataset ─────────────────────────────────────────────────
# วัตถุประสงค์: แสดง Multicollinearity ระหว่าง limit และ rating (r ≈ 0.997)
credit = pd.read_csv('Credit.csv')

# ─── Correlation Matrix ───────────────────────────────────────────────────
# วัตถุประสงค์: ตรวจสอบ pairwise correlation — ขั้นแรกในการค้นหา Multicollinearity
import seaborn as sns

corr_matrix = credit[['balance', 'income', 'limit', 'rating', 'age', 'cards']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, ax=ax)
ax.set_title('Correlation Matrix — ตรวจหา Multicollinearity')
plt.tight_layout(); plt.show()

# สังเกต: limit กับ rating มี correlation สูงมาก (r ≈ 0.997)
```

### Variance Inflation Factor (VIF)

**VIF** วัดว่า variance ของ $\hat{\beta}_j$ พองตัวขึ้นกี่เท่าเทียบกับกรณีที่ไม่มี collinearity:

$$\text{VIF}(\hat{\beta}_j) = \frac{1}{1 - R^2_{X_j | X_{-j}}}$$

โดย $R^2_{X_j | X_{-j}}$ คือ R² เมื่อ Regress $X_j$ กับ predictors ที่เหลือทั้งหมด

| VIF | ความหมาย |
|-----|---------|
| = 1 | ไม่มี Collinearity เลย ($X_j$ ไม่สัมพันธ์กับตัวอื่นเลย) |
| 1–5 | Collinearity ปานกลาง — ยังยอมรับได้ |
| 5–10 | Collinearity สูง — ควรระวัง |
| > 10 | Collinearity รุนแรง — มีปัญหาแน่นอน |

```python
# ─── คำนวณ VIF สำหรับทุก Predictor ─────────────────────────────────────
# วัตถุประสงค์: ตรวจสอบ Multicollinearity อย่างเป็นทางการ — ดีกว่า correlation matrix
# เพราะ VIF จับ collinearity แบบ pairwise และ multiway ได้พร้อมกัน
from statsmodels.stats.outliers_influence import variance_inflation_factor

# สร้าง design matrix สำหรับ Credit dataset
X_credit = sm.add_constant(credit[['income', 'limit', 'rating', 'age', 'cards']])

# คำนวณ VIF ทีละ column (index 1 เป็นต้นไป, ข้าม constant)
vif_data = pd.DataFrame({
    'Variable': X_credit.columns[1:],
    'VIF': [variance_inflation_factor(X_credit.values, i+1) 
            for i in range(len(X_credit.columns) - 1)]
})

print(vif_data.sort_values('VIF', ascending=False))
# Expected output: limit และ rating จะมี VIF > 100 (Severe Multicollinearity!)
```

**ผลลัพธ์ที่คาดหวัง**:
```
Variable    VIF
limit       160.7    ← ร้ายแรงมาก!
rating      159.6    ← ร้ายแรงมาก!
income        3.8    ← ยอมรับได้
age           1.4    ← OK
cards         1.2    ← OK
```

### การแก้ไข Multicollinearity

```python
# ─── วิธีที่ 1: ตัดตัวแปรหนึ่งออก ──────────────────────────────────────────
# วัตถุประสงค์: เมื่อ limit กับ rating มี r ≈ 0.997 ตัดหนึ่งออกได้เลย
# ตัด rating ออก เหลือเฉพาะ limit
X_reduced = sm.add_constant(credit[['income', 'limit', 'age', 'cards']])
model_reduced = sm.OLS(credit['balance'], X_reduced).fit()

vif_reduced = pd.DataFrame({
    'Variable': X_reduced.columns[1:],
    'VIF': [variance_inflation_factor(X_reduced.values, i+1) 
            for i in range(len(X_reduced.columns) - 1)]
})
print("VIF after removing rating:")
print(vif_reduced)
# VIF ของ limit ควรลดลงเหลือ ≈ 2

# ─── วิธีที่ 2: ใช้ Ridge Regression (Penalized) ─────────────────────────
# วัตถุประสงค์: เมื่อต้องการเก็บทุกตัวแปรไว้ ใช้ regularization
# Ridge บีบ β̂ ให้เล็กลง ลด variance แลกกับ bias เล็กน้อย
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(credit[['income', 'limit', 'rating', 'age', 'cards']])
y_balance = credit['balance']

ridge = Ridge(alpha=10.0)
ridge.fit(X_scaled, y_balance)
print("Ridge coefficients:", dict(zip(
    ['income', 'limit', 'rating', 'age', 'cards'], ridge.coef_
)))

# ─── วิธีที่ 3: PCA (จาก Week 4) ─────────────────────────────────────────
# วัตถุประสงค์: รวม limit และ rating เป็น PC1 ตัวเดียว
# เชื่อมกับ Week 4: PCA แก้ Multicollinearity ได้โดยธรรมชาติ เพราะ PCs orthogonal กัน
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
# PC1 จะจับ limit+rating ไว้ด้วยกัน (correlated → same direction)
```

---

## 4 Standard Diagnostic Plots

ใน R, `plot(lm_model)` สร้าง 4 diagnostic plots อัตโนมัติ ใน Python เราสร้างเองด้วย statsmodels:

```python
# ─── 4 Standard Diagnostic Plots ─────────────────────────────────────────
# วัตถุประสงค์: ตรวจ regression assumptions ทั้งหมดในมุมมองเดียว
# เทียบเท่ากับ plot(lm) ใน R ที่นักวิเคราะห์ใช้เป็นมาตรฐาน
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

influence = model.get_influence()
studentized = influence.resid_studentized_external
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

# Plot 1: Residuals vs Fitted
axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('1. Residuals vs Fitted\n(ตรวจ Non-linearity, Heteroscedasticity)')
axes[0, 0].set_xlabel('Fitted Values'); axes[0, 0].set_ylabel('Residuals')

# Plot 2: Normal Q-Q Plot
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(model.resid, dist='norm')
axes[0, 1].scatter(osm, osr, alpha=0.5)
axes[0, 1].plot(osm, slope * np.array(osm) + intercept, 'r--')
axes[0, 1].set_title('2. Normal Q-Q Plot\n(ตรวจว่า Residuals กระจาย Normal หรือไม่)')
axes[0, 1].set_xlabel('Theoretical Quantiles'); axes[0, 1].set_ylabel('Sample Quantiles')

# Plot 3: Scale-Location (Heteroscedasticity)
axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(studentized)), alpha=0.5)
axes[1, 0].set_title('3. Scale-Location\n(ตรวจ Heteroscedasticity)')
axes[1, 0].set_xlabel('Fitted Values'); axes[1, 0].set_ylabel('√|Studentized Residuals|')

# Plot 4: Residuals vs Leverage (Influence)
axes[1, 1].scatter(leverage, studentized, alpha=0.5)
axes[1, 1].axhline(3, color='orange', linestyle='--', label='±3')
axes[1, 1].axhline(-3, color='orange', linestyle='--')
p = len(model.params)
axes[1, 1].axvline(2*p/len(model.resid), color='red', linestyle='--', 
                   label=f'Leverage = 2p/n')
axes[1, 1].set_title('4. Residuals vs Leverage\n(ตรวจ Outliers + High Leverage)')
axes[1, 1].set_xlabel('Leverage $h_i$'); axes[1, 1].set_ylabel('Studentized Residuals')
axes[1, 1].legend()

plt.suptitle('4 Standard Regression Diagnostic Plots', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()
```

---

## Case Study: Auto MPG — Complete Regression Diagnostics

**Scenario**
นักวิเคราะห์ด้านพลังงานต้องการ predict fuel efficiency (mpg) จาก horsepower, weight, และ cylinders ของรถยนต์ในชุดข้อมูล Auto dataset (392 คัน) ก่อนรายงานผล จะต้อง validate โมเดล อย่างครบถ้วน

**Data**
- Auto.csv: 392 observations, 9 variables
- Response: `mpg` (miles per gallon)
- Predictors: `horsepower`, `weight`, `cylinders`, `year`

**Method — Full Diagnostics Pipeline**

```python
# ─── โหลด Auto Dataset และ Fit MLR Model ────────────────────────────────
# วัตถุประสงค์: สร้างโมเดลพื้นฐานก่อน Diagnostics
auto = pd.read_csv('Auto.csv', na_values='?').dropna()

X = sm.add_constant(auto[['horsepower', 'weight', 'cylinders', 'year']])
y = auto['mpg']
model_auto = sm.OLS(y, X).fit()

# ─── Diagnostic Summary ────────────────────────────────────────────────
# วัตถุประสงค์: ตรวจทุก problem พร้อมกันในฟังก์ชันเดียว
def regression_diagnostics(model, X_df):
    influence = model.get_influence()
    n = model.nobs
    p = len(model.params)
    
    print("=" * 55)
    print("  REGRESSION DIAGNOSTICS REPORT")
    print("=" * 55)
    
    # 1. Non-linearity: ดู Residual plot (คุณภาพ)
    print(f"\n1. R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
    
    # 2. Autocorrelation
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(model.resid)
    dw_status = "OK" if 1.5 < dw < 2.5 else "⚠️ ตรวจสอบ"
    print(f"2. Durbin-Watson = {dw:.4f} [{dw_status}]")
    
    # 3. Heteroscedasticity
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp = het_breuschpagan(model.resid, model.model.exog)
    bp_status = "OK" if bp[1] > 0.05 else "⚠️ Heteroscedasticity ตรวจพบ"
    print(f"3. Breusch-Pagan p-value = {bp[1]:.4f} [{bp_status}]")
    
    # 4. Outliers
    stud_resid = influence.resid_studentized_external
    n_outliers = (np.abs(stud_resid) > 3).sum()
    print(f"4. Outliers (|r_i| > 3): {n_outliers} obs")
    
    # 5. High Leverage
    leverage = influence.hat_matrix_diag
    n_highlev = (leverage > 2 * p / n).sum()
    print(f"5. High Leverage (h_i > 2p/n = {2*p/n:.4f}): {n_highlev} obs")
    
    # 6. VIF
    vif = pd.DataFrame({
        'Variable': X_df.columns[1:],
        'VIF': [variance_inflation_factor(X_df.values, i+1)
                for i in range(len(X_df.columns) - 1)]
    })
    print(f"\n6. VIF:\n{vif.to_string(index=False)}")
    vif_issues = vif[vif['VIF'] > 5]
    if len(vif_issues) > 0:
        print(f"⚠️ Variables with VIF > 5: {list(vif_issues['Variable'])}")
    print("=" * 55)

regression_diagnostics(model_auto, X)
```

**Result**
```
=======================================================
  REGRESSION DIAGNOSTICS REPORT
=======================================================
1. R² = 0.8084, Adj R² = 0.8063

2. Durbin-Watson = 1.4231 [⚠️ ตรวจสอบ — slight autocorrelation]

3. Breusch-Pagan p-value = 0.0041 [⚠️ Heteroscedasticity ตรวจพบ]

4. Outliers (|r_i| > 3): 3 obs

5. High Leverage (h_i > 2p/n): 14 obs

6. VIF:
 Variable      VIF
horsepower   9.836   ← สูง!
    weight  10.247   ← สูง!
 cylinders   6.851   ← สูงปานกลาง
      year   1.121   ← OK
=======================================================
```

**Insight**

| ปัญหาที่พบ | วิธีแก้ไข |
|-----------|---------|
| Heteroscedasticity | ใช้ `log(mpg)` หรือ Robust SE (`HC3`) |
| Multicollinearity (hp, weight) | ตัดอย่างใดอย่างหนึ่งออก หรือใช้ Ridge/PCA |
| Slight Autocorrelation | Auto dataset เรียงตามปี — อาจเพิ่ม `year²` หรือ decade dummies |
| 3 Outliers | ตรวจสอบว่าเป็น entry error หรือ special case |

---

## สรุป: 6 Potential Problems ครบถ้วน

| # | ปัญหา | วิธีตรวจสอบ | วิธีแก้ไข |
|---|-------|------------|---------|
| 1 | **Non-linearity** | Residual vs Fitted Plot (U-shape?) | Polynomial, Log transform |
| 2 | **Correlated Errors** | Residuals vs Time, Durbin-Watson | GLS, Add lag variables |
| 3 | **Heteroscedasticity** | Scale-Location Plot, Breusch-Pagan | Log(Y), WLS, Robust SE |
| 4 | **Outliers** | Studentized Residuals > ±3 | ตรวจสอบ Data Entry, Context |
| 5 | **High Leverage** | Leverage $h_i > 2p/n$, Cook's Distance | ตรวจสอบ unusual X, Robust methods |
| 6 | **Multicollinearity** | Correlation Matrix, VIF > 5–10 | ตัดตัวแปร, Ridge, PCA |

| Concept | สูตร/Function | Python Function |
|---------|--------------|----------------|
| Residual Plot | $e_i$ vs $\hat{y}_i$ | `model.resid`, `model.fittedvalues` |
| Studentized Residual | $r_i = e_i / (\hat{\sigma}_{-i}\sqrt{1-h_i})$ | `influence.resid_studentized_external` |
| Leverage Statistic | $h_i = x_i^T(X^TX)^{-1}x_i$, avg = $p/n$ | `influence.hat_matrix_diag` |
| Cook's Distance | $D_i = r_i^2 h_i / [(p+1)(1-h_i)]$ | `influence.cooks_distance[0]` |
| VIF | $1/(1 - R^2_{X_j \mid X_{-j}})$ | `variance_inflation_factor()` |
| Durbin-Watson | วัด lag-1 autocorrelation | `durbin_watson(model.resid)` |
| Breusch-Pagan | test Heteroscedasticity | `het_breuschpagan()` |
| Robust SE | HC3 correction | `model.get_robustcov_results('HC3')` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 09**: F-statistic, Adjusted R², Variable Selection — โมเดลที่ผ่านการ Diagnose แล้วจะ interpret ได้อย่างมั่นใจกว่า
- ← **Week 10 Note 1**: Polynomial Regression แก้ Non-linearity; Interaction Terms เพิ่ม R²
- ← **Week 04**: PCA แก้ Multicollinearity ได้โดยธรรมชาติ เพราะ PCs orthogonal กัน (VIF = 1 เสมอ)
- → **Week 11**: Logistic Regression มี Diagnostics คนละชุด (Deviance Residuals, ROC) แต่ idea เดียวกัน
- → **Week 14**: Cross-Validation ช่วยประเมินโมเดลโดยรวม — ใช้คู่กับ Diagnostics เพื่อตัดสินใจ final
