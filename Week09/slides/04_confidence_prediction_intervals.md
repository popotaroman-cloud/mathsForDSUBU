# Slide Deck: Confidence Intervals vs Prediction Intervals
> Week 09 | CLO3 | ISLP Ch.3.2.3 | 9 slides

---
## Slide 1 — Title
**CI vs PI: ประมาณ Mean Response กับ Individual Response**  
Week 9 | CLO3 | ISLP 3.2.3  
LLo: แยกความต่างระหว่าง CI และ PI และอธิบายความหมายได้

---
## Slide 2 — สองคำถามเกี่ยวกับ Prediction
**Key Message**: CI ตอบ "mean ของ Sales เท่าไร?" ส่วน PI ตอบ "ตลาดนี้จะขายได้เท่าไร?"

**สถานการณ์**: ทีม marketing ต้องการรู้ว่า ตลาดที่ใช้งบ TV = $100k จะมียอดขายเท่าไร

**สองคำถามที่แตกต่าง**:

| คำถาม | Interval | ตอบอะไร |
|-------|----------|---------|
| "ค่าเฉลี่ย Sales ของตลาดทั้งหมดที่ใช้ TV=$100k คือเท่าไร?" | **CI** (Confidence Interval) | uncertainty ของ f(X₀) = β₀ + β₁×100 |
| "ตลาดนี้ (เฉพาะ) จะขายได้เท่าไร?" | **PI** (Prediction Interval) | uncertainty ของ ŷ + ε |

**PI กว้างกว่า CI เสมอ** เพราะ PI รวม uncertainty ของ ε ไว้ด้วย

---
## Slide 3 — สูตร: CI สำหรับ Mean Response
**Key Message**: CI ของ mean response กว้างขึ้นเมื่อ X₀ ห่างจาก x̄

$$CI: \hat{y}_0 \pm t^* \cdot SE_{mean}$$

$$SE_{mean} = \hat{\sigma}\sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum(x_i - \bar{x})^2}}$$

**ความหมาย**: CI บอก range ที่ **average** Sales ของตลาดทั้งหมดที่มี TV = x₀ อยู่ใน

**สังเกต SE_mean**:
- n ใหญ่ → SE ต่ำ (ข้อมูลมากขึ้น → mean แม่นขึ้น)
- x₀ ใกล้ x̄ → SE ต่ำ (ตรงกลาง data → แม่น)
- x₀ ไกล x̄ → SE สูง (extrapolation → ไม่แม่น)

---
## Slide 4 — สูตร: PI สำหรับ Individual Response
**Key Message**: PI กว้างกว่า CI เสมอ เพราะต้องรวม irreducible error ε ด้วย

$$PI: \hat{y}_0 \pm t^* \cdot SE_{pred}$$

$$SE_{pred} = \hat{\sigma}\sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum(x_i - \bar{x})^2}}$$

**ความต่าง**: SE_pred มี "+1" เพิ่มใต้ root → บวก variance ของ ε

**Irreducible error**: แม้ model สมบูรณ์แค่ไหน ε ยังมีเสมอ (Bias-Variance Week 6)  
→ PI ไม่มีวันแคบเท่า CI

---
## Slide 5 — Visual: CI และ PI บน Scatter Plot
**Key Message**: CI เป็น narrow band รอบ fitted line; PI กว้างกว่ามาก ครอบ data points

[FIGURE: scatter plot Sales vs TV พร้อม:
- Fitted line (แดง)
- CI band (น้ำเงิน แคบ) — ครอบ fitted line
- PI band (เขียว กว้าง) — ครอบ data points ส่วนใหญ่]

**สังเกต**:
- CI และ PI แคบที่สุดตรงกลาง (x₀ = x̄) และขยายออกที่ขอบ
- PI กว้างกว่า CI ชัดเจน
- ~95% ของ data points ควรอยู่ใน PI 95%

---
## Slide 6 — Python: สร้าง CI และ PI
**Key Message**: statsmodels `get_prediction()` ให้ทั้ง CI และ PI ในคำสั่งเดียว

```python
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_csv('Advertising.csv')
model = smf.ols('Sales ~ TV', data=df).fit()

# ─── สร้าง CI และ PI สำหรับ range ของ TV ─────────────────────
# วัตถุประสงค์: visualize uncertainty ของ prediction ทั้ง mean และ individual
tv_range = pd.DataFrame({'TV': np.linspace(df['TV'].min(), df['TV'].max(), 100)})
pred = model.get_prediction(tv_range)
frame = pred.summary_frame(alpha=0.05)

# ─── Plot ──────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.scatter(df['TV'], df['Sales'], alpha=0.4, label='Data')
plt.plot(tv_range['TV'], frame['mean'], 'r-', lw=2, label='Fitted line')
plt.fill_between(tv_range['TV'], frame['mean_ci_lower'], frame['mean_ci_upper'],
                 alpha=0.3, color='blue', label='95% CI (mean)')
plt.fill_between(tv_range['TV'], frame['obs_ci_lower'], frame['obs_ci_upper'],
                 alpha=0.15, color='green', label='95% PI (individual)')
plt.legend(); plt.xlabel('TV'); plt.ylabel('Sales'); plt.tight_layout(); plt.show()
```

---
## Slide 7 — ตัวอย่างตัวเลข: TV = $100k
**Key Message**: CI บอก mean Sales ±0.7k; PI บอก individual Sales ±6.3k — กว้างกว่ามาก

```python
# ─── predict สำหรับ TV = 100 ────────────────────────────────
# วัตถุประสงค์: ดู CI และ PI สำหรับจุดเฉพาะ
new_data = pd.DataFrame({'TV': [100]})
pred = model.get_prediction(new_data).summary_frame(alpha=0.05)
print(pred[['mean', 'mean_ci_lower', 'mean_ci_upper',
            'obs_ci_lower', 'obs_ci_upper']])
```

**ผลลัพธ์** (ประมาณ):
| | ค่า |
|--|-----|
| ŷ (prediction) | 11.79 พันหน่วย |
| **CI**: (11.07, 12.51) | ±0.72 พันหน่วย |
| **PI**: (5.33, 18.25) | ±6.46 พันหน่วย |

**ตีความ**:
- "ตลาดที่มี TV budget $100k มียอดขาย **เฉลี่ย** 11,790 หน่วย (95% CI: 11,070–12,510)"
- "ตลาด **นี้โดยเฉพาะ** คาดว่ามียอดขาย 11,790 หน่วย (95% PI: 5,330–18,250)"

---
## Slide 8 — เลือก CI หรือ PI?
**Key Message**: เลือกตามคำถาม — mean หรือ individual — อย่าสับสน

| สถานการณ์ | ใช้ |
|-----------|-----|
| ต้องการ estimate ค่าเฉลี่ยของ group | **CI** |
| ต้องการ predict value ของ case ใหม่ | **PI** |
| ต้องการ range ที่ observation ใหม่จะตกอยู่ | **PI** |
| ต้องการ range ของ mean Y สำหรับ X₀ | **CI** |

**ตัวอย่างจริง**:
- "Sales เฉลี่ยในทุกตลาดที่ใช้งบ TV $200k คือเท่าไร?" → **CI**
- "ถ้าเปิดตลาดใหม่ที่ TV $200k จะขายได้เท่าไร?" → **PI**

**⚠ Common mistake**: ใช้ CI ทั้งที่ต้องการ PI → interval แคบเกินจริง → overconfident!

---
## Slide 9 — Summary
**Key Message**: PI กว้างกว่า CI เสมอ — อย่าใช้ CI เพื่อ predict individual outcome

**สิ่งที่เรียนรู้วันนี้**:
- **CI**: uncertainty ของ mean response f(X₀) — แคบกว่า, เป็นประโยคเกี่ยวกับ "average"
- **PI**: uncertainty ของ individual response ŷ + ε — กว้างกว่า, เกี่ยวกับ "case เดียว"
- **SE_pred** = SE_mean + σ̂ (extra uncertainty จาก ε)
- ทั้งคู่แคบที่สุดตรง x̄ และขยายออกเมื่อ x₀ ห่างจาก x̄

| Interval | สูตร SE | Python |
|----------|---------|--------|
| CI | σ̂√(1/n + (x₀−x̄)²/Σ(xᵢ−x̄)²) | `pred.summary_frame()['mean_ci_*']` |
| PI | σ̂√(1 + 1/n + (x₀−x̄)²/Σ(xᵢ−x̄)²) | `pred.summary_frame()['obs_ci_*']` |

**Week Summary** (Slides 1–4):
```
MLR: β̂ = (XᵀX)⁻¹Xᵀy
  → F-test → "model มีประโยชน์ไหม?"
  → t-test → "predictor ตัวนี้จำเป็นไหม?"
  → R²_adj, AIC, BIC → "เลือก model ที่ดีที่สุด"
  → CI, PI → "predict แล้วมั่นใจแค่ไหน?"
```

**สัปดาห์ต่อไป — Week 10**: Regression Diagnostics — residual analysis, leverage, outliers
