# Slide Deck: Variable Selection และ Model Fit Assessment
> Week 09 | CLO3 | ISLP Ch.3.2.2–3.2.3 | 10 slides

---
## Slide 1 — Title
**Variable Selection: เลือก Predictor ที่ดีที่สุดด้วย Adjusted R², AIC, BIC**  
Week 9 | CLO3 | ISLP 3.2.2–3.2.3  
LLo: เลือก predictor ที่สำคัญและเปรียบเทียบ model ด้วย Adjusted R² ได้

---
## Slide 2 — ปัญหา: เพิ่ม Predictor แล้ว R² สูงขึ้นเสมอ
**Key Message**: R² ไม่เหมาะสำหรับเปรียบเทียบ model ที่มี predictor จำนวนต่างกัน

```python
# R² เพิ่มทุกครั้งที่เพิ่ม predictor แม้ predictor นั้นไม่มีประโยชน์
import numpy as np, pandas as pd, statsmodels.formula.api as smf
df = pd.read_csv('Advertising.csv')

m1 = smf.ols('Sales ~ TV', data=df).fit()
m2 = smf.ols('Sales ~ TV + Radio', data=df).fit()
m3 = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit()

print(f'R²(TV):                   {m1.rsquared:.4f}')
print(f'R²(TV+Radio):             {m2.rsquared:.4f}')
print(f'R²(TV+Radio+Newspaper):   {m3.rsquared:.4f}')
# 0.6119, 0.8972, 0.8972 — Newspaper ไม่เพิ่ม R²!
```

**ปัญหา**: R² = 0 never decreases — ไม่ลงโทษ predictor ที่ไม่จำเป็น

---
## Slide 3 — Adjusted R²
**Key Message**: Adjusted R² ลงโทษ predictor ที่ไม่ช่วย — ใช้เปรียบเทียบ model ต่างจำนวน predictor

**สูตร**:
$$R^2_{adj} = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$$

- น้อยกว่า R² เสมอ (แต่ใกล้เคียงถ้า predictor มีประโยชน์)
- เพิ่ม predictor ที่ไม่ช่วย → R²_adj **ลดลง**

**เปรียบเทียบ 3 model**:

| Model | R² | **R²_adj** |
|-------|----|-----------|
| Sales ~ TV | 0.6119 | 0.6099 |
| Sales ~ TV + Radio | 0.8972 | 0.8962 |
| Sales ~ TV + Radio + Newspaper | 0.8972 | **0.8956** ↓ |

**Newspaper ทำให้ R²_adj ลด** → ไม่ควรใส่ใน model

---
## Slide 4 — AIC และ BIC
**Key Message**: AIC และ BIC เป็น information criteria — เลือก model ที่ AIC/BIC ต่ำที่สุด

$$AIC = -2\ln(\hat{L}) + 2p$$
$$BIC = -2\ln(\hat{L}) + p\ln(n)$$

- L̂: maximum likelihood (OLS → proportional to RSS)
- BIC ลงโทษ parameter มากกว่า AIC เมื่อ n ใหญ่
- เลือก model ที่ **AIC/BIC ต่ำที่สุด**

```python
# ─── เปรียบเทียบ AIC และ BIC ──────────────────────────────
# วัตถุประสงค์: ใช้ AIC/BIC เลือก model ที่ดีที่สุด
for name, m in [('TV', m1), ('TV+Radio', m2), ('TV+Radio+News', m3)]:
    print(f'{name:20s}: AIC={m.aic:.1f}, BIC={m.bic:.1f}, R²_adj={m.rsquared_adj:.4f}')
```

---
## Slide 5 — Forward Selection
**Key Message**: Forward selection เริ่มจากไม่มี predictor แล้วเพิ่มทีละตัว

**Algorithm**:
```
1. Start: ไม่มี predictor (null model: Sales ~ Intercept)
2. ลอง: เพิ่ม predictor 1 ตัวทุกตัวที่เหลือ
3. เลือก: predictorที่ทำให้ RSS ต่ำที่สุด (หรือ AIC ต่ำที่สุด)
4. Repeat: ลอง เพิ่มอีก 1 ตัว
5. หยุด: เมื่อ AIC ไม่ลดอีก
```

**ตัวอย่าง Advertising**:
```
Step 1: เพิ่ม TV     → RSS ลดจาก 5417 → 2103 (ดีที่สุด)
Step 2: เพิ่ม Radio  → RSS ลดจาก 2103 → 557  (ดีที่สุด)
Step 3: ลอง Newspaper → RSS 556.83 — ไม่ลดมีนัยสำคัญ → หยุด
```

**Model สุดท้าย**: Sales ~ TV + Radio

---
## Slide 6 — Backward Elimination
**Key Message**: Backward เริ่มจาก full model แล้วลบ predictor ทีละตัว

**Algorithm**:
```
1. Start: full model (ทุก predictor)
2. ดู: p-value ของแต่ละ predictor
3. ลบ: predictor ที่มี p-value สูงที่สุด (ถ้า > threshold)
4. Repeat: ลบอีก 1 ตัว
5. หยุด: ทุก predictor ที่เหลือมี p < threshold
```

**ตัวอย่าง Advertising (threshold = 0.05)**:
```
Step 1: Full model — Newspaper p = 0.860 (สูงที่สุด)
Step 2: ลบ Newspaper → TV + Radio model
Step 3: TV p ≈ 0, Radio p ≈ 0 → ทุกตัว significant → หยุด
```

**Model สุดท้าย**: Sales ~ TV + Radio (เหมือน forward selection)

---
## Slide 7 — Python: เปรียบเทียบ Models Systematically
**Key Message**: ลอง models ทั้งหมดที่เป็นไปได้แล้วเปรียบ AIC/R²_adj

```python
from itertools import combinations

# ─── เปรียบเทียบทุก subset ของ predictors ──────────────────
# วัตถุประสงค์: Best subset selection สำหรับ p เล็ก ๆ
predictors = ['TV', 'Radio', 'Newspaper']
results = []

for r in range(1, len(predictors)+1):
    for combo in combinations(predictors, r):
        formula = 'Sales ~ ' + ' + '.join(combo)
        m = smf.ols(formula, data=df).fit()
        results.append({
            'model': ' + '.join(combo),
            'p': len(combo),
            'R2_adj': round(m.rsquared_adj, 4),
            'AIC': round(m.aic, 1)
        })

import pandas as pd
res_df = pd.DataFrame(results).sort_values('AIC')
print(res_df.to_string(index=False))
```

---
## Slide 8 — ผล Best Subset Selection
**Key Message**: TV + Radio เป็น best model — R²_adj สูงที่สุด AIC ต่ำที่สุด

| Model | p | R²_adj | AIC |
|-------|---|--------|-----|
| **TV + Radio** | 2 | **0.8962** | **780.4** |
| TV + Radio + Newspaper | 3 | 0.8956 | 781.4 |
| TV | 1 | 0.6099 | 1022.5 |
| Radio + Newspaper | 2 | 0.3346 | 1139.8 |
| Radio | 1 | 0.3294 | 1142.0 |
| TV + Newspaper | 2 | 0.6462 | 1008.2 |
| Newspaper | 1 | 0.0448 | 1282.4 |

[FIGURE: bar chart เปรียบเทียบ AIC ของทุก subset]

**สรุป**: **Sales ~ TV + Radio** คือ optimal model

---
## Slide 9 — ข้อควรระวัง
**Key Message**: Variable selection มีข้อจำกัด — ต้องระวัง overfitting และ theory

**⚠ ข้อควรระวัง**:

1. **Overfitting**: เลือก model จาก training data → อาจไม่ generalize
   → ต้องใช้ cross-validation (Week 14) สำหรับ model ใหญ่ ๆ

2. **Confounding**: ลบ predictor ที่ correlated กับตัวอื่นอาจทำให้ coefficient ที่เหลือ biased

3. **Theory first**: ถ้ามี domain knowledge → ใส่ predictor ตามทฤษฎีก่อน อย่า "data mine"

4. **p-value inflation**: Backward selection หลายรอบ → p-value ไม่น่าเชื่อถือ 100%

5. **Large p**: ถ้า p ใหญ่มาก (เช่น genomics p = 20,000) → ต้องใช้ regularization (Lasso/Ridge)

---
## Slide 10 — Summary
**Key Message**: R²_adj และ AIC/BIC เป็นเครื่องมือเลือก model ที่ balance fit กับ complexity

**สิ่งที่เรียนรู้วันนี้**:
- **R²_adj**: 1 − RSS/(n−p−1) / TSS/(n−1) — ลงโทษ predictor ที่ไม่จำเป็น
- **AIC/BIC**: information criteria — เลือก model AIC/BIC ต่ำสุด
- **Forward Selection**: เพิ่มทีละตัวจนไม่ดีขึ้น
- **Backward Elimination**: ลบทีละตัวจนทุกตัว significant

| Criteria | ลงโทษ | ใช้เมื่อ |
|----------|-------|---------|
| R² | ไม่ลงโทษ | ดู variance explained (ห้ามเปรียบ model ต่าง p) |
| R²_adj | ต่ำมาก | เปรียบ models ต่าง p บน data เดิม |
| AIC | ปานกลาง | prediction — อนุญาตโมเดลซับซ้อนกว่า |
| BIC | มาก | parsimony — เลือก model ง่ายกว่า |

**สัปดาห์ต่อไป — Slide 4**: CI vs PI — ประมาณ mean response vs individual response
