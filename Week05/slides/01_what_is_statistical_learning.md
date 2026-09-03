# Slide Deck: What Is Statistical Learning?
> Week 5 | CLO2 | ISLP Ch.2.1 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 5**  
Statistical Learning: กรอบแนวคิด Y = f(X) + ε  
CLO2: อธิบายกรอบแนวคิด Statistical Learning และแยกแยะ problem types ได้

---

## Slide 2 — Week Overview
**Key Message**: Statistical Learning คือ framework ที่รองรับ Machine Learning ทั้งหมด — เข้าใจ Y = f(X) + ε คือเข้าใจหัวใจของ Data Science

สัปดาห์นี้เราก้าวข้ามจาก Linear Algebra (Week 1–4) สู่ Statistical Learning ซึ่งเป็น framework ที่ใช้อธิบาย Machine Learning algorithms ทั้งหมด แนวคิดหลักคือ "ข้อมูลทุกอย่างสร้างมาจาก Y = f(X) + ε" โดย f คือความสัมพันธ์จริงที่เราต้องการค้นหา และ ε คือ noise ที่หลีกเลี่ยงไม่ได้ Statistical Learning แบ่งเป็น 2 เป้าหมายหลัก: Prediction (ทำนาย Y ให้แม่นยำ) และ Inference (เข้าใจว่า X กระทบ Y อย่างไร) เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถกำหนด X, Y, f ในปัญหาจริงได้ อธิบาย reducible vs irreducible error ได้ และเลือก approach ที่เหมาะสมได้ ทักษะนี้สำคัญมากเพราะก่อนเขียน code ทุกครั้ง ต้องคิดก่อนว่า "problem นี้คือ Regression หรือ Classification? Supervised หรือ Unsupervised?"

**สิ่งที่จะเรียนรู้:**
- Framework: Y = f(X) + ε
- Prediction vs Inference: ต้องการอะไร?
- Reducible vs Irreducible Error: อะไรปรับปรุงได้?
- How to estimate f: parametric vs non-parametric
- [EXAMPLE: 3 motivating examples จาก ISLP: Wage, Stock, Cancer]

---

## Slide 3 — The Framework: Y = f(X) + ε
**Key Message**: ทุก ML problem สามารถ formulate เป็น Y = f(X) + ε — เข้าใจ notation นี้คือเข้าใจภาษากลางของ Data Science

**Framework:**
```
Y = f(X) + ε

X = (X₁, X₂, ..., Xₚ)  — input / features / predictors
Y                        — output / response / target
f                        — unknown systematic function (what we want!)
ε                        — irreducible error: E[ε]=0, independent of X
```

**ตัวอย่างจริง:**
```
Wage prediction:
Y = salary ($)
X = (education, age, job title, industry, ...)
f = true relationship between education/experience and salary
ε = individual variation (luck, personal network, etc.)
```

**สิ่งที่ไม่สามารถ predict ได้:**
- ε รวม variation ที่ไม่มีใน X
- แม้รู้ f จริง ก็ยัง predict ผิดได้เพราะ ε
- [FIGURE: scatter plot ของ Y vs X(1D) แสดง true f และ data points กระจายรอบๆ f]

---

## Slide 4 — Why Do We Estimate f?
**Key Message**: มี 2 เหตุผลที่เราต้องการ f̂: Prediction (ต้องการ Ŷ แม่นยำ) หรือ Inference (ต้องการเข้าใจความสัมพันธ์)

**Goal 1: Prediction**
```
Ŷ = f̂(X)   ← ต้องการ Ŷ ที่ใกล้ Y มากที่สุด
```
- ไม่สนใจว่า f̂ มีรูปแบบอะไร ขอแค่ predict ได้แม่น
- ตัวอย่าง: ทำนายราคาหุ้น, ตรวจจับ spam, วินิจฉัยโรค

**Goal 2: Inference**
```
"Education ส่งผลต่อ salary มากแค่ไหน?"
"ปัจจัยไหนมีผลต่อ cancer risk มากที่สุด?"
```
- ต้องการ interpretable f̂ ที่อธิบาย mechanism ได้
- ตัวอย่าง: วิจัยทางการแพทย์, policy analysis, social science

**ทำไมสำคัญ?** — เพราะเป้าหมายต่างกัน → method ต่างกัน:
- Prediction → ยอมรับ black-box (neural network, random forest)
- Inference → ต้องการ interpretable (linear regression, decision tree)

---

## Slide 5 — Reducible vs Irreducible Error
**Key Message**: Error ของ model มี 2 ส่วน — เราปรับปรุงได้แค่ reducible error ส่วน irreducible error เป็น fundamental limit

**Expected Prediction Error:**
```
E[(Y - f̂(X))²] = [f(X) - f̂(X)]²  +  Var(ε)
                   ─────────────────    ────────
                   Reducible Error      Irreducible Error
                   (ปรับปรุงได้)        (ปรับปรุงไม่ได้)
```

**Reducible Error:**
- เกิดจาก f̂ ≠ f (model ไม่ดีพอ)
- ลดได้โดย: เก็บข้อมูลมากขึ้น, เลือก model ดีกว่า, tune hyperparameter

**Irreducible Error (Var(ε)):**
- เกิดจาก factors ที่ไม่อยู่ใน X
- ไม่สามารถลดได้ แม้จะรู้ f จริง
- **ข้อสำคัญ**: ทำไม perfect prediction ไม่มีจริง

**[FIGURE: แสดง f จริง, f̂, Y = f+ε — arrows แสดง reducible (gap f vs f̂) และ irreducible (scatter รอบ f)]**

---

## Slide 6 — How to Estimate f: Two Approaches
**Key Message**: Parametric ง่ายกว่าแต่ assume รูปแบบ; Non-parametric ยืดหยุ่นกว่าแต่ต้องการข้อมูลมากกว่า

**Approach 1 — Parametric:**
```
1. Assume functional form: f(X) = β₀ + β₁X₁ + β₂X₂ + ...
2. Fit (train): estimate β₀, β₁, ..., βₚ จาก data
   → reduce problem to estimating p+1 parameters
```
- ✓ ง่าย, interpretable, ทำงานได้แม้ data น้อย
- ✗ ถ้า model form ผิด → high bias

**Approach 2 — Non-parametric:**
```
ไม่ assume รูปแบบ f — fit curve ให้ใกล้ data มากที่สุด
ตัวอย่าง: KNN, Splines, Kernel regression
```
- ✓ ยืดหยุ่น, จับ pattern ซับซ้อน
- ✗ ต้องการข้อมูลมากกว่ามาก, interpret ยากกว่า

**[FIGURE: ซ้าย = linear fit (parametric), ขวา = wavy fit (non-parametric) บน data เดียวกัน]**

---

## Slide 7 — ตัวอย่างจริง: 3 Motivating Examples (ISLP)
**Key Message**: ISLP เปิดด้วย 3 examples นี้เพราะแต่ละอันแสดง use case ต่างกัน ของ Statistical Learning

**Example 1 — Wage Data:**
- X: education, age, job class, year
- Y: salary (continuous)
- Goal: **Inference** — "ตัวแปรไหนส่งผลต่อ salary?"
- Method: Linear Regression

**Example 2 — Stock Market:**
- X: past 5 days' returns (% change)
- Y: market Up/Down tomorrow
- Goal: **Prediction** — แม่นยำสูงสุด
- Method: Logistic Regression, LDA, QNN

**Example 3 — Cancer Gene Expression:**
- X: expression ของ 6,830 genes
- Y: cancer type (4 classes)
- Goal: **Prediction + Discovery** — classify + find important genes
- Challenge: p >> n (more features than samples) → PCA first!

---

## Slide 8 — Python: Exploring Advertising Dataset
**Key Message**: Advertising dataset เป็น "hello world" ของ Statistical Learning — สาธิต Y = f(X) + ε ด้วย real data

**โหลดและสำรวจข้อมูล:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# โหลด dataset
df = pd.read_csv('Advertising.csv', index_col=0)

print(df.shape)        # (200, 4)
print(df.describe())   # stats ทุก column

# X = TV, Radio, Newspaper advertising budget (K$)
# Y = Sales (K units)
```

**Scatter plots:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, feature in enumerate(['TV', 'Radio', 'Newspaper']):
    axes[i].scatter(df[feature], df['Sales'], alpha=0.5)
    axes[i].set_xlabel(f'{feature} Budget ($K)')
    axes[i].set_ylabel('Sales (K units)')
    axes[i].set_title(f'{feature} vs Sales')
plt.tight_layout()
```

**สิ่งที่สังเกตได้:**
- TV มี linear relationship กับ Sales ชัดเจน
- Radio มี positive trend
- Newspaper weak/no relationship

---

## Slide 9 — Key Concepts Summary
**Key Message**: 4 คำถามที่ต้องถามก่อนทำ ML ทุกครั้ง — นี่คือ checklist ของ Data Scientist

**4 คำถามก่อนเริ่มทำ ML:**
```
1. X = อะไร? (inputs/features)
   → กำหนดให้ชัดเจน, clean หรือยัง?

2. Y = อะไร? (output/target)
   → Quantitative (regression) หรือ Categorical (classification)?
   → มี Y label หรือไม่? (Supervised vs Unsupervised)

3. Goal = อะไร? (Prediction or Inference)
   → ต้องการ ŷ ที่แม่นยำ หรือ ต้องการเข้าใจ mechanism?

4. Method = อะไร? (Parametric or Non-parametric)
   → Data มีมากพอไหม? ต้องการ interpretability ไหม?
```

**[FIGURE: flowchart แสดง decision tree สำหรับเลือก method]**

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Framework**: Y = f(X) + ε — ทุก ML problem มีรูปแบบนี้
- **Prediction vs Inference**: เป้าหมายต่างกัน → method ต่างกัน
- **Reducible Error**: ลดได้โดยปรับ model
- **Irreducible Error**: Var(ε) — fundamental limit, ลดไม่ได้
- **Parametric vs Non-parametric**: trade-off ระหว่าง simplicity กับ flexibility

| คำถาม | คำตอบ |
|------|------|
| ทำไมต้องประมาณ f? | Prediction และ/หรือ Inference |
| Error ลดได้ไหม? | Reducible ได้, Irreducible ไม่ได้ |
| ใช้ method ไหน? | Parametric: ง่าย / Non-param: ยืดหยุ่น |

**สัปดาห์ถัดไป (Slide Deck 2):** Parametric vs Non-parametric ในเชิงลึก + Accuracy-Interpretability Trade-off
