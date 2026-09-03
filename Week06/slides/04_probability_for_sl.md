# Slide Deck: Probability Foundations for Statistical Learning
> Week 6 | CLO2 | ISLP Math Bridge | 9 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 6**  
Probability Foundations: เครื่องมือคณิตศาสตร์สำหรับ Statistical Learning  
CLO2: อธิบาย Conditional Probability, Bayes Theorem และ Distributions หลักที่ใช้ใน ML ได้

---

## Slide 2 — ทำไมต้องรู้ Probability?
**Key Message**: Probability คือภาษากลางของ Statistical Learning — ทุก model ใน ISLP มีรากฐานจาก probability

ตลอดเทอมนี้เราจะพบ probability ในทุกที่: Logistic Regression คือการประมาณ Pr(Y=1|X), Bayes Classifier ใช้ posterior probability, LDA สมมติ normal distribution ของ features, Naive Bayes ใช้ Bayes Theorem โดยตรง ดังนั้น Deck นี้เป็น "math bridge" ที่ต้องรู้ก่อนเดินหน้าไปยัง Regression และ Classification ขั้นสูง เป้าหมายคือให้นักศึกษา review concepts สำคัญ 4 อย่าง: Conditional Probability, Bayes Theorem, Normal Distribution, และ t/χ²/F distributions ที่ใช้ใน inference

**Probability ปรากฏที่ไหนใน course นี้:**

| Week | Concept | Probability ที่ใช้ |
|------|---------|------------------|
| 6 | Bayes Classifier | Pr(Y=j\|X) |
| 11 | Logistic Regression | P = 1/(1+e^{-z}) |
| 12 | LDA / QDA | Gaussian prior + likelihood |
| 13 | Naive Bayes | Bayes Theorem |
| 9-11 | Regression inference | t-distribution, F-distribution |

---

## Slide 3 — Conditional Probability
**Key Message**: Pr(A|B) คือ "ความน่าจะเป็นของ A ถ้าเรารู้ว่า B เกิดขึ้นแล้ว" — เป็น core ของ Bayesian thinking

**นิยาม:**
```
Pr(A|B) = Pr(A ∩ B) / Pr(B)   (B ≠ 0)

อ่านว่า: "probability ของ A given B"
```

**ตัวอย่าง Medical:**
```
A = patient เป็น cancer
B = test result เป็น positive

Pr(A|B) = Pr(cancer และ positive) / Pr(positive)
        = "ถ้า test positive → น่าจะเป็น cancer แค่ไหน?"
```

**ทำไมสำคัญใน ML:**
```
Pr(Y = spam | X = email content)   → Spam Filter
Pr(Y = 1 | X = patient features)   → Medical Diagnosis
Pr(Y = Up | X = past returns)       → Stock Prediction
```

→ นี่คือสิ่งที่ทุก supervised learning model พยายามประมาณ!

---

## Slide 4 — Bayes Theorem
**Key Message**: Bayes Theorem เปลี่ยน "คำถามที่ตอบยาก" เป็น "คำถามที่ตอบง่ายกว่า" โดยใช้ข้อมูลที่มี

**Bayes Theorem:**
```
Pr(A|B) = Pr(B|A) × Pr(A) / Pr(B)

Prior    = Pr(A)        ← ความน่าจะเป็นก่อนเห็น evidence
Likelihood = Pr(B|A)   ← น่าจะเป็นแค่ไหนที่เห็น B ถ้า A จริง
Marginal  = Pr(B)       ← normalization constant
Posterior = Pr(A|B)     ← ความน่าจะเป็นหลังเห็น evidence
```

**ตัวอย่าง: COVID Test**
```
Pr(COVID) = 0.01     ← prevalence (prior)
Pr(+|COVID) = 0.99   ← sensitivity (likelihood)
Pr(+|no COVID) = 0.05 ← false positive rate

Pr(+) = 0.99×0.01 + 0.05×0.99 = 0.0594

Pr(COVID|+) = (0.99 × 0.01) / 0.0594 ≈ 0.167

→ ถ้า test + ยังมีโอกาสเป็น COVID แค่ 16.7%!
```

**Connection กับ ML:**  
Naive Bayes Classifier ใช้ Bayes Theorem โดยตรง (Week 13)

---

## Slide 5 — Normal Distribution
**Key Message**: Normal Distribution N(μ, σ²) เป็น distribution ที่สำคัญที่สุดใน Statistics เพราะปรากฏในหลายสิถานการณ์จาก Central Limit Theorem

**N(μ, σ²) — PDF:**
```
f(x) = (1 / (σ√(2π))) × exp(−(x − μ)² / (2σ²))

μ = mean (center ของ distribution)
σ² = variance (spread ของ distribution)
σ = standard deviation
```

**[FIGURE: Normal curves สำหรับ N(0,1), N(2,1), N(0,4) แสดงผลของ μ และ σ]**

**ทำไม Normal สำคัญใน ML:**
- OLS (Least Squares) มี optimal properties ถ้า ε ~ N(0, σ²)
- LDA สมมติว่า features ใน class แต่ละ class มี normal distribution
- Sampling distribution ของ β̂ → Normal (CLT) → t-test ใช้ได้

**68-95-99.7 Rule:**
```
μ ± 1σ → 68% ของข้อมูล
μ ± 2σ → 95% ของข้อมูล  ← p-value 0.05
μ ± 3σ → 99.7% ของข้อมูล
```

---

## Slide 6 — t-Distribution และ Hypothesis Testing
**Key Message**: t-distribution ใช้เมื่อ σ ไม่รู้และ n เล็ก — เป็น basis ของ t-test ที่ใช้ใน regression inference

**t-Distribution:**
```
t = (β̂ − β₀) / SE(β̂)  ~ t(n − p − 1)

n − p − 1 = degrees of freedom (df)
SE(β̂) = standard error ของ estimate
β₀ = 0 สำหรับ H₀: β = 0 (no effect)
```

**[FIGURE: t-distribution vs Normal: t มี heavy tails, เมื่อ df เพิ่ม → Normal]**

**ใช้อย่างไรใน Regression (Week 9):**
```
ตัวอย่าง: Advertising → Sales
β̂₁ = 0.048 (TV effect)
SE(β̂₁) = 0.003
t-stat = 0.048 / 0.003 = 16.0

df = 200 − 1 − 1 = 198
p-value < 0.001 → reject H₀: TV มีผลต่อ Sales จริง!
```

---

## Slide 7 — χ² และ F-Distribution
**Key Message**: χ² ใช้ test variance, F ใช้ compare models — ทั้งคู่ปรากฏใน ANOVA และ Multiple Regression

**χ²-Distribution:**
```
X² ~ χ²(k)   (k = degrees of freedom)
= sum of k independent N(0,1)² random variables

ใช้ใน: Goodness-of-fit test, Independence test (categorical data)
```

**F-Distribution:**
```
F = (SS_Model / df_model) / (SS_Error / df_error)  ~ F(df₁, df₂)

ใช้ใน: F-test ใน Multiple Linear Regression (Week 9)
H₀: β₁ = β₂ = ... = βₚ = 0  (ไม่มีตัวแปรไหนมีผลเลย)
```

**F-test ใน Regression:**
```
F = (TSS − RSS) / p  /  RSS / (n − p − 1)

ถ้า F ใหญ่มาก → reject H₀ → อย่างน้อย 1 predictor มีผลจริง
F_statistic สำหรับ Advertising: F = 570 >> F_critical
→ reject H₀ ชัดเจน → TV, Radio, Newspaper มีผลต่อ Sales
```

---

## Slide 8 — Summary ของ Distributions
**Key Message**: รู้จัก distributions หลักทั้ง 4 ตัว ว่าใช้เมื่อไหรและเชื่อมกับ ML อย่างไร

| Distribution | รูปแบบ | ใช้เมื่อ | ใน course |
|-------------|-------|---------|----------|
| Normal N(μ,σ²) | Bell curve symmetric | ε, β̂, features | Week 5, 9, 12 |
| t(df) | Normal แต่ heavy tail | n เล็ก, σ ไม่รู้ | Week 9-11 (t-test) |
| χ²(k) | Right-skewed, ≥0 | Test variance, categorical | Week 9 |
| F(df₁,df₂) | Right-skewed, ≥0 | Compare nested models | Week 9 (F-test) |

**Key Relationships:**
```
t(∞) = N(0,1)      ← t distribution → Normal เมื่อ df → ∞
χ²(1) = Z²          ← Z ~ N(0,1) squared
F(1,n) = t²(n)      ← F with df₁=1 เชื่อมกับ t-test
```

---

## Slide 9 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Conditional Probability**: Pr(A|B) = Pr(A∩B)/Pr(B) — basis ของ classification
- **Bayes Theorem**: Posterior ∝ Likelihood × Prior — ใช้ใน Naive Bayes
- **Normal N(μ,σ²)**: distribution หลักที่สมมติใน regression และ LDA
- **t-distribution**: สำหรับ hypothesis test ของ regression coefficients
- **F-distribution**: สำหรับ test model significance (F-test)

**สิ่งที่ต้องจำก่อน Week 9:**
```
1. Pr(Y=j|X) ← สิ่งที่ classification model ทำ
2. ε ~ N(0, σ²) ← assumption สำหรับ Linear Regression
3. t-test ← ทดสอบว่า β ≠ 0 จริงไหม?
4. F-test ← ทดสอบว่า model ทั้งหมด significant ไหม?
```

**สัปดาห์ถัดไป (Week 7):** Exploratory Data Analysis (EDA) — วิธีสำรวจและเข้าใจข้อมูลก่อนสร้าง model: distribution, correlation, outlier, missing values
