# Slide Deck: Course Summary — คณิตศาสตร์ทั้งหมดในภาพเดียว
> Week 15 | CLO1+2+3+4 | Integration | 11 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 15 — Final Week**  
บูรณาการคณิตศาสตร์: จาก Vector ถึง Machine Learning  
CLO1+2+3+4: เชื่อม 15 สัปดาห์เข้าด้วยกันในภาพเดียว

---

## Slide 2 — The Journey: 15 สัปดาห์ในภาพเดียว
**Key Message**: ทุก topic ในวิชานี้ไม่ได้แยกกัน — เชื่อมกันเป็น pipeline เดียวตั้งแต่ Week 1 ถึง Week 15

สัปดาห์สุดท้ายนี้เราจะมองย้อนกลับไปดูภาพรวมทั้งหมดของวิชา แต่ละ topic ที่เรียนมาเป็นชิ้นส่วนที่เชื่อมกัน Linear Algebra ใน Week 1–4 เป็นภาษาที่ Machine Learning ใช้สื่อสาร Statistical Learning ใน Week 5–7 เป็น framework ที่ช่วย formulate ปัญหา Regression และ Classification ใน Week 8–13 คือ tools ที่ใช้แก้ปัญหา และ Model Selection ใน Week 14–15 คือวิธีที่เราตัดสินใจว่า tool ไหนดีที่สุด เป้าหมายสุดท้ายคือให้นักศึกษา integrate ทุกสิ่งนี้ในการวิเคราะห์ข้อมูลจริง

```
Week 1–4   CLO1: Linear Algebra    → ภาษาของ ML (Vector, Matrix, SVD, PCA)
Week 5–7   CLO2: Statistical Learn → Framework (Y=f(X)+ε, EDA, Bias-Variance)
Week 8–13  CLO3: Regression/Class  → Tools (SLR, MLR, Logistic, LDA)
Week 14–15 CLO4: Model Selection   → Decisions (CV, Bootstrap, Regularization)
```

---

## Slide 3 — CLO1: Linear Algebra — What We Learned
**Key Message**: Linear Algebra คือภาษาพื้นฐานที่ Machine Learning ใช้ในการ represent และ compute

**Week 1 — Vectors:**
- Vector = การนำเสนอ observation ใน data science
- Dot product, norm, cosine similarity → basis ของ recommendation systems

**Week 2 — Matrix:**
- Matrix = dataset (rows = obs, columns = features)
- Gaussian elimination → แก้ระบบสมการ linear (basis ของ OLS)

**Week 3 — Subspaces:**
- Column space, null space → เข้าใจว่า solution space คืออะไร
- Projection → Least Squares ← หัวใจของ Linear Regression

**Week 4 — Eigenvalues, SVD, PCA:**
- Ax = λx → compress information ของ dataset
- SVD: A = UΣVᵀ → low-rank approximation, noise reduction
- PCA → dimensionality reduction สำหรับ visualization และ preprocessing

**Connection:**
```
Data Matrix X → Covariance C = XᵀX/(n-1) → Eigendecomposition → PCA
                                                                     ↓
                                                            Feature reduction
                                                            → Better models
```

---

## Slide 4 — CLO2: Statistical Learning — What We Learned
**Key Message**: Statistical Learning framework ให้ "คำถามที่ถูกต้อง" ก่อนเริ่มสร้าง model

**Week 5 — Framework Y = f(X) + ε:**
- ทุก ML problem มีโครงสร้างนี้
- Prediction goal vs Inference goal → เลือก method ต่างกัน
- Reducible vs Irreducible Error

**Week 6 — Bias-Variance Trade-Off:**
- E[MSE] = Bias² + Variance + Var(ε)
- U-curve ของ Test MSE → หา optimal complexity
- Overfitting / Underfitting diagnosis

**Week 7 — EDA + Statistical Inference:**
- EDA ก่อนสร้าง model เสมอ — distribution, outliers, correlation
- Hypothesis testing (t-test, F-test) → ตัดสินใจ statistically

**The 4 Questions ก่อนเริ่ม ML:**
```
1. X = อะไร?           → features / predictors
2. Y = อะไร?           → regression หรือ classification?
3. Goal = prediction หรือ inference?
4. Parametric หรือ Non-parametric?
```

---

## Slide 5 — CLO3: Regression & Classification — What We Learned
**Key Message**: 6 สัปดาห์ให้ tools ครอบคลุมทั้ง regression, classification, GLM, และ diagnostics

**Week 8–10 — Regression:**
```
SLR: y = β₀ + β₁x + ε              → 1 predictor
MLR: y = β₀ + β₁x₁ + ... + βₚxₚ + ε → p predictors
Diagnostics: Residual plots, R², RSE, VIF
Polynomial, Interaction terms, Categorical variables
```

**Week 11–12 — Classification:**
```
Logistic Regression: log(p/1-p) = β₀ + β₁x₁ + ...
LDA/QDA: Gaussian assumption ของ features ในแต่ละ class
KNN: non-parametric, vote จาก K neighbors
Metrics: Accuracy, Precision, Recall, AUC, Confusion Matrix
```

**Week 13 — GLM + ROC:**
```
Poisson Regression: สำหรับ count data (E[Y] = e^{Xβ})
ROC Curve: Trade-off ระหว่าง Sensitivity และ Specificity
```

---

## Slide 6 — CLO4: Model Selection — What We Learned
**Key Message**: Week 14 เสร็จสมบูรณ์ framework ของ ML — วิธีเลือก model อย่างเป็นระบบ

**Week 14 — Cross-Validation & Bootstrap:**
```
k-Fold CV: แบ่ง data เป็น k ส่วน → estimate Test Error ที่น่าเชื่อถือ
LOOCV: ใช้ n=1 เป็น test ทุก round → low bias แต่ high variance
Bootstrap: sampling with replacement → estimate uncertainty ของ β̂

ใช้เมื่อ: เลือก K ใน KNN, เลือก degree ใน polynomial, เลือก regularization α
```

**Week 15 — Integration:**
```
Pipeline: EDA → Feature Engineering → Model Selection → CV → Best Model
Final Answer: ไม่ใช่ model ที่ train MSE ต่ำสุด แต่ CV test MSE ต่ำสุด
```

---

## Slide 7 — Connection Map: ทุก Topic เชื่อมกัน
**Key Message**: ภาพรวมความสัมพันธ์ระหว่าง topics — เห็นได้ว่า math ไม่ได้แยกกันเป็นชิ้นๆ

```
VECTOR/MATRIX (W1-2)
    ↓
SUBSPACES/PROJECTION (W3) ──────────────────── LEAST SQUARES
    ↓                                                ↓
SVD/PCA (W4) ─── DIMENSION REDUCTION ─────── LINEAR REGRESSION (W8)
    ↓                                                ↓
STATISTICAL LEARNING (W5) ─── Y=f(X)+ε ──── MULTIPLE REG (W9)
    ↓                                                ↓
BIAS-VARIANCE (W6) ────── OVERFITTING ──── DIAGNOSTICS (W10)
    ↓                                                ↓
EDA/INFERENCE (W7) ──── P-VALUE/T-TEST ─── LOGISTIC REG (W11)
                                                     ↓
CROSS-VALIDATION (W14) ── MODEL SELECTION ─ LDA/QDA (W12)
                                                     ↓
FINAL PROJECT (W15) ──── INTEGRATION ───── GLM/ROC (W13)
```

---

## Slide 8 — ตัวเลขสำคัญที่ต้องจำ
**Key Message**: สูตรและค่าสำคัญที่ปรากฏซ้ำๆ ตลอดวิชา — ควรจำและเข้าใจ

**สูตรพื้นฐาน 5 สูตร:**
```
1. OLS:           β̂ = (XᵀX)⁻¹Xᵀy          ← เชื่อม LA กับ Regression
2. Test MSE:      E[(Y−f̂)²] = Bias² + Var + Var(ε)
3. Logistic:      P(Y=1|X) = 1 / (1 + e^{−Xβ})
4. k-Fold CV:     CV = (1/k) Σ MSEₖ
5. Eigendecomp:   Ax = λx → A = PDP⁻¹
```

**Metrics ที่ใช้บ่อย:**
| Task | Primary Metric | Secondary |
|------|---------------|-----------|
| Regression | RMSE, R² | MAE |
| Binary Classification | Accuracy | AUC-ROC, Precision, Recall |
| Multi-class | Accuracy | Macro F1 |
| Model Selection | CV MSE | Std Dev of CV |

---

## Slide 9 — Final Project: Putting It All Together
**Key Message**: Final Project คือการพิสูจน์ว่าสามารถ integrate ทุก CLO บน dataset จริงได้

**Pipeline ของ Final Project:**
```
1. เลือก Dataset (≥500 rows, ≥5 features)
       ↓
2. EDA + Statistical Summary (CLO2)
       ↓
3. Linear Algebra Analysis: PCA หรือ SVD (CLO1)
       ↓
4. Model Building: Regression หรือ Classification (CLO3)
       ↓
5. Cross-Validation: เลือก Best Model (CLO4)
       ↓
6. Data Storytelling: สรุป Insights
```

**Deliverables:**
- Jupyter Notebook ที่รัน clean โดยไม่มี error (10%)
- Presentation 10 นาที (10%)

---

## Slide 10 — Presentation Structure (10 นาที)
**Key Message**: 10 นาทีต้องครอบคลุมทุก CLO — structure ที่แนะนำช่วยให้ไม่หลุดประเด็น

```
[0-2 min] Dataset & Problem Statement
  → ชื่อ dataset, แหล่งที่มา, ทำไมถึงน่าสนใจ
  → Problem: จะ predict หรือ classify อะไร?

[2-7 min] Mathematical Analysis
  → CLO1: แสดง PCA/SVD ผล + interpretation
  → CLO2: EDA insights + Bias-Variance curve
  → CLO3: Model ที่ใช้ + ผล (MSE/Accuracy)
  → CLO4: Cross-Validation → Best Model

[7-9 min] Conclusions & Insights
  → สรุปสิ่งที่ค้นพบ 3 ข้อสำคัญ
  → Limitations และ Future Work

[9-10 min] Q&A
```

---

## Slide 11 — Summary & Congratulations
**สิ่งที่เรียนรู้ตลอด 15 สัปดาห์**

```
CLO1 ✅ Linear Algebra: Vector, Matrix, Eigen, SVD, PCA
CLO2 ✅ Statistical Learning: Y=f(X)+ε, Bias-Variance, EDA
CLO3 ✅ Regression & Classification: SLR, MLR, Logistic, LDA
CLO4 ✅ Model Selection: Cross-Validation, Bootstrap
```

**From Vectors to Machine Learning — Complete!**

คณิตศาสตร์ที่เรียนในวิชานี้เป็นรากฐานของ Data Science ทั้งหมด ไม่ว่าจะเป็น Deep Learning, NLP, หรือ Reinforcement Learning ล้วนสร้างบนพื้นฐานที่เรียนไปแล้วทั้งนั้น

**สิ่งที่ต้องทำต่อ → (Slide Deck 4: Next Steps)**
