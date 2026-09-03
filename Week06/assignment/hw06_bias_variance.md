# Homework 06: Bias-Variance Trade-Off and Model Assessment
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO2 | **LLo**: คำนวณ MSE, อธิบาย Bias-Variance Trade-Off และประเมินความแม่นยำของ model ได้  
**กำหนดส่ง**: ก่อน Lab Week 7  
**คะแนนรวม**: 100 คะแนน  

---

สัปดาห์นี้เราได้เรียนรู้ว่า error ของ model ประกอบด้วย 3 ส่วน: Bias² (ความเอนเอียง), Variance (ความไม่เสถียร) และ Irreducible Error (noise พื้นฐาน) ความเข้าใจ Trade-Off นี้เป็นหัวใจของ Statistical Learning เพราะช่วยให้ Data Scientist เลือก model ที่เหมาะสมได้ แทนที่จะเลือก model ที่ "แม่น" บน training set เพียงอย่างเดียว ในโจทย์ชุดนี้คุณจะ simulate, คำนวณ, และอธิบาย Bias-Variance Trade-Off จากทั้งมุมมองทางทฤษฎีและจาก Python simulation

---

## ข้อ 1 — MSE Decomposition โดยมือ (25 คะแนน)

ในข้อนี้คุณจะคำนวณ Bias² + Variance + Irreducible Error จาก simulation เล็กๆ เพื่อยืนยันสูตร E[(Y − f̂)²] = Bias²(f̂) + Var(f̂) + Var(ε)

**กำหนด:** True function f(X) = 2X  และ ε ~ N(0, 1) (Var(ε) = 1)  
เราสร้าง model โดยการ fit linear model บน training sets ขนาด n=5 สองชุดแยกกัน:

| Training Set | (X, Y) pairs |
|-------------|-------------|
| Set A | (1, 2.3), (2, 4.8), (3, 6.1), (4, 8.5), (5, 9.9) |
| Set B | (1, 1.7), (2, 3.2), (3, 7.4), (4, 7.6), (5, 10.8) |

สมมติว่า model form คือ f̂(x) = β̂₁x (ผ่านจุด origin ไม่มี intercept)  
และ least squares estimate คือ β̂₁ = Σ(xᵢyᵢ) / Σ(xᵢ²)

**a.** คำนวณ β̂₁ สำหรับ Set A และ Set B  
**b.** คำนวณ prediction ที่ x₀ = 3 จากทั้ง 2 models  
**c.** True value f(3) = 2×3 = 6. คำนวณ Bias(f̂(3)) = mean(predictions) − f(3)  
**d.** คำนวณ Variance(f̂(3)) = variance ของ predictions จาก 2 training sets  
**e.** คำนวณ Expected Test MSE ≈ Bias²(f̂(3)) + Variance(f̂(3)) + Var(ε) = ?

**Deliverable**: แสดงการคำนวณทุกขั้นตอนอย่างชัดเจน ไม่รับแค่คำตอบสุดท้าย

---

## ข้อ 2 — Python Simulation: U-Curve (35 คะแนน)

ในข้อนี้คุณจะ reproduce simulation ของ Bias-Variance Trade-Off จาก ISLP Figure 2.12 โดยใช้ Python เพื่อยืนยันด้วยตัวเองว่า U-curve เกิดขึ้นจริง

**Dataset**: สร้าง synthetic data จาก f_true(X) = 4sin(X) + X/2 บน X ∈ [0, 2π] พร้อม ε ~ N(0, 1)

```python
import numpy as np
np.random.seed(2024)

def f_true(X):
    return 4 * np.sin(X) + X / 2

n_total  = 120
noise_std = 1.0
X_all = np.sort(np.random.uniform(0, 2 * np.pi, n_total))
y_all = f_true(X_all) + np.random.normal(0, noise_std, n_total)
```

**a.** แบ่ง data เป็น train (70%) และ test (30%) ด้วย `train_test_split(..., random_state=99)`

**b.** Fit polynomial models สำหรับ degree = [1, 2, 3, 4, 5, 6, 8, 10, 14]:
   - ใช้ `make_pipeline(PolynomialFeatures(deg), LinearRegression())`
   - คำนวณ Train MSE และ Test MSE สำหรับแต่ละ degree
   - Plot U-curve (Train MSE และ Test MSE vs Degree) พร้อม horizontal line แสดง Var(ε) = 1.0

**c.** ระบุ degree ที่ให้ minimum Test MSE

**d.** Simulation Bias-Variance Decomposition: ทำ 300 simulation rounds โดย:
   - สำหรับแต่ละ round สร้าง training set ใหม่ (n=84) จาก same f_true + noise ต่างกัน
   - Fit polynomial degrees 1, 3, 5, 8, 14 บนแต่ละ training set
   - คำนวณ Bias², Variance, Irreducible สำหรับแต่ละ degree บน fixed test points
   - Plot Bias² + Variance + Irreducible ทั้งหมดใน graph เดียวกัน

**Deliverable**:
- Python code (.py หรือ .ipynb) พร้อม comment ทุก block
- 2 plots: (1) U-curve จากข้อ b, (2) Bias-Variance decomposition จากข้อ d
- ตารางสรุป: degree | Bias² | Variance | Irreducible | Total E[MSE]

---

## ข้อ 3 — Short Essay: อธิบาย Trade-Off (25 คะแนน)

ในข้อนี้คุณจะอธิบายแนวคิดเหล่านี้ในรูปแบบ short essay เพื่อทดสอบว่าเข้าใจ "why" ไม่ใช่แค่ "how"

**a. (10 คะแนน)** อธิบายว่าทำไม Training MSE จึงลดลงเรื่อยๆ เมื่อ model ซับซ้อนขึ้น แต่ Test MSE กลับเพิ่มขึ้นในที่สุด โดยเชื่อมกับแนวคิด Bias² และ Variance ชัดเจน (อย่างน้อย 4–5 ประโยค)

**b. (8 คะแนน)** จากผล simulation ข้อ 2 ถ้าเพิ่ม noise_std จาก 1.0 เป็น 3.0 คาดว่า U-curve จะเปลี่ยนอย่างไร? โดยเฉพาะ optimal degree จะเปลี่ยนทิศทางไหน? อธิบายเหตุผล

**c. (7 คะแนน)** ใน ISLP Chapter 2.2.2 กล่าวว่า "it is not hard to show that the expected test MSE... can always be decomposed into three fundamental quantities." อธิบายว่า "always" หมายความว่าอย่างไร — ผลนี้ apply กับ model ทุกประเภท (Linear Regression, KNN, Neural Network) หรือไม่? ทำไม?

**Deliverable**: เขียนเป็น paragraph (ภาษาไทยหรืออังกฤษ) พร้อมระบุว่าตรงกับ concept ในตำราหน้าไหน

---

## ข้อ 4 — Open-Ended: Kaggle Dataset (15 คะแนน)

ในข้อนี้คุณจะนำ Bias-Variance Trade-Off ไปใช้กับ dataset จริงที่คุณสนใจ

**เลือก dataset หนึ่ง**:
- **Option A**: California Housing Dataset (`from sklearn.datasets import fetch_california_housing`)
- **Option B**: Diabetes Dataset (`from sklearn.datasets import load_diabetes`)
- **Option C**: Dataset ของคุณเอง (regression problem, ≥500 rows, ≥5 features)

**ทำดังนี้**:
1. โหลดข้อมูลและเลือก **1 feature ที่มี correlation สูงสุดกับ target** เป็น X
2. Fit polynomial models degree 1–10 บน 80% training data
3. Plot U-curve ของ Train/Test MSE
4. ระบุ: (a) optimal degree, (b) ถ้า degree นี้คือ "best" แล้วทำไมไม่ใช้ degree 10 เสมอ?
5. เขียนสรุป 3–5 ประโยคว่าข้อมูลชุดนี้มีลักษณะ near-linear หรือ nonlinear

**Deliverable**:
- Python code พร้อม comment
- Plot U-curve
- สรุป 3–5 ประโยค

---

## รูปแบบการส่ง

| ไฟล์ที่ต้องส่ง | รายละเอียด |
|--------------|-----------|
| `hw06_Q1.pdf` หรือ `.md` | การคำนวณข้อ 1 ทุกขั้นตอน |
| `hw06_Q2.ipynb` หรือ `.py` | Code ข้อ 2 พร้อม plots |
| `hw06_Q3.pdf` หรือ `.md` | Essay ข้อ 3 |
| `hw06_Q4.ipynb` หรือ `.py` | Code + plot ข้อ 4 |

**รวมเป็น zip file ชื่อ** `hw06_StudentID.zip` และอัปโหลดใน LMS

---

## เกณฑ์การให้คะแนนโดยย่อ

| ข้อ | เกณฑ์หลัก | คะแนน |
|----|----------|------|
| 1 | ความถูกต้องของการคำนวณ + แสดงขั้นตอน | 25 |
| 2 | Code ถูกต้อง + plots ถูกต้อง + table | 35 |
| 3 | อธิบาย concept ได้ถูกต้อง + ชัดเจน | 25 |
| 4 | ใช้ dataset จริง + วิเคราะห์ได้ | 15 |
| **รวม** | | **100** |

ดูเกณฑ์โดยละเอียดใน `hw06_rubric.md`
