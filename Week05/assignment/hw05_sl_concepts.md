# Homework 05: Statistical Learning Concepts
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO2 | **LLo**: อธิบายกรอบแนวคิด Statistical Learning แยกแยะ Supervised/Unsupervised และ Regression/Classification ได้  
**กำหนดส่ง**: สัปดาห์ที่ 6 (ก่อนเริ่มชั่วโมง Lecture)  
**รูปแบบการส่ง**: Markdown หรือ PDF (พร้อม code ที่ run ได้ใน .ipynb ถ้ามี)

---

## บทนำ

สัปดาห์นี้เราเรียนรู้ **Statistical Learning Framework** ซึ่งเป็นแนวคิดพื้นฐานที่รองรับ Machine Learning ทั้งหมด กรอบนี้มองว่าข้อมูลทุกอย่างสร้างจาก Y = f(X) + ε โดยที่ f คือ "ความจริงที่ซ่อนอยู่" และ ε คือ noise ที่หลีกเลี่ยงไม่ได้ เราแบ่ง algorithm ออกเป็น Supervised vs Unsupervised ขึ้นกับว่ามี Y หรือไม่ และแบ่ง problem เป็น Regression vs Classification ขึ้นกับลักษณะของ Y การบ้านนี้ทดสอบว่านักศึกษาสามารถนำ framework นี้ไปจำแนกและอธิบาย real-world problems ได้ ทักษะเหล่านี้จะใช้ตลอดทั้งวิชาเพราะช่วยในการ "วิเคราะห์ปัญหาก่อนเขียน code"

---

## ข้อ 1 — จำแนกประเภท Machine Learning Problems (30 คะแนน)

ในข้อนี้คุณจะฝึกจำแนก real-world problems เป็น Supervised/Unsupervised และ Regression/Classification เพื่อแสดงให้เห็นว่าเข้าใจ framework ของ Statistical Learning

**สำหรับแต่ละ scenario ต่อไปนี้ (10 ข้อ):**
1. ระบุว่าเป็น: **Supervised** หรือ **Unsupervised**
2. ถ้า Supervised: **Regression** หรือ **Classification**
3. ระบุ **X (inputs)** และ **Y (output)** ถ้ามี
4. อธิบายสั้นๆ ว่า **ทำไม** (1–2 ประโยค)

| # | Scenario | Type | X | Y | เหตุผล |
|---|---------|------|---|---|--------|
| 1 | บริษัทอสังหาริมทรัพย์ต้องการทำนายราคาบ้านจากพื้นที่, ทำเล, จำนวนห้องนอน | | | | |
| 2 | ธนาคารต้องการแบ่งกลุ่มลูกค้าออกเป็น high/medium/low value โดยไม่มี label ล่วงหน้า | | | | |
| 3 | โรงพยาบาลต้องการทำนายว่าผู้ป่วยเป็น COVID หรือไม่ จากอาการ 15 อย่าง | | | | |
| 4 | Netflix ต้องการแนะนำหนังที่ผู้ใช้น่าจะชอบ โดยใช้ประวัติการดูของผู้ใช้คนอื่น | | | | |
| 5 | นักอุตุนิยมวิทยาต้องการพยากรณ์อุณหภูมิพรุ่งนี้จาก 20 ปัจจัยบรรยากาศ | | | | |
| 6 | ระบบ spam filter แยกว่า email เป็น spam/not spam จาก word frequency | | | | |
| 7 | นักวิจัยต้องการหา "กลุ่มยีน" ที่ทำงานคล้ายกันจาก RNA-seq data 20,000 genes | | | | |
| 8 | Facebook ทำนายว่าผู้ใช้จะ click โฆษณาหรือไม่ จาก demographic + history | | | | |
| 9 | นักเศรษฐศาสตร์ต้องการวัดผลกระทบของการศึกษาต่อรายได้ (ต้องการ interpret coefficient) | | | | |
| 10 | ระบบตรวจจับการฉ้อโกงบัตรเครดิต โดยดูว่า transaction ปกติหรือผิดปกติ โดยไม่มี labeled data | | | | |

---

## ข้อ 2 — Parametric vs Non-parametric: Trade-off Analysis (30 คะแนน)

ในข้อนี้คุณจะวิเคราะห์ trade-off ระหว่าง parametric และ non-parametric approach เพื่อแสดงว่าเข้าใจว่าควรเลือก approach ไหนในสถานการณ์ต่างๆ

**Dataset**: ใช้ Advertising dataset (`TV`, `Radio`, `Newspaper` → `Sales`)

**(a)** โหลด dataset และสำรวจข้อมูล:
```python
import pandas as pd
df = pd.read_csv('https://www.statlearning.com/s/Advertising.csv', index_col=0)
# หรือ download จาก ISLP website
```
แสดง: shape, describe(), scatter plot ของ TV vs Sales และ Radio vs Sales

**(b)** **Parametric approach — Linear Regression:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```
- Split data: 80% train, 20% test
- Fit model: Sales ≈ β₀ + β₁×TV
- แสดง: coefficients, R², test MSE
- Plot: data + fitted line

**(c)** **Non-parametric approach — KNN Regression:**
```python
from sklearn.neighbors import KNeighborsRegressor
```
- Fit KNN: k=1, 5, 10 บน training data
- คำนวณ test MSE สำหรับแต่ละ k
- Plot: data + KNN predictions (k=1 vs k=10)

**(d)** **Comparison table:**

| Method | Training MSE | Test MSE | Interpretable? | Parameters? |
|--------|-------------|---------|----------------|-------------|
| Linear Reg (β₀ + β₁TV) | | | Yes | 2 |
| KNN (k=1) | | | No | — |
| KNN (k=5) | | | No | — |
| KNN (k=10) | | | No | — |

**(e)** อธิบาย (2–3 ประโยค): ทำไม KNN k=1 ถึง overfit? และในกรณีนี้ควรใช้ parametric หรือ non-parametric?

---

## ข้อ 3 — Reducible vs Irreducible Error (25 คะแนน)

ในข้อนี้คุณจะทำ simulation เพื่อแสดงให้เห็นว่า irreducible error ทำให้ prediction ไม่มีทาง perfect แม้จะรู้ f จริง

**สิ่งที่ต้องทำ:**

**(a)** สร้าง "true function" และ simulate data:
```python
np.random.seed(42)
n = 100
X = np.linspace(0, 10, n)
f_true = 2 * X + np.sin(X)          # true f (known)
epsilon = np.random.normal(0, 2, n)  # irreducible error, σ=2
Y = f_true + epsilon                 # observed Y
```

**(b)** Plot Y vs X, วาด true f(X) ด้วยสีต่างกัน  
แสดงให้เห็นว่า observation ไม่ตรงกับ f_true เพราะ ε

**(c)** แม้รู้ f จริง, prediction error = irreducible error:
- คำนวณ MSE ถ้าใช้ f_true โดยตรง: `np.mean((Y - f_true)**2)`
- เปรียบเทียบกับ `np.var(epsilon)` (ควรใกล้เคียงกัน)
- อธิบาย: ทำไม MSE ≠ 0 แม้รู้ f จริง?

**(d)** Fit Linear Regression บน data และ compute MSE:
- `f_hat = LinearRegression().fit(X.reshape(-1,1), Y)`
- คำนวณ: reducible error = MSE(f_hat) - Var(ε)
- บอกว่า model ดีขึ้นได้อีกแค่ไหน

---

## ข้อ 4 — Open-ended: เลือก Real Dataset และ Define Problem (15 คะแนน)

ในข้อนี้คุณจะฝึก "คิดเหมือน Data Scientist" โดยเลือก dataset จริงและกำหนด problem statement ที่สมบูรณ์

**เลือก 1 dataset จาก:**
- `sklearn.datasets.load_breast_cancer()` — ข้อมูลมะเร็งเต้านม
- `sklearn.datasets.fetch_california_housing()` — ราคาบ้านในแคลิฟอร์เนีย
- Titanic dataset (จาก Kaggle) — survival prediction
- Dataset อื่นที่สนใจ (แจ้งผู้สอนก่อน)

**สิ่งที่ต้องทำ:**
1. **โหลด + explore**: shape, features, target variable description
2. **Problem Statement**: อธิบายว่า X คืออะไร, Y คืออะไร, f คืออะไร
3. **จำแนกประเภท**: Supervised/Unsupervised? Regression/Classification? ทำไม?
4. **จำแนก approach**: Parametric หรือ Non-parametric เหมาะกว่า? ทำไม?
5. **สำรวจ data**: สร้าง visualization อย่างน้อย 2 อย่าง (histogram, scatter, box plot)
6. **ตั้งคำถาม**: ตั้ง 2 คำถามที่อยากตอบจาก dataset นี้ (จะตอบใน HW ถัดไป)

---

## เกณฑ์การให้คะแนนรวม

| ข้อ | หัวข้อ | คะแนน |
|-----|--------|--------|
| ข้อ 1 | จำแนกประเภท ML Problems | 30 |
| ข้อ 2 | Parametric vs Non-parametric | 30 |
| ข้อ 3 | Reducible vs Irreducible Error | 25 |
| ข้อ 4 | Open-ended Problem Definition | 15 |
| **รวม** | | **100** |

---

*Homework 05 | วิชา 1145 201 | กำหนดส่ง: สัปดาห์ที่ 6*
