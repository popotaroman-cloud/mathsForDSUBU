# Homework 03: Subspaces และ Least Squares
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO1 | **LLo**: อธิบาย Vector Space, Subspace, Rank และคำนวณ Projection / Least Squares ได้  
**กำหนดส่ง**: สัปดาห์ที่ 4  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราได้เรียนรู้ว่า Linear Regression ที่เราจะใช้ตั้งแต่ Week 8 เป็นต้นไปนั้น มีรากฐานทางคณิตศาสตร์มาจาก **Least Squares** และ **Normal Equations** บน design matrix ในการบ้านฉบับนี้คุณจะได้วิเคราะห์ subspaces ของ matrix จริง คำนวณ projection และ fit โมเดลด้วย least squares เป้าหมายคือให้นักศึกษาเชื่อมโยงแนวคิด Null Space, Column Space, Rank กับการ detect multicollinearity รวมถึงเข้าใจว่า least squares คือการ project response vector ลงบน column space ของ design matrix

---

## ข้อ 1 — Four Fundamental Subspaces (3 คะแนน)

ในข้อนี้คุณจะวิเคราะห์ matrix A เพื่อหา 4 fundamental subspaces ตาม Strang's Big Picture เพื่อให้เข้าใจ structure ของ matrix ก่อนนำไปใช้ใน regression

$$A = \begin{pmatrix} 1 & 2 & 3 & 0 \\ 2 & 4 & 6 & 1 \\ 3 & 6 & 9 & 2 \end{pmatrix}$$

**สิ่งที่ต้องทำ**:
1. หา rank(A), nullity(A) และ verify Rank-Nullity Theorem: rank + nullity = n = 4
2. หา dimension ของ Column Space C(A) และ Left Null Space N(Aᵀ)
3. ใช้ SVD (`np.linalg.svd(A)`) หา basis ของ Null Space N(A) (จาก V matrix, rows ที่สอดคล้องกับ zero singular values)
4. Verify null space vector: A @ null_vec ≈ 0
5. สรุปใน **table** แสดง 4 subspaces พร้อม dimension

**Deliverable**: Code + output + table สรุป 4 subspaces

---

## ข้อ 2 — Least Squares on Real Data (4 คะแนน)

ในข้อนี้คุณจะ fit Linear Regression model กับ data จริง โดยใช้ Normal Equations โดยตรง เพื่อแสดงว่า regression คือ least squares projection ทางเลือก: ใช้ Advertising dataset หรือ data ที่ให้ด้านล่าง

**Dataset**: ข้อมูลงบโฆษณา TV (x, พัน$) และ Sales (y, พันหน่วย) — 10 ตัวอย่าง:

```python
tv    = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
sales = [12, 18, 25, 31, 38, 44, 52, 58, 65, 72]
```

**สิ่งที่ต้องทำ**:
1. สร้าง design matrix **A** = [**1**, **x**] (เพิ่ม intercept column)
2. Solve Normal Equations **AᵀA β̂ = Aᵀy** manually: `np.linalg.solve(A.T@A, A.T@y)`
3. Solve ด้วย `np.linalg.lstsq(A, y)` — verify ว่าได้ β̂ เหมือนกัน
4. คำนวณ **residuals** e = y - Aβ̂, RSS = ||e||², TSS = ||y - ȳ||²
5. คำนวณ **R²** = 1 - RSS/TSS (ตีความ)
6. Plot: scatter data + fitted line + residual line segments

**Deliverable**: Code + output + รายงาน **3–5 ประโยค** ตีความ β̂₀, β̂₁, และ R²

---

## ข้อ 3 — Case Study: Polynomial Regression (3 คะแนน)

ในข้อนี้คุณจะ fit **polynomial regression** โดยใช้ least squares เพื่อแสดงว่า polynomial features เพียงแค่ขยาย design matrix ให้มีมากกว่า 2 columns — ไม่มีอะไรใหม่ แค่ linear algebra เดิม

**Scenario**: นักวิทยาศาสตร์วัดอุณหภูมิ (°C) ของปฏิกิริยาเคมีในช่วงเวลา t (นาที):

```python
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
T = np.array([20, 25, 32, 40, 45, 43, 38, 30, 22], dtype=float)
```

**สิ่งที่ต้องทำ**:
1. Fit **linear** model: T = β₀ + β₁t ด้วย lstsq
2. Fit **quadratic** model: T = β₀ + β₁t + β₂t² ด้วย lstsq
3. Fit **cubic** model: T = β₀ + β₁t + β₂t² + β₃t³ ด้วย lstsq
4. สำหรับแต่ละโมเดล คำนวณ R² และ RSE = √(RSS/(n-p))
5. Plot: data + 3 fitted curves + ตาราง R² และ RSE เปรียบเทียบ
6. **ตีความ**: โมเดลไหนเหมาะสมที่สุด? ทำไม cubic จึงไม่ดีที่สุดเสมอ?

**Deliverable**: Code + output + คำอธิบาย **1 ย่อหน้า** เรื่อง trade-off ระหว่าง complexity และ fit quality

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้วทุก cell
- ตั้งชื่อ: `hw03_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | Four Fundamental Subspaces | 3 |
| 2 | Least Squares on Data | 4 |
| 3 | Polynomial Regression | 3 |
| **รวม** | | **10** |
