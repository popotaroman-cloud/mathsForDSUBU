# Homework 02: Matrix Operations และ Linear Systems
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO1 | **LLo**: ดำเนินการทางแมทริกซ์ แก้ระบบสมการ Ax = b ด้วย Elimination และคำนวณ Determinant ได้  
**กำหนดส่ง**: สัปดาห์ที่ 3 (ก่อนชั่วโมง Lab)  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราได้เรียนรู้ว่า **Matrix** ไม่ใช่แค่ตารางตัวเลข แต่คือเครื่องมือในการแทน transformation และแก้ระบบสมการ ซึ่งเป็นหัวใจของ Machine Learning ตั้งแต่ Linear Regression ไปจนถึง Deep Neural Networks ในการบ้านฉบับนี้คุณจะได้ฝึกดำเนินการทางแมทริกซ์ แก้ระบบสมการ Ax = b ด้วยทั้ง NumPy และ implement เอง รวมถึงตีความ Determinant ในบริบทของ Data Science เป้าหมายคือให้นักศึกษาเชื่อมโยง math notation กับ Python code ได้อย่างมั่นใจ ทักษะเหล่านี้เป็นพื้นฐานก่อนเรียน Vector Spaces, Eigenvalues, และ SVD ในสัปดาห์ถัดมา

---

## ข้อ 1 — Matrix Properties (3 คะแนน)

ในข้อนี้คุณจะ verify คุณสมบัติสำคัญของ Matrix multiplication เพื่อให้เข้าใจว่าทำไม Matrix algebra จึงต่างจาก scalar algebra ซึ่งสำคัญมากเวลาอ่าน derivation ของ ML algorithms

**Dataset**: กำหนด matrices:

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ 2 & 3 \end{pmatrix}, \quad C = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$$

**สิ่งที่ต้องทำ**:
1. คำนวณ AB และ BA — แสดงว่า AB ≠ BA (non-commutative)
2. Verify: (AB)C = A(BC) (associative)
3. Verify: (AB)ᵀ = BᵀAᵀ (transpose of product)
4. หา Aᵀ และตรวจสอบว่า (Aᵀ)ᵀ = A
5. สร้าง **symmetric matrix** S ขนาด 3×3 โดยใช้ S = AᵀA (สำหรับ A random 4×3)

**Deliverable**: Code Python + output + อธิบาย **1 ย่อหน้า** ว่าทำไม non-commutativity ถึงสำคัญใน ML

---

## ข้อ 2 — Solving Ax = b (4 คะแนน)

ในข้อนี้คุณจะแก้ระบบสมการเชิงเส้นและวิเคราะห์ว่า Determinant บอกอะไรเกี่ยวกับ solution เพื่อเชื่อมโยงกับ multicollinearity ใน Week 9–10

**ระบบสมการที่ 1** (Unique solution):
```
3x + 2y + z = 10
x - y + 2z = -1
2x + 3y - z = 8
```

**ระบบสมการที่ 2** (No unique solution หรือ infinite solutions):
```
x + 2y + z = 3
2x + 4y + 2z = 6
3x + 6y + 3z = 9
```

**สิ่งที่ต้องทำ**:
1. แก้ระบบสมการที่ 1 ด้วย `np.linalg.solve()` และ verify ด้วย A @ x == b
2. คำนวณ det(A) สำหรับทั้ง 2 ระบบ — อธิบายว่าทำไมระบบที่ 2 ถึงแก้ไม่ได้
3. หา LU decomposition ของ matrix A จากระบบที่ 1 ด้วย `scipy.linalg.lu()` — verify ว่า PA = LU
4. Implement **back substitution** (ไม่ต้องทำ forward elimination) สำหรับ upper triangular matrix U จาก step 3

**Deliverable**: Code Python + output + อธิบาย **2–3 ประโยค** สำหรับแต่ละ step

---

## ข้อ 3 — Case Study: Budget Allocation (3 คะแนน)

ในข้อนี้คุณจะประยุกต์ความรู้ Matrix และ Linear System กับปัญหาจริงด้านการวางแผนงบประมาณ เพื่อแสดงให้เห็นว่า Math → Decision Making ทำงานอย่างไร

**Scenario**: บริษัท Startup ต้องการจัดสรรงบประมาณการตลาดรายเดือน 3 ช่องทาง: TV (x₁), Social Media (x₂), Search Ads (x₃)

**เงื่อนไข** (สมการจาก historical data):
```
x₁ + 2x₂ + x₃ = 50        (งบรวม = 50 ล้านบาท)
3x₁ + x₂ + 2x₃ = 80       (ผลตอบแทน ROI ตาม formula = 80)
x₁ + 3x₂ + 4x₃ = 100      (reach target = 100 ล้าน impressions)
```

**สิ่งที่ต้องทำ**:
1. สร้าง matrix A และ vector b แล้วแก้หา x = [x₁, x₂, x₃]
2. Verify ผลด้วย `np.allclose(A @ x, b)`
3. คำนวณ condition number ด้วย `np.linalg.cond(A)` — ถ้า > 100 แสดงว่า ill-conditioned
4. **เพิ่มงบรวมเป็น 55 ล้านบาท** (b[0] = 55 แต่เงื่อนไขอื่นเดิม) — งบแต่ละช่องทางเปลี่ยนอย่างไร?
5. ตีความผล: ช่องทางไหน cost-effective ที่สุด?

**Deliverable**: Code Python + output + สรุป **1 ย่อหน้า** สำหรับผู้บริหารที่ไม่ใช่นักคณิตศาสตร์

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้วทุก cell
- ตั้งชื่อไฟล์: `hw02_XXXXXXXX.ipynb`
- อัปโหลดใน LMS ภายในกำหนด

---

## เกณฑ์การให้คะแนน (ดูรายละเอียดใน hw02_rubric.md)

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | Matrix Properties | 3 |
| 2 | Solving Ax = b | 4 |
| 3 | Case Study | 3 |
| **รวม** | | **10** |
