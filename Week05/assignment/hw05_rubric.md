# Rubric — HW05: Statistical Learning Concepts

**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | CLO2  
**ผู้สอน**: _______________________  
**ภาคการศึกษา**: 1/2568

---

## ข้อ 1 — จำแนกประเภท ML Problems (30 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Supervised vs Unsupervised** (10 ข้อ) | ถูกต้อง ≥ 9/10 ข้อ พร้อมเหตุผลชัดเจน | ถูกต้อง 7–8/10 | ถูกต้อง 5–6/10 | ถูกต้อง < 5/10 | 10 |
| **Regression vs Classification** | ถูกต้องทุก Supervised problem พร้อมเหตุผล | ถูก 75%+ | ถูก 50%+ | ถูก < 50% | 10 |
| **X และ Y identification** | ระบุ X, Y ได้ครบและสมเหตุสมผลทุกข้อ | ขาด X หรือ Y บางข้อ | ระบุได้บางส่วน | ส่วนใหญ่ผิด | 10 |

**เฉลยที่ควรได้:**
1. Supervised, Regression, X=พื้นที่/ทำเล/ห้อง, Y=ราคา
2. Unsupervised (Clustering), ไม่มี Y
3. Supervised, Classification, X=อาการ 15 อย่าง, Y=COVID/ไม่ใช่
4. Unsupervised/Semi-supervised (Recommender), note: อาจ supervised ถ้ามี explicit rating
5. Supervised, Regression, X=ปัจจัยบรรยากาศ, Y=อุณหภูมิ
6. Supervised, Classification, X=word frequency, Y=spam/not
7. Unsupervised (Clustering), ไม่มี Y
8. Supervised, Classification, X=demographic+history, Y=click/not
9. Supervised, Regression (Inference-focused), X=ปีการศึกษา, Y=รายได้
10. Unsupervised (Anomaly detection), ไม่มี labeled Y

---

## ข้อ 2 — Parametric vs Non-parametric (30 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Data exploration** | describe() + 2 scatter plots พร้อม title/labels | มี plot แต่ขาด label | plot อย่างหนึ่ง | ไม่มี | 5 |
| **Linear Regression** | coefficients + R² + test MSE ถูกต้อง, plot fitted line | ผล OK แต่ขาด plot | มีแค่ fit model | ไม่มีหรือ error | 8 |
| **KNN Regression** | test MSE สำหรับ k=1,5,10 ครบ, plot comparison | k=1,5,10 ครบแต่ขาด plot | MSE บางค่า | ไม่มีหรือ error | 8 |
| **Comparison table** | ตารางครบ, ตัวเลขถูกต้อง | ตารางครบแต่บางค่าผิด | ตารางไม่ครบ | ไม่มีตาราง | 5 |
| **Explanation** | อธิบาย overfitting ของ k=1 ได้ถูกต้อง, choice justified | อธิบายได้แต่ไม่ครบ | อธิบายผิวเผิน | ไม่มีหรือผิด | 4 |

---

## ข้อ 3 — Reducible vs Irreducible Error (25 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Simulation + Plot** | plot Y vs X + f_true อ่านง่าย, มี legend | plot ดีแต่ขาด f_true | มีแค่ scatter | ไม่มีหรือ error | 8 |
| **Irreducible error analysis** | MSE(f_true) ≈ Var(ε), อธิบายได้ว่าทำไม | คำนวณถูกแต่อธิบายไม่ครบ | คำนวณได้บางส่วน | ผิดหรือไม่มี | 10 |
| **Linear Regression + reducible** | fit model + คำนวณ reducible error ถูกต้อง | fit ได้แต่ reducible error ผิด | มีแค่ fit | ไม่มี | 7 |

**หมายเหตุ**: MSE(f_true) ≈ σ² ≈ 4 (เพราะ σ=2, Var=4) — ถ้านักศึกษาได้ค่าใกล้ 4 ถือว่าถูกต้อง

---

## ข้อ 4 — Open-ended Problem Definition (15 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Data exploration** | shape + features + target ครบ + ≥2 visualizations | EDA ดีแต่ขาด visualization | มีแค่ head/describe | ไม่มี EDA | 5 |
| **Problem statement (X, Y, f)** | X, Y, f ระบุชัดเจน สมเหตุสมผล | ระบุได้แต่ f ไม่ชัด | ระบุ X, Y แต่ไม่มี f | ไม่มี | 5 |
| **Classification + Approach choice** | จำแนกถูกต้อง, เหตุผล approach สมเหตุสมผล | จำแนกถูกแต่ approach ไม่มีเหตุผล | จำแนกผิดบางส่วน | ผิดหรือไม่มี | 3 |
| **2 Research questions** | คำถาม 2 ข้อ ตอบได้ด้วย ML, น่าสนใจ | คำถาม 2 ข้อ แต่ทั่วไป | 1 คำถาม | ไม่มีหรือตอบไม่ได้ด้วย ML | 2 |

---

## หมายเหตุสำหรับผู้สอน

- **ข้อ 1 #4 (Netflix)**: รับทั้ง Supervised (ถ้ามี explicit rating) และ Unsupervised (collaborative filtering ไม่มี Y ชัดเจน)
- **ข้อ 2**: ถ้านักศึกษา standardize ข้อมูลก่อน KNN ให้ bonus point (เพราะเป็น best practice)
- **ข้อ 3**: Var(epsilon) ≠ MSE(f_true) ตรงๆ เพราะ finite sample — รับ ±20% ถือว่าถูกต้อง
- **ข้อ 4**: รับ dataset ใดก็ได้ที่มี numerical features อย่างน้อย 3 ตัว

---

*Rubric — HW05 | วิชา 1145 201 | ภาคการศึกษา 1/2568*
