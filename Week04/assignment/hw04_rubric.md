# Rubric — HW04: PCA Report

**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | CLO1  
**ผู้สอน**: _______________________  
**ภาคการศึกษา**: 1/2568

---

## ข้อ 1 — Eigendecomposition ด้วยมือ (25 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Center + Covariance** | Center ถูกต้อง, C ถูกต้อง 4 decimal, อธิบาย Bessel's correction | Center ถูกต้อง, C มีข้อผิดพลาดเล็กน้อย | Center ถูกแต่ C มีข้อผิดพลาด | Center หรือ C ผิด | 8 |
| **Characteristic equation** | เขียนและขยาย det(C-λI)=0 ครบถ้วน, แสดง polynomial | เขียนได้แต่ขาดขั้นตอนบางส่วน | เขียนได้บางส่วน | ไม่มีหรือผิดทั้งหมด | 7 |
| **Eigenvalues + Eigenvectors** | หา λ ถูกต้อง 2 ค่า, eigenvector normalized ถูกต้อง | หา λ ถูกแต่ eigenvector ผิด หรือไม่ normalize | หา λ ถูก 1 ค่า | ผิดทั้งหมด | 7 |
| **Python verification** | verify ด้วย eigh, อธิบาย sign convention | verify แต่ไม่อธิบาย | verify แต่ผลไม่ตรง | ไม่มี verification | 3 |

---

## ข้อ 2 — PCA from Scratch + sklearn (35 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Dataset selection & EDA** | เลือก dataset เหมาะสม, แสดง shape/features/stats ครบ | เลือก dataset แต่ EDA ไม่ครบ | มี dataset แต่ขาด EDA | ไม่มี dataset หรือผิดประเภท | 5 |
| **PCA from scratch** | implement ครบ 5 ขั้นตอน, code มี comment, ผลถูกต้อง | ขาด 1 ขั้นตอน หรือ comment ไม่ครบ | ขาด 2+ ขั้นตอน | ไม่มีหรือ error | 12 |
| **sklearn comparison** | verify ผล from-scratch vs sklearn, อธิบาย sign convention | verify แต่ไม่อธิบาย sign | verify แบบ partial | ไม่มี | 5 |
| **Scree Plot** | สวยงาม, มี cumulative line, 95% threshold, labels ครบ | มี bar + cumulative แต่ขาด label/threshold | มีแค่ bar chart | ไม่มีหรือผิด | 7 |
| **2D Scatter + Interpretation** | scatter สวย, color by class, labels ครบ, interpretation ถูกต้อง | scatter ดีแต่ interpretation ผิวเผิน | มี scatter แต่ไม่มี color/interpretation | ไม่มีหรือ error | 6 |

---

## ข้อ 3 — SVD Compression Analysis (25 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **k_min table (5 digits)** | ตารางครบ 5 digits, k_min ถูกต้อง, compression ratio ถูกต้อง | ครบ 5 digits แต่ ratio ผิด 1–2 ค่า | มีแค่ 3 digits | ไม่มีตาราง | 10 |
| **Reconstruction visualization** | แสดง k=1,2,3,4,5,8 ชัดเจน, title บอก % energy | แสดงครบแต่ title/label ไม่ครบ | แสดงได้บางส่วน | ไม่มีหรือ error | 7 |
| **RMSE graph** | graph 5 digits บน plot เดียว, legend, grid, อธิบาย diminishing return | graph ดีแต่ขาด legend หรืออธิบาย | graph 1–2 digit | ไม่มี | 8 |

---

## ข้อ 4 — Open-ended Case Study (15 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **5 ส่วนของ Case Study** | ครบ 5 ส่วน (Scenario/Data/Method/Result/Insight), ลึกซึ้ง | ครบ 5 ส่วนแต่บางส่วนผิวเผิน | มีแค่ 3–4 ส่วน | น้อยกว่า 3 ส่วน | 8 |
| **Code + Visualization** | code ทำงานได้ มี comment, graph ชัดเจน, ตัวเลขจริง | code ทำงานได้แต่ comment/graph ไม่ครบ | code มี error บางส่วน | ไม่มี code | 5 |
| **Insight & Creativity** | Insight แปลกใหม่, เชื่อมกับ real-world, มีข้อเสนอต่อ | Insight ดีแต่ไม่มีข้อเสนอต่อ | Insight ทั่วไป | ไม่มี insight | 2 |

---

## โบนัส (สูงสุด 5 คะแนน)

| เกณฑ์ | คะแนนโบนัส |
|-------|-----------|
| ทำครบทั้ง 3 Option ใน ข้อ 4 (แทนที่จะเลือก 1) | +3 |
| ใช้ dataset ที่ไม่ได้อยู่ในรายการที่แนะนำ และน่าสนใจมาก | +2 |
| Notebook presentation ดีมาก (markdown อ่านง่าย, cell organization ดี) | +1 |

---

## หมายเหตุสำหรับผู้สอน

- **Sign convention**: eigenvector direction ไม่ unique (สามารถ flip sign) — หักคะแนนได้เฉพาะถ้า magnitude ผิด
- **eigh vs eig**: ถ้านักศึกษาใช้ eig กับ covariance matrix แล้วผลถูก ให้คะแนนเต็ม แต่ comment ว่าควรใช้ eigh
- **sklearn vs from-scratch**: ถ้า from-scratch ผิดแต่เหตุผลจากเรื่อง numerical precision ไม่หักคะแนน
- **Code comments**: ถ้า code ทำงานถูกต้องแต่ไม่มี comment หักได้ 20% ของ cell นั้น

---

*Rubric — HW04 | วิชา 1145 201 | ภาคการศึกษา 1/2568*
