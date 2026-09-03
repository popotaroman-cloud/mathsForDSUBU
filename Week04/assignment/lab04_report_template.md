# Lab Report Template — Lab 04: Eigenvalues, SVD & PCA
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**ชื่อ-นามสกุล**: _______________________  
**รหัสนักศึกษา**: _______________________  
**วันที่ทำ Lab**: _______________________  
**Section**: _______________________

---

## 1. สรุปผลการทดลอง

### 1.1 TODO 1 — Eigenvalues ของ Covariance Matrix

**ผลลัพธ์ที่ได้:**

| Feature | Eigenvalue | % Variance Explained |
|---------|-----------|---------------------|
| PC1 | | |
| PC2 | | |
| PC3 | | |

**สิ่งที่สังเกตได้:**
> _(เขียนอธิบาย 2–3 ประโยค: eigenvalue ที่ใหญ่ที่สุดคืออะไร? อธิบาย variance ได้กี่ %?)_

---

### 1.2 TODO 2 — Reduced SVD & Low-rank Approximation

**ผลลัพธ์:**

| k | Frobenius Error | % Variance Explained |
|---|----------------|---------------------|
| 1 | | |
| 2 | | |
| Full | ≈ 0 | 100% |

**สิ่งที่สังเกตได้:**
> _(อธิบาย: ทำไม k=2 ถึง explain variance ได้เกือบ 100%? แสดงว่า matrix นี้มีสมบัติอะไร?)_

---

### 1.3 TODO 3 — Explained Variance Ratio (Iris)

**ตาราง variance explained:**

| PC | Eigenvalue | Explained% | Cumulative% |
|----|-----------|------------|------------|
| PC1 | | | |
| PC2 | | | |
| PC3 | | | |
| PC4 | | | |

**k สำหรับ 95% variance**: _____ PCs

**สิ่งที่สังเกตได้:**
> _(PC1+PC2 รวมกัน = ____% แปลว่าอะไรสำหรับ Iris dataset?)_

---

### 1.4 TODO 4 — Scree Plot + 2D PCA Scatter

**[แทรก screenshot ของ plot ที่นี่]**

**การอ่าน Scree Plot:**
> _(Elbow point อยู่ที่ PC ไหน? หมายความว่าอะไร?)_

**การอ่าน 2D Scatter:**
> _(Species ไหนแยกออกจากกันได้ชัดเจนที่สุด? ทำไม?)_

---

### 1.5 TODO 5 — SVD Image Compression

**k_min สำหรับ 95% energy**: _____  
**Storage: 64 values → _____ values (_____x compression)**

**[แทรก screenshot ของ 3-panel plot ที่นี่]**

**สิ่งที่สังเกตได้:**
> _(k_min มีค่าน้อยมาก — อธิบายว่าทำไม digit image มี low rank?)_

---

## 2. คำถาม Reflection (ตอบให้ครบทั้ง 2 ข้อ)

**คำถาม 1:** จาก Scree Plot — ถ้าใช้แค่ PC1 เดียว จะสูญเสีย information ไปกี่ %?  
ในแง่ classification task นั่นหมายความว่าอะไร?

> **คำตอบ:**
>
> _(เขียนตอบที่นี่ อย่างน้อย 3 ประโยค)_

---

**คำถาม 2:** k_min ≈ ? สำหรับ digit image 8×8  
ทำไม digit image จึงมี effective rank ต่ำมาก? insight นี้บอกอะไรเกี่ยวกับ natural images?

> **คำตอบ:**
>
> _(เขียนตอบที่นี่ อย่างน้อย 3 ประโยค)_

---

## 3. ปัญหาที่พบและวิธีแก้

> _(เขียนปัญหาที่พบระหว่างทำ lab และวิธีที่แก้ไข หรือ "ไม่มีปัญหา" ก็ได้)_

---

## 4. สิ่งที่เรียนรู้เพิ่มเติม

> _(สิ่งที่ค้นพบด้วยตัวเอง, connection กับสิ่งที่เรียนมาก่อน, หรือคำถามที่ยังสงสัย)_

---

*ส่งพร้อม Notebook ที่ Run All Cells แล้ว | Lab 04 | วิชา 1145 201*
