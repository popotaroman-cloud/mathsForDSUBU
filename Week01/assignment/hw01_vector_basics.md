# Homework 01: Vector Basics และ NumPy เบื้องต้น
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO1 | **LLo**: อธิบายความสำคัญของคณิตศาสตร์ในวิทยาการข้อมูล และสร้าง Vector ได้  
**กำหนดส่ง**: สัปดาห์ที่ 2 (ก่อนชั่วโมง Lab)  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราได้เรียนรู้พื้นฐานที่สำคัญที่สุดของวิชาคณิตศาสตร์สำหรับวิทยาการข้อมูล นั่นคือ **Vector** และ **NumPy** ซึ่งเป็นหัวใจของทุกอย่างใน Machine Learning ในการบ้านฉบับนี้คุณจะได้ฝึกสร้าง Vector จากสถานการณ์จริง คำนวณ magnitude และ cosine similarity และตีความผลในบริบทของ Data Science เป้าหมายคือให้นักศึกษาเชื่อมโยงทฤษฎีที่เรียนในชั้นเรียนกับการใช้งาน NumPy ได้จริง โดยเฉพาะการแทนข้อมูลด้วย Vector ซึ่งเป็นทักษะที่จะใช้ตลอดทั้งภาคการศึกษา งานที่ได้ฝึกในการบ้านนี้เป็นพื้นฐานก่อนที่เราจะเรียน Matrix Operations และ Dot Product ในสัปดาห์ถัดไป

---

## ข้อ 1 — Feature Vector ของนักศึกษา (3 คะแนน)

ในข้อนี้คุณจะสร้าง **feature vector** ที่แทนข้อมูลของนักศึกษา 1 คน เพื่อแสดงให้เห็นว่า Machine Learning มองข้อมูลคนเป็น Vector ได้อย่างไร ทักษะนี้เป็นรากฐานของ Recommendation Systems, Clustering, และ Classification ทุกประเภท

**Dataset**: นักศึกษาคนหนึ่งมีข้อมูลดังต่อไปนี้:
- คะแนนสอบวิชาคณิตศาสตร์: **82 คะแนน**
- คะแนนสอบวิชา Python Programming: **91 คะแนน**
- จำนวนชั่วโมงที่ใช้ศึกษาต่อสัปดาห์: **15 ชั่วโมง**
- GPA: **3.45**

**สิ่งที่ต้องทำ**:
1. สร้าง numpy array ชื่อ `student_vec` จากข้อมูลด้านบน (**dtype=float64**)
2. Print: shape, dtype, ndim, และค่าสูงสุดและต่ำสุดของ vector นี้
3. Normalize vector นี้ให้มี magnitude = 1 (unit vector) โดยใช้สูตร: $\hat{v} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$
4. ตรวจสอบว่า `np.linalg.norm(unit_vec)` ≈ 1.0 หรือไม่ และ Print ผล

**Deliverable**: Code Python พร้อม output และอธิบาย **1 ย่อหน้า** ว่า unit vector มีประโยชน์อะไรใน ML

---

## ข้อ 2 — Cosine Similarity ระหว่าง User Profiles (4 คะแนน)

ในข้อนี้คุณจะใช้ Cosine Similarity เพื่อวัดความคล้ายคลึงระหว่าง User Profiles ของนักศึกษา 3 คน เพื่อแสดงให้เห็นว่า Recommendation System ของ Netflix หรือ Spotify ทำงานอย่างไรในระดับพื้นฐาน

**Dataset**: นักศึกษา 3 คนมี feature vectors ดังนี้ (features: Math score, Python score, Study hours/week, GPA):

```
Alice = [85, 88, 18, 3.8]
Bob   = [82, 91, 15, 3.45]
Carol = [55, 60,  8, 2.5]
```

**สิ่งที่ต้องทำ**:
1. สร้าง numpy arrays สำหรับ Alice, Bob, Carol
2. เขียน function `cosine_similarity(v1, v2)` ที่คำนวณ cosine similarity
3. คำนวณ similarity สำหรับทุกคู่: (Alice, Bob), (Alice, Carol), (Bob, Carol)
4. สร้าง similarity matrix ขนาด 3×3 และ print ในรูปแบบตาราง
5. ตอบคำถาม: **นักศึกษาคู่ไหนคล้ายกันมากที่สุด?** เพราะเหตุใด?

**Deliverable**: Code Python + output + คำอธิบาย **2–3 ประโยค**

---

## ข้อ 3 — Case Study: Vector ใน Real-World Data Science (3 คะแนน)

ในข้อนี้คุณจะวิเคราะห์ว่า Vector ถูกนำไปใช้ใน Domain ที่คุณสนใจอย่างไร โดยการค้นคว้าและอธิบายด้วยตัวเอง เพื่อให้เชื่อมโยงทฤษฎีกับสถานการณ์จริงที่ตรงกับความสนใจของคุณ

**สิ่งที่ต้องทำ**:

เลือก **1 Use Case** จากรายการด้านล่าง (หรือเสนอ Use Case ของตัวเองได้):

| ตัวเลือก | Domain | Vector คืออะไร |
|---------|--------|----------------|
| A | E-Commerce | User purchase history เป็น vector ของจำนวนครั้งซื้อสินค้าแต่ละหมวด |
| B | Healthcare | ผู้ป่วยแต่ละรายเป็น vector ของค่า blood test ต่าง ๆ |
| C | Sports Analytics | นักฟุตบอลแต่ละคนเป็น vector ของ performance stats |
| D | Social Media | โพสต์ของผู้ใช้เป็น vector ของ word frequencies |

จากนั้น:
1. **อธิบาย** ว่า data ถูกแปลงเป็น vector อย่างไร (feature แต่ละตัวคืออะไร)
2. **สร้าง** ตัวอย่างข้อมูล 3 records จำลอง (ไม่ต้องใช้ data จริง) เป็น numpy arrays
3. **คำนวณ** cosine similarity ระหว่าง records ทั้งหมดและตีความผล
4. **อธิบาย** ว่าระบบจะใช้ similarity นี้ทำอะไรได้ในสถานการณ์จริง

**Deliverable**: Code Python + output + คำอธิบาย **1 ย่อหน้า** ต่อขั้นตอน

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้วทุก cell
- ตั้งชื่อไฟล์: `hw01_XXXXXXXX.ipynb` (แทน X ด้วยรหัสนักศึกษา)
- อัปโหลดใน LMS ภายในกำหนด

---

## เกณฑ์การให้คะแนน (ดูรายละเอียดใน hw01_rubric.md)

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | Feature Vector + Unit Vector | 3 |
| 2 | Cosine Similarity + Similarity Matrix | 4 |
| 3 | Case Study Analysis | 3 |
| **รวม** | | **10** |
