# Slide Deck: บทบาทของคณิตศาสตร์ใน Data Science
> Week 01 | CLO1 | 10 slides

---
## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล — Week 1**  
บทนำ: ทำไม Data Scientist ต้องเรียนคณิตศาสตร์?  
CLO1: อธิบายความสำคัญของคณิตศาสตร์ใน Data Science และสร้าง Vector ได้

---
## Slide 2 — Week Overview
**Key Message**: คณิตศาสตร์ไม่ใช่อุปสรรค แต่เป็นภาษาของ Machine Learning

สัปดาห์นี้เราจะเริ่มต้นการเดินทางสู่โลกของ Data Science ด้วยการทำความเข้าใจว่าคณิตศาสตร์เชื่อมกับ AI และ ML อย่างไร เป้าหมายของสัปดาห์นี้คือสร้างแรงจูงใจและ mental model ที่ถูกต้องก่อนที่เราจะลงลึกในหัวข้อต่าง ๆ ในสัปดาห์ถัดไป ทุกสิ่งที่เราจะเรียนในวิชานี้ตั้งแต่ Vector ถึง SVD ล้วนนำไปสู่การสร้าง AI ได้จริง

**สิ่งที่จะเรียนวันนี้**:
- ภาพรวมคณิตศาสตร์ 3 สาขา: Linear Algebra, Calculus, Statistics
- ทำไม Vector และ Matrix จึงเป็นหัวใจของ ML
- ตัวอย่างจริงที่คุณใช้อยู่ทุกวันแต่ไม่รู้ว่ามีคณิตศาสตร์อยู่ข้างหลัง

---
## Slide 3 — คณิตศาสตร์ 3 เสาหลักของ ML
**Key Message**: Linear Algebra + Calculus + Statistics = Machine Learning

```
┌─────────────────────────────────────────────────┐
│                Machine Learning                 │
│                                                 │
│   Linear Algebra  +  Calculus  +  Statistics   │
│   ─────────────     ─────────    ─────────────  │
│   Data as vectors   Optimization  Uncertainty   │
│   Transformations   Gradient     Probability    │
│   Compression       Learning     Inference      │
└─────────────────────────────────────────────────┘
```

- **Linear Algebra** → แทนข้อมูลด้วย Vector/Matrix, PCA, Neural Network weights
- **Calculus** → Gradient Descent, Backpropagation, Optimization
- **Statistics** → Hypothesis testing, Confidence interval, Model evaluation
- [FIGURE: Venn diagram แสดง overlap ของ 3 สาขา]

---
## Slide 4 — ข้อมูลทุกอย่างคือ Vector
**Key Message**: Vector คือภาษากลางที่คอมพิวเตอร์เข้าใจข้อมูล

| ข้อมูล | Vector representation |
|--------|----------------------|
| รูปภาพ 28×28 pixel | vector ขนาด 784 |
| ประโยคภาษาอังกฤษ | word embedding ขนาด 768 (BERT) |
| ผู้ป่วย 1 คน | lab results เป็น vector 50+ features |
| หนัง 1 เรื่อง | rating pattern เป็น vector ของ users |

- **Demo**: แสดง MNIST digit ที่กลายเป็น array ตัวเลข
- [FIGURE: รูปภาพเลข "5" และ numpy array ที่ correspond]

---
## Slide 5 — Real Use Cases
**Key Message**: AI ที่คุณใช้ทุกวันล้วนอาศัย Vector operations

1. **Netflix Recommendation**: cosine similarity ระหว่าง user vectors
2. **Google Translate**: word embedding + matrix transformation ข้ามภาษา
3. **Face Unlock บนโทรศัพท์**: eigenfaces (PCA) เปรียบเทียบ face vector
4. **ChatGPT**: attention mechanism คือ dot product ของ query/key/value vectors

- [EXAMPLE: แสดง simplified ว่า Netflix คำนวณ similarity ยังไง]
- คำถามสำหรับนักศึกษา: "คุณใช้ app ไหนที่ใช้ Vector อยู่บ้าง?"

---
## Slide 6 — แผนการเรียน 15 สัปดาห์
**Key Message**: ทุก Week เชื่อมกัน — เรียนครบแล้วทำ Final Project ได้

```
Weeks 1–4: Linear Algebra Foundation
  W1: Vector basics  →  W2: Matrix ops  →  W3: Subspaces  →  W4: SVD/PCA

Weeks 5–7: Statistical Learning Framework
  W5: SL overview  →  W6: Bias-Variance  →  W7: EDA + Inference

Weeks 8–10: Regression
  W8: SLR  →  W9: MLR  →  W10: Diagnostics

Weeks 11–13: Classification & GLM
  W11: Logistic  →  W12: LDA/QDA  →  W13: GLM/ROC

Week 14–15: Model Selection + Final Project
  W14: CV/Bootstrap  →  W15: Final Presentation
```

---
## Slide 7 — เครื่องมือที่เราจะใช้
**Key Message**: Python + NumPy + statsmodels + sklearn = toolkit ของ Data Scientist

| เครื่องมือ | ใช้ทำอะไร | เริ่มใช้ใน Week |
|-----------|---------|--------------|
| **NumPy** | Vector/Matrix operations | Week 1 |
| **Matplotlib** | Visualization | Week 1 |
| **Pandas** | Data manipulation | Week 7 |
| **statsmodels** | Statistical models + inference | Week 7 |
| **scikit-learn** | Machine learning models | Week 8 |
| **ISLP** | ISLP datasets + utilities | Week 8 |

- ทุก library นี้จะถูกใช้ใน Lab และ Assignment
- [EXAMPLE: `import numpy as np; v = np.array([1, 2, 3])`]

---
## Slide 8 — วิธีประเมินผล
**Key Message**: เรียนรู้ผ่านการปฏิบัติ — คะแนนส่วนใหญ่มาจาก Lab + Project

| CLO | เนื้อหา | Week | คะแนน |
|-----|--------|------|-------|
| CLO1 | Linear Algebra | 1–4 | 20% (สอบกลางภาค) |
| CLO2 | Statistical Learning | 5–7 | 10% (สอบกลางภาค) |
| CLO3 | Regression & Classification | 8–13 | 20% (สอบปลายภาค) |
| CLO4 | Model Validation | 14 | 20% (สอบปลายภาค) |
| — | Lab Exercises | ทุก Week | 20% |
| — | Final Project | Week 15 | 20% (รายงาน+นำเสนอ) |

---
## Slide 9 — Tips สำหรับ Lab 01
**Key Message**: อย่ากลัว error — error คือการเรียนรู้

**สิ่งที่จะทำใน Lab 01:**
1. ติดตั้งและ import NumPy
2. สร้าง vector จาก list และ special functions
3. คำนวณ magnitude, dot product, cosine similarity
4. 3 TODOs: Easy → Medium → Hard
5. Case Study: User similarity ใน Recommendation System

**Tips:**
- อ่าน error message ทุกบรรทัดก่อนถาม
- ใช้ `help(np.array)` หรือ `?np.dot` ใน Jupyter
- ลอง `print(v.shape, v.dtype)` ก่อนทำ operation ใด ๆ

---
## Slide 10 — Summary & Next Week
**Key Message**: คณิตศาสตร์ 3 สาขา (LA + Calc + Stats) คือรากฐานที่ทำให้ทุก algorithm ไม่ใช่ black box — เริ่มต้น Week 1 นี้คือก้าวแรกสู่การเข้าใจ ML อย่างแท้จริง

**สิ่งที่เรียนรู้วันนี้**:
- คณิตศาสตร์ 3 สาขา (LA + Calc + Stats) เป็นพื้นฐาน ML
- ข้อมูลทุกชนิดถูกแทนด้วย Vector
- NumPy คือเครื่องมือหลักสำหรับ vector computation ใน Python

**Checklist ก่อนออกห้อง**:
- [ ] สามารถอธิบายได้ว่า Vector คืออะไรและใช้ทำอะไรใน DS
- [ ] `import numpy as np; print(np.array([1,2,3]).shape)` run ได้
- [ ] เข้าใจว่า shape, dtype, ndim ต่างกันอย่างไร

**สัปดาห์ต่อไป — Week 2**: Matrix Operations, Gaussian Elimination, LU Decomposition
