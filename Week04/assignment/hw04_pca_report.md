# Homework 04: PCA Report — Dimensionality Reduction บน Dataset จริง
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO1 | **LLo**: คำนวณ Eigenvalue/Eigenvector, SVD และประยุกต์ใช้ PCA ลด dimension ชุดข้อมูลจริงได้  
**กำหนดส่ง**: สัปดาห์ที่ 5 (ก่อนเริ่มชั่วโมง Lecture)  
**รูปแบบการส่ง**: Jupyter Notebook (.ipynb) พร้อม output ของทุก cell + รายงาน PDF/Markdown

---

## บทนำ

สัปดาห์นี้เราเรียนรู้เรื่อง **Eigenvalues/Eigenvectors**, **SVD** และ **PCA** ซึ่งเป็นเครื่องมือ Linear Algebra ที่สำคัญที่สุดในวงการ Data Science Eigenvalue บอกเราว่า matrix transformation ยืด/หดข้อมูลในแต่ละทิศทางเท่าไร SVD แยก matrix ใดๆ ออกเป็นโครงสร้างที่มีความหมาย และ PCA ใช้สิ่งเหล่านี้เพื่อหาทิศทางที่ข้อมูลกระจายตัวมากที่สุด การบ้านนี้จะทดสอบว่านักศึกษาสามารถนำทั้ง 3 เรื่องไปประยุกต์ใช้กับ dataset จริงที่เลือกเองได้ เป้าหมายคือให้นักศึกษาสามารถอธิบาย PCA ได้ทั้งในแง่คณิตศาสตร์ (eigendecomposition) และในแง่ผลลัพธ์เชิงปฏิบัติ (visualization, interpretation) ทักษะนี้จะเป็นพื้นฐานสำคัญสำหรับ Feature Engineering และ Unsupervised Learning ในวิชาถัดไป

---

## ข้อ 1 — Eigendecomposition ด้วยมือ (25 คะแนน)

ในข้อนี้คุณจะคำนวณ eigenvalues และ eigenvectors ของ covariance matrix ด้วยมือ (หรือ Python step-by-step) เพื่อแสดงให้เห็นว่าเข้าใจทุกขั้นตอนของกระบวนการ ไม่ใช่แค่เรียก sklearn

**Dataset**: ใช้ข้อมูล 2D ง่ายๆ นี้ (10 จุด):

```
X = [[2.5, 2.4],
     [0.5, 0.7],
     [2.2, 2.9],
     [1.9, 2.2],
     [3.1, 3.0],
     [2.3, 2.7],
     [2.0, 1.6],
     [1.0, 1.1],
     [1.5, 1.6],
     [1.1, 0.9]]
```

**สิ่งที่ต้องทำ:**

**(a)** Center data: คำนวณ mean ของแต่ละ column และลบออก  
**(b)** Covariance matrix: คำนวณ C = (1/(n-1)) × X_centered.T @ X_centered  
แสดงค่า C ทั้ง 4 elements พร้อม 4 decimal places  
**(c)** Characteristic equation: เขียน det(C - λI) = 0 และขยาย polynomial  
**(d)** แก้สมการหา λ₁ และ λ₂ (แสดงทุกขั้นตอน)  
**(e)** หา eigenvector สำหรับแต่ละ eigenvalue โดย solve (C - λI)x = 0  
**(f)** Normalize eigenvectors ให้มีความยาว = 1  
**(g)** Verify ด้วย Python: `np.linalg.eigh(C)` ให้ผลตรงกันหรือไม่?

**Deliverable**: Python code พร้อม output ทุก step + คำอธิบาย

---

## ข้อ 2 — PCA from Scratch บน Dataset ที่เลือกเอง (35 คะแนน)

ในข้อนี้คุณจะเลือก dataset ที่มี **อย่างน้อย 5 features** และทำ PCA ทั้ง from scratch และด้วย sklearn เพื่อเปรียบเทียบ ข้อนี้วัดว่าคุณสามารถนำ PCA ไปใช้กับ real data ได้จริงหรือไม่

**Dataset ที่แนะนำ** (เลือก 1):
- `sklearn.datasets.load_wine()` — 13 features
- `sklearn.datasets.load_breast_cancer()` — 30 features
- `sklearn.datasets.fetch_california_housing()` — 8 features
- Dataset จาก Kaggle/UCI ที่มี ≥ 5 numerical features (แจ้งผู้สอนก่อน)

**สิ่งที่ต้องทำ:**

**(a)** โหลด dataset และแสดง: shape, feature names, sample count  
**(b)** **PCA from scratch** (implement ทุกขั้นตอน):
  1. Center data
  2. Covariance matrix
  3. Eigendecomposition ด้วย `np.linalg.eigh`
  4. Sort eigenvectors by eigenvalue (descending)
  5. Project onto top-2 PCs
  
**(c)** **PCA ด้วย sklearn** และ verify ว่าให้ผลเดียวกับ from-scratch  
**(d)** **Scree Plot**: แสดง explained variance per PC + cumulative line + 95% threshold  
**(e)** **2D Scatter Plot**: plot data ใน PCA space (ถ้า dataset มี class label ให้ color by class)  
**(f)** **Interpretation**: อธิบายว่า PC1 และ PC2 แสดงถึง pattern อะไรใน data (ดูจาก loadings)

**Deliverable**: Jupyter Notebook พร้อม output ทุก step

---

## ข้อ 3 — SVD Image Compression Analysis (25 คะแนน)

ในข้อนี้คุณจะวิเคราะห์ SVD compression อย่างเป็นระบบ โดยทำ experiment บน digit images หลายตัวเพื่อหาว่า k ที่เหมาะสมคือเท่าไร และ compression ratio นั้นดีแค่ไหน

**Dataset**: `sklearn.datasets.load_digits()` — digit images 8×8

**สิ่งที่ต้องทำ:**

**(a)** เลือก digit 5 ตัว (0–4) คำนวณ k_min สำหรับ 95% energy ของแต่ละตัว  
บันทึกผลในตาราง: | Digit | k_min | Storage original | Storage compressed | Ratio |  
**(b)** สร้าง reconstruction quality comparison: ใช้ digit '3' แสดง k=1,2,3,4,5,8  
**(c)** สร้าง graph: x=k, y=RMSE (Root Mean Square Error) ระหว่าง original กับ reconstruction  
สำหรับ digit ทั้ง 5 ตัว บน graph เดียวกัน  
**(d)** อธิบาย: ทำไม RMSE ถึงลดลงเมื่อ k เพิ่ม? และจุดไหนคือ diminishing return?

**สูตร RMSE**: `np.sqrt(np.mean((A - A_k)**2))`

**Deliverable**: Python code + graphs + ตาราง + คำอธิบาย

---

## ข้อ 4 — Open-ended Case Study: PCA ใน Real Application (15 คะแนน)

ในข้อนี้คุณจะเลือก 1 application จริงของ PCA/SVD ใน Data Science และวิเคราะห์อย่างลึก โจทย์นี้ไม่มีคำตอบถูกผิด แต่วัดความสามารถในการคิดวิเคราะห์และเชื่อมโยง concept

**เลือก 1 จาก 3 หัวข้อนี้:**

**Option A — Eigenfaces**  
ศึกษาว่า Face Recognition ด้วย Eigenfaces ทำงานอย่างไร  
ใช้ `sklearn.datasets.fetch_olivetti_faces()` หรือ AT&T dataset  
ทำ PCA, แสดง "eigenfaces" (top-k eigenvectors visualized as images), ทดสอบ reconstruction

**Option B — Recommender System (Matrix Factorization)**  
สร้าง user-item rating matrix แบบง่าย (synthesized หรือ MovieLens 100K)  
ใช้ SVD ทำ matrix factorization, predict ratings, วัด RMSE บน test set

**Option C — Genomics / Text Data**  
ใช้ PCA บน text data (TF-IDF matrix) หรือ gene expression data จาก UCI  
วิเคราะห์ว่า PCA grouping ตรงกับ category จริงหรือไม่

**ทุก Option ต้องมี:**
1. **Scenario**: ปัญหาจริงคืออะไร? ใครใช้?
2. **Data**: dataset อะไร? ขนาดเท่าไร?
3. **Method**: ใช้ PCA/SVD อย่างไร? (พร้อม code)
4. **Result**: ผลลัพธ์เป็นอย่างไร? (ตัวเลข + visualization)
5. **Insight**: สรุปอะไรได้? จะพัฒนาต่อได้อย่างไร?

---

## เกณฑ์การให้คะแนนรวม

| ข้อ | หัวข้อ | คะแนน |
|-----|--------|--------|
| ข้อ 1 | Eigendecomposition ด้วยมือ | 25 |
| ข้อ 2 | PCA from Scratch + sklearn | 35 |
| ข้อ 3 | SVD Compression Analysis | 25 |
| ข้อ 4 | Open-ended Case Study | 15 |
| **รวม** | | **100** |

---

## คำแนะนำในการทำ

- ทุก code cell ต้องมี comment อธิบาย **ทำอะไร** และ **เพื่ออะไร**
- ทุก graph ต้องมี title, axis labels, legend
- ตอบคำถามเชิงวิเคราะห์เป็นภาษาไทยหรืออังกฤษก็ได้ (ขอให้ชัดเจน)
- Run All Cells ก่อนส่ง ห้ามส่ง notebook ที่มี error

---

*Homework 04 | วิชา 1145 201 | กำหนดส่ง: สัปดาห์ที่ 5*
