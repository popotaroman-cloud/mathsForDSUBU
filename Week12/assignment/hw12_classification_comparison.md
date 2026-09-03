# Homework 12: Classification Methods Comparison
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO3 | **LLo**: อธิบายและเปรียบเทียบ LDA, QDA, Naive Bayes และ KNN เลือก classifier ที่เหมาะสมกับปัญหาได้  
**กำหนดส่ง**: สัปดาห์ที่ 13  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์ที่ผ่านมาเราเรียน Logistic Regression ซึ่งเป็น **discriminative classifier** ที่ model Pr(Y|X) โดยตรง สัปดาห์นี้เราขยายไปสู่ **generative classifiers** — LDA, QDA และ Naive Bayes — ซึ่ง model Pr(X|Y) แล้วใช้ Bayes Theorem แปลงกลับ นอกจากนี้เรายังเรียน KNN ซึ่งเป็น non-parametric method ที่ไม่ assume distribution ใดๆ ในการบ้านนี้คุณจะทดสอบทุก method บน dataset จริง เปรียบเทียบ performance ผ่าน confusion matrix, ROC curve, AUC และสรุปว่าทำไม method หนึ่งถึงดีกว่าอีก method ในสถานการณ์ที่กำหนด ทักษะการเลือก classifier ที่ถูกต้องเป็นหัวใจสำคัญของ Machine Learning Engineer ในอุตสาหกรรม

---

## ข้อ 1 — LDA บน Wine Quality Dataset (3 คะแนน)

ในข้อนี้คุณจะใช้ LDA กับ **Wine dataset** จาก sklearn ซึ่งเป็น classic multi-class classification ที่ต้องการจำแนกไวน์ออกเป็น 3 class ตาม chemical properties LDA เหมาะมากกับปัญหานี้เพราะ assumes Gaussian distributions และ shared covariance ซึ่งสอดคล้องกับ chemical measurements

**Dataset**: `sklearn.datasets.load_wine()` — 178 samples, 13 features, 3 classes (wine cultivar)

**สิ่งที่ต้องทำ**:
1. โหลด Wine dataset และแสดง class distribution + feature names
2. Split: test_size=0.3, random_state=42, **stratify=y**
3. Standardize features ด้วย `StandardScaler` (fit บน train เท่านั้น)
4. Fit `LinearDiscriminantAnalysis` และแสดง:
   - Class priors (π̂ₖ)
   - Class means (μ̂ₖ) สำหรับ 2 features แรก
5. สร้าง **confusion matrix** และ classification report
6. Plot: **LDA projection** — transform data ลง 2D ด้วย `lda.transform()` แล้ว scatter plot แยก 3 class ด้วย color

**Expected**: Accuracy ≈ 0.98 (Wine dataset เป็น well-separated)

**Deliverable**: Code + confusion matrix plot + LDA projection scatter + คำอธิบาย 2–3 ประโยค

---

## ข้อ 2 — LDA vs QDA vs Naive Bayes: Assumption Testing (4 คะแนน)

ในข้อนี้คุณจะทดสอบว่า assumption ของแต่ละ method ส่งผลต่อ performance อย่างไร โดยใช้ dataset 2 ชุดที่มีลักษณะต่างกัน เป้าหมายคือ**เข้าใจว่า assumption ที่ถูกต้องหรือไม่ถูกต้องเปลี่ยน accuracy อย่างไร**

**Dataset ที่ 1**: `sklearn.datasets.make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)` — linearly separable

**Dataset ที่ 2**: `sklearn.datasets.make_moons(n_samples=1000, noise=0.2, random_state=42)` — non-linear boundary

**สิ่งที่ต้องทำ**:
1. สำหรับแต่ละ dataset: split 80/20, standardize
2. Fit 4 classifiers: `LogisticRegression`, `LDA`, `QDA`, `GaussianNB`
3. สร้าง **comparison table** (Accuracy, F1, AUC) สำหรับทั้ง 2 datasets:

| Method | Dataset1-Acc | Dataset1-AUC | Dataset2-Acc | Dataset2-AUC |
|--------|-------------|-------------|-------------|-------------|
| Logistic Regression | | | | |
| LDA | | | | |
| QDA | | | | |
| Naive Bayes | | | | |

4. Plot **Decision Boundary** สำหรับ Dataset 2 (make_moons) — ใช้ meshgrid:
   - Plot 4 subplots (2×2) แสดง decision boundary ของแต่ละ method
5. อธิบาย (3–5 ประโยค):
   - ทำไม LDA ถึงทำงานดีบน Dataset 1 แต่ไม่ดีบน Dataset 2?
   - QDA ดีกว่า LDA บน Dataset 2 หรือไม่? เพราะอะไร?
   - Naive Bayes ทำงานอย่างไรเมื่อ features มี correlation?

**Deliverable**: Code + ตาราง + decision boundary plots + คำอธิบาย

---

## ข้อ 3 — KNN + Full Pipeline (3 คะแนน)

ในข้อนี้คุณจะสร้าง full classification pipeline สำหรับ **Heart Disease dataset** ซึ่งเป็น use case ทางการแพทย์ที่ Recall สำคัญมาก (miss heart disease = อันตราย) แล้วเลือก classifier ที่เหมาะสมพร้อมเหตุผล

**Dataset**: 
```python
from sklearn.datasets import fetch_openml
heart = fetch_openml('heart-statlog', version=1, as_frame=True)
# หรือใช้ synthetic ถ้า fetch ไม่ได้:
from sklearn.datasets import make_classification
X_heart, y_heart = make_classification(
    n_samples=270, n_features=13, n_informative=8, 
    n_redundant=2, random_state=99)
```

**สิ่งที่ต้องทำ**:
1. โหลด dataset (ใช้ synthetic ถ้า fetch ไม่ได้) + EDA เบื้องต้น
2. Split: test_size=0.2, random_state=42, standardize
3. Fit **KNN ด้วย K=1,3,5,7,10,15** และ plot F1 vs K
4. เลือก optimal K จากกราฟ
5. Fit ทุก 5 classifiers (LR, LDA, QDA, NB, KNN(best K))
6. สร้าง **full comparison table** (Accuracy, Precision, Recall, F1, AUC)
7. Plot ROC Curve ของทุก 5 classifiers ในกราฟเดียวกัน
8. **Business recommendation** (2–3 ประโยค): classifier ไหนที่คุณจะเลือกสำหรับ heart disease screening? ทำไม? threshold ควรเป็นเท่าไร?

**Deliverable**: Code + F1 vs K plot + comparison table + ROC curve + คำแนะนำ

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว พร้อม output ทุก cell
- ตั้งชื่อ: `hw12_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | LDA บน Wine + projection plot | 3 |
| 2 | Assumption testing + decision boundary | 4 |
| 3 | KNN pipeline + recommendation | 3 |
| **รวม** | | **10** |
