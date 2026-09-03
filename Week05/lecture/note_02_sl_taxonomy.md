# Note 2: อนุกรมวิธานของ Statistical Learning — Parametric/Non-parametric, Supervised/Unsupervised
> Week 5 | CLO2 | ISLP Reference: Ch.2.1.2–2.1.5

## บทนำ

สัปดาห์นี้เราจะสร้าง "แผนที่" ของ Statistical Learning เพื่อให้นักศึกษารู้ว่ามี algorithm ประเภทใดบ้าง และแต่ละประเภทเหมาะกับโจทย์แบบไหน เป้าหมายคือสามารถอ่านโจทย์จริงแล้วบอกได้ว่าควรใช้ Supervised หรือ Unsupervised, Regression หรือ Classification, Parametric หรือ Non-parametric ซึ่งตรงกับ LLo ของสัปดาห์นี้ ความสามารถนี้สำคัญมากใน Data Science จริงเพราะ "เลือก method ผิด → ผลผิด" เสมอ ก่อนเขียน code ต้องคิดให้ชัดก่อนว่าโจทย์ตรงกับ framework ไหน

---

## 5.2 How Do We Estimate f? — Parametric vs Non-parametric  *(ISLP 2.1.2)*

ในส่วนนี้เราจะเปรียบเทียบสองแนวทางหลักในการประมาณ f เพื่อให้นักศึกษาเลือกแนวทางที่เหมาะสมกับข้อมูลและโจทย์

### Parametric Approach (วิธีเชิงพารามิเตอร์)

**ขั้นตอน**:
1. **สมมติรูปแบบ** ของ f ก่อน เช่น สมมติว่า f เป็น linear:
$$f(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$$
2. **ประมาณพารามิเตอร์** $\beta_0, \beta_1, \ldots, \beta_p$ จากข้อมูล training

**ข้อดี**:
- ตีความได้ง่าย (β บอกทิศทางและขนาดของความสัมพันธ์)
- ใช้ข้อมูลน้อยกว่า
- คำนวณเร็ว

**ข้อเสีย**:
- ถ้าสมมติฐานผิด (f ไม่ linear จริง) → Bias สูง
- ความยืดหยุ่นจำกัด

**ตัวอย่างตัวเลข**: สมมติ $\beta_0 = 2.9$, $\beta_1 = 0.046$, $\beta_2 = 0.189$

$$\text{Sales} = 2.9 + 0.046 \cdot \text{TV} + 0.189 \cdot \text{Radio}$$

การแทนค่า TV=230.1, Radio=37.8:
$$\hat{\text{Sales}} = 2.9 + 0.046(230.1) + 0.189(37.8) = 2.9 + 10.58 + 7.14 = 20.62 \text{ (จริง: 22.1)}$$

Error = |22.1 - 20.62| = 1.48 พันหน่วย

### Non-parametric Approach (วิธีไม่เชิงพารามิเตอร์)

**แนวคิด**: ไม่สมมติรูปแบบของ f ล่วงหน้า แต่ให้ data "บอก" ว่า f ควรเป็นอย่างไร

ตัวอย่าง: **K-Nearest Neighbors (KNN)** สำหรับ regression
$$\hat{f}(x_0) = \frac{1}{K} \sum_{i \in \mathcal{N}_0} y_i$$
โดย $\mathcal{N}_0$ คือกลุ่ม K จุดที่ใกล้ $x_0$ ที่สุด

**ข้อดี**: ยืดหยุ่น, จับ pattern ซับซ้อนได้

**ข้อเสีย**: ต้องการข้อมูลมาก, interpretability ต่ำ, computationally expensive

**ตัวอย่างตัวเลข**: K=3, สำหรับ point ใหม่ X=5.0
- 3 เพื่อนบ้านใกล้สุด: y = {12.3, 11.8, 13.1}
- $\hat{f}(5.0) = (12.3 + 11.8 + 13.1)/3 = 12.4$

**DS Connection**: Linear Regression เป็น parametric ที่ใช้บ่อยที่สุดเพราะ interpretable ส่วน KNN, Random Forest, Neural Network เป็น non-parametric ที่ flexible กว่าแต่ตีความยากกว่า

---

## 5.3 Prediction Accuracy vs Model Interpretability  *(ISLP 2.1.3)*

ในส่วนนี้เราจะดู spectrum ของ methods เพื่อให้นักศึกษาเลือก level of flexibility ที่เหมาะกับงาน

### Flexibility Spectrum

```
Low Flexibility                                    High Flexibility
(High Interpretability)                       (Low Interpretability)
     │                                                    │
  Linear        Ridge/Lasso    GAMs     Trees    Neural
Regression    Regression                         Networks
     │              │            │        │          │
  ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → 
```

**เมื่อไรควร Interpretable?**
- ปัญหา Inference: ต้องการรู้ว่า feature ไหนสำคัญ
- การตัดสินใจทางการแพทย์/กฎหมาย ต้องอธิบายได้
- ข้อมูลน้อย

**เมื่อไรควร Flexible?**
- ปัญหา Prediction ล้วน ๆ (ไม่ต้องอธิบาย)
- ข้อมูลมาก, pattern ซับซ้อน
- เช่น: image recognition, speech recognition

**DS Connection**: ใน industry ของไทย งาน regulatory compliance (ธนาคาร, ประกัน) มักต้องการ interpretable model ส่วนงาน tech startup ที่ต้องการ accuracy สูง อาจใช้ neural network ได้

---

## 5.4 Supervised vs Unsupervised Learning  *(ISLP 2.1.4)*

ในส่วนนี้เราจะแบ่งประเภท learning ตามว่ามี label Y หรือไม่ เพื่อให้นักศึกษาวิเคราะห์โจทย์ได้ถูกต้อง

### Supervised Learning (การเรียนรู้แบบมีผู้สอน)

มีคู่ข้อมูล $(x_i, y_i)$ สำหรับ $i = 1, 2, \ldots, n$

**เป้าหมาย**: เรียนรู้ฟังก์ชัน $\hat{f}$ จาก training data เพื่อ predict Y สำหรับ input X ใหม่

**ตัวอย่าง**:
| Input (X) | Output (Y) | Task |
|-----------|-----------|------|
| คุณสมบัติบ้าน | ราคาบ้าน | Regression |
| รูปภาพ | cat/dog | Classification |
| Email features | spam/ham | Classification |
| ข้อมูล stock | ราคาพรุ่งนี้ | Regression |

### Unsupervised Learning (การเรียนรู้แบบไม่มีผู้สอน)

มีเพียง $x_i$ — **ไม่มี Y** หรือ label

**เป้าหมาย**: ค้นหาโครงสร้างหรือ pattern ที่ซ่อนอยู่ใน X

**ตัวอย่างวิธี**:
- **Clustering** (K-Means, DBSCAN): จัดกลุ่มข้อมูลที่คล้ายกัน
- **Dimensionality Reduction** (PCA — Week 4): ลดมิติข้อมูล
- **Density Estimation**: ประมาณ distribution ของข้อมูล

**ตัวอย่างจริง**: ร้านค้าออนไลน์ใช้ clustering จัดกลุ่มลูกค้าโดยไม่มี label — พบว่ามี 3 กลุ่ม: ลูกค้าซื้อบ่อย/มูลค่าสูง, ลูกค้าซื้อนาน ๆ ครั้ง, ลูกค้าที่ดูแต่ไม่ซื้อ

**DS Connection**: ใน Data Science จริง งาน Unsupervised มักเป็น step แรกก่อนทำ Supervised เช่น Cluster ลูกค้าก่อน แล้วสร้าง classifier แยกกันสำหรับแต่ละกลุ่ม

---

## 5.5 Regression vs Classification  *(ISLP 2.1.5)*

ในส่วนนี้เราจะแยกแยะสองประเภทของ Supervised Learning ตามลักษณะของ Y เพื่อเลือก algorithm ที่ถูกต้อง

### Regression (ค่าต่อเนื่อง)

$Y$ เป็น **quantitative** (ตัวเลขต่อเนื่อง)

$$Y \in \mathbb{R} \quad \text{เช่น: } \{100, 250.5, -30, 1000000\}$$

**ตัวอย่าง**: ราคาบ้าน, อุณหภูมิ, รายได้, คะแนนสอบ

**Methods**: Linear Regression, Ridge, Lasso, Polynomial Regression, KNN Regression

### Classification (ค่าจัดประเภท)

$Y$ เป็น **qualitative** (หมวดหมู่)

$$Y \in \{C_1, C_2, \ldots, C_K\} \quad \text{เช่น: } \{\text{spam, ham}\} \text{ หรือ } \{\text{cat, dog, bird}\}$$

**Binary Classification**: $K = 2$ เช่น spam/ham, healthy/sick

**Multi-class Classification**: $K > 2$ เช่น digit recognition (0–9), cancer type

**Methods**: Logistic Regression, KNN, Decision Trees, SVM, Neural Networks

### Quick Test: เป็น Regression หรือ Classification?

| โจทย์ | Y | ประเภท |
|-------|---|--------|
| พยากรณ์ราคาหุ้น SET พรุ่งนี้ | ราคา (บาท) | Regression |
| วินิจฉัยโรคมะเร็ง | มี/ไม่มี | Classification |
| พยากรณ์ GDP ไทยปี 2570 | GDP (พัน ล้านบาท) | Regression |
| วิเคราะห์ sentiment ความคิดเห็น | Positive/Negative/Neutral | Classification |
| ทำนายคะแนน ONET นักเรียน | คะแนน (0–100) | Regression |

**DS Connection**: ขอบเขตระหว่าง Regression และ Classification ไม่ได้ชัดเสมอไป เช่น พยากรณ์ว่าหุ้นจะขึ้นหรือลง (Classification) เทียบกับพยากรณ์ % การเปลี่ยนแปลง (Regression) — การเลือกส่งผลต่อ metric และ algorithm ที่ใช้

---

## สรุป (Summary)

| แนวคิด | นิยาม | ตัวอย่าง Python |
|--------|-------|----------------|
| Parametric | สมมติรูป f, fit parameters | `LinearRegression()` |
| Non-parametric | ไม่สมมติรูป f | `KNeighborsRegressor()` |
| Supervised | มีคู่ (X, Y) | `model.fit(X, y)` |
| Unsupervised | มีแค่ X | `KMeans().fit(X)` |
| Regression | Y ต่อเนื่อง | `LinearRegression()` |
| Classification | Y จัดประเภท | `LogisticRegression()` |

```python
# ─── Taxonomy Demo: แสดง 4 ประเภทของ SL ─────────────────────────
# วัตถุประสงค์: เห็นภาพรวมว่า sklearn รองรับทุก category อย่างไร
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Data ตัวอย่าง (2D for clarity)
np.random.seed(0)
X = np.random.randn(50, 2)
y_reg = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(50)  # continuous Y
y_cls = (y_reg > 0).astype(int)                        # binary Y

# Supervised Regression (Parametric)
lin_reg = LinearRegression().fit(X, y_reg)
print(f"Linear Regression β: {lin_reg.coef_}")  # interpretable

# Supervised Regression (Non-parametric)
knn_reg = KNeighborsRegressor(n_neighbors=5).fit(X, y_reg)
print(f"KNN Regression prediction: {knn_reg.predict([[0, 0]])}")

# Supervised Classification
log_cls = LogisticRegression().fit(X, y_cls)
print(f"Logistic prediction: {log_cls.predict([[1, -1]])}")

# Unsupervised (ไม่มี y)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print(f"Cluster labels: {kmeans.labels_[:10]}")
```

---

## เชื่อมกับสัปดาห์อื่น

- ← Week 1–4: Matrix operations, eigenvalues — เป็นพื้นฐาน math ของทุก method ในแผนที่นี้
- → Week 6: เมื่อเลือก method แล้ว จะประเมิน accuracy ด้วย MSE และ Bias-Variance ได้อย่างไร?
- → Week 9: Linear Regression — parametric supervised regression method แรกที่เรียนเต็มรูปแบบ
- → Week 12: Logistic Regression — parametric supervised classification
- → Week 13: Naive Bayes — probabilistic classification (Bayes theorem จาก Week 6)
