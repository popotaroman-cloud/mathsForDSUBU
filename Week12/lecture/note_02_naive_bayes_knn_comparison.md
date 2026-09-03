# Note 12-2: Naive Bayes, KNN Classifier และการเปรียบเทียบ Classification Methods
> Week 12 | CLO3 | ISLP Reference: Ch.4.4.4, Ch.4.5

---

## บทนำ

ในส่วนที่สองของ Week 12 นี้เราจะเรียนรู้อีกสองวิธีสำคัญใน classification ได้แก่ **Naive Bayes** และ **K-Nearest Neighbors (KNN)** ซึ่งมีแนวคิดที่แตกต่างกันโดยสิ้นเชิง Naive Bayes เป็น generative model เหมือน LDA แต่มีสมมติฐานง่ายกว่า คือสมมติให้ features เป็นอิสระต่อกันภายใน class — แม้จะ "naive" แต่ทำงานได้ดีในทางปฏิบัติโดยเฉพาะกับข้อมูล text ส่วน KNN เป็น non-parametric classifier ที่ไม่สมมติเกี่ยวกับ distribution ของข้อมูลเลย เป้าหมายของ Note นี้คือให้นักศึกษาเข้าใจหลักการ tradeoff ของแต่ละวิธี และสามารถเลือก classifier ที่เหมาะสมกับปัญหาจริงได้ ในงาน Data Science ปัจจุบัน Naive Bayes ยังคงใช้อยู่ใน spam filtering, sentiment analysis และ document classification ส่วน KNN ใช้ใน recommendation systems และ anomaly detection

---

## 12.4 Naive Bayes Classifier  *(ISLP 4.4.4)*

ในส่วนนี้เราจะเรียนรู้ Naive Bayes ซึ่งเป็น generative model ที่ใช้สมมติฐานง่ายมากเพื่อให้คำนวณได้เร็วและทำงานได้ดีแม้กับข้อมูลที่มี features จำนวนมาก

### ปัญหาของ LDA/QDA กับ p ที่ใหญ่

เมื่อ p (จำนวน features) มีขนาดใหญ่ การประมาณ covariance matrix Σ (ขนาด p×p) ต้องการข้อมูลมาก:
- LDA ต้องการประมาณ p(p+1)/2 parameters สำหรับ Σ
- ถ้า p = 1000 (เช่น ข้อมูล text) → ต้องประมาณ 500,500 parameters!

### Naive Independence Assumption

Naive Bayes แก้ปัญหาด้วยการสมมติว่า features **เป็นอิสระต่อกัน** ภายในแต่ละ class:

$$f_k(x) = \prod_{j=1}^{p} f_{kj}(x_j)$$

โดย fₖⱼ คือ marginal density ของ Xⱼ ใน class k

### Classifier Formula

นำสมมติฐานนี้ใส่ใน Bayes Theorem:

$$\Pr(Y=k|X=x) \propto \pi_k \cdot \prod_{j=1}^{p} f_{kj}(x_j)$$

หรือในรูป log:

$$\log \Pr(Y=k|X) \propto \log\pi_k + \sum_{j=1}^{p} \log f_{kj}(x_j)$$

### Variants ของ Naive Bayes

| Variant | สมมติฐาน fₖⱼ | ใช้กับข้อมูล |
|---------|-------------|------------|
| **Gaussian NB** | Xⱼ | Y=k ~ N(μₖⱼ, σ²ₖⱼ) | ข้อมูล continuous ทั่วไป |
| **Bernoulli NB** | Xⱼ | Y=k ~ Bernoulli(pₖⱼ) | binary features (word present/absent) |
| **Multinomial NB** | Xⱼ | Y=k ~ Multinomial | word counts ใน document |
| **Kernel NB** | KDE (non-parametric) | ข้อมูล continuous ที่ไม่ Gaussian |

### ตัวอย่างเชิงตัวเลข: Gaussian Naive Bayes

**ปัญหา**: จำแนก email เป็น spam/not-spam ด้วย 3 features:
- X₁: จำนวนคำว่า "free" ในอีเมล
- X₂: จำนวน link
- X₃: ขนาด attachment (MB)

**Training data** (spam class):
- μ₁,spam = 5.2, σ₁,spam = 1.8
- μ₂,spam = 8.1, σ₂,spam = 2.5
- μ₃,spam = 2.3, σ₃,spam = 1.2

**Classify email ใหม่**: x = [6, 7, 1]

$$\log\Pr(\text{spam}|x) \propto \log(0.3) + \log f_{\text{spam},1}(6) + \log f_{\text{spam},2}(7) + \log f_{\text{spam},3}(1)$$

คำนวณแต่ละ term แล้วเปรียบเทียบกับ class "not-spam"

### ทำไม "Naive" แต่ทำงานได้ดี?

แม้สมมติฐาน independence จะไม่เป็นจริงในทางปฏิบัติ แต่:
1. ถ้า correlation ระหว่าง features มีน้อย → approximation ดีพอ
2. Naive Bayes ต้องการข้อมูลน้อย → ทำงานดีกับ p ใหญ่
3. **Bias-Variance**: NB มี bias สูง (เพราะ assumption เข้มงวด) แต่ variance ต่ำมาก
4. ในทางปฏิบัติ: variance ต่ำมักสำคัญกว่า bias เล็กน้อย

### เชื่อมกับ Week 6 (Bayes Theorem)

โสตรของ Naive Bayes คือ Bayes Theorem ตรงๆ: Prior × Likelihood → Posterior
- **Prior** πₖ: ก่อนเห็น x เราเชื่อว่า class k มีความน่าจะเป็นเท่าไร
- **Likelihood** fₖ(x): ถ้าข้อมูลเป็น class k ควรเห็น x แบบนี้แค่ไหน
- **Posterior** Pr(Y=k|X=x): หลังเห็น x แล้วเชื่อว่าเป็น class k แค่ไหน

```python
# ─── Gaussian Naive Bayes ────────────────────────────────────────────
# วัตถุประสงค์: ฝึก Gaussian NB สมมติ Gaussian likelihood ต่อ feature
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ISLP import load_data

# โหลด Default dataset
Default = load_data('Default')
Default['student_num'] = (Default['student'] == 'Yes').astype(int)
X = Default[['balance', 'income', 'student_num']]
y = Default['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Fit Gaussian NB — ไม่ต้องการ scaling!
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ดู learned parameters
print("Class priors:", gnb.class_prior_)         # π̂ₖ
print("Class means:\n", gnb.theta_)              # μ̂ₖⱼ (mean ต่อ feature ต่อ class)
print("Class variances:\n", gnb.var_)            # σ̂²ₖⱼ

y_pred_gnb = gnb.predict(X_test)
print("GNB Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))
```

---

## 12.5 K-Nearest Neighbors Classifier  *(ISLP 2.2.3 + 4.7.6)*

ในส่วนนี้เราจะเรียนรู้ KNN ซึ่งเป็น non-parametric classifier ที่ไม่มีสมมติฐานใดๆ เกี่ยวกับ distribution ของข้อมูล ทำให้ยืดหยุ่นมากที่สุดในบรรดา classifiers ทั้งหมดที่เรียนมา

### หลักการ KNN

สำหรับ test point x₀:
1. หา K neighbors ที่ใกล้ที่สุดใน training set: 𝒩₀ = {K nearest points}
2. ประมาณความน่าจะเป็น posterior:

$$\Pr(Y=j | X=x_0) = \frac{1}{K} \sum_{i \in \mathcal{N}_0} \mathbf{1}(y_i = j)$$

3. จัดประเภทไปยัง class ที่มี majority vote สูงสุด

### ผลของ K ต่อ Bias-Variance

| K | Bias | Variance | Decision Boundary | เหมาะกับ |
|---|------|----------|------------------|---------|
| K=1 | ต่ำมาก | สูงมาก | ซับซ้อนมาก | ข้อมูลที่สะอาดมาก |
| K=5 | ปานกลาง | ปานกลาง | ค่อนข้าง smooth | ส่วนใหญ่ |
| K=10 | ปานกลาง | ปานกลาง | smooth | ทั่วไป |
| K=100 | สูง | ต่ำ | เรียบมาก | n ใหญ่มาก |
| K=n | สูงมาก | 0 | constant | predict majority always |

**Golden rule**: เลือก K ด้วย cross-validation (Week 14)!

### ตัวอย่างเชิงตัวเลข: KNN กับ 2D Data

สมมติข้อมูล 5 จุด:
- (1,1) → Class A
- (1,2) → Class A
- (2,1) → Class A
- (4,4) → Class B
- (4,5) → Class B

จะจัดประเภท x₀ = (2,3)?

**K=1**: ใกล้สุดคือ (1,2) distance = √(1+1) = 1.41 → Class A
**K=3**: ใกล้สุด 3 จุดคือ (1,2), (1,1), (2,1) → 3 Class A → Class A
**K=5**: ทุกจุด → 3A, 2B → Class A

### Distance Metric ที่ใช้

- **Euclidean** (default): d(x,y) = √Σ(xⱼ−yⱼ)²
- **Manhattan**: d(x,y) = Σ|xⱼ−yⱼ|
- **Minkowski**: d(x,y) = (Σ|xⱼ−yⱼ|^p)^(1/p)

**สำคัญ**: KNN ไวต่อ scale ของ features มาก → ต้อง StandardScaler ก่อนเสมอ!

```python
# ─── KNN Classifier หลาย K ──────────────────────────────────────────
# วัตถุประสงค์: เปรียบเทียบ KNN ที่ K ต่างกัน เพื่อเห็น bias-variance trade-off
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# ต้อง scale ก่อน KNN เสมอ! (distance-based)
k_values = [1, 3, 5, 10, 20, 50]
results = {}

for k in k_values:
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # zero mean, unit variance
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    results[k] = acc
    print(f"K={k:3d}: Accuracy = {acc:.4f}")

# K ที่ดีที่สุด
best_k = max(results, key=results.get)
print(f"\nBest K = {best_k} (Accuracy = {results[best_k]:.4f})")
```

---

## 12.6 การเปรียบเทียบ Classification Methods  *(ISLP 4.5)*

ในส่วนนี้เราจะสรุปและเปรียบเทียบ classifier ทั้งหมดที่เรียนมา เพื่อสร้าง framework การเลือก classifier ที่เหมาะสมกับปัญหาจริง

### ตาราง Comparison

| Method | Assumption | Decision Boundary | Parameters | ดีเมื่อ |
|--------|-----------|------------------|------------|---------|
| **Logistic Regression** | none (discriminative) | Linear | p+1 | n ใหญ่, features ต่าง scale |
| **LDA** | Gaussian, shared Σ | Linear | p×K + p(p+1)/2 | n เล็ก, classes well-separated |
| **QDA** | Gaussian, per-class Σₖ | Quadratic | p×K + K×p(p+1)/2 | n ใหญ่, Σ ต่างกันมาก |
| **Naive Bayes** | Independent features | Flexible | K×2p (Gaussian) | p ใหญ่, text/NLP, n เล็ก |
| **KNN** | none (non-parametric) | Non-linear | 0 (memory-based) | boundary ซับซ้อน, n ใหญ่ |

### เมื่อไรควรใช้อะไร?

**ใช้ LDA เมื่อ:**
- n ค่อนข้างเล็ก
- Features มี distribution ใกล้ Gaussian
- ต้องการ interpretability (linear coefficients)
- Covariances ของแต่ละ class ไม่ต่างกันมาก

**ใช้ QDA เมื่อ:**
- n ค่อนข้างใหญ่
- Covariances ต่างกันอย่างชัดเจน
- ต้องการ boundary ที่โค้งได้

**ใช้ Naive Bayes เมื่อ:**
- p (จำนวน features) ใหญ่มาก
- Features เป็น independent จริงๆ (เช่น ข้อมูล text)
- ต้องการ speed ในการ train/predict

**ใช้ KNN เมื่อ:**
- n ใหญ่มาก
- Decision boundary ซับซ้อนมาก
- ไม่รู้ distribution ของข้อมูลล่วงหน้า
- ยอมรับ computation cost ใน prediction ได้

### ความสัมพันธ์ระหว่าง Logistic Regression และ LDA

น่าแปลกใจที่ Logistic Regression และ LDA ให้ **linear classifier** เหมือนกัน แต่ต่างกันที่:
- **LDA**: ประมาณ parameters จาก class-conditional distribution (Gaussian assumption)
- **Logistic Regression**: ประมาณ parameters โดยตรงจาก likelihood ของ P(Y|X)

ผลลัพธ์มักใกล้เคียงกัน แต่ LDA ดีกว่าเมื่อ Gaussian assumption จริง, Logistic ดีกว่าเมื่อไม่ใช่

```python
# ─── เปรียบเทียบ Classifiers ทั้งหมดบน Smarket ──────────────────────
# วัตถุประสงค์: วัด accuracy ของ 5 methods บน test set เดียวกัน
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from ISLP import load_data
import pandas as pd

# เตรียมข้อมูล Smarket
Smarket = load_data('Smarket')
features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
train_mask = Smarket['Year'] < 2005

X_train = Smarket.loc[train_mask, features]
X_test  = Smarket.loc[~train_mask, features]
y_train = Smarket.loc[train_mask, 'Direction']
y_test  = Smarket.loc[~train_mask, 'Direction']

# ─── Define Classifiers ──────────────────────────────────────────────
# วัตถุประสงค์: สร้าง dict ของ classifiers ทั้ง 5 วิธีพร้อม pipeline
classifiers = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42))
    ]),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB(),
    'KNN (K=5)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

# ─── Train + Evaluate ────────────────────────────────────────────────
# วัตถุประสงค์: วัด accuracy ของแต่ละ classifier บน test set
comparison = []
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    comparison.append({'Method': name, 'Test Accuracy': acc})

results_df = pd.DataFrame(comparison).sort_values('Test Accuracy', ascending=False)
print(results_df.to_string(index=False))
```

---

## สรุป

| แนวคิด | สูตร | Python |
|--------|------|--------|
| Naive Bayes | Pr(Y=k|X) ∝ πₖ × Πⱼ fₖⱼ(xⱼ) | `GaussianNB()` |
| KNN Posterior | Pr(Y=j|X=x₀) = (1/K) Σᵢ∈𝒩₀ I(yᵢ=j) | `KNeighborsClassifier(n_neighbors=K)` |
| Feature Scaling | (x − μ)/σ | `StandardScaler()` |
| Pipeline | scaler → classifier | `Pipeline([...])` |
| Cross-validation for K | CV error vs K curve | `cross_val_score(...)` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 12-Note 1**: LDA (shared Σ) และ QDA (per-class Σₖ) — Naive Bayes ผ่อนคลายสมมติฐานต่อไปอีก
- ← **Week 6 (Bayes Theorem)**: Naive Bayes ใช้ prior × likelihood → posterior ตรงๆ
- → **Week 13**: ROC Curve — วิธีประเมิน classifier ที่ดีกว่า accuracy เดียว
- → **Week 14**: Cross-Validation — วิธีเลือก K ที่ดีที่สุดสำหรับ KNN และเปรียบเทียบ classifiers อย่างยุติธรรม
