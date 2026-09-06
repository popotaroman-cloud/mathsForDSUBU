# Note 12-1: Linear Discriminant Analysis (LDA) และ Quadratic Discriminant Analysis (QDA)
> Week 12 | CLO3 | ISLP Reference: Ch.4.4.1–4.4.3

---

## บทนำ

สัปดาห์นี้เราจะเรียนรู้เกี่ยวกับ **Generative Models** สำหรับการจำแนกประเภท (Classification) โดยเฉพาะ LDA และ QDA ซึ่งเป็นวิธีที่ใช้กันอย่างแพร่หลายใน Data Science เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถอธิบายหลักการทำงานของ LDA และ QDA ได้ รวมถึงเลือกใช้ได้อย่างเหมาะสม ในทางปฏิบัติ Generative Models มีข้อได้เปรียบเมื่อข้อมูลมีจำนวนน้อยหรือเมื่อ classes แยกกันได้ชัดเจน ตัวอย่างการใช้งานจริงในทาง Data Science ได้แก่ การพยากรณ์ทิศทางราคาหุ้น (ขึ้น/ลง) การวินิจฉัยโรคจากอาการ และการตรวจจับ spam email LDA และ QDA มีพื้นฐานมาจาก Bayes Theorem: ใช้ความรู้เกี่ยวกับ distribution ของ features ภายในแต่ละ class เพื่อคำนวณความน่าจะเป็น posterior แล้วจัดประเภทตามความน่าจะเป็นสูงสุด การเข้าใจ LDA ยังช่วยให้เข้าใจ covariance matrix และ eigendecomposition ที่เรียนไปแล้วใน Week 4 อีกด้วย

---

## 12.1 Generative vs Discriminative Models  *(ISLP 4.4 Intro)*

ในส่วนนี้เราจะเปรียบเทียบสองแนวทางหลักในการสร้าง classifier เพื่อให้เข้าใจว่า LDA แตกต่างจาก Logistic Regression อย่างไร

**Discriminative Models** เช่น Logistic Regression สร้างโมเดล Pr(Y|X) โดยตรง คือเรียนรู้ boundary ระหว่าง classes โดยตรงจากข้อมูล

**Generative Models** เช่น LDA, QDA, Naive Bayes ใช้แนวทางต่างออกไป:
1. สร้างโมเดล Pr(X|Y=k) — distribution ของ features ภายในแต่ละ class k
2. ประมาณ prior Pr(Y=k) = πₖ
3. ใช้ **Bayes Theorem** เพื่อคำนวณ posterior:

$$\Pr(Y=k | X=x) = \frac{\Pr(X=x | Y=k) \cdot \Pr(Y=k)}{\Pr(X=x)} = \frac{f_k(x) \cdot \pi_k}{\sum_{l=1}^{K} f_l(x) \cdot \pi_l}$$

โดยที่ fₖ(x) = Pr(X=x|Y=k) คือ **likelihood** ของการสังเกต x ภายใต้ class k

**ข้อได้เปรียบของ Generative Models:**
- Stable เมื่อ classes แยกกันได้ชัดเจน (Logistic Regression อาจมีปัญหา)
- ทำงานดีกว่าเมื่อ n มีขนาดเล็ก
- สามารถสร้างข้อมูล synthetic ได้ (เพราะรู้ distribution)

---

## 12.2 Linear Discriminant Analysis (LDA) สำหรับ p = 1  *(ISLP 4.4.1)*

ในส่วนนี้เราจะเริ่มจาก LDA กรณีที่มี feature เดียว (p=1) เพื่อให้เข้าใจ concept ก่อน แล้วจึงขยายไปยังกรณีทั่วไป

### สมมติฐาน
LDA สมมติว่า feature X ภายในแต่ละ class k มี distribution เป็น **Gaussian**:

$$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right)$$

โดย **ใช้ σ² ร่วมกัน** ทุก class (shared variance)

### Discriminant Function

แทนสูตรเข้าใน Bayes Theorem และเอา log จะได้ **log-posterior**:

$$\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$$

จัดประเภท x ไปยัง class k ที่ทำให้ δₖ(x) มีค่ามากที่สุด

### Decision Boundary (กรณี K=2, π₁=π₂)

เมื่อ δ₁(x) = δ₂(x):

$$x = \frac{\mu_1 + \mu_2}{2}$$

นั่นคือ boundary อยู่ที่ **จุดกึ่งกลางของค่าเฉลี่ยทั้งสอง class** — เป็น linear boundary!

### ตัวอย่างเชิงตัวเลข

สมมติ: Class 1 (Down): μ₁ = -0.3, Class 2 (Up): μ₂ = 0.4, σ² = 1.0, π₁ = π₂ = 0.5

Decision boundary = (-0.3 + 0.4) / 2 = **0.05**

- ถ้า x < 0.05 → จัดเป็น "Down"
- ถ้า x > 0.05 → จัดเป็น "Up"

### การประมาณค่า Parameters

จากข้อมูล training:

$$\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i \quad \text{(ค่าเฉลี่ยของ class k)}$$

$$\hat{\sigma}^2 = \frac{1}{n-K} \sum_{k=1}^{K} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)^2 \quad \text{(pooled variance)}$$

$$\hat{\pi}_k = \frac{n_k}{n} \quad \text{(สัดส่วนของ class k)}$$

---

## 12.3 LDA สำหรับ p > 1  *(ISLP 4.4.2)*

ในส่วนนี้เราจะขยาย LDA ไปยังกรณีที่มีหลาย features เพื่อให้ใช้งานได้จริงกับปัญหา Data Science

### สมมติฐาน (Multivariate)
$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_k)^T \Sigma^{-1} (x-\mu_k)\right)$$

โดย **ใช้ covariance matrix Σ ร่วมกัน** ทุก class — นี่คือ "Linear" ใน LDA

### Discriminant Function (p > 1)

$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)$$

จัดประเภทไปยัง class k ที่ทำให้ δₖ(x) มีค่ามากที่สุด

**สังเกต**: สูตรนี้เป็น **linear** ใน x → decision boundary เป็น hyperplane ตรง!

### การประมาณค่า Parameters

$$\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i \quad \text{(vector mean ของ class k)}$$

$$\hat{\Sigma} = \frac{1}{n-K} \sum_{k=1}^{K} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T \quad \text{(pooled covariance)}$$

$$\hat{\pi}_k = \frac{n_k}{n}$$

### เชื่อมกับ Week 4 (Covariance Matrix)

Σ̂ คือ covariance matrix ที่เราเรียนใน Week 4! การทำ LDA ต้องคำนวณ Σ⁻¹ (matrix inverse) และ xᵀΣ⁻¹μₖ (quadratic form) ซึ่งเป็นแนวคิดเดียวกับ eigendecomposition ที่ใช้ใน PCA

### Confusion Matrix และการปรับ Threshold

LDA ใช้ threshold เริ่มต้นที่ Pr(Y=k|X) > 0.5 (majority vote) แต่สามารถปรับได้

**ตัวอย่างจริง (ISLP Table 4.4)**: fit LDA บน Default dataset ทั้งหมด (`default ~ balance + student`) แล้วดู confusion matrix บน training data (n=10,000):

| | True: No | True: Yes | Total |
|--|--|--|--|
| **Predicted No** | 9644 (TN) | 252 (FN) | 9896 |
| **Predicted Yes** | 23 (FP) | 81 (TP) | 104 |
| **Total** | 9667 | 333 | 10000 |

**Training error rate** = (23+252)/10000 = **2.75%** — ดูต่ำมาก! แต่:
- **Sensitivity** (recall ของ class Yes) = 81/333 = **24.3%** — LDA จับ default ได้แค่ 1 ใน 4 เท่านั้น!
- **Specificity** = 1 − 23/9667 = **99.8%** — จับ non-default ได้เกือบสมบูรณ์แบบ

**ทำไม accuracy สูงแต่ sensitivity ต่ำมาก?** เพราะ Bayes classifier (และ LDA ที่ approximate มัน) ใช้ threshold 0.5 ซึ่ง minimize **total error rate รวม** โดยไม่สนว่า error เกิดจาก class ไหน — เมื่อ class Yes มีแค่ 3.33% ของข้อมูล การทำนาย "No" เกือบตลอดก็ยังให้ accuracy สูงอยู่ดี (null classifier ที่ทาย "No" เสมอก็ได้ error แค่ 3.33% อยู่แล้ว!) แต่สำหรับธนาคารที่อยากจับลูกค้าที่จะ default ให้ได้มากที่สุด ผลลัพธ์นี้ใช้งานไม่ได้เลย

**ปรับ threshold เป็น 0.2** (ISLP Table 4.5) — ทำนาย default ถ้า Pr(Yes|X) > 0.2 แทน 0.5:

| | True: No | True: Yes | Total |
|--|--|--|--|
| **Predicted No** | 9432 (TN) | 138 (FN) | 9570 |
| **Predicted Yes** | 235 (FP) | 195 (TP) | 430 |
| **Total** | 9667 | 333 | 10000 |

ตอนนี้ sensitivity = 195/333 = **58.6%** (ดีขึ้นมาก จาก 24.3%!) แต่ overall error rate เพิ่มเป็น (235+138)/10000 = **3.73%** (จาก 2.75%) — **trade-off ที่ชัดเจน**: ยอมรับ error รวมสูงขึ้นนิดหน่อย เพื่อจับ default ได้มากขึ้นเยอะ ธนาคารต้องตัดสินใจเลือก threshold จาก **cost จริงของแต่ละ error type** ไม่ใช่แค่ดู overall accuracy

- ถ้าต้องการ recall สูง (เช่น การตรวจโรค, credit risk): ลด threshold → จับ True Positive มากขึ้น แลกกับ False Positive ที่มากขึ้น
- ถ้าต้องการ precision สูง: เพิ่ม threshold
- การเลือก threshold ที่ดีที่สุดในทุกค่า threshold พร้อมกัน ดูได้จาก **ROC Curve** (Week 13)

---

## 12.4 Quadratic Discriminant Analysis (QDA)  *(ISLP 4.4.3)*

ในส่วนนี้เราจะผ่อนคลายสมมติฐานของ LDA โดยให้แต่ละ class มี covariance matrix เป็นของตัวเอง เพื่อให้โมเดลยืดหยุ่นมากขึ้น

### สมมติฐาน QDA

แต่ละ class k มี covariance matrix **Σₖ เป็นของตัวเอง**:

$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k)\right)$$

### Discriminant Function (QDA)

$$\delta_k(x) = -\frac{1}{2} x^T \Sigma_k^{-1} x + x^T \Sigma_k^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma_k^{-1} \mu_k - \frac{1}{2} \log|\Sigma_k| + \log(\pi_k)$$

สังเกต term แรก: **-½ xᵀΣₖ⁻¹x** เป็น quadratic ใน x → decision boundary เป็น **เส้นโค้ง** (parabola, ellipse, hyperbola)

### LDA vs QDA: Bias-Variance Trade-off

| | LDA | QDA |
|---|---|---|
| **สมมติฐาน** | Shared Σ | Per-class Σₖ |
| **Decision Boundary** | Linear | Quadratic |
| **Parameters** | p×K + p(p+1)/2 | p×K + K×p(p+1)/2 |
| **Bias** | สูงกว่า | ต่ำกว่า |
| **Variance** | ต่ำกว่า | สูงกว่า |
| **ดีกว่าเมื่อ** | n เล็ก, Σ ไม่ต่างกัน | n ใหญ่, Σ ต่างกันมาก |

**หลักการเลือก**: ถ้าสมมติฐาน shared Σ ใกล้เคียงความจริง → LDA ดีกว่า (lower variance ชนะ) แต่ถ้า covariances ต่างกันมาก → QDA จำเป็น

### ตัวอย่างเชิงตัวเลข: LDA vs QDA

สมมติข้อมูลหุ้น 2 features: [Lag1, Lag2]
- Class "Down": μ₁ = [-0.4, -0.3], Σ₁ = [[1.0, 0.2], [0.2, 1.0]]
- Class "Up": μ₂ = [0.3, 0.4], Σ₂ = [[1.2, -0.1], [-0.1, 0.8]]

**LDA** ใช้ Σ̂ (pooled) → linear boundary
**QDA** ใช้ Σ̂₁ และ Σ̂₂ แยกกัน → quadratic boundary

---

## Case Study: การพยากรณ์ทิศทางหุ้น S&P500

**Scenario**: ธนาคารต้องการระบบ AI พยากรณ์ว่าดัชนี S&P500 จะปิดสูงขึ้น (Up) หรือต่ำลง (Down) จากวันก่อนหน้า เพื่อช่วยตัดสินใจลงทุน

**Data**: Smarket dataset จาก ISLP — ข้อมูลรายวัน S&P500 ปี 2001–2005
- Features: Lag1, Lag2, Lag3, Lag4, Lag5 (% return ย้อนหลัง 1–5 วัน), Volume
- Response: Direction (Up/Down)
- จำนวน: 1,250 observations

**Method**: เปรียบเทียบ LDA vs QDA บน train (2001–2004) / test (2005)

**Result**:
- LDA accuracy บน test set: 56.0%
- QDA accuracy บน test set: 59.9%
- QDA ชนะ! เพราะ covariance structure ของ Up vs Down อาจแตกต่างกัน

**Insight**: ในปัญหา stock market direction ที่ signal-to-noise ratio ต่ำมาก แม้ accuracy 56–60% ก็มีคุณค่าทางการเงินอย่างมาก เพราะสามารถสร้าง trading strategy ที่ profitable ได้ การเลือก QDA บอกเราว่า covariance ของ features แตกต่างกันระหว่างวันขึ้นและวันลง

```python
# ─── โหลดและเตรียม Smarket Dataset ─────────────────────────────────
# วัตถุประสงค์: โหลดข้อมูลตลาดหุ้นและแบ่ง train/test ตามปี
import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# โหลด Smarket dataset
Smarket = load_data('Smarket')
print(Smarket.head())
print(Smarket.shape)   # (1250, 9)

# ─── แบ่ง Train/Test ตามปี ───────────────────────────────────────────
# วัตถุประสงค์: ใช้ปี 2001-2004 สำหรับ train, ปี 2005 สำหรับ test
features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
train_mask = Smarket['Year'] < 2005

X_train = Smarket.loc[train_mask, features]
X_test  = Smarket.loc[~train_mask, features]
y_train = Smarket.loc[train_mask, 'Direction']
y_test  = Smarket.loc[~train_mask, 'Direction']

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ─── Fit LDA Model ──────────────────────────────────────────────────
# วัตถุประสงค์: สร้าง LDA classifier โดยสมมติ shared covariance matrix
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# ดู estimated parameters
print("LDA Class Means:\n", lda.means_)      # μ̂ₖ
print("LDA Priors:", lda.priors_)            # π̂ₖ

# ─── ทำนายและประเมิน LDA ─────────────────────────────────────────────
# วัตถุประสงค์: วัดความแม่นยำของ LDA บน test set
y_pred_lda = lda.predict(X_test)
print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
print("LDA Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lda))
print(classification_report(y_test, y_pred_lda))

# ─── Fit QDA Model ──────────────────────────────────────────────────
# วัตถุประสงค์: สร้าง QDA classifier ที่มี per-class covariance matrix
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

y_pred_qda = qda.predict(X_test)
print("QDA Accuracy:", accuracy_score(y_test, y_pred_qda))
print("QDA Confusion Matrix:\n", confusion_matrix(y_test, y_pred_qda))

# ─── เปรียบเทียบ LDA vs QDA ─────────────────────────────────────────
# วัตถุประสงค์: สรุปผลการเปรียบเทียบโมเดลทั้งสอง
results = pd.DataFrame({
    'Model': ['LDA', 'QDA'],
    'Test Accuracy': [accuracy_score(y_test, y_pred_lda),
                      accuracy_score(y_test, y_pred_qda)]
})
print(results)
```

---

## สรุป

| แนวคิด | สูตร | Python |
|--------|------|--------|
| LDA Discriminant | δₖ(x) = xᵀΣ⁻¹μₖ − ½μₖᵀΣ⁻¹μₖ + log πₖ | `LinearDiscriminantAnalysis()` |
| QDA Discriminant | δₖ(x) = −½xᵀΣₖ⁻¹x + xᵀΣₖ⁻¹μₖ − ... | `QuadraticDiscriminantAnalysis()` |
| Prior Estimate | π̂ₖ = nₖ/n | `.priors_` |
| Mean Estimate | μ̂ₖ = (1/nₖ)Σ xᵢ (i ∈ class k) | `.means_` |
| Pooled Covariance | Σ̂ = 1/(n−K) Σₖ Σᵢ (xᵢ−μ̂ₖ)(xᵢ−μ̂ₖ)ᵀ | `.covariance_` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 11 (Logistic Regression)**: Logistic เป็น discriminative model — เปรียบเทียบกับ LDA ซึ่งเป็น generative model
- ← **Week 4 (Covariance Matrix)**: Σ ใน LDA คือ covariance matrix จาก Week 4 — เชื่อมทฤษฎีกับการประยุกต์ใช้จริง
- → **Week 12-Note 2**: Naive Bayes (ผ่อนคลายสมมติฐาน Gaussian) และ KNN (non-parametric approach)
- → **Week 13**: ROC Curve ใช้ประเมิน LDA/QDA/Logistic บน threshold ต่างๆ
- → **Week 14**: Cross-Validation ใช้เลือกว่าควรใช้ LDA หรือ QDA
