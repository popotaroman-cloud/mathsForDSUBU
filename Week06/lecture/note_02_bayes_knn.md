# Note 2: Classification Setting, Bayes Classifier และ KNN
> Week 6 | CLO2 | ISLP Reference: Ch.2.2.3 The Classification Setting

---

## บทนำ

บันทึกนี้ขยาย Bias-Variance Trade-Off ไปสู่ **Classification** และแนะนำ **Bayes Classifier** ซึ่งเป็น "เพดานทางทฤษฎี" — classifier ที่ดีที่สุดที่เป็นไปได้ เราจะเห็นว่าแม้แต่ Bayes Classifier ที่สมบูรณ์แบบก็ยังมี error rate ที่ลดไม่ได้ (Bayes Error Rate) เป้าหมายคือให้นักศึกษาเข้าใจ Error Rate ใน Classification, อธิบาย Bayes Theorem ได้ และเชื่อมกับ KNN ซึ่งเป็น non-parametric method ตัวแรก ในชีวิตจริง Bayes Theorem เป็นพื้นฐานของ Spam Filter, Medical Diagnosis, และ Naive Bayes Classifier ซึ่งเราจะเรียนในสัปดาห์ที่ 12

---

## Section 1: Classification Setting และ Error Rate  *(ISLP 2.2.3)*

ในส่วนนี้เราจะปรับจาก Regression (predict ตัวเลข) มาสู่ **Classification** (predict category) และเรียนรู้วิธีวัด performance ใน setting ใหม่นี้

### 1.1 Training Error Rate

สำหรับ classifier $\hat{C}(x)$ ที่ predict class $j$:

$$\text{Training Error Rate} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(y_i \neq \hat{C}(x_i))$$

โดย $\mathbf{1}(\cdot)$ คือ Indicator Function (= 1 ถ้าเงื่อนไขจริง, 0 ถ้าเท็จ)

ตัวอย่าง: 100 ตัวอย่าง, predict ผิด 8 ตัว → Error Rate = 0.08 = 8%

**Test Error Rate** (สิ่งที่เราสนใจจริง ๆ):
$$\text{Test Error Rate} = \frac{1}{m}\sum_{i=1}^{m}\mathbf{1}(y_i^* \neq \hat{C}(x_i^*))$$

### 1.2 ความสัมพันธ์กับ MSE

เหมือนกับ Regression — เราต้องการ minimize Test Error Rate ไม่ใช่ Training Error Rate และ Bias-Variance Trade-Off ยังคงมีอยู่ใน Classification:
- Low $K$ ใน KNN: Low Bias, High Variance → Overfitting
- High $K$ ใน KNN: High Bias, Low Variance → Underfitting

---

## Section 2: Bayes Classifier  *(ISLP 2.2.3)*

ในส่วนนี้เราจะเรียนรู้ **Bayes Classifier** ซึ่งเป็น theoretical optimal classifier ที่ต้องรู้ true probability distribution — เพื่อใช้เป็น benchmark ว่า model จริง ๆ ใกล้เคียง optimal แค่ไหน

### 2.1 นิยาม Bayes Classifier

**Bayes Classifier** จัดสรร observation $x_0$ ให้ class $j$ ที่ให้ **Posterior Probability สูงสุด**:

$$\hat{C}_{\text{Bayes}}(x_0) = \arg\max_j \Pr(Y = j \mid X = x_0)$$

### 2.2 Bayes Error Rate

$$\text{Bayes Error Rate} = 1 - \mathbb{E}\left[\max_j \Pr(Y = j \mid X)\right]$$

**ความหมาย**: Error rate ต่ำสุดที่ทำได้ด้วย classifier ใด ๆ — เทียบเท่ากับ Irreducible Error ใน Regression

**ตัวอย่าง** (Binary Classification):

สมมติ $x_0$ มี $\Pr(Y=1|X=x_0) = 0.8$ และ $\Pr(Y=0|X=x_0) = 0.2$  
Bayes Classifier → predict $Y=1$  
Error rate ที่ $x_0$ = 0.2 (ดีที่สุดเท่าที่ทำได้)

**Bayes Decision Boundary**: เส้น/พื้นผิวที่ $\Pr(Y=1|X) = \Pr(Y=0|X) = 0.5$

**ปัญหาจริง**: เราไม่รู้ $\Pr(Y = j \mid X = x)$ — ต้องประมาณจาก data! KNN เป็นวิธีหนึ่ง

---

## Section 3: Probability Foundations สำหรับ Statistical Learning

ในส่วนนี้เราจะทบทวน **Conditional Probability** และ **Bayes Theorem** ซึ่งเป็นรากฐานของ Bayes Classifier และ Naive Bayes (Week 12) เพื่อให้มีเครื่องมือทางคณิตศาสตร์ที่จำเป็น

### 3.1 Conditional Probability

$$\Pr(A|B) = \frac{\Pr(A \cap B)}{\Pr(B)} \quad \text{(ถ้า } \Pr(B) > 0\text{)}$$

"ความน่าจะเป็นที่ $A$ จะเกิดขึ้น **เมื่อรู้ว่า** $B$ เกิดขึ้นแล้ว"

**ตัวอย่าง**:  
ถุงมี Ball แดง 3 ลูก น้ำเงิน 2 ลูก  
$\Pr(\text{แดง}) = 3/5$  
$\Pr(\text{แดงครั้งที่ 2} \mid \text{แดงครั้งที่ 1}) = 2/4 = 1/2$ (ไม่ใส่คืน)

### 3.2 Bayes Theorem

$$\Pr(A|B) = \frac{\Pr(B|A)\Pr(A)}{\Pr(B)}$$

**Components ใน Machine Learning**:

| Term | ชื่อ | ความหมาย |
|------|------|---------|
| $\Pr(A)$ | **Prior** | ความน่าจะเป็นก่อนมี data |
| $\Pr(B\|A)$ | **Likelihood** | ความน่าจะเป็นของ data ถ้า $A$ เป็นจริง |
| $\Pr(B)$ | **Evidence** | ความน่าจะเป็นของ data (normalizing constant) |
| $\Pr(A\|B)$ | **Posterior** | ความน่าจะเป็นหลังเห็น data |

**ML Workflow**: Prior + Data (Likelihood) → Posterior → Decision

### 3.3 Bayes Theorem ใน Medical Diagnosis

**ตัวอย่าง step-by-step**:

- โรคมะเร็งพบในประชากร 1% → $\Pr(\text{cancer}) = 0.01$ (Prior)
- Test มี sensitivity 90% → $\Pr(\text{positive}|\text{cancer}) = 0.90$ (Likelihood)
- Test มี false positive 5% → $\Pr(\text{positive}|\text{no cancer}) = 0.05$

ถาม: ถ้า test positive ความน่าจะเป็นที่เป็นมะเร็งจริงคือเท่าไหร่?

**ใช้ Bayes Theorem**:

$$\Pr(\text{cancer}|\text{positive}) = \frac{\Pr(\text{positive}|\text{cancer})\Pr(\text{cancer})}{\Pr(\text{positive})}$$

คำนวณ denominator ด้วย Total Probability:
$$\Pr(\text{positive}) = \Pr(\text{pos}|\text{cancer})\Pr(\text{cancer}) + \Pr(\text{pos}|\text{no cancer})\Pr(\text{no cancer})$$
$$= (0.90)(0.01) + (0.05)(0.99) = 0.009 + 0.0495 = 0.0585$$

$$\Pr(\text{cancer}|\text{positive}) = \frac{(0.90)(0.01)}{0.0585} = \frac{0.009}{0.0585} \approx 0.154 = 15.4\%$$

**Insight สำคัญ**: แม้ test จะ positive แต่ยังมีโอกาสเป็นมะเร็งแค่ **15.4%** เพราะ Prior probability ต่ำมาก — นี่คือเหตุผลที่แพทย์ไม่ได้วินิจฉัยจาก test เดียว!

```python
# ─── Bayes Theorem Calculator ──────────────────────────────────────────
# วัตถุประสงค์: คำนวณ posterior probability จาก prior และ likelihood
def bayes_update(prior, likelihood_pos, likelihood_neg):
    """
    คำนวณ P(Disease | Test=Positive)
    prior: P(Disease)
    likelihood_pos: P(Test=Pos | Disease)
    likelihood_neg: P(Test=Pos | No Disease)
    """
    # Evidence: P(Test=Positive) = ผลรวม weighted likelihood
    evidence = likelihood_pos * prior + likelihood_neg * (1 - prior)

    # Posterior: Bayes Theorem
    posterior = (likelihood_pos * prior) / evidence
    return posterior

# Medical test scenario
prior = 0.01           # 1% prevalence
sensitivity = 0.90     # true positive rate
fpr = 0.05             # false positive rate

posterior = bayes_update(prior, sensitivity, fpr)
print(f"P(cancer | positive test) = {posterior:.4f} ({posterior*100:.2f}%)")

# แสดงผลของ Prior ต่าง ๆ (sensitivity analysis)
# วัตถุประสงค์: เข้าใจว่า Prior มีผลต่อ Posterior อย่างไร
print("\nPosterior probability at different prevalence rates:")
for prev in [0.001, 0.01, 0.05, 0.10, 0.20]:
    post = bayes_update(prev, sensitivity, fpr)
    print(f"  Prevalence {prev*100:4.1f}% → Posterior = {post*100:.2f}%")
```

---

## Section 4: K-Nearest Neighbors (KNN)  *(ISLP 2.2.3)*

ในส่วนนี้เราจะเรียนรู้ **KNN** ซึ่งเป็น non-parametric method ที่ **ประมาณ Posterior Probability** โดยใช้ data จริงใกล้เคียง

### 4.1 หลักการของ KNN

**Algorithm**:
1. กำหนด test point $x_0$
2. หา $K$ training points ที่ใกล้ $x_0$ มากที่สุด (ใช้ Euclidean distance)
3. ประมาณ $\Pr(Y = j \mid X = x_0)$ โดย fraction ของ neighbors ที่เป็น class $j$:
$$\Pr(Y = j \mid X = x_0) = \frac{1}{K}\sum_{i \in \mathcal{N}_0}\mathbf{1}(y_i = j)$$
4. จัดสรร class ที่มี probability สูงสุด

### 4.2 ผลของ K ต่อ Bias-Variance

| $K$ | Decision Boundary | Variance | Bias |
|-----|------------------|----------|------|
| 1 | ขรุขระมาก (Voronoi) | สูง | ต่ำ |
| 5–20 | smooth ขึ้น | ปานกลาง | ปานกลาง |
| $n$ | เส้นตรง (constant) | ต่ำ | สูง |

**กฎทอง**: $K$ ที่ดีมักอยู่ระหว่าง $\sqrt{n}$ ถึง $n/5$

```python
# ─── KNN และ Bias-Variance Trade-Off ────────────────────────────────────
# วัตถุประสงค์: แสดงผลของ K ต่อ decision boundary และ test error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)

# สร้าง classification data
X_cls, y_cls = make_moons(n_samples=300, noise=0.25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.3)

# ทดสอบ KNN ที่ K ต่าง ๆ
K_values = [1, 3, 5, 10, 20, 50, 100]
print(f"{'K':>5} | {'Train Err':>10} | {'Test Err':>10}")
print("-" * 32)
for K in K_values:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_tr, y_tr)
    train_err = 1 - knn.score(X_tr, y_tr)
    test_err = 1 - knn.score(X_te, y_te)
    print(f"{K:>5} | {train_err:>10.4f} | {test_err:>10.4f}")
```

**DS Connection**: KNN เป็น conceptual basis ของ Instance-Based Learning และ Memory-Based Reasoning ซึ่งใช้ใน Recommender Systems (หา users ที่ "คล้าย" กับ target user) แต่ใน practice KNN ช้าเกินสำหรับ large-scale data — ใช้ Approximate Nearest Neighbors (FAISS, Annoy) แทน

---

## Case Study: Spam Filter ด้วย Bayes Theorem

**Scenario**  
Email provider ต้องการ filter spam email โดยใช้ Bayesian approach

**Data**  
ข้อมูล email 1,000 ฉบับ — 200 spam, 800 ham  
Feature: คำว่า "free" ปรากฏหรือไม่ (binary)

| | Spam (200) | Ham (800) |
|--|-----------|---------|
| "free" ปรากฏ | 150 (75%) | 80 (10%) |
| "free" ไม่ปรากฏ | 50 (25%) | 720 (90%) |

**Method — Naïve Bayes**:

Email ใหม่มีคำว่า "free" → คำนวณ:

$$\Pr(\text{spam} \mid \text{"free"}) = \frac{\Pr(\text{"free"} \mid \text{spam})\Pr(\text{spam})}{\Pr(\text{"free"})}$$

$$= \frac{(0.75)(0.20)}{(0.75)(0.20) + (0.10)(0.80)} = \frac{0.15}{0.15 + 0.08} = \frac{0.15}{0.23} \approx 0.652$$

**Result**  
ถ้า email มีคำว่า "free" → 65.2% ว่าเป็น spam

**Insight**  
- แม้แต่ feature เดียวก็ให้ข้อมูลได้มาก ถ้าใช้ Bayes ถูกต้อง
- Naive Bayes (Week 12) ขยายวิธีนี้ไปหลาย features โดย assume independence
- Gmail ใช้ Bayesian spam filter ในยุคแรก ก่อนจะพัฒนาไปใช้ Neural Network

---

## สรุป (Summary)

| แนวคิด | สูตร | ความหมาย |
|--------|------|---------|
| Training Error Rate | $\frac{1}{n}\sum \mathbf{1}(y_i \neq \hat{C}(x_i))$ | ไม่ใช่ goal จริง |
| Test Error Rate | วัดบน unseen data | goal จริง |
| Bayes Classifier | $\arg\max_j \Pr(Y=j \mid X=x_0)$ | optimal theoretical |
| Bayes Error Rate | $1 - \mathbb{E}[\max_j \Pr(Y=j \mid X)]$ | irreducible error |
| Bayes Theorem | $\Pr(A\|B) = \frac{\Pr(B\|A)\Pr(A)}{\Pr(B)}$ | posterior probability |
| KNN | majority vote ของ $K$ neighbors | approximate Bayes |
| KNN ขนาด $K$ ↑ | bias ↑, variance ↓ | Bias-Variance trade |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 6 (note 1)**: Bias-Variance Trade-Off ใน Regression ขยายมาสู่ Classification Setting นี้
- ← **Week 5**: supervised learning framework — input $X$, output $Y$, model $\hat{f}$
- → **Week 7**: EDA และ Statistical Inference — ต้องเข้าใจ Probability distributions ก่อน test hypothesis
- → **Week 11**: Logistic Regression คือการ model $\Pr(Y=1 \mid X)$ โดยตรง — นี่คือ Bayes Classifier ที่ประมาณจาก data
- → **Week 12**: Naive Bayes และ LDA คือ explicit probabilistic models ที่ implement Bayes Theorem อย่างเต็มรูปแบบ
