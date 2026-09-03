# Slide Deck: Classification Overview
> Week 11 | CLO3 | ISLP Ch.4.1 | 8 slides

---
## Slide 1 — Title
**Classification: จาก Regression สู่การจำแนกประเภท**  
Week 11 | CLO3 | ISLP Ch.4.1  
LLo: อธิบายความต่างระหว่าง regression และ classification และเลือก classifier ที่เหมาะสมได้

---
## Slide 2 — Week Overview
**Key Message**: Classification ตอบคำถาม "ใคร/อะไร?" ส่วน Regression ตอบ "เท่าไร?" — ทั้งสองเป็น Supervised Learning

สัปดาห์นี้เราจะเปลี่ยนจาก regression ที่มี **quantitative response** (Sales, mpg) มาสู่ **classification** ที่มี **qualitative response** (Default: Yes/No, Spam: Yes/No) ปัญหา classification พบมากที่สุดใน Data Science ตั้งแต่การตรวจจับ spam email, การวินิจฉัยโรค, การจำแนกประเภทสินค้า ไปจนถึงการ detect credit card fraud จุดสำคัญคือ classifier ต้องการ estimate **Pr(Y=k|X)** ซึ่งต้องใช้ method ที่ต่างจาก linear regression เราจะเรียน Logistic Regression เป็นหลักซึ่งเป็น foundation สำหรับ classifier ที่ซับซ้อนกว่าใน Week 12–13

**ภาพรวม Weeks 11–12**: Logistic → LDA/QDA → KNN → GLM/ROC

---
## Slide 3 — Regression vs Classification
**Key Message**: ทั้งสองเป็น Supervised Learning แต่ output type ต่างกัน

| | Regression | Classification |
|--|-----------|---------------|
| **Response Y** | Quantitative (continuous) | Qualitative (categorical) |
| **Examples** | Sales, mpg, price, age | Spam/Not Spam, Default/No Default |
| **Output** | ตัวเลข | class label หรือ probability |
| **Loss** | MSE = Σ(y-ŷ)² | Misclassification rate, Log-loss |
| **Methods** | OLS, Ridge, LASSO | Logistic, LDA, KNN, Trees |

[FIGURE: side-by-side: regression (scatter + fitted line) vs classification (scatter + decision boundary)]

---
## Slide 4 — Real-World Classification Problems
**Key Message**: Classification อยู่ทุกที่ใน Data Science — ต้องเลือก classifier ให้เหมาะกับปัญหา

| Domain | Problem | Y (Classes) |
|--------|---------|-------------|
| Finance | Credit default prediction | Yes / No |
| Marketing | Email spam detection | Spam / Not Spam |
| Medical | Disease diagnosis | Disease / Healthy |
| Computer Vision | Image recognition | Cat / Dog / Bird / ... |
| NLP | Sentiment analysis | Positive / Negative / Neutral |
| HR | Employee churn | Leave / Stay |

**ISLP Datasets**:
- `Default.csv`: Credit card default (2 classes)
- `Smarket.csv`: Stock market direction (Up/Down)
- `Wage.csv`: Income level (Low/Med/High — 3 classes)

---
## Slide 5 — Classifiers ใน ISLP Ch.4
**Key Message**: มี classifiers หลายตัว เลือกตาม assumption และลักษณะ data

| Classifier | Assumption | เมื่อใช้ |
|------------|-----------|---------|
| **Logistic Regression** (Week 11) | none about X distribution | binary/multinomial, interpretable |
| **LDA** (Week 12) | X ~ Normal, shared Σ | Gaussian X, want linear boundary |
| **QDA** (Week 12) | X ~ Normal, class-specific Σ | Gaussian X, need quadratic boundary |
| **KNN** (Week 5) | non-parametric | complex boundary, enough data |
| **Naive Bayes** (Week 12) | X independent given Y | high-dimension, text |

**สัปดาห์นี้**: Logistic Regression — เพราะอ่านและตีความได้ง่ายที่สุด

---
## Slide 6 — Bayesian Perspective
**Key Message**: classifier ทุกตัวพยายาม estimate Pr(Y=k|X) — Bayes classifier ที่ optimal ทำแบบนี้

**Bayes Classifier** (ideal, ใช้ได้เมื่อรู้ true distribution):
$$\text{Classify to } k^* = \arg\max_k \Pr(Y=k|X)$$

- Bayes error rate = 1 − max_k Pr(Y=k|X) ← irreducible error (Week 6)
- ไม่สามารถทำได้จริงเพราะ Pr(Y|X) ไม่รู้ → ต้องประมาณ

**สามวิธีประมาณ Pr(Y=k|X)**:
1. **Direct**: Logistic Regression — model Pr(Y=1|X) โดยตรง
2. **Generative**: LDA/QDA/Naive Bayes — model Pr(X|Y) แล้วใช้ Bayes theorem
3. **Non-parametric**: KNN — estimate จาก nearest neighbors

---
## Slide 7 — Default Dataset: ตัวอย่างตลอดสัปดาห์
**Key Message**: Default dataset บอกว่าลูกค้าคนไหนจะผิดนัดชำระ credit card

**Default.csv** (10,000 คน):

| Variable | Type | Description |
|----------|------|-------------|
| default | Y (binary) | Yes/No — ผิดนัดชำระ |
| student | X (binary) | Yes/No — นักศึกษา |
| balance | X (continuous) | ยอดค้างชำระ credit card |
| income | X (continuous) | รายได้ต่อปี |

```python
import pandas as pd, seaborn as sns
default = pd.read_csv('Default.csv')
print(default['default'].value_counts())
# No: 9667 (96.7%), Yes: 333 (3.3%) — imbalanced!
sns.scatterplot(data=default, x='balance', y='income',
                hue='default', alpha=0.3, palette=['steelblue','red'])
```

---
## Slide 8 — Summary
**Key Message**: Classification = supervised learning กับ qualitative Y — ต้องการ method ที่ต่างจาก regression

**สิ่งที่เรียนรู้วันนี้**:
- **Classification**: Y categorical — binary (K=2) หรือ multi-class (K>2)
- **Goal**: estimate Pr(Y=k|X) แล้ว classify ตาม max probability
- **Bayes classifier**: optimal แต่ theoretical — classifiers จริง approximate มัน
- **ISLP Default dataset**: binary classification — default (Yes/No) จาก balance, income, student

**สัปดาห์ต่อไป — Slide 2**: ทำไม Linear Regression ไม่เหมาะสำหรับ Classification?
