# Slide Deck: Classification Metrics
> Week 11 | CLO3 | 10 slides

---
## Slide 1 — Title
**Classification Metrics: วัดประสิทธิภาพของ Classifier**  
Week 11 | CLO3  
LLo: อ่านและตีความ confusion matrix, precision, recall, F1-score ได้

---
## Slide 2 — ทำไม Accuracy ไม่พอ?
**Key Message**: Accuracy ทำให้เข้าใจผิดเมื่อ class imbalanced — Default 3.3% → accuracy 96.7% แม้ไม่ทำอะไร

**ตัวอย่าง Default dataset** (9,667 No + 333 Yes):
- Classifier ที่ predict "No" ทุกกรณี → accuracy = 96.7% — ดูสูง!
- แต่จับ Default ได้ 0 ราย → ไม่มีประโยชน์เลย

**ต้องการ metrics ที่**:
- แยก Type I error (FP) กับ Type II error (FN)
- ระบุ trade-off ระหว่าง Precision และ Recall
- เหมาะกับ imbalanced data

---
## Slide 3 — Confusion Matrix
**Key Message**: confusion matrix แสดง 4 ประเภทของ prediction — foundation ของ metrics ทั้งหมด

$$\begin{array}{c|cc}
& \text{Predict No} & \text{Predict Yes} \\ \hline
\text{Actual No}  & TN & FP \\
\text{Actual Yes} & FN & TP
\end{array}$$

| Term | ย่อ | ความหมาย |
|------|-----|---------|
| True Positive | TP | จริงเป็น Yes และ predict Yes ✓ |
| True Negative | TN | จริงเป็น No และ predict No ✓ |
| False Positive | FP | จริงเป็น No แต่ predict Yes ✗ (Type I error) |
| False Negative | FN | จริงเป็น Yes แต่ predict No ✗ (Type II error) |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['No','Yes']).plot()
```

---
## Slide 4 — Precision, Recall, Specificity
**Key Message**: เลือก metric ตาม "ผิดแบบไหนเจ็บปวดกว่า" — FP vs FN

$$\text{Precision} = \frac{TP}{TP+FP} \quad \text{(ใน predicted Yes มีจริง Yes กี่%)}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP+FN} \quad \text{(ใน actual Yes จับได้กี่%)}$$

$$\text{Specificity} = \frac{TN}{TN+FP} \quad \text{(ใน actual No บอกถูกกี่%)}$$

| Scenario | สำคัญกว่า | เหตุผล |
|----------|---------|-------|
| Medical test (disease) | Recall สูง | ไม่ควร miss disease case (FN แย่กว่า) |
| Spam filter | Precision สูง | ไม่ควรกรอง email ดีทิ้ง (FP แย่กว่า) |
| Credit fraud detection | Recall สูง | ไม่ควร miss fraud (FN cost สูง) |

---
## Slide 5 — F1-Score และ Accuracy
**Key Message**: F1 = harmonic mean ของ Precision และ Recall — balance ทั้งสอง

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP+FP+FN}$$

**ตัวอย่าง Default** (threshold = 0.5):
| Metric | ค่า | ตีความ |
|--------|-----|-------|
| Accuracy | 0.974 | 97.4% ถูกต้อง |
| Precision | 0.65 | 65% ของที่ predict Default เป็น Default จริง |
| Recall | 0.33 | จับ Default ได้แค่ 33% |
| F1 | 0.44 | balance ต่ำ — precision กับ recall ต่างกันมาก |

**Insight**: Recall ต่ำมาก → ธนาคารจะ miss default ถึง 67% → ต้องปรับ threshold

---
## Slide 6 — Threshold Tuning
**Key Message**: ลด threshold → Recall ↑ Precision ↓ — trade-off ที่ต้องตัดสินใจตาม business context

```python
# ─── เปรียบ threshold ต่าง ๆ ──────────────────────────────
# วัตถุประสงค์: หา threshold ที่เหมาะสมกับ business objective
from sklearn.metrics import precision_score, recall_score, f1_score

proba = logit_model.predict_proba(X_test)[:, 1]
thresholds = [0.3, 0.4, 0.5, 0.6]
results = []

for t in thresholds:
    pred = (proba >= t).astype(int)
    results.append({
        'threshold': t,
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred)
    })

print(pd.DataFrame(results).round(3))
```

**Default detection**: ธนาคารอาจเลือก threshold = 0.3 เพื่อ recall สูงกว่า แม้ precision ลด

---
## Slide 7 — Python: Classification Report
**Key Message**: `classification_report` แสดง precision, recall, F1 ทุก class ในคำสั่งเดียว

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X = default[['balance', 'income', 'student_num']]
y = default['default_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─── Confusion matrix + report ─────────────────────────────
# วัตถุประสงค์: ดู metrics ครบทุกตัวพร้อมกัน
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
```

---
## Slide 8 — Imbalanced Data: Class Weights
**Key Message**: ถ้า class imbalanced — ใช้ `class_weight='balanced'` หรือ resampling

```python
# ─── Handle imbalanced data ────────────────────────────────
# วัตถุประสงค์: ให้ model ให้ความสำคัญกับ minority class (Default=Yes) มากขึ้น

# วิธีที่ 1: class_weight='balanced' — ลงโทษ FN ของ majority class น้อยลง
model_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
model_balanced.fit(X_train, y_train)
y_pred_bal = model_balanced.predict(X_test)

print('Balanced model:')
print(classification_report(y_test, y_pred_bal, target_names=['No Default', 'Default']))

# วิธีที่ 2: SMOTE (Synthetic Minority Oversampling) — สร้าง synthetic samples
# from imblearn.over_sampling import SMOTE
# X_res, y_res = SMOTE().fit_resample(X_train, y_train)
```

---
## Slide 9 — สรุป: เลือก Metric อย่างไร?
**Key Message**: ไม่มี "best metric" universal — ต้องดู business context และ cost ของ error

| สถานการณ์ | Primary Metric | เหตุผล |
|-----------|---------------|-------|
| Medical diagnosis | Recall | Miss disease → patient อันตราย |
| Spam filter | Precision | Block หรือ email สำคัญ → user เสียหาย |
| Fraud detection | Recall | Miss fraud → ธนาคารเสียหาย |
| Search ranking | Precision@K | ต้องการ result ที่ดีอยู่ top-K |
| Overall balance | F1 | ไม่มี priority ชัดเจน |
| Ranking/Score | AUC-ROC | เปรียบ model ที่ threshold ต่างกัน (Week 13) |

---
## Slide 10 — Week Summary
**Key Message**: Classification metrics ช่วย diagnose ว่า classifier ดีจริงหรือแค่ดูดีบน accuracy

**สิ่งที่เรียนรู้สัปดาห์นี้**:
- **Classification**: qualitative Y → estimate Pr(Y=k|X)
- **Logistic Regression**: sigmoid, log-odds, MLE, z-statistic
- **Multiple Logistic**: ตีความ "holding others fixed", confounding
- **Multinomial**: Softmax สำหรับ K > 2 classes
- **Metrics**: CM, Accuracy, Precision, Recall, F1 — เลือกตาม use case

| Metric | สูตร | Python |
|--------|------|--------|
| Accuracy | (TP+TN)/n | `accuracy_score(y,ŷ)` |
| Precision | TP/(TP+FP) | `precision_score(y,ŷ)` |
| Recall | TP/(TP+FN) | `recall_score(y,ŷ)` |
| F1 | 2·P·R/(P+R) | `f1_score(y,ŷ)` |
| Report | all | `classification_report(y,ŷ)` |

**สัปดาห์ต่อไป — Week 12**: LDA, QDA, Naive Bayes — generative classifiers
