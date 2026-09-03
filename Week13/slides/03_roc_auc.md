# Slide Deck 3: ROC Curve and AUC
> Week 13 | 9 slides | CLO3

---

## Slide 1 — Title

**ROC Curve + AUC: Threshold-Free Classifier Comparison**  
ISLP Ch.4.7 Context | Week 13 | CLO3

---

## Slide 2 — The Problem with Fixed Threshold

**Key Message**: Accuracy ที่ threshold=0.5 ไม่เพียงพอในการเปรียบเทียบ classifier

**Recall from Week 11:**
- Default threshold = 0.5: classify positive ถ้า Pr(Y=1|X) ≥ 0.5
- แต่ optimal threshold ขึ้นกับ cost of FN vs FP!

**ตัวอย่าง:**

| Threshold | Precision | Recall | Accuracy |
|-----------|-----------|--------|---------|
| 0.5 | 0.78 | 0.23 | 0.97 |
| 0.2 | 0.42 | 0.72 | 0.93 |
| 0.1 | 0.28 | 0.89 | 0.86 |

**คำถาม:** ถ้า threshold เปลี่ยนได้ — classifier ไหนดีกว่า?

**ต้องการ metric ที่ไม่ขึ้นกับ threshold → ROC Curve + AUC**

**[FIGURE: 2 classifiers บน plot เดียว — A มี F1=0.7 ที่ threshold=0.5, B มี F1=0.65 แต่ถ้าปรับ threshold B ดีกว่า]**

---

## Slide 3 — Confusion Matrix Refresher

**Key Message**: ก่อน ROC ต้องเข้าใจ TP, TN, FP, FN ก่อน

**Confusion Matrix (binary, positive = class 1):**

|  | Predicted 0 | Predicted 1 |
|--|------------|------------|
| **Actual 0** | TN (True Negative) | FP (False Positive) |
| **Actual 1** | FN (False Negative) | TP (True Positive) |

**Metrics derived:**

| Metric | Formula | ชื่ออื่น |
|--------|---------|--------|
| **Sensitivity (TPR)** | TP/(TP+FN) | Recall |
| **Specificity (TNR)** | TN/(TN+FP) | — |
| **FPR** | FP/(FP+TN) | 1 − Specificity |
| **Precision (PPV)** | TP/(TP+FP) | — |

**ROC Curve ใช้:** TPR (y-axis) vs FPR (x-axis) ที่ threshold ต่างๆ

**[FIGURE: confusion matrix 2×2 พร้อม color code — green TP/TN, red FP/FN]**

---

## Slide 4 — ROC Curve Construction

**Key Message**: ROC Curve สร้างจากการ scan threshold ทุกค่าตั้งแต่ 0 ถึง 1

**อัลกอริทึม:**
1. Fit classifier → get predicted probabilities Pr(Y=1|X) สำหรับ test set
2. Sort thresholds จากสูงสุดถึงต่ำสุด
3. สำหรับแต่ละ threshold t:
   - Classify: ŷᵢ = 1 ถ้า p̂ᵢ ≥ t, 0 ถ้าไม่ใช่
   - คำนวณ FPR = FP/(FP+TN), TPR = TP/(TP+FN)
4. Plot (FPR, TPR) ทุกจุด → ROC Curve

**Extreme points:**
- t = 1.0: predict ทุกอย่างเป็น 0 → TPR=0, FPR=0 (origin)
- t = 0.0: predict ทุกอย่างเป็น 1 → TPR=1, FPR=1 (top-right)

**[FIGURE: ROC curve ที่มีจุด highlight ที่ t=0.5, t=0.2, t=0.8 พร้อมลูกศรชี้]**

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
```

---

## Slide 5 — AUC: Area Under the Curve

**Key Message**: AUC สรุป ROC curve เป็นตัวเลขเดียว — classifier ที่ดีมี AUC ใกล้ 1

**AUC Definition:**
$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

**Interpretation ของ AUC:**
> AUC = โอกาสที่ classifier rank a random positive instance สูงกว่า a random negative instance

**Scale:**

| AUC | คุณภาพ |
|-----|-------|
| 1.0 | Perfect (ไม่มีใน real world) |
| 0.9–0.99 | Excellent |
| 0.8–0.89 | Good |
| 0.7–0.79 | Fair |
| 0.6–0.69 | Poor |
| 0.5 | Random (diagonal) |
| < 0.5 | Worse than random (flip predictions!) |

**[FIGURE: 3 ROC curves: AUC=0.5 (diagonal), AUC=0.8 (curved), AUC=0.95 (near top-left corner)]**

---

## Slide 6 — Selecting Optimal Threshold

**Key Message**: หลังเลือก classifier ด้วย AUC แล้ว ค่อยหา optimal threshold

**Method 1: Youden's J Statistic**
$$J = \text{TPR} - \text{FPR} = \text{Sensitivity} + \text{Specificity} - 1$$
→ เลือก threshold ที่ J สูงสุด (balanced ระหว่าง sensitivity + specificity)

```python
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]
```

**Method 2: Cost-based (FN cost ≠ FP cost)**
$$\text{minimize}: c_{FN} \cdot \text{FNR} + c_{FP} \cdot \text{FPR}$$

**Method 3: F-beta Score**
$$F_\beta = (1+\beta^2) \frac{P \cdot R}{\beta^2 P + R}$$
- β=1: F1 (equal weight)
- β=2: F2 (weight Recall 2×)
- β=0.5: F0.5 (weight Precision 2×)

**[FIGURE: ROC curve ที่ตัดด้วยเส้น Youden's J maximum — จุด optimal อยู่บน curve]**

---

## Slide 7 — Precision-Recall Curve

**Key Message**: สำหรับ imbalanced data ใช้ Precision-Recall curve แทน ROC ได้

**ROC Curve ปัญหา:**
- เมื่อ negative class ใหญ่มาก (imbalanced) ROC curve ดูดีกว่าความเป็นจริง
- FPR = FP/(FP+**TN**) → TN ใหญ่ → FPR เล็กแม้ FP มาก

**Precision-Recall Curve:**
- X-axis: Recall (TPR) — sensitivity to positive class
- Y-axis: Precision — quality of positive predictions
- ไม่ใช้ TN → ไม่ bias โดย negative majority

**[FIGURE: side-by-side — ROC curve ดูดีมาก vs P-R curve เดียวกันดูไม่ดีนัก สำหรับ imbalanced data]**

**Rule of Thumb:**
- Balanced classes: ใช้ ROC + AUC
- Imbalanced classes: ใช้ Precision-Recall + AP (Average Precision)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
```

---

## Slide 8 — Multi-Class ROC

**Key Message**: ROC + AUC ขยายสู่ K > 2 classes ด้วย One-vs-Rest

**One-vs-Rest (OvR):**
- สร้าง K ROC curves: class k vs rest
- Macro AUC = average ของ K AUC values

```python
from sklearn.metrics import roc_auc_score

# Multi-class AUC
auc_macro = roc_auc_score(
    y_test_multiclass,
    y_proba_multiclass,
    multi_class='ovr',      # One-vs-Rest
    average='macro'         # average over classes
)
print(f'Multi-class AUC (macro): {auc_macro:.4f}')
```

**ใช้เมื่อ K > 2:**
- Wine classification (3 cultivars)
- Digit recognition (10 digits)
- Disease stage (3 stages)

**[FIGURE: 3 ROC curves (class 0 vs rest, class 1 vs rest, class 2 vs rest) บน plot เดียว]**

---

## Slide 9 — Summary: ROC + AUC

**Key Message**: ROC + AUC เป็น standard ของ industry สำหรับ binary classifier evaluation

**ROC + AUC สรุป:**
- ROC Curve: TPR vs FPR ที่ threshold ต่างๆ
- AUC: area under ROC, ∈ [0,1], ใกล้ 1 = ดี
- Threshold-independent → เปรียบเทียบ classifier ได้ยุติธรรม

**Workflow:**
1. Fit classifiers ทั้งหมด
2. เปรียบเทียบ AUC → เลือก best classifier
3. Plot ROC → เลือก optimal threshold ตาม business need
4. Report: confusion matrix + metrics ที่ chosen threshold

**Python:**
```python
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
J = tpr - fpr
best_threshold = thresholds[np.argmax(J)]
```

**Connection to Next Week:**  
Week 14: Cross-Validation ใช้ AUC เป็น scoring metric ในการเลือก K ของ KNN และ hyperparameters อื่นๆ
