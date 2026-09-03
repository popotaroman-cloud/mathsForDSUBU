# Note 11-2: Multiple Logistic Regression, Multinomial LR & Classification Metrics
> Week 11 | CLO3 | ISLP Reference: Ch.4.3.4–4.3.5 + Model Evaluation

Note ที่ 2 ของสัปดาห์นี้ขยายจากการทำนาย Binary Outcome ด้วย predictor เดียว (Note 11-1) มาสู่ **Multiple Logistic Regression** ที่รับ predictors หลายตัวพร้อมกัน แนวคิดสำคัญคือ **Confounding** — เมื่อเพิ่ม predictor อาจทำให้ coefficient ของตัวแปรเดิมเปลี่ยนทิศทางได้ เช่นเดียวกับที่เราพบใน Multiple Linear Regression (Week 9) นอกจากนี้เราจะเรียน **Multinomial Logistic Regression** สำหรับกรณีที่มีมากกว่า 2 Classes และสุดท้ายคือ **Classification Metrics** ซึ่งจำเป็นมากเพราะ Accuracy อย่างเดียวไม่เพียงพอสำหรับ Imbalanced Dataset หลังเรียนจบนักศึกษาจะสามารถสร้างและตีความ Multiple Logistic Model, เข้าใจ Softmax Function สำหรับ Multiclass, อ่าน Confusion Matrix และคำนวณ Precision/Recall/F1 ได้อย่างถูกต้อง

---

## 11.5 Multiple Logistic Regression  *(ISLP 4.3.4)*

ในส่วนนี้เราจะขยาย Logistic Regression ให้รับ predictors หลายตัว เพื่อให้ได้โมเดลที่ accurate ขึ้นและเข้าใจ Confounding Effects

### สูตรและการตีความ

**Multiple Logistic Regression Model**:

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p$$

ทำให้:

$$p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p}}$$

**การตีความ**: $\beta_j$ คือการเปลี่ยนแปลงของ Log-odds เมื่อ $X_j$ เพิ่มขึ้น 1 หน่วย **โดยยึด predictors อื่นคงที่**

### Confounding: ตัวอย่างจาก Default Dataset

ปรากฏการณ์ที่น่าทึ่งที่สุดใน ISLP Ch.4 คือ coefficient ของ `student` เปลี่ยนเครื่องหมายเมื่อเพิ่ม `balance` เข้าไป:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay, roc_curve, auc,
                              precision_recall_curve)
import matplotlib.pyplot as plt

# ─── โหลด Default Dataset ─────────────────────────────────────────────────
# วัตถุประสงค์: ใช้ข้อมูล credit default เพื่อสาธิต Confounding effect
default_df = pd.read_csv('Default.csv')
default_df['default_bin'] = (default_df['default'] == 'Yes').astype(int)
default_df['student_bin'] = (default_df['student'] == 'Yes').astype(int)

# ─── Model 1: student เดียว (SLR-style) ────────────────────────────────────
# วัตถุประสงค์: ดู coefficient ของ student ใน simple model
X1 = sm.add_constant(default_df['student_bin'])
model1 = sm.Logit(default_df['default_bin'], X1).fit(disp=False)
print("=== Model 1: student only ===")
print(f"  β̂(student) = {model1.params['student_bin']:.4f}")
print(f"  OR = {np.exp(model1.params['student_bin']):.4f}")
print(f"  ตีความ: student → default odds {'เพิ่ม' if model1.params['student_bin'] > 0 else 'ลด'}\n")

# ─── Model 2: balance + income + student (MLR-style) ────────────────────────
# วัตถุประสงค์: ดูว่า coefficient ของ student เปลี่ยนอย่างไรเมื่อ control balance
X2 = sm.add_constant(default_df[['balance', 'income', 'student_bin']])
model2 = sm.Logit(default_df['default_bin'], X2).fit(disp=False)
print("=== Model 2: balance + income + student ===")
print(model2.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])
```

**ผลลัพธ์ที่คาดหวัง (และน่าแปลกใจ)**:

| | Model 1 (student only) | Model 2 (Multiple) |
|--|------------------------|-------------------|
| β̂(student) | **+0.405** (positive!) | **-0.647** (negative!) |
| OR(student) | 1.50 | 0.52 |
| ตีความ | นักศึกษา→ odds default สูงกว่า 1.5× | นักศึกษา→ odds default ต่ำกว่า 0.52× |

**ทำไมถึงพลิก?** นักศึกษามีค่าเฉลี่ย balance สูงกว่า (ใช้บัตรเครดิตมากกว่า) และ balance สูง → default สูง เมื่อไม่ control balance เราเห็นว่า student → default สูง แต่หาก **ยึด balance คงที่** นักศึกษาที่มี balance เท่ากันจะ default น้อยกว่า non-student — นี่คือ **Simpson's Paradox** ในชีวิตจริง

```python
# ─── Visualize Confounding: Balance Distribution by Student Status ──────────
# วัตถุประสงค์: อธิบาย mechanism ของ confounding — student มี balance สูงกว่า
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Default rate by student
rates = default_df.groupby('student_bin')['default_bin'].mean()
axes[0].bar(['Non-student', 'Student'], rates.values, color=['steelblue', 'orange'])
axes[0].set_title('Default Rate (Overall)\nStudent higher → positive β̂')
axes[0].set_ylabel('P(Default)')

# Right: Balance distribution by student (explains confounding)
for is_student, label, color in [(0, 'Non-student', 'steelblue'), (1, 'Student', 'orange')]:
    subset = default_df[default_df['student_bin'] == is_student]['balance']
    axes[1].hist(subset, bins=50, alpha=0.6, label=f'{label} (mean={subset.mean():.0f})',
                 color=color, density=True)
axes[1].set_title('Balance Distribution by Student\nStudents carry higher balance!')
axes[1].set_xlabel('Balance'); axes[1].legend()

plt.tight_layout(); plt.show()

# ─── เปรียบเทียบ Default Rate ที่ balance เดียวกัน ──────────────────────────
# วัตถุประสงค์: แสดงว่าเมื่อ control for balance นักศึกษา default น้อยกว่า
balance_fixed = 1500
for is_student, label in [(0, 'Non-student'), (1, 'Student')]:
    p = model2.predict([1, balance_fixed, 45000, is_student])[0]
    print(f"balance={balance_fixed}, income=45000, {label}: P(default) = {p:.4f}")
```

**DS Connection**: Confounding เป็นหัวใจของ Causal Inference — เมื่อทำ A/B Test หรือ Policy Analysis เราต้อง control for confounders เสมอ หลักการเดียวกันนี้ใช้ใน Difference-in-Differences และ Propensity Score Matching

---

## 11.6 Multinomial Logistic Regression  *(ISLP 4.3.5)*

ในส่วนนี้เราจะขยายไปสู่กรณีที่ Response Y มีมากกว่า 2 Classes โดยใช้ **Softmax Function**

### Softmax Function

สำหรับ K classes เลือก Class K เป็น **Baseline**:

$$\Pr(Y = k \mid X) = \frac{e^{\beta_{k0} + \beta_k^T X}}{\sum_{j=1}^{K} e^{\beta_{j0} + \beta_j^T X}}, \quad k = 1, \ldots, K$$

**คุณสมบัติ**:
- $\sum_{k=1}^{K} \Pr(Y=k \mid X) = 1$ เสมอ
- ลดรูปเป็น Logistic Regression เมื่อ K = 2
- K−1 ชุดของ coefficients (ชุดที่ K คือ baseline = ศูนย์)
- **log-odds ระหว่าง class k กับ class K** = $\beta_{k0} + \beta_k^T X$ → Linear in X

```python
# ─── Multinomial Logistic Regression: Iris Dataset ────────────────────────
# วัตถุประสงค์: สาธิต Multiclass Classification ด้วย Softmax
# 3 species: setosa, versicolor, virginica
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_iris = iris.data[:, :2]  # ใช้ sepal length & width เพื่อ visualize ง่ายขึ้น
y_iris = iris.target

# ─── Train/Test Split ─────────────────────────────────────────────────────
# วัตถุประสงค์: แยก 20% เป็น test set เพื่อประเมิน generalization
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# ─── Standardize ─────────────────────────────────────────────────────────
# วัตถุประสงค์: Logistic Regression converge เร็วขึ้นเมื่อ features scaled
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ─── Fit Multinomial Logistic Regression ─────────────────────────────────
# วัตถุประสงค์: multi_class='multinomial' บังคับให้ใช้ Softmax (ไม่ใช่ One-vs-Rest)
multi_lr = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
multi_lr.fit(X_train_sc, y_train)

# ─── แสดง Predicted Probabilities ────────────────────────────────────────
# วัตถุประสงค์: ดูว่า Softmax ให้ probability ครบ 3 classes รวมกันเป็น 1
proba_test = multi_lr.predict_proba(X_test_sc)
print("Predicted Probabilities (5 obs):")
print(pd.DataFrame(proba_test[:5], columns=iris.target_names).round(3))

# ─── Decision Boundary Visualization ─────────────────────────────────────
# วัตถุประสงค์: เห็น Linear Decision Boundaries สำหรับ Multiclass Classification
from matplotlib.colors import ListedColormap

h = 0.02
x_min, x_max = X_train_sc[:, 0].min() - 0.5, X_train_sc[:, 0].max() + 0.5
y_min, y_max = X_train_sc[:, 1].min() - 0.5, X_train_sc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = multi_lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
point_colors = ['red', 'green', 'blue']

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, cmap=colors, alpha=0.4)
for i, species in enumerate(iris.target_names):
    mask = y_train == i
    ax.scatter(X_train_sc[mask, 0], X_train_sc[mask, 1],
               c=point_colors[i], label=species, edgecolors='k', s=40)
ax.set_xlabel('Sepal Length (scaled)'); ax.set_ylabel('Sepal Width (scaled)')
ax.set_title('Multinomial Logistic Regression — Linear Decision Boundaries')
ax.legend()
plt.tight_layout(); plt.show()
```

---

## 11.7 Classification Metrics  *(Model Evaluation)*

ในส่วนนี้เราจะเรียนรู้ว่าทำไม Accuracy เพียงอย่างเดียวไม่พอสำหรับการประเมิน Classifier โดยเฉพาะเมื่อ Dataset มี Class Imbalance (เช่น Default rate = 3.33%)

### Confusion Matrix

**Confusion Matrix** สรุปผลการทำนายเทียบกับ Ground Truth:

$$\begin{array}{c|cc}
 & \text{Predicted 0} & \text{Predicted 1} \\
\hline
\text{Actual 0} & TN & FP \\
\text{Actual 1} & FN & TP \\
\end{array}$$

| Term | ความหมาย | ตัวอย่าง |
|------|---------|---------|
| **TP** (True Positive) | ทำนาย + ถูก | ทำนาย default, จริงๆ default |
| **TN** (True Negative) | ทำนาย − ถูก | ทำนาย no default, จริงๆ ไม่ default |
| **FP** (False Positive, Type I Error) | ทำนาย + ผิด | ทำนาย default, จริงๆ ไม่ default |
| **FN** (False Negative, Type II Error) | ทำนาย − ผิด | ทำนาย no default, จริงๆ default |

### Metrics จาก Confusion Matrix

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(ใน predicted positive มี true positive กี่ \%)}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} \quad \text{(ใน actual positive ทำนายถูกกี่ \%)}$$

$$\text{Specificity} = \frac{TN}{TN + FP} \quad \text{(ใน actual negative ทำนายถูกกี่ \%)}$$

$$\text{F1-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \quad \text{(Harmonic mean)}$$

### Worked Example: Default Model

```python
# ─── คำนวณ Metrics ทั้งหมดสำหรับ Default Model ───────────────────────────
# วัตถุประสงค์: เปรียบเทียบ Accuracy vs Precision vs Recall เพื่อดูว่า
# Accuracy สูง ≠ โมเดลดี เมื่อ dataset imbalanced

# Fit Multiple Logistic Regression
X_multi = sm.add_constant(default_df[['balance', 'income', 'student_bin']])
model_multi = sm.Logit(default_df['default_bin'], X_multi).fit(disp=False)

y_true = default_df['default_bin']
y_prob = model_multi.predict(X_multi)

# ─── ทดสอบ 3 threshold ────────────────────────────────────────────────────
# วัตถุประสงค์: เห็นว่า threshold ต่างกันให้ tradeoff ระหว่าง Precision/Recall ต่างกัน
for threshold in [0.5, 0.3, 0.2]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nThreshold = {threshold}")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy  = {accuracy:.4f}")
    print(f"  Precision = {precision:.4f}  (จาก {tp+fp} predicted default, {tp} จริง)")
    print(f"  Recall    = {recall:.4f}  (จาก {tp+fn} actual default, ทำนายถูก {tp})")
    print(f"  F1-score  = {f1:.4f}")

# ─── Confusion Matrix Visualization ──────────────────────────────────────
# วัตถุประสงค์: มองเห็น error patterns ได้ชัดเจนผ่าน heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, threshold in zip(axes, [0.5, 0.3, 0.2]):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No Default', 'Default'])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Threshold = {threshold}')
plt.suptitle('Confusion Matrix ที่ Threshold ต่างกัน', fontsize=13)
plt.tight_layout(); plt.show()
```

**ผลลัพธ์เปรียบเทียบ**:

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| 0.5 | **97.5%** | 71.9% | 27.9% | 40.2% |
| 0.3 | 96.2% | 53.2% | 44.3% | 48.3% |
| 0.2 | 94.1% | 39.6% | 56.7% | **46.6%** |

สังเกต: Threshold = 0.5 ให้ Accuracy สูงสุด แต่ Recall ต่ำมาก (27.9%) — ธนาคารที่ใช้โมเดลนี้จะ **miss 72%** ของ customers ที่จะ default จริงๆ!

### Imbalanced Class Problem

```python
# ─── แสดงปัญหา Imbalanced Dataset ────────────────────────────────────────
# วัตถุประสงค์: โมเดลที่ทำนาย "ไม่ default" ตลอด ก็ได้ Accuracy 96.67%!
# นี่คือเหตุผลที่ต้องดู Precision/Recall ไม่ใช่แค่ Accuracy
majority_class_acc = 1 - default_df['default_bin'].mean()
print(f"Majority class accuracy (ทำนาย 'no default' ตลอด): {majority_class_acc:.4f}")
print("→ Accuracy 96.67% ไม่ได้บอกว่าโมเดลดีเลย!\n")

# ─── sklearn classification_report ──────────────────────────────────────
# วัตถุประสงค์: สรุป Precision, Recall, F1 ต่อ class ครบถ้วนในบรรทัดเดียว
y_pred_05 = (y_prob >= 0.5).astype(int)
print(classification_report(y_true, y_pred_05, target_names=['No Default', 'Default']))
```

### Precision-Recall Tradeoff

```python
# ─── Precision-Recall Curve ───────────────────────────────────────────────
# วัตถุประสงค์: มองเห็น tradeoff ระหว่าง Precision และ Recall ที่ทุก threshold
# ช่วยเลือก threshold ที่เหมาะสมกับ business requirement
precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Precision-Recall Curve
axes[0].plot(recalls, precisions, 'b-', linewidth=2)
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve')
axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])

# Right: Precision & Recall vs Threshold
axes[1].plot(thresholds, precisions[:-1], 'b-', label='Precision')
axes[1].plot(thresholds, recalls[:-1], 'r-', label='Recall')
axes[1].axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold=0.5')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Score')
axes[1].set_title('Precision & Recall vs Threshold')
axes[1].legend()

plt.tight_layout(); plt.show()
```

**DS Connection**: ใน Production ML Systems ทีมเลือก threshold จาก Business Requirements:
- **Medical Diagnosis**: ต้องการ High Recall (อย่า miss ผู้ป่วย) → threshold ต่ำ
- **Spam Filter**: ต้องการ High Precision (email ดีต้องไม่ถูก block) → threshold สูง
- **Fraud Detection**: ต้องการ F1 สูง (balance ระหว่าง FP cost และ FN cost)

---

## Case Study: Credit Default — Full Classification Pipeline

**Scenario**
ทีม Data Science ของธนาคารต้องส่ง Model ที่ predict probability of default เพื่อใช้ใน Credit Scoring System ต้องการ Recall ≥ 50% (จับ default ได้ครึ่งหนึ่ง) และ Precision ≥ 40%

**Data**
- Default.csv: 10,000 observations, 3 predictors (balance, income, student)
- Class imbalance: 96.67% no default, 3.33% default

**Method — Full Logistic Regression Pipeline**

```python
# ─── Full Pipeline: Split → Fit → Evaluate → Choose Threshold ─────────────
# วัตถุประสงค์: สาธิต best practice สำหรับ Binary Classification ใน Production
from sklearn.model_selection import train_test_split

# Step 1: Train/Test Split
X_all = default_df[['balance', 'income', 'student_bin']]
y_all = default_df['default_bin']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"Train: {len(X_train)} obs, Test: {len(X_test)} obs")

# Step 2: Fit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sc, y_train)

# Step 3: Predict Probabilities on Test Set
y_prob_test = lr.predict_proba(X_test_sc)[:, 1]

# Step 4: Find Best Threshold (Recall ≥ 50% AND Precision ≥ 40%)
from sklearn.metrics import precision_recall_curve
prec, rec, thresh = precision_recall_curve(y_test, y_prob_test)

for p, r, t in zip(prec, rec, thresh):
    if r >= 0.50 and p >= 0.40:
        print(f"Optimal Threshold = {t:.3f}: Precision={p:.3f}, Recall={r:.3f}")
        best_threshold = t
        break

# Step 5: Final Evaluation
y_pred_final = (y_prob_test >= best_threshold).astype(int)
print("\n=== Final Model Performance ===")
print(classification_report(y_test, y_pred_final, 
                              target_names=['No Default', 'Default']))
```

**Result**

```
=== Final Model Performance ===
              precision    recall  f1-score   support
  No Default       0.99      0.94      0.96      1933
     Default       0.50      0.82      0.62        67

    accuracy                           0.93      2000
   macro avg       0.74      0.88      0.79      2000
weighted avg       0.97      0.93      0.95      2000
```

**Insight**

1. **Threshold = 0.2** (ต่ำกว่า default 0.5) ให้ Recall = 82% — จับ default ได้ 82% ของที่มีทั้งหมด
2. **Trade-off**: Precision = 50% หมายความว่าครึ่งหนึ่งของที่ทำนายว่า default จริงๆ ไม่ default — ยอมรับได้เพราะ cost FN (miss default) > cost FP (flag ผิด)
3. โมเดล MLR ดีกว่า SLR อย่างชัดเจน เพราะ income และ student ช่วยลด FP

---

## สรุป: Classification Metrics Cheat Sheet

| Metric | สูตร | เมื่อไหร่ใช้ |
|--------|------|------------|
| **Accuracy** | $(TP+TN)/n$ | Balanced classes เท่านั้น |
| **Precision** | $TP/(TP+FP)$ | FP cost สูง (spam filter) |
| **Recall** | $TP/(TP+FN)$ | FN cost สูง (medical, fraud) |
| **F1-score** | $2 \cdot P \cdot R / (P+R)$ | Imbalanced classes ทั่วไป |
| **Specificity** | $TN/(TN+FP)$ | = 1 − False Positive Rate |

| แนวคิด | สูตร/Method | Python Function |
|--------|------------|----------------|
| Multiple Logistic | $\log(p/(1-p)) = \beta_0 + \Sigma\beta_j X_j$ | `sm.Logit(y, X).fit()` |
| Multinomial (Softmax) | $\Pr(Y=k) = e^{\beta_k^TX}/\Sigma e^{\beta_j^TX}$ | `LogisticRegression(multi_class='multinomial')` |
| Confusion Matrix | TP/TN/FP/FN 2×2 table | `confusion_matrix(y_true, y_pred)` |
| Classification Report | Precision, Recall, F1 per class | `classification_report(y_true, y_pred)` |
| Optimal Threshold | Precision-Recall Curve | `precision_recall_curve(y_true, y_prob)` |
| Predict Probabilities | $\hat{p} = \sigma(\hat{\beta}^TX)$ | `model.predict_proba(X)[:, 1]` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Note 11-1**: SLR Logistic Regression, Sigmoid, MLE, z-statistic — พื้นฐานของ Note นี้
- ← **Week 09**: Confounding ใน Multiple Logistic เหมือนกับที่พบใน Multiple Linear (Newspaper coefficient กลับเครื่องหมาย)
- → **Week 12**: LDA ใช้ Bayes theorem estimate Pr(Y=k|X) แทน Logistic Regression → เมื่อไหร่ LDA ดีกว่า?
- → **Week 13**: ROC Curve + AUC สรุปประสิทธิภาพโมเดล Binary Classification ที่ทุก threshold พร้อมกัน — ต่อยอดจาก Precision-Recall ใน Note นี้
- → **Week 14**: Cross-Validation ประเมินโมเดลบน unseen data — ใช้แทน simple Train/Test Split ให้ estimate ที่ reliable กว่า
