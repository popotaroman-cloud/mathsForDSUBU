# Note 13-2: ROC Curve, AUC & Full Classification Pipeline
> Week 13 | CLO3 | ISLP Reference: Ch.4.7 Lab: Classification

Note ที่ 2 ของสัปดาห์นี้จะรวมทุกอย่างที่เรียนมาใน Classification (Weeks 11–13) เข้าสู่ **Full Pipeline** สมบูรณ์แบบ แนวคิดหลักคือ **ROC Curve** ซึ่งสรุปประสิทธิภาพของ Binary Classifier ที่ทุก Threshold พร้อมกันในกราฟเดียว และ **AUC** ที่เป็น Single Number สำหรับเปรียบเทียบโมเดลต่างกัน จากนั้นเราจะ run Full Pipeline บน Stock Market dataset เปรียบเทียบ Logistic, LDA, QDA, Naive Bayes และ KNN ทั้งหมดพร้อมกัน เป้าหมายคือให้นักศึกษาสามารถเลือก Model ที่ดีที่สุดจาก Evidence ที่ครบถ้วนและอธิบายเหตุผลในการเลือกได้ ทักษะ Full Pipeline นี้คือสิ่งที่ Data Scientist ทำในการสัมภาษณ์งาน และใน Kaggle Competition ทุกครั้ง

---

## 13.4 ROC Curve  *(ISLP 4.7.3)*

ในส่วนนี้เราจะเรียนรู้ ROC Curve ซึ่งแก้ปัญหาที่ว่า Accuracy, Precision, Recall วัดที่ threshold เดียว แต่ ROC มองที่ **ทุก threshold พร้อมกัน**

### แนวคิดของ ROC

สำหรับ Binary Classifier ที่ output probability $\hat{p}(X)$ ถ้าเราเลื่อน threshold จาก 0 ถึง 1:

- Threshold สูง → Predict positive น้อยมาก → TPR ต่ำ, FPR ต่ำ
- Threshold ต่ำ → Predict positive เยอะมาก → TPR สูง, FPR สูง

**ROC Curve** plot TPR vs FPR ที่ทุก threshold:

$$\text{TPR (Sensitivity)} = \frac{TP}{TP + FN}$$

$$\text{FPR (1 − Specificity)} = \frac{FP}{FP + TN}$$

### 3 Reference Points บน ROC Curve

| จุด | ความหมาย |
|-----|---------|
| $(0, 0)$ | Threshold = 1 → ไม่ทำนาย positive เลย |
| $(1, 1)$ | Threshold = 0 → ทำนาย positive ทั้งหมด |
| $(0, 1)$ | Perfect Classifier (ต้องการ!) |
| เส้นแนวทแยง | Random Classifier (AUC = 0.5) |

### Area Under the Curve (AUC)

**AUC** คือพื้นที่ใต้ ROC Curve → Single number สำหรับเปรียบเทียบโมเดล:

| AUC | ความหมาย |
|-----|---------|
| 1.0 | Perfect Classifier |
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Fair |
| 0.6–0.7 | Poor |
| 0.5 | Random Classifier (no skill) |
| < 0.5 | Worse than random (invert predictions!) |

**AUC Interpretation**: ความน่าจะเป็นที่ classifier ให้ score ของ positive observation สูงกว่า negative observation ที่เลือกแบบสุ่ม

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

# ─── โหลด Default Dataset เพื่อสาธิต ROC Curve ─────────────────────────
# วัตถุประสงค์: เปรียบเทียบ ROC ของ Logistic Regression กับ Random Classifier
default_df = pd.read_csv('Default.csv')
default_df['default_bin'] = (default_df['default'] == 'Yes').astype(int)
default_df['student_bin'] = (default_df['student'] == 'Yes').astype(int)

X = default_df[['balance', 'income', 'student_bin']]
y = default_df['default_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ─── Fit Logistic Regression ─────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sc, y_train)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

# ─── สร้าง ROC Curve ─────────────────────────────────────────────────────
# วัตถุประสงค์: plot TPR vs FPR ที่ทุก threshold เพื่อดู overall performance
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR / Sensitivity)')
ax.set_title('ROC Curve — Default Dataset')
ax.legend()
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout(); plt.show()

print(f"AUC = {roc_auc:.4f}")

# ─── หา Optimal Threshold จาก ROC Curve ─────────────────────────────────
# วัตถุประสงค์: Youden's J statistic = TPR - FPR (maximize)
j_stat = tpr - fpr
optimal_idx = np.argmax(j_stat)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
print(f"  TPR = {tpr[optimal_idx]:.4f}, FPR = {fpr[optimal_idx]:.4f}")
```

---

## 13.5 Full Classification Pipeline  *(ISLP 4.7)*

ในส่วนนี้เราจะ run Full Pipeline เปรียบเทียบ Classifiers ทั้ง 5 ประเภทบน Smarket dataset (Stock Market Direction Prediction)

### Smarket Dataset

**Smarket** dataset จาก ISLP มี 1,250 observations (ปี 2001–2005) ของ S&P 500 index:
- Response: `Direction` (Up/Down) — ตลาดหุ้นพรุ่งนี้ขึ้นหรือลง?
- Predictors: `Lag1`–`Lag5` (% returns ของ 5 วันก่อน), `Volume` (shares traded in billions)

```python
# ─── โหลด Smarket Dataset ─────────────────────────────────────────────────
# วัตถุประสงค์: Stock market classification — ข้อมูลจาก ISLP Ch.4.7
smarket = pd.read_csv('Smarket.csv')
print(smarket.head())
print(f"\nClass distribution: {smarket['Direction'].value_counts()}")

# ─── Preprocessing ────────────────────────────────────────────────────────
# วัตถุประสงค์: เตรียม features และ target สำหรับ model comparison
X_sm = smarket[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y_sm = (smarket['Direction'] == 'Up').astype(int)

# ─── Train/Test Split: ใช้ปี 2001-2004 train, 2005 test ──────────────────
# วัตถุประสงค์: ใช้ temporal split แทน random split เพราะเป็น time series
train_mask = smarket['Year'] < 2005
X_train_sm, X_test_sm = X_sm[train_mask], X_sm[~train_mask]
y_train_sm, y_test_sm = y_sm[train_mask], y_sm[~train_mask]
print(f"Train: {len(X_train_sm)} obs (2001-2004), Test: {len(X_test_sm)} obs (2005)")

# ─── Feature Scaling ─────────────────────────────────────────────────────
# วัตถุประสงค์: Scale เพื่อ KNN และ LDA ที่ sensitive ต่อ scale ของ features
scaler_sm = StandardScaler()
X_train_sc_sm = scaler_sm.fit_transform(X_train_sm)
X_test_sc_sm = scaler_sm.transform(X_test_sm)
```

### Fit 5 Classifiers พร้อมกัน

```python
# ─── สร้าง Dictionary ของ Classifiers ────────────────────────────────────
# วัตถุประสงค์: เปรียบเทียบ 5 methods ที่เรียนมาตลอด Classification Weeks
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB(),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

# ─── Train และ Evaluate ทุก Classifier ───────────────────────────────────
# วัตถุประสงค์: สร้าง comparison table ที่ครบถ้วน
results = []
roc_data = {}

for name, clf in classifiers.items():
    # Fit
    clf.fit(X_train_sc_sm, y_train_sm)
    
    # Predict
    y_pred = clf.predict(X_test_sc_sm)
    y_prob = clf.predict_proba(X_test_sc_sm)[:, 1]
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test_sm, y_pred).ravel()
    accuracy = (tp + tn) / len(y_test_sm)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    roc_auc_score_val = roc_auc_score(y_test_sm, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': roc_auc_score_val
    })
    
    # Store ROC data for plotting
    fpr_val, tpr_val, _ = roc_curve(y_test_sm, y_prob)
    roc_data[name] = (fpr_val, tpr_val, roc_auc_score_val)
    
    print(f"\n{name}:")
    print(f"  Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
          f"Recall={recall:.4f}, F1={f1:.4f}, AUC={roc_auc_score_val:.4f}")

# ─── Summary Table ────────────────────────────────────────────────────────
# วัตถุประสงค์: เปรียบเทียบทุกโมเดลในตารางเดียว
results_df = pd.DataFrame(results).set_index('Model')
print("\n=== Model Comparison ===")
print(results_df.round(4))
```

### ROC Curves เปรียบเทียบ

```python
# ─── Plot ROC Curves: ทุก Classifier บน Figure เดียวกัน ─────────────────
# วัตถุประสงค์: visual comparison ที่ดีกว่าการดู numbers เพียงอย่างเดียว
# จุดที่ curve อยู่สูงกว่า = model ดีกว่าที่ FPR เดียวกัน
colors = ['blue', 'green', 'red', 'orange', 'purple']

fig, ax = plt.subplots(figsize=(8, 7))
for (name, (fpr_val, tpr_val, auc_val)), color in zip(roc_data.items(), colors):
    ax.plot(fpr_val, tpr_val, color=color, linewidth=2,
            label=f'{name} (AUC = {auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves: 5 Classifiers — Smarket Dataset')
ax.legend(loc='lower right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout(); plt.show()
```

---

## 13.6 Imbalanced Classes & Practical Tips

ในส่วนนี้เราจะเรียนรู้วิธีจัดการกับ Imbalanced Dataset ซึ่งเป็นปัญหาที่พบบ่อยมากในงาน Data Science จริง

### ปัญหา Imbalanced Dataset

เมื่อ Class หนึ่งมี observations น้อยมาก (เช่น Fraud = 0.1%, Normal = 99.9%):

- Classifier มักจะ predict majority class ตลอด
- Accuracy ดูสูง แต่ Recall ของ minority class ต่ำมาก
- AUC เป็น metric ที่ดีกว่าในกรณีนี้ (ไม่ขึ้นกับ class distribution)

```python
# ─── วิธีที่ 1: ปรับ class_weight ──────────────────────────────────────────
# วัตถุประสงค์: บอกให้ model ให้ความสำคัญกับ minority class มากขึ้น
# 'balanced' คำนวณ weight = n_samples / (n_classes * np.bincount(y))
lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_balanced.fit(X_train_sc_sm, y_train_sm)

y_pred_balanced = lr_balanced.predict(X_test_sc_sm)
print("Logistic Regression (class_weight='balanced'):")
print(classification_report(y_test_sm, y_pred_balanced, 
                              target_names=['Down', 'Up']))

# ─── วิธีที่ 2: ปรับ Threshold ──────────────────────────────────────────────
# วัตถุประสงค์: ใช้ threshold ต่ำลงเพื่อเพิ่ม Recall ของ minority class
# เหมาะเมื่อ FN cost > FP cost
lr_standard = LogisticRegression(max_iter=1000)
lr_standard.fit(X_train_sc_sm, y_train_sm)
y_prob_standard = lr_standard.predict_proba(X_test_sc_sm)[:, 1]

for threshold in [0.5, 0.4, 0.3]:
    y_pred_t = (y_prob_standard >= threshold).astype(int)
    from sklearn.metrics import f1_score
    f1_val = f1_score(y_test_sm, y_pred_t)
    print(f"Threshold={threshold}: F1={f1_val:.4f}")

# ─── Feature Scaling Reminder ────────────────────────────────────────────
# วัตถุประสงค์: แสดงผลกระทบของ scaling ต่อ KNN
print("\nKNN without scaling:")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train_sm, y_train_sm)  # no scaling
acc_unscaled = knn_unscaled.score(X_test_sm, y_test_sm)

print("KNN with scaling:")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_sc_sm, y_train_sm)  # with scaling
acc_scaled = knn_scaled.score(X_test_sc_sm, y_test_sm)

print(f"Accuracy without scaling: {acc_unscaled:.4f}")
print(f"Accuracy with scaling: {acc_scaled:.4f}")
```

### เมื่อไหร่ใช้ Classifier แบบไหน?

| Classifier | เมื่อไหร่ได้เปรียบ | ข้อเสีย |
|-----------|-----------------|---------|
| **Logistic Regression** | p ไม่ใหญ่มาก, ต้องการ Interpretability | ต้องการ Linear Decision Boundary |
| **LDA** | n เล็ก, features มี Normal distribution | สมมติ Equal Covariance |
| **QDA** | Classes มี Covariance ต่างกัน | p ใหญ่ → overfit |
| **Naive Bayes** | p ใหญ่มาก, features เป็นอิสระ | Independence assumption มักไม่จริง |
| **KNN** | Decision boundary ซับซ้อน | Slow prediction, sensitive to scale |

---

## Case Study: Stock Market Direction Prediction

**Scenario**
Hedge Fund ต้องการสร้างโมเดลทำนายทิศทาง S&P 500 วันถัดไป (Up/Down) จาก lagged returns เพื่อใช้ใน Quantitative Trading Strategy

**Data**
- Smarket.csv: 1,250 daily observations, 2001–2005
- Features: Lag1–Lag5 (% returns), Volume
- Target: Direction (Up/Down)
- Train: 2001–2004 (998 obs), Test: 2005 (252 obs)

**Method**

```python
# ─── Best Model Selection และ Final Report ────────────────────────────────
# วัตถุประสงค์: เลือก model จาก evidence ที่ครบถ้วน พร้อมเหตุผล
print("=" * 60)
print("  FINAL MODEL COMPARISON REPORT — SMARKET DATASET")
print("=" * 60)
print(results_df.round(4).to_string())

best_model_name = results_df['AUC'].idxmax()
best_auc = results_df.loc[best_model_name, 'AUC']
print(f"\nBest Model (by AUC): {best_model_name} (AUC = {best_auc:.4f})")

# ─── Confusion Matrix ของ Best Model ─────────────────────────────────────
best_clf = classifiers[best_model_name]
y_pred_best = best_clf.predict(X_test_sc_sm)
cm = confusion_matrix(y_test_sm, y_pred_best)
print(f"\nConfusion Matrix ({best_model_name}):")
print(pd.DataFrame(cm, index=['Actual Down', 'Actual Up'],
                   columns=['Pred Down', 'Pred Up']))
```

**Result** (คาดหวัง)

```
=== Model Comparison ===
                    Accuracy  Precision  Recall     F1    AUC
Logistic Regression   0.5238     0.5619  0.5806  0.5711  0.553
LDA                   0.5595     0.5769  0.6290  0.6019  0.566
QDA                   0.5992     0.5814  0.6935  0.6327  0.599
Naive Bayes           0.5476     0.5649  0.6290  0.5952  0.558
KNN (k=5)             0.5357     0.5484  0.6048  0.5752  0.534
```

**Insight**

1. **QDA ดีที่สุด** (AUC = 0.599, Accuracy = 59.9%) — decision boundary ที่ flexible กว่าเหมาะกับ stock market ที่ distributions ต่างกันใน Up/Down
2. **AUC ≈ 0.5–0.6** ทุก model → Stock market แทบ Random Walk — เป็น "baseline difficulty" ที่ Efficient Market Hypothesis ทำนายไว้
3. ความสำเร็จเล็กน้อยของ QDA ≈ 10% edge เหนือ random → อาจเพียงพอสำหรับ Quant Trading ถ้า execution cost ต่ำพอ
4. **ข้อสังเกต**: ใน real trading ต้องระวัง transaction costs, slippage, look-ahead bias

---

## สรุป: ROC + Full Pipeline

| แนวคิด | สูตร/Method | Python Function |
|--------|------------|----------------|
| TPR (Recall) | $TP/(TP+FN)$ | `recall_score()` |
| FPR | $FP/(FP+TN)$ | `1 - specificity_score()` |
| ROC Curve | TPR vs FPR ทุก threshold | `roc_curve(y_true, y_prob)` |
| AUC | พื้นที่ใต้ ROC Curve | `roc_auc_score(y_true, y_prob)` |
| Optimal Threshold | argmax(TPR − FPR) | Youden's J Statistic |
| Train/Test Split (Temporal) | year < 2005 → train | boolean mask |
| Feature Scaling | StandardScaler | `StandardScaler().fit_transform()` |
| Class Imbalance | class_weight='balanced' | `LogisticRegression(class_weight='balanced')` |

### Pipeline สำหรับ Classification Project

```
1. Load Data → Explore (distributions, class balance, missing values)
2. Preprocess → encode categoricals, handle NA, scale features
3. Split → Train/Test (stratify by class, temporal split if time series)
4. Fit → ≥3 classifiers (Logistic, LDA/QDA, KNN at minimum)
5. Evaluate → Confusion Matrix, Precision/Recall/F1, AUC
6. Compare → ROC Curves ทั้งหมดบน plot เดียวกัน
7. Select → Best model โดยพิจารณา AUC, F1, Business Requirements
8. Tune → ปรับ Threshold ตาม FP/FN cost
9. Report → ระบุ Model ที่เลือก + เหตุผล + Performance บน Test Set
```

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 11**: Logistic Regression + Confusion Matrix + Precision/Recall — Note นี้ต่อยอดด้วย ROC
- ← **Week 12**: LDA, QDA, KNN — เปรียบเทียบกันครบในสัปดาห์นี้
- ← **Note 13-1**: GLM Framework — Logistic Regression เป็น GLM; ทุก model ที่ compare ก็อยู่ใน statistical framework เดียวกัน
- → **Week 14**: Cross-Validation จะทำให้ Model Comparison น่าเชื่อถือกว่า Single Train/Test Split เพราะ estimate performance บน held-out data หลาย folds
