# Note 1: MSE, Overfitting และ Bias-Variance Trade-Off
> Week 6 | CLO2 | ISLP Reference: Ch.2.2 Assessing Model Accuracy

---

## บทนำ

สัปดาห์นี้เราจะเรียนรู้หนึ่งในแนวคิดที่สำคัญที่สุดในทั้งวิชา: **Bias-Variance Trade-Off** ซึ่งเป็นคำอธิบายพื้นฐานว่าทำไม Machine Learning models บางตัวถึง "ทำงานได้ดีบน training data แต่พังบน test data" เป้าหมายคือให้นักศึกษาสามารถคำนวณ MSE, อธิบาย Decomposition $\text{MSE} = \text{Variance} + \text{Bias}^2 + \text{Irreducible Error}$, และวินิจฉัยได้ว่า model มีปัญหา Overfitting หรือ Underfitting แนวคิดนี้เป็นพื้นฐานของการเลือก model complexity ที่เหมาะสม — ตั้งแต่ degree ของ polynomial ไปจนถึงจำนวน hidden layers ใน Neural Network ในชีวิตจริง Data Scientist ทุกคนต้องเข้าใจ Bias-Variance เพื่อตัดสินใจว่าควร collect data เพิ่ม (แก้ Variance) หรือเปลี่ยน model (แก้ Bias)

---

## Section 1: วัดความแม่นยำของ Model — MSE  *(ISLP 2.2.1)*

ในส่วนนี้เราจะเรียนรู้ว่า **Mean Squared Error (MSE)** คืออะไร และทำไมต้องแยกระหว่าง Training MSE กับ Test MSE — ซึ่งเป็นความแตกต่างสำคัญที่ผู้เริ่มต้นมักเข้าใจผิด

### 1.1 Mean Squared Error (MSE)

**นิยาม**: สำหรับ regression model $\hat{f}$ ที่ predict บน data point $(x_i, y_i)$:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{f}(x_i))^2$$

**Training MSE**: คำนวณบน data ที่ใช้ train:
$$\text{MSE}_{\text{train}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{f}(x_i))^2 \quad \text{(data เดิม)}$$

**Test MSE**: คำนวณบน data ที่ model ไม่เคยเห็น:
$$\text{MSE}_{\text{test}} = \frac{1}{m}\sum_{i=1}^{m}(y_i^* - \hat{f}(x_i^*))^2 \quad \text{(unseen data)}$$

**กฎทอง**: เราสนใจ **Test MSE** เพราะนั่นคือสิ่งที่จะเกิดขึ้นจริงเมื่อ deploy model

### 1.2 Overfitting และ Underfitting

**Overfitting**: model ซับซ้อนเกินไป — จำ noise ใน training data  
→ Training MSE ต่ำ, Test MSE สูง

**Underfitting**: model เรียบง่ายเกินไป — ไม่จับ pattern จริงได้  
→ Training MSE สูง, Test MSE สูง

**ตัวอย่าง Polynomial Fitting**:

```python
# ─── จำลอง Overfitting vs Underfitting ──────────────────────────────────
# วัตถุประสงค์: แสดง Training/Test MSE เมื่อเพิ่ม degree ของ polynomial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

np.random.seed(42)

# สร้าง data จาก true function f(x) = sin(2πx) + noise
# วัตถุประสงค์: มี ground truth ที่รู้จึงสามารถประเมิน bias จริงได้
n = 50
X = np.sort(np.random.uniform(0, 1, n)).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X.ravel())
y = y_true + np.random.normal(0, 0.3, n)

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# fit polynomial ดีกรีต่าง ๆ และวัด MSE
degrees = [1, 3, 5, 10, 15]
train_mses = []
test_mses = []

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train, y_train)

    train_mse = np.mean((y_train - model.predict(X_train))**2)
    test_mse = np.mean((y_test - model.predict(X_test))**2)
    train_mses.append(train_mse)
    test_mses.append(test_mse)
    print(f"Degree {d:2d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")

# Plot MSE vs complexity
plt.figure(figsize=(8, 4))
plt.plot(degrees, train_mses, 'b-o', label='Training MSE')
plt.plot(degrees, test_mses, 'r-o', label='Test MSE')
plt.xlabel('Polynomial Degree (Model Flexibility)')
plt.ylabel('MSE')
plt.title('Bias-Variance Trade-Off: MSE vs Model Complexity')
plt.legend(); plt.grid(True)
plt.savefig("bias_variance_mse.png", dpi=100, bbox_inches='tight')
```

**DS Connection**: พฤติกรรม U-shape ของ Test MSE เป็น pattern ที่เห็นในทุก ML model ตั้งแต่ degree ของ polynomial ไปจนถึง depth ของ decision tree และจำนวน hidden units ใน Neural Network

---

## Section 2: Bias-Variance Decomposition  *(ISLP 2.2.2)*

ในส่วนนี้เราจะ derive อย่างเป็นทางการว่า Expected Test MSE แตกออกเป็น 3 ส่วน เพื่อให้เข้าใจว่า "ทำไม model ถึงทำนายผิด" อย่างแม่นยำ

### 2.1 การ Decompose MSE

สมมติ True model: $Y = f(X) + \varepsilon$ โดย $\mathbb{E}[\varepsilon] = 0, \text{Var}(\varepsilon) = \sigma^2$

สำหรับ test point $x_0$ ใหม่:

$$\mathbb{E}\left[(y_0 - \hat{f}(x_0))^2\right] = \underbrace{\text{Var}(\hat{f}(x_0))}_{\text{Variance}} + \underbrace{[\text{Bias}(\hat{f}(x_0))]^2}_{\text{Bias}^2} + \underbrace{\sigma^2}_{\text{Irreducible}}$$

### 2.2 นิยามแต่ละส่วน

**Variance** ของ $\hat{f}(x_0)$:
$$\text{Var}(\hat{f}(x_0)) = \mathbb{E}\left[(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2\right]$$

ความหมาย: $\hat{f}$ จะเปลี่ยนไปแค่ไหนถ้าเราใช้ training set ชุดอื่น

**Bias** ของ $\hat{f}(x_0)$:
$$\text{Bias}(\hat{f}(x_0)) = \mathbb{E}[\hat{f}(x_0)] - f(x_0)$$

ความหมาย: ค่าเฉลี่ยของ prediction ต่างจาก true value เท่าไหร่ (systematic error)

**Irreducible Error** $\sigma^2$:
$$\text{Var}(\varepsilon) = \sigma^2$$

ความหมาย: noise ในข้อมูลที่ไม่ว่าจะใช้ model ดีแค่ไหนก็ลดไม่ได้

### 2.3 Derivation (สั้น)

$$\mathbb{E}[(y_0 - \hat{f})^2] = \mathbb{E}[(f + \varepsilon - \hat{f})^2]$$

เพิ่ม $\pm\mathbb{E}[\hat{f}]$:
$$= \mathbb{E}[(f - \mathbb{E}[\hat{f}])^2] + \mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2] + \mathbb{E}[\varepsilon^2] + 2\cdot\mathbb{E}[\varepsilon]\cdot(\cdots)$$

เพราะ $\mathbb{E}[\varepsilon] = 0$ และ $\varepsilon$ independent:
$$= \text{Bias}^2 + \text{Variance} + \sigma^2 \checkmark$$

### 2.4 Trade-Off ระหว่าง Bias และ Variance

| Flexibility | Variance | Bias | Test MSE |
|-------------|----------|------|---------|
| ต่ำ (เรียบง่าย) | ต่ำ | สูง | สูง (Underfitting) |
| เหมาะสม | medium | medium | **ต่ำสุด** |
| สูง (ซับซ้อน) | สูง | ต่ำ | สูง (Overfitting) |

```
Test MSE
   │           *                Irreducible Error
   │        *     *           ─────────────────────
   │     *           *
   │  *                 *  *
   └────────────────────────── Model Flexibility
       ↑           ↑
   Underfit    Optimal    Overfit
```

```python
# ─── จำลอง Bias-Variance Decomposition ──────────────────────────────────
# วัตถุประสงค์: แสดง Variance, Bias², Irreducible Error อย่างชัดเจน
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# True function
f_true = lambda x: np.sin(2 * np.pi * x)
sigma = 0.3          # irreducible noise
n_train = 30         # training sample size
n_sim = 100          # จำนวนครั้งที่ simulate
x_test = np.linspace(0, 1, 50)

degrees = list(range(1, 12))
biases_sq = []
variances = []

for d in degrees:
    preds = []  # เก็บ predictions จาก 100 simulations
    for _ in range(n_sim):
        # สร้าง training set ใหม่แต่ละรอบ
        x_train = np.random.uniform(0, 1, n_train)
        y_train = f_true(x_train) + np.random.normal(0, sigma, n_train)

        # fit polynomial degree d
        coeffs = np.polyfit(x_train, y_train, d)
        y_pred = np.polyval(coeffs, x_test)
        preds.append(y_pred)

    preds = np.array(preds)           # shape: (n_sim, 50)
    mean_pred = preds.mean(axis=0)    # average prediction
    y_true_test = f_true(x_test)

    # Bias² = (mean prediction - true)²
    bias_sq = np.mean((mean_pred - y_true_test)**2)
    # Variance = average squared deviation from mean prediction
    variance = np.mean(np.var(preds, axis=0))

    biases_sq.append(bias_sq)
    variances.append(variance)

# Plot Bias², Variance, Total Expected Error
irreducible = sigma**2
total = np.array(biases_sq) + np.array(variances) + irreducible

plt.figure(figsize=(8, 4))
plt.plot(degrees, biases_sq, 'b-', label='Bias²', linewidth=2)
plt.plot(degrees, variances, 'r-', label='Variance', linewidth=2)
plt.plot(degrees, total, 'k--', label='Total MSE', linewidth=2)
plt.axhline(irreducible, color='gray', linestyle=':', label=f'Irreducible (σ²={irreducible})')
plt.xlabel('Polynomial Degree'); plt.ylabel('Error')
plt.title('Bias-Variance Decomposition')
plt.legend(); plt.grid(True)
plt.savefig("bias_variance_decomp.png", dpi=100, bbox_inches='tight')
print("Plot saved!")
```

**DS Connection**: เส้น U-shape ของ Total MSE คือแนวคิดที่อยู่เบื้องหลังการทำ Cross-Validation (Week 14) ซึ่งเป็นวิธีหา optimal flexibility โดยไม่ต้องรู้ True MSE ล่วงหน้า

---

## Section 3: Variance สูง vs Bias สูง — แก้อย่างไร?

ในส่วนนี้เราจะเรียนรู้ **วิธีวินิจฉัย** ว่า model มีปัญหาแบบใด และแก้ได้อย่างไร เพราะในชีวิตจริงการเลือกวิธีแก้ให้ถูกต้องประหยัดเวลาได้มาก

### 3.1 วินิจฉัยจาก Learning Curves

**High Bias (Underfitting)** — Signs:
- Training MSE สูง
- Test MSE ≈ Training MSE (ไม่ต่างกันมาก)
- เพิ่ม training data ไม่ช่วย

**High Variance (Overfitting)** — Signs:
- Training MSE ต่ำมาก
- Test MSE >> Training MSE (ช่องว่างกว้าง)
- เพิ่ม training data **ช่วยได้**

### 3.2 วิธีแก้

| ปัญหา | แก้โดย |
|-------|-------|
| High Bias | เพิ่ม model complexity, เพิ่ม features, ลด regularization |
| High Variance | เพิ่ม training data, ลด complexity, regularization (L1/L2) |

**DS Connection**: ในยุค Deep Learning — Large Language Models (LLMs) เช่น GPT-4 มี parameters หลายพัน billion → Variance สูงมาก แต่แก้ด้วยการใช้ training data ขนาดใหญ่มาก (internet-scale) นี่คือ Bias-Variance Trade-Off ในระดับ massive

---

## Case Study: Medical Diagnosis — Overfitting บน Small Dataset

**Scenario**  
ทีมวิจัยต้องการสร้าง model ทำนายว่าผู้ป่วยมีโรคมะเร็งหรือไม่ จากผล blood tests 20 ตัว โดยมีข้อมูลผู้ป่วยแค่ 50 ราย

**Data**  
- 50 patients × 20 features (blood markers)
- Target: cancer (1) / no cancer (0)
- แบ่ง: 35 train, 15 test

**Method**

```python
# ─── Overfitting Demo — Medical Dataset ──────────────────────────────────
# วัตถุประสงค์: แสดง high variance problem เมื่อ n samples << p features
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# สร้าง synthetic medical data
n_samples, n_features = 50, 20
X_med = np.random.randn(n_samples, n_features)
# True decision: based on first 3 features only
y_med = (X_med[:, 0] + X_med[:, 1] - X_med[:, 2] > 0).astype(int)

# แบ่ง train/test
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_med, y_med, test_size=0.3, random_state=1)

# ทดสอบ models ต่าง ๆ
models = {
    "Logistic (L2)": LogisticRegression(C=1.0, max_iter=1000),
    "Logistic (No Reg)": LogisticRegression(C=1e6, max_iter=1000),
    "Decision Tree (max_depth=None)": DecisionTreeClassifier(),
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
}

print(f"{'Model':<35} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
print("-" * 65)
for name, model in models.items():
    model.fit(X_tr, y_tr)
    train_acc = model.score(X_tr, y_tr)
    test_acc = model.score(X_te, y_te)
    gap = train_acc - test_acc
    print(f"{name:<35} {train_acc:>10.3f} {test_acc:>10.3f} {gap:>8.3f}")
```

**Result** (ตัวอย่าง):
```
Model                               Train Acc   Test Acc      Gap
Logistic (L2)                           0.771      0.733    0.038
Logistic (No Reg)                       1.000      0.533    0.467  ← Overfit!
Decision Tree (max_depth=None)          1.000      0.467    0.533  ← Overfit!
KNN (k=1)                               1.000      0.467    0.533  ← Overfit!
KNN (k=7)                               0.714      0.667    0.047
```

**Insight**
- High flexibility models (no regularization, k=1, deep tree) → perfect training accuracy แต่ test accuracy แย่
- Regularization (L2) ช่วยลด Variance และให้ test accuracy ที่ดีกว่า
- ข้อสรุป: **เมื่อ n << p ต้องใช้ regularization เสมอ** — นี่คือ fundamental rule ใน Clinical Machine Learning

---

## สรุป (Summary)

| แนวคิด | สูตร | ความหมาย |
|--------|------|---------|
| Training MSE | $\frac{1}{n}\sum(y_i - \hat{f}(x_i))^2$ | ไม่ใช่ goal จริง |
| Test MSE | $\frac{1}{m}\sum(y_i^* - \hat{f}(x_i^*))^2$ | goal จริง |
| Bias | $\mathbb{E}[\hat{f}(x_0)] - f(x_0)$ | systematic error |
| Variance | $\mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2]$ | sensitivity ต่อ data |
| Decomposition | Bias² + Variance + σ² | total test error |
| Overfitting | Train MSE ↓↓, Test MSE ↑ | Variance สูง |
| Underfitting | Train MSE ↑, Test MSE ↑ | Bias สูง |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 5**: Statistical Learning Framework — $f$ คือ true function ที่เราพยายาม estimate
- → **Week 6 (note 2)**: Bayes Classifier และ KNN เป็นตัวอย่างของ Bias-Variance trade-off ใน classification
- → **Week 14**: Cross-Validation คือวิธีประเมิน Test MSE จาก training data — ใช้ได้เพราะเข้าใจ Bias-Variance แล้ว
- → **Week 9**: Ridge/Lasso Regression เป็น formal วิธีลด Variance ด้วย Regularization
