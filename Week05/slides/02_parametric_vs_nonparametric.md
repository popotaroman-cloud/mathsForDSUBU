# Slide Deck: Parametric vs Non-parametric Approaches
> Week 5 | CLO2 | ISLP Ch.2.1.2 | 9 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 5**  
Parametric vs Non-parametric: เลือก approach อย่างไรให้เหมาะกับปัญหา  
CLO2: เปรียบเทียบ parametric และ non-parametric approach พร้อมอธิบาย trade-off ได้

---

## Slide 2 — The Core Question
**Key Message**: Parametric กับ Non-parametric ไม่ใช่ว่าอันไหนดีกว่า — แต่อันไหน "เหมาะกว่า" กับปัญหาและข้อมูลที่มี

ในการประมาณ f เราต้องเลือกว่าจะ "assume อะไรล่วงหน้า" เกี่ยวกับ f หรือไม่ Parametric approach สมมุติรูปแบบ f ก่อน (เช่น linear) แล้วประมาณแค่ parameters ที่ไม่กี่ตัว Non-parametric approach ไม่สมมุติอะไร ยอมให้ data "พูด" ว่า f ควรมีรูปร่างอย่างไร ทั้งสองวิธีมี trade-off ที่ชัดเจน และนักวิทยาการข้อมูลที่ดีต้องเลือกได้ถูก เป้าหมายของ deck นี้คือให้นักศึกษาเข้าใจ trade-off อย่างลึกซึ้งและสามารถ justify การเลือก approach ในปัญหาจริงได้

**Parametric**: assume f form → fit parameters → use model  
**Non-parametric**: no assumption → fit data directly → flexible

---

## Slide 3 — Parametric Approach: Step-by-step
**Key Message**: Parametric ลด problem จาก "หา f ใดๆ" เป็น "หา β ไม่กี่ตัว" — ทำให้ tractable แต่ต้อง assume ให้ถูก

**ขั้นตอน:**
```
Step 1: เลือก model form
        ตัวอย่าง: linear model
        f(X) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ

Step 2: Fit (Train)
        ใช้ training data เพื่อ estimate β₀, β₁, ..., βₚ
        minimize: Σᵢ(Yᵢ - f̂(Xᵢ))²  (Least Squares)

Step 3: ได้ f̂ ที่ใช้งานได้
        f̂(X_new) = β̂₀ + β̂₁X₁ + ... + β̂ₚXₚ
```

**ตัวอย่าง — Advertising:**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['TV']], df['Sales'])
print(f'β₀={model.intercept_:.3f}, β₁={model.coef_[0]:.3f}')
# β₀=7.033, β₁=0.048
```

**ข้อดี**: ง่าย, interpretable, ทำงานได้ดีกับ data น้อย  
**ข้อเสีย**: ถ้า assume ผิด → high bias ไม่ว่า data จะมีมากแค่ไหน

---

## Slide 4 — Non-parametric Approach: KNN Example
**Key Message**: KNN ไม่ assume รูปแบบ — predict ด้วยค่าเฉลี่ยของ k neighbors ที่ใกล้ที่สุด

**K-Nearest Neighbors (KNN Regression):**
```
f̂(x₀) = (1/K) Σᵢ∈N(x₀) Yᵢ

N(x₀) = K neighbors ที่ใกล้ x₀ ที่สุด
K = จำนวน neighbors (hyperparameter)
```

**[FIGURE: แสดง 2D scatter, วงกลมรอบ x₀ แสดง K nearest neighbors, f̂(x₀) = average Y ของจุดในวงกลม]**

**Python:**
```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

**ผลของ K:**
- K เล็ก (K=1): wiggly fit, low bias, high variance → overfit
- K ใหญ่ (K=100): smooth fit, high bias, low variance → underfit
- K เหมาะสม: balance ระหว่าง bias และ variance

---

## Slide 5 — Comparing Fits: Parametric vs Non-parametric
**Key Message**: [FIGURE ที่เห็นชัดที่สุด] — KNN K=1 overfit ชัดเจน, KNN K=10 ใกล้เคียง linear model

**[FIGURE: 3 panels บน dataset เดียวกัน]**
```
Panel 1: Linear model (parametric)
         - Clean line ตรงๆ
         - Bias ถ้า true f ไม่ linear

Panel 2: KNN K=1 (non-parametric, very flexible)
         - Wiggly, interpolate ทุกจุด training
         - Training MSE = 0 (!), Test MSE สูง

Panel 3: KNN K=10 (non-parametric, moderate)
         - Smooth curve
         - ใกล้เคียง linear สำหรับ near-linear data
```

**ข้อสังเกต:**
- ไม่มี "best" method — ขึ้นกับ true shape ของ f
- สำหรับ near-linear f: Linear Regression ชนะ (simpler, less variance)
- สำหรับ complex f: Non-parametric อาจชนะ (ถ้า data เยอะพอ)

---

## Slide 6 — When to Choose What
**Key Message**: กฎง่ายๆ: ถ้า goal = inference → parametric; ถ้า goal = prediction + data เยอะ → อาจ non-parametric

**เลือก Parametric เมื่อ:**
- ต้องการ interpret coefficients (inference goal)
- Data น้อย (น้อยกว่า 1000 observations)
- เชื่อว่า f ใกล้เคียง linear
- ต้องการ model ที่ deploy ง่าย

**เลือก Non-parametric เมื่อ:**
- ต้องการ prediction accuracy สูงสุด (prediction goal)
- Data มีมากพอ (อย่างน้อยหลายพัน)
- ไม่แน่ใจว่า f มีรูปแบบอะไร
- Noise ต่ำ (signal ชัดเจน)

**Decision Table:**

| สถานการณ์ | แนะนำ |
|----------|------|
| น้อยกว่า 1,000 samples | Parametric |
| Interpretability สำคัญ | Parametric |
| Data ≥ 10,000 + black-box OK | Non-parametric |
| Clinical decision support | Parametric (trust & explain) |
| Image/text/audio | Deep learning (non-parametric) |

---

## Slide 7 — Overfitting: The Main Risk of Non-parametric
**Key Message**: Non-parametric method ที่ flexible เกินไปจะ memorize training data แต่ generalize ไม่ได้

**Overfitting:**
```
Model เรียน noise ใน training data
→ Training MSE ต่ำมาก
→ Test MSE สูงมาก
→ "จำ training set" แทนที่จะ "เรียนรู้ f"
```

**[FIGURE: Training MSE vs Test MSE เป็น function ของ flexibility — training ลดตลอด, test มี U-shape]**

**ตัวอย่าง KNN:**
- K=1: Training MSE = 0 (interpolates perfectly)
- K=1: Test MSE สูงมาก (overfit)
- K=10: Test MSE ดีกว่ามาก
- K=200: Test MSE เพิ่มขึ้น (underfit)

**Prevention:**
- ใช้ train/test split หรือ Cross-Validation
- เลือก K ที่ minimize test MSE (Week 14)

---

## Slide 8 — Python: Comparing Methods on Advertising Data
**Key Message**: ดู code จริงที่ fit ทั้ง 2 approaches และเปรียบเทียบ MSE เพื่อเห็น trade-off

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['TV']].values
y = df['Sales'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parametric: Linear Regression
lr = LinearRegression().fit(X_train, y_train)
mse_lr = mean_squared_error(y_test, lr.predict(X_test))
print(f'Linear Reg — Test MSE: {mse_lr:.3f}')

# Non-parametric: KNN
for k in [1, 5, 10, 50]:
    knn = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    mse_knn = mean_squared_error(y_test, knn.predict(X_test))
    print(f'KNN k={k:3d} — Test MSE: {mse_knn:.3f}')
```

**ผลที่คาดหวัง:** KNN k=5 หรือ k=10 จะใกล้เคียงกับ Linear Regression สำหรับ near-linear data นี้

---

## Slide 9 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Parametric**: assume f form → estimate parameters → interpretable, needs less data
- **Non-parametric**: no assumption → flexible, needs more data, overfitting risk
- **KNN**: predict = average of K nearest neighbors — K controls flexibility
- **Overfitting**: training MSE ต่ำแต่ test MSE สูง → model จำแทนที่จะเรียนรู้
- **Trade-off**: Flexibility ↑ → Variance ↑, Bias ↓

| | Parametric | Non-parametric |
|--|-----------|--------------|
| Assume f? | Yes | No |
| Data needed | น้อย | มาก |
| Interpretable | Yes | Hard |
| Flexibility | Low | High |
| Overfit risk | Low | High |

**สัปดาห์ถัดไป (Slide Deck 3):** Accuracy-Interpretability Trade-off ในเชิงลึก — flexibility spectrum จาก Linear Regression ถึง Deep Learning
