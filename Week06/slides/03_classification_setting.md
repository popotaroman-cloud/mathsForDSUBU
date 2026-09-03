# Slide Deck: The Classification Setting
> Week 6 | CLO2 | ISLP Ch.2.2.3 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 6**  
The Classification Setting: Error Rate, Bayes Classifier, KNN  
CLO2: อธิบาย Error Rate, Bayes Classifier และผลของ K ต่อ KNN ได้

---

## Slide 2 — จาก MSE สู่ Error Rate
**Key Message**: ใน Classification เราใช้ Error Rate แทน MSE — แต่แนวคิด Bias-Variance Trade-Off ยังคงเดิม

ใน Deck 1 และ 2 เราพูดถึง Regression ที่ Y เป็น quantitative (ตัวเลข) ใช้ MSE เป็น metric ใน Classification Y เป็น qualitative (class label) เช่น spam/ham, malignant/benign, cat/dog ดังนั้น metric เปลี่ยนเป็น **Error Rate** แต่ปรัชญา Bias-Variance Trade-Off ยังคงเหมือนเดิม — model ที่ complex เกินไปจะ overfit ส่วน model ที่ง่ายเกินไปจะ underfit

**Training Error Rate:**
```
Error_train = (1/n) Σᵢ I(yᵢ ≠ ŷᵢ)

I(yᵢ ≠ ŷᵢ) = 1 ถ้า predict ผิด
             = 0 ถ้า predict ถูก
```

**Test Error Rate:**
```
Error_test = average I(y₀ ≠ ŷ₀) บน test observations
→ สิ่งที่เราต้องการ minimize
```

**เชื่อมกับ MSE:**  
Error Rate สามารถมองเป็น "0-1 MSE" เพราะ (0−1)² = 1 เมื่อผิด, (1−1)² = 0 เมื่อถูก

---

## Slide 3 — Bayes Classifier: Theoretical Optimal
**Key Message**: Bayes Classifier คือ classifier ที่ดีที่สุดในเชิงทฤษฎี — assign class ที่มี posterior probability สูงสุด

**Bayes Classifier:**
```
ŷ = arg max_j Pr(Y = j | X = x₀)

สำหรับ binary (2 class):
ŷ = Class 1 ถ้า Pr(Y=1|X=x₀) > 0.5
ŷ = Class 2 ถ้า Pr(Y=1|X=x₀) ≤ 0.5
```

**ทำไม "optimal"?**
- Assign class ที่ "น่าจะเป็นที่สุด" ณ แต่ละจุด x₀
- ได้รับการพิสูจน์ว่า minimize Test Error Rate
- Bayes Decision Boundary = เส้นแบ่ง class ตาม Pr(Y=j|X)

**Bayes Error Rate:**
```
1 − E[max_j Pr(Y = j | X)]

= irreducible error ของ Classification
= ต่ำที่สุดที่ classifier ใดๆ ทำได้ (analogous กับ Var(ε))
```

**ปัญหาในทางปฏิบัติ:** เราไม่รู้ Pr(Y=j|X) จริงๆ → ต้องประมาณ

---

## Slide 4 — Bayes Decision Boundary
**Key Message**: Bayes Boundary แบ่ง feature space ตาม posterior probability — ไม่เป็น linear เสมอไป

**[FIGURE: 2D scatter plot แสดง 2 classes + Bayes Decision Boundary (เส้นโค้ง)]**

```
สีม่วง: Class 1 observations
สีเขียว: Class 2 observations

Bayes Boundary (เส้นประ): Pr(Y=1|X) = 0.5
→ ขวาของเส้น: Class 1 (probability สูงกว่า)
→ ซ้ายของเส้น: Class 2 (probability สูงกว่า)

Bayes Error Rate = 0.133 (ตัวอย่างจาก ISLP Figure 2.13)
```

**ความสำคัญ:**
- Bayes Boundary คือ "เฉลย" ที่สมบูรณ์แบบ
- ทุก classifier พยายาม approximate Bayes Boundary นี้
- KNN, Logistic Regression, SVM = วิธีต่างๆ ในการประมาณ boundary นี้

---

## Slide 5 — KNN Classifier: ประมาณ Pr(Y=j|X)
**Key Message**: KNN ประมาณ posterior probability จาก K neighbors ที่ใกล้ที่สุด — นำ idea จาก regression มาสู่ classification

**KNN Classifier Algorithm:**
```
Given: x₀ (test point), training data {(xᵢ, yᵢ)}

Step 1: หา K observations ที่ใกล้ x₀ มากที่สุด → N₀
Step 2: ประมาณ probability สำหรับแต่ละ class j:
        Pr̂(Y = j | X = x₀) = (1/K) Σᵢ∈N₀ I(yᵢ = j)

Step 3: ŷ = class j ที่มี Pr̂ สูงสุด
```

**ตัวอย่าง K=3:**
```
x₀ = new test point
3 neighbors ที่ใกล้สุด: {Class 1, Class 1, Class 2}

Pr̂(Y=1|X=x₀) = 2/3 ≈ 0.67
Pr̂(Y=2|X=x₀) = 1/3 ≈ 0.33

ŷ = Class 1  (probability สูงกว่า)
```

---

## Slide 6 — ผลของ K ต่อ Decision Boundary
**Key Message**: K เล็ก → boundary ซับซ้อน (overfit), K ใหญ่ → boundary เรียบ (underfit) — อยู่ใน Bias-Variance framework เดิม

**[FIGURE: 3 panels แสดง KNN decision boundary สำหรับ K=1, K=10, K=100 บน dataset เดิม]**

```
K=1   → Decision Boundary ซับซ้อนมาก, เป็น "islands" รอบทุกจุด training
        Train Error = 0%, Test Error สูง ← Overfit

K=10  → Boundary smooth กว่า, ใกล้เคียง Bayes Boundary
        Train Error ต่ำพอสมควร, Test Error ต่ำ ← Near Optimal

K=100 → Boundary เรียบมาก, classify ทุกจุดเป็น class เดียว
        Train Error ≈ Test Error สูง ← Underfit
```

**Pattern เหมือน polynomial degree ทุกประการ:**
- K เล็ก = high flexibility = high Variance
- K ใหญ่ = low flexibility = high Bias

---

## Slide 7 — KNN: Train/Test Error vs K
**Key Message**: Plot Error Rate vs K แสดง U-curve ชัดเจน — พิสูจน์ว่า Bias-Variance Trade-Off ใช้ได้กับ Classification ด้วย

**[FIGURE: Train Error (น้ำเงิน) และ Test Error (แดง) vs K บน log scale]**

```
แกน X: K (1/K = flexibility, K=1 rightmost = most flexible)
แกน Y: Error Rate

Train Error: ต่ำที่ K=1 (0%) เพิ่มขึ้นเมื่อ K เพิ่ม
Test Error:  สูงที่ K=1 (overfit), ลงจนถึง min แล้วขึ้น (U-shape)
Bayes Error: dashed line = irreducible lower bound
```

**ผลจาก ISLP (Figure 2.16):**

| K | Train Error | Test Error |
|---|------------|----------|
| 1 | 0% | 16.7% |
| 10 | 4.5% | 11.5% ← near optimal |
| 100 | 13.9% | 19.4% |
| Bayes | — | 10.4% ← lower bound |

---

## Slide 8 — เลือก K อย่างไร?
**Key Message**: ไม่มีสูตรตายตัวสำหรับ K — ใช้ Cross-Validation เพื่อเลือก K ที่ minimize Test Error

**วิธีที่ผิด:**
```
❌ เลือก K ที่ minimize Train Error
→ จะเลือก K=1 เสมอ (Train Error = 0%)
→ Overfit ร้ายแรง
```

**วิธีที่ถูก (Preview Week 14):**
```
✓ K-Fold Cross-Validation:
  1. แบ่ง training set เป็น 5 ส่วน
  2. ทดสอบแต่ละ K บน validation fold
  3. เลือก K ที่ minimize CV Error
```

**Rule of Thumb สำหรับ Exploration:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ─── หา optimal K ด้วย 5-fold CV ──────────────────────────
# วัตถุประสงค์: estimate test error โดยไม่ต้องใช้ test set
K_values = range(1, 31)
cv_errors = []
for k in K_values:
    knn   = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_errors.append(1 - score.mean())

optimal_K = K_values[cv_errors.index(min(cv_errors))]
```

---

## Slide 9 — Case Study: Spam Classification กับ KNN
**Key Message**: KNN บน spam dataset แสดงว่า optimal K ≈ 7–15 ขึ้นกับ feature representation

**Scenario:** Email spam classification  
**Data:** 5,000+ emails, X = TF-IDF features (50 dimensions)  
**Method:** KNN Classifier

**ผลการทดสอบ K ต่างๆ:**

| K | Train Accuracy | Test Accuracy |
|---|--------------|-------------|
| 1 | 100% | 83.2% — Overfit |
| 5 | 97.3% | 88.4% |
| 10 | 95.1% | 89.7% ← near optimal |
| 20 | 92.8% | 89.2% |
| 50 | 88.5% | 85.1% — Underfit |

**ข้อสังเกต:**
- KNN accuracy ≈ 89.7% กับ K=10
- Logistic Regression ≈ 97% — ชนะ KNN บน text data
- เหตุผล: text features มี high dimension → KNN suffer จาก curse of dimensionality
- เชื่อมกับ Week 4 PCA: ลด dimension ก่อน KNN จะช่วยได้

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Error Rate**: metric ของ classification = fraction ที่ predict ผิด
- **Bayes Classifier**: optimal classifier ที่ assign class ด้วย max posterior probability
- **Bayes Error Rate**: irreducible error ของ classification (analog กับ Var(ε))
- **KNN Classifier**: ประมาณ Pr(Y=j|X) จาก K nearest neighbors
- **K Effect**: K เล็ก=overfit, K ใหญ่=underfit, U-curve ของ Test Error

```
K → ∞: Bayes Classifier? No! → ทำนาย majority class เสมอ (oversmooth)
K = 1: Voronoi Tessellation — each point is its own region
K optimal: found by Cross-Validation (Week 14)
```

| Method | Analogy | Hyperparameter |
|--------|---------|---------------|
| Polynomial Regression | — | degree d |
| KNN Regression | — | K |
| KNN Classification | ← today | K |
| จะเรียนต่อไป | Logistic Regression, LDA | — |

**สัปดาห์ถัดไป (Slide Deck 4):** Probability Foundations — Conditional Probability, Bayes Theorem, Key Distributions
