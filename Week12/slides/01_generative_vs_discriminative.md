# Slide Deck 1: Generative vs Discriminative Models
> Week 12 | 8 slides | CLO3

---

## Slide 1 — Title

**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล — Week 12**  
Generative Models: LDA, QDA, Naive Bayes & KNN  
CLO3: เลือก classifier ที่เหมาะสมกับปัญหาได้

---

## Slide 2 — Week Overview

**Key Message**: Classification มีหลาย approach — รู้จัก assumption ของแต่ละ method ก่อนเลือกใช้

สัปดาห์นี้เราขยาย toolbox ของ classification จาก Logistic Regression ไปสู่ **Generative Models** ซึ่งเป็น family ที่มองปัญหา classification จากมุมมองที่ต่างออกไป แทนที่จะ model Pr(Y|X) โดยตรง เราจะ model Pr(X|Y) ก่อน แล้วใช้ **Bayes Theorem** คำนวณ Pr(Y|X) กลับมา เป้าหมายของสัปดาห์นี้คือให้นักศึกษาเข้าใจ assumption ของ LDA, QDA, Naive Bayes และ KNN และเลือก classifier ที่เหมาะสมกับข้อมูลที่กำหนดได้ ในอุตสาหกรรม Data Science ความเข้าใจ assumption เป็นสิ่งสำคัญ — ใช้ LDA ผิดที่อาจให้ผลแย่กว่า Logistic Regression ทั้งที่ theory ของ LDA แข็งแกร่งกว่า Classifiers เหล่านี้ใช้ในการ detect fraud, diagnose disease, classify email spam และ predict stock direction

- **Discriminative**: Model Pr(Y|X) โดยตรง (Logistic Regression)
- **Generative**: Model Pr(X|Y) + Pr(Y) → Bayes → Pr(Y|X)
- 4 methods สัปดาห์นี้: LDA, QDA, Naive Bayes, KNN
- Comparison table: when to use each

---

## Slide 3 — Two Philosophies of Classification

**Key Message**: Discriminative เรียนรู้ boundary, Generative เรียนรู้ distribution ของแต่ละ class

**Discriminative Approach (Logistic Regression)**
- Model Pr(Y=1|X) โดยตรง
- เรียนรู้ decision boundary เท่านั้น
- ไม่สนใจว่า X distribute อย่างไรใน class แต่ละ class

**Generative Approach (LDA, QDA, NB)**
- Step 1: Model Pr(X|Y=k) — X มี distribution อะไรในแต่ละ class?
- Step 2: Prior Pr(Y=k) = πₖ
- Step 3: Bayes Theorem → Pr(Y=k|X) ∝ Pr(X|Y=k) · πₖ

**[FIGURE: 2D scatter plot แสดง 2 classes — LR วาด boundary เส้นเดียว, LDA วาด 2 Gaussian ellipses + boundary ตรงกลาง]**

---

## Slide 4 — Bayes Theorem for Classification

**Key Message**: Bayes Theorem แปลง Pr(X|Y) ที่ model ได้ กลับเป็น Pr(Y|X) ที่ต้องการ

**Bayes Theorem:**
$$\Pr(Y=k|X=x) = \frac{\Pr(X=x|Y=k) \cdot \Pr(Y=k)}{\sum_{j=1}^{K} \Pr(X=x|Y=j) \cdot \Pr(Y=j)}$$

**3 components:**
| Symbol | ชื่อ | ความหมาย |
|--------|------|---------|
| Pr(Y=k\|X=x) | **Posterior** | สิ่งที่ต้องการ (classify) |
| Pr(X=x\|Y=k) | **Likelihood** | distribution ของ X ใน class k |
| Pr(Y=k) = πₖ | **Prior** | สัดส่วน class k ใน training data |

**[EXAMPLE: ถ้า Pr(X=high_balance\|Default=Yes) = 0.3, Pr(Default=Yes) = 0.033 → Posterior]**

---

## Slide 5 — Why Generative?

**Key Message**: Generative models ดีกว่า Logistic ในสถานการณ์เฉพาะ

**Advantages of Generative Models:**

1. **Stable เมื่อ classes well-separated**
   - Logistic Regression: β̂ → ∞ (numerical instability)
   - LDA: estimate μ̂ₖ, Σ̂ แยกกัน → stable

2. **ดีกว่าเมื่อ n เล็ก**
   - LDA: ต้องการ estimate น้อย parameters กว่า QDA
   - ตัวอย่าง: medical study มี 50 patients per class

3. **Naturally handles multi-class (K > 2)**
   - ขยายสู่ K classes โดยตรง ไม่ต้องทำ one-vs-rest

4. **Probabilistic outputs**
   - Posterior Pr(Y=k|X) ใช้ใน decision-making ที่ต้องการ confidence level

**[FIGURE: timing diagram — LR ดีเมื่อ n ใหญ่, LDA ดีเมื่อ n เล็กหรือ classes well-separated]**

---

## Slide 6 — Overview: 4 Classifiers

**Key Message**: ทุก method ต่างกันที่ assumption เกี่ยวกับ Pr(X|Y)

| Method | Pr(X|Y=k) Assumption | Boundary | Parameters |
|--------|---------------------|----------|-----------|
| **LDA** | N(μₖ, **Σ**) — shared covariance | **Linear** | μₖ, Σ (shared) |
| **QDA** | N(μₖ, **Σₖ**) — per-class covariance | **Quadratic** | μₖ, Σₖ (per-class) |
| **Naive Bayes** | Πⱼ fₖⱼ(xⱼ) — **independent** features | Flexible | μₖⱼ, σ²ₖⱼ |
| **KNN** | ไม่มี distribution assumption | Non-linear | K เท่านั้น |

**[FIGURE: 4 panels แสดง decision boundary ของแต่ละ method บน same 2D data]**

---

## Slide 7 — Connection to Bayes Classifier

**Key Message**: LDA/QDA เป็น approximation ของ Bayes Optimal Classifier

**Bayes Optimal Classifier:**
- Assign observation ไปยัง class k ที่ทำให้ Pr(Y=k|X) สูงสุด
- Error rate ต่ำที่สุดที่เป็นไปได้ (irreducible)
- แต่ Pr(X|Y) จริงๆ ไม่ทราบ → ต้อง estimate

**LDA: ถ้า Gaussian shared Σ เป็นจริง → optimal!**

**LDA error → Bayes error เมื่อ n → ∞**

**[FIGURE: Bayes classifier (dashed), LDA (solid) บน 2D — LDA เข้าใกล้ Bayes เมื่อ sample size เพิ่ม]**

---

## Slide 8 — Summary + Preview

**Key Message**: เข้าใจ assumption ก่อนเลือก method

**สิ่งที่เรียนรู้วันนี้:**
- Discriminative vs Generative: 2 philosophies ของ classification
- Bayes Theorem: แปลง Pr(X|Y) → Pr(Y|X)
- Generative แข็งแกร่งกว่าเมื่อ: n เล็ก, classes well-separated
- Overview: LDA, QDA, NB, KNN ต่างกันที่ assumption เกี่ยวกับ Pr(X|Y)

**สัปดาห์ต่อไป (Week 13):**  
KNN Regression, GLM, ROC Curve และ model comparison ขั้นสูง
