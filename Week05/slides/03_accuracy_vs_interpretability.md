# Slide Deck: Prediction Accuracy vs Model Interpretability
> Week 5 | CLO2 | ISLP Ch.2.1.3 | 9 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 5**  
Accuracy vs Interpretability: ไม่มี Free Lunch ใน Machine Learning  
CLO2: อธิบาย flexibility spectrum และเลือก model ตาม goal ได้

---

## Slide 2 — The Fundamental Trade-off
**Key Message**: Model ที่ flexible มาก predict แม่นกว่าในหลายกรณี แต่อธิบายได้ยากขึ้น — นี่คือ trade-off พื้นฐานที่ทุก Data Scientist ต้องเข้าใจ

ใน Statistical Learning มี trade-off สำคัญระหว่าง "ความแม่นยำในการ predict" และ "ความสามารถในการอธิบาย model" model ที่ flexible (เช่น Deep Neural Network) มักให้ prediction แม่นกว่าสำหรับ complex problems แต่ interpret ได้ยากมาก ส่วน model ที่ simple (เช่น Linear Regression) อธิบายได้ชัดเจนแต่อาจ predict ได้ไม่แม่นเท่า การเลือกใช้ขึ้นอยู่กับว่า goal คือ Prediction (แม่นยำสูงสุด) หรือ Inference (เข้าใจ mechanism) ตัวอย่างเช่น แพทย์ต้องการ interpretable model ที่อธิบายได้ว่า "ทำไมถึง diagnose ว่าเป็นมะเร็ง" แต่ Netflix สนแค่ว่า "recommend หนังอะไรที่คนจะดู" เป้าหมายของ deck นี้คือให้นักศึกษาเข้าใจ spectrum นี้และเลือก model type ได้ถูกต้อง

**[FIGURE: spectrum diagram: Linear Reg → Lasso → GAM → Tree → Bagging/RF → SVM → Neural Network, แกน x = Flexibility, แกน y = Interpretability (inverse)]**

---

## Slide 3 — Flexibility Spectrum
**Key Message**: ML methods เรียงจาก least flexible (most interpretable) ไปถึง most flexible (least interpretable)

**Spectrum จาก ISLP (least → most flexible):**
```
LEAST FLEXIBLE (Most Interpretable)
    Linear Regression       — β per feature
    Lasso / Ridge           — regularized linear
    Logistic Regression     — for classification
    Linear Discriminant Analysis (LDA)
    GAM (Generalized Additive Models)
    Decision Trees          — interpretable rules
    Bagging, Random Forests
    Boosting (GBM, XGBoost)
    Support Vector Machines
    Deep Neural Networks    — most flexible
MOST FLEXIBLE (Least Interpretable)
```

**ข้อสังเกต:**
- ความ flexible ไม่ได้แปลว่า "ดีกว่า" — ขึ้นกับ problem
- Highly flexible + insufficient data = overfitting
- Simple model + correct assumption = can beat complex model

---

## Slide 4 — Interpretability: Why Does It Matter?
**Key Message**: ในหลาย domain ความสามารถในการอธิบาย model สำคัญกว่า accuracy เล็กน้อย — ตัวอย่างที่ชัดเจน: medical, legal, financial

**เมื่อ Interpretability สำคัญมาก:**

**Medical Diagnosis:**
```
แพทย์: "ทำไม model ถึงบอกว่าผู้ป่วยเป็นมะเร็ง?"
→ ต้องอธิบายได้ (มิฉะนั้นแพทย์ไม่เชื่อ)
→ Linear/Logistic Regression: "PSA level >=4 + age >=60 → risk"
→ Neural Network: อธิบายไม่ได้ → แพทย์ไม่เชื่อถือ
```

**Legal / Financial:**
- Basel III: ธนาคารต้องอธิบาย credit decision ได้
- GDPR (EU): ผู้ใช้มีสิทธิ์ขอคำอธิบาย automated decision
- "Model ทำไมถึง reject loan ผม?" → ต้องตอบได้

**เมื่อ Interpretability สำคัญน้อยกว่า:**
- Spam filter: แค่ต้องการ accuracy สูง
- Image recognition: ไม่มีใครถาม "ทำไมเห็นแมว"
- Stock trading: แค่ต้องการ profit

---

## Slide 5 — Bias-Variance Conceptual Preview
**Key Message**: Flexibility ↑ → Bias ↓ แต่ Variance ↑ — optimal model อยู่ที่จุด trade-off ที่ minimize test error

**Conceptual picture (ดูรายละเอียดใน Week 6):**
```
Test MSE ≈ Bias²(f̂) + Variance(f̂) + Var(ε)
              ↑              ↑           ↑
         ลดด้วย         ลดด้วย     ลดไม่ได้
         Flexibility    Simplicity
```

**[FIGURE: U-shape ของ test MSE เป็น function ของ flexibility:
- ซ้าย: underfitting (high bias, low variance)
- กลาง: optimal flexibility
- ขวา: overfitting (low bias, high variance)]**

**ใน practice:**
- เราไม่รู้ exact U-shape นี้
- Cross-Validation ช่วยหา optimal flexibility (Week 14)

---

## Slide 6 — Case Study: When Simple Beats Complex
**Key Message**: งานวิจัยหลายชิ้นพบว่า simple model ชนะ complex model เมื่อ data น้อยหรือ signal-to-noise ratio ต่ำ

**Scenario: Advertising → Sales (200 observations)**

| Model | Test MSE | Interpretable? | # Parameters |
|-------|---------|----------------|-------------|
| Linear Reg (TV only) | ≈11.0 | ✓ Yes | 2 |
| Linear Reg (all 3) | ≈1.7 | ✓ Yes | 4 |
| KNN k=1 | ≈17.0 | ✗ No | — |
| KNN k=5 | ≈9.5 | ✗ No | — |
| KNN k=10 | ≈10.3 | ✗ No | — |

**สรุป**: สำหรับ dataset ขนาดเล็กนี้ Multiple Linear Regression ชนะ KNN ทุก k  
→ เพราะ TV→Sales ใกล้ linear จริงๆ

**Lesson**: เริ่มด้วย simple model เสมอ ถ้า simple ไม่พอค่อยขยับไป complex

---

## Slide 7 — Explainable AI (XAI): กลาง?
**Key Message**: XAI ให้ "เหตุผล" สำหรับ complex model โดยไม่ต้องลด complexity — แต่ยังเป็น approximation ไม่ใช่ true explanation

**ปัญหา:** Deep Learning แม่นยำ แต่ interpret ไม่ได้  
**วิธีแก้ (partial):** Explainable AI techniques

**SHAP (SHapley Additive exPlanations):**
```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
# แสดง: "feature X₁ เพิ่ม prediction ขึ้น +0.5, X₂ ลด -0.3, ..."
```

**LIME:** อธิบาย individual prediction ด้วย local linear approximation

**ข้อจำกัด:**
- SHAP/LIME เป็น approximation ไม่ใช่ true mechanism
- Regulatory standard ส่วนใหญ่ยังต้องการ inherently interpretable model
- "Explainability" ≠ "Transparency"

---

## Slide 8 — Decision Guide: เลือก Model อย่างไร
**Key Message**: Framework 4 คำถามสำหรับเลือก model — เริ่มจาก simplest ที่ work แล้วค่อย scale up

**Framework สำหรับเลือก Method:**
```
คำถาม 1: Goal คืออะไร?
├─ Inference → เลือก interpretable (Linear/Logistic)
└─ Prediction → ไปคำถาม 2

คำถาม 2: Data มีมากพอไหม?
├─ < 1000 obs → เลือก parametric (regularized linear)
└─ > 10000 obs → อาจลองสิ่งที่ flexible กว่า

คำถาม 3: f ซับซ้อนแค่ไหน?
├─ Near-linear → Linear works
└─ Nonlinear → Tree, SVM, Neural Net

คำถาม 4: Explainability required?
├─ Yes (medical, legal, finance) → Linear/Tree
└─ No (image, text, game) → Deep Learning OK
```

---

## Slide 9 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**
- **Flexibility spectrum**: Linear Reg (interpretable) → Deep Learning (flexible)
- **Trade-off**: ↑ Flexibility → ↓ Bias แต่ ↑ Variance → U-shaped test error
- **Interpretability matters**: medical, legal, financial domains require it
- **No Free Lunch**: ไม่มี model ที่ดีที่สุดสำหรับทุก problem
- **Golden rule**: เริ่มจาก simple model ก่อนเสมอ

| Goal | Recommended Model |
|------|-----------------|
| Inference + small data | Linear/Logistic Regression |
| Prediction + tabular data | Tree methods, Gradient Boosting |
| Prediction + image/text | Neural Networks |
| Explainability required | Linear, Decision Tree |

**สัปดาห์ถัดไป (Slide Deck 4):** Supervised vs Unsupervised + Regression vs Classification — จำแนก ML problems ทั้งหมด
