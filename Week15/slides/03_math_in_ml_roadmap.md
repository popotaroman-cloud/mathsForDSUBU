# Slide Deck: คณิตศาสตร์ใน ML — Roadmap สู่อนาคต
> Week 15 | CLO1+2+3+4 | Career Preparation | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 15**  
คณิตศาสตร์ใน ML Roadmap: วิชานี้เป็นจุดเริ่มต้นของอะไร?  
CLO1+2+3+4: เชื่อมสิ่งที่เรียนกับ ML สายอาชีพและวิชาถัดไป

---

## Slide 2 — ภาพรวม: Math as a Foundation
**Key Message**: คณิตศาสตร์ที่เรียนในวิชานี้เป็น "layer 0" ของ Data Science stack ทั้งหมด — ทุก ML algorithm สร้างบนพื้นฐานนี้

```
╔══════════════════════════════════════╗
║  Advanced AI (GPT, Diffusion, RL)    ║  ← ปีที่ 4
╠══════════════════════════════════════╣
║  Deep Learning (CNN, RNN, GAN)       ║  ← ปีที่ 3-4
╠══════════════════════════════════════╣
║  Machine Learning (SVM, Boosting,    ║  ← ปีที่ 3
║  Clustering, Neural Networks)        ║
╠══════════════════════════════════════╣
║  Statistical Learning (วิชานี้ W5-15)║  ← ปีที่ 2 ✅
╠══════════════════════════════════════╣
║  Linear Algebra + Stats (วิชานี้ W1-7)║  ← ปีที่ 2 ✅
╚══════════════════════════════════════╝
```

ทุก algorithm ที่เรียนในอนาคตใช้ Vector, Matrix, Eigenvalue, Probability, และ Optimization — ทั้งหมดนี้เรียนไปแล้วในวิชานี้

---

## Slide 3 — CLO1 → Future: Linear Algebra ใช้ที่ไหนต่อ
**Key Message**: Linear Algebra ปรากฏในทุก advanced ML algorithm — เรียนแล้วใช้จริงตลอด

**Neural Networks:**
```
Forward pass: a⁽¹⁾ = σ(W⁽¹⁾x + b⁽¹⁾)   ← Matrix multiplication
Backprop: δ = Wᵀδ × σ'(z)              ← Transpose + chain rule
```

**SVD / PCA → Deep Learning:**
```
Attention (Transformer): QKᵀ/√d   ← Dot product + scaling
Low-rank approximation → LoRA ในการ fine-tune LLM
```

**Graph Neural Networks:**
```
Adjacency matrix A → eigenvalues → Graph Fourier Transform
PageRank → eigenvector ของ transition matrix
```

**Computer Vision:**
```
Convolution = กรณีพิเศษของ matrix multiplication
Image compression = SVD low-rank approximation (ที่เรียน Week 4!)
```

---

## Slide 4 — CLO2 → Future: Statistical Learning ใช้ที่ไหนต่อ
**Key Message**: Bias-Variance Trade-Off เป็น central theme ที่ reappear ใน advanced ML ทุกรูปแบบ

**Regularization:**
```
L1 (Lasso): ||β||₁ penalty → sparse features → feature selection
L2 (Ridge): ||β||₂² penalty → shrink coefficients → reduce variance
ElasticNet: L1 + L2 combined

เชื่อมกับ: Bias-Variance Trade-Off (Week 6) — regularization เพิ่ม Bias เพื่อลด Variance
```

**Ensemble Methods:**
```
Bagging: train หลาย models → average → ลด Variance
Boosting: แก้ error ของ model ก่อน → ลด Bias
Random Forest, XGBoost, LightGBM → ขยายจาก Decision Tree
```

**Deep Learning:**
```
Dropout → เทียบได้กับ Bagging (random subset of neurons)
Early Stopping → หยุดก่อน overfit ← ใช้ Test MSE monitor
Batch Normalization → stabilize training
```

---

## Slide 5 — CLO3 → Future: Regression/Classification ใช้ที่ไหนต่อ
**Key Message**: Regression และ Logistic Regression เป็น building blocks ของ advanced models

**จาก Linear → Neural Network:**
```
Linear Regression:    y = Xβ
                           ↓
Logistic Regression:  y = σ(Xβ)
                           ↓
Neural Network:       y = σₙ(Wₙ...σ₁(W₁X + b₁)...+ bₙ)
                           ↑ หลาย layers ของ logistic หน่วย
```

**จาก Logistic → Text Classification:**
```
Bag-of-Words + Logistic = baseline text classifier
TF-IDF features + Logistic = spam filter (ISLP Case Study)
BERT + Logistic/Softmax head = state-of-art text classifier
```

**Time Series:**
```
Autoregression: yₜ = β₀ + β₁yₜ₋₁ + β₂yₜ₋₂ + ε  ← MLR in time
ARIMA, LSTM → extensions of linear regression in time
```

---

## Slide 6 — CLO4 → Future: Model Selection ใช้ที่ไหนต่อ
**Key Message**: Cross-Validation เป็นหัวใจของ Hyperparameter Tuning ใน ML ทุกระดับ

**Hyperparameter Tuning:**
```
Grid Search CV:   ทดสอบทุก combination ของ hyperparameters
Random Search CV: sample แบบ random → ประหยัดกว่า
Bayesian Search:  ใช้ probabilistic model guide การค้นหา
```

**AutoML:**
```
AutoML platforms (AutoSklearn, TPOT, H2O) ใช้ CV เพื่อ:
- เลือก model type อัตโนมัติ
- Tune hyperparameters อัตโนมัติ
- Feature engineering อัตโนมัติ
```

**MLOps:**
```
A/B Testing = Cross-Validation ใน production
Model monitoring = detect เมื่อ test distribution shift
Retraining triggers = ใช้ CV เปรียบเทียบ old vs new model
```

---

## Slide 7 — Math Requirements สำหรับวิชาถัดไป
**Key Message**: วิชา ML/Deep Learning ขั้นสูงต้องการ math เพิ่มเติม — รู้ไว้ล่วงหน้าเพื่อเตรียมตัว

**วิชาถัดไปและ math ที่ต้องการ:**

| วิชา | Linear Algebra | Statistics | Calculus/Optimization | Probability |
|-----|--------------|-----------|----------------------|------------|
| Machine Learning | ✅ ใช้มาก | ✅ ใช้มาก | ⬜ ต้องเพิ่ม | ✅ ใช้มาก |
| Deep Learning | ✅ ใช้มาก | ✅ medium | ✅ ต้องการมาก | ✅ ใช้มาก |
| NLP | ✅ ใช้มาก | ✅ medium | ✅ medium | ✅ ใช้มาก |
| Computer Vision | ✅ ใช้มาก | ✅ medium | ✅ ต้องการมาก | ✅ medium |

**สิ่งที่ควรเรียนเพิ่ม:**
- **Calculus + Gradient Descent**: วิธีที่ Neural Networks เรียนรู้ (Backpropagation)
- **Convex Optimization**: SVM, Lasso
- **Information Theory**: Entropy, KL Divergence → ใช้ใน Decision Trees, VAE

---

## Slide 8 — Career Paths: คณิตศาสตร์ใช้ในอาชีพไหน
**Key Message**: Data Science มีหลาย role และแต่ละ role ใช้ math ต่างกัน

| Role | Linear Algebra | Stats/ML | Coding | Domain |
|------|--------------|---------|--------|--------|
| **Data Analyst** | medium | ✅ EDA/Stats | SQL, Python | ✅ Strong |
| **ML Engineer** | ✅ ใช้บ่อย | ✅ ✅ | ✅ Strong | medium |
| **Data Scientist** | ✅ ใช้บ่อย | ✅ ✅ | ✅ Strong | ✅ Strong |
| **ML Researcher** | ✅ ✅ ขั้นสูง | ✅ ✅ ขั้นสูง | ✅ Strong | medium |
| **AI Consultant** | medium | ✅ ML overview | medium | ✅ ✅ Strong |

**สิ่งที่ทุก role ต้องมี:**
- ✅ เข้าใจ Bias-Variance Trade-Off (รู้จัก overfitting)
- ✅ ประเมิน model ด้วย CV ได้ (ไม่ใช้แค่ training accuracy)
- ✅ อ่าน correlation/regression output ได้ถูกต้อง
- ✅ อธิบาย model ให้ non-technical stakeholders ได้

---

## Slide 9 — Tools Ecosystem
**Key Message**: คณิตศาสตร์คือ "why" — tools เป็นแค่ "how" ที่เปลี่ยนได้ตลอด

```
MATH ──────────────────────────────────── TOOLS

Linear Algebra ──────────────────── NumPy, PyTorch, JAX
Statistics ──────────────────────── scipy.stats, statsmodels
ML Framework ───────────────────── scikit-learn, XGBoost
Deep Learning ───────────────────── PyTorch, TensorFlow, Keras
Data Manipulation ───────────────── pandas, polars
Visualization ───────────────────── matplotlib, seaborn, plotly
Deployment ──────────────────────── FastAPI, Streamlit, Docker
Cloud ────────────────────────────── AWS SageMaker, GCP Vertex
```

**Philosophy:** เรียน Math ลึก แต่เรียน Tools แบบ breadth — tools เปลี่ยนทุก 2–3 ปี แต่ math เปลี่ยนช้ามาก

---

## Slide 10 — Summary: What You Now Have
**สิ่งที่นักศึกษาได้จากวิชานี้**

```
Technical Skills:
✅ Linear Algebra → PCA, matrix ops, eigenvalues
✅ Statistical Learning → MSE, Bias-Variance, EDA  
✅ Regression/Classification → อ่าน/ตีความผลได้
✅ Model Selection → CV, bootstrap

Conceptual Skills:
✅ Formulate ML problems (X, Y, f, ε)
✅ Diagnose model issues (overfit/underfit)
✅ เลือก model type ที่เหมาะกับปัญหา
✅ อธิบาย "ทำไม" ไม่ใช่แค่ "อย่างไร"

Practical Skills:
✅ Python (NumPy, pandas, sklearn, matplotlib)
✅ Jupyter Notebook workflow
✅ Data storytelling
```

**สัปดาห์ถัดไป (Slide Deck 4):** แหล่งเรียนรู้เพิ่มเติม, Certificate paths, Career roadmap
