# Note 15-1: สรุปรายวิชา — Mathematics for Data Science ทั้งหมด
> Week 15 | CLO1–4 | ภาพรวมและ Connections ระหว่าง Topics ทั้ง 14 สัปดาห์

สัปดาห์สุดท้ายของรายวิชา 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล เราจะหยุดมองย้อนกลับไปตลอดเส้นทาง 14 สัปดาห์ที่ผ่านมา เพื่อเห็นภาพรวมว่า Mathematics, Statistics, และ Machine Learning เชื่อมกันอย่างไร Note นี้ไม่ใช่การท่องซ้ำ แต่เป็นการมองเห็น **Big Picture** — ว่าทำไม Linear Algebra จากสัปดาห์ที่ 2-4 ถึงปรากฏในทุก Algorithm ของ ML, ว่า Bias-Variance จาก Week 6 เป็น Framework ที่ตอบทุกคำถามเรื่อง Model Complexity, และว่า Cross-Validation จาก Week 14 คือสะพานเชื่อมทฤษฎีกับ Practice เป้าหมายของ Note นี้คือให้นักศึกษามองเห็น Unified Story ของรายวิชาและสามารถอธิบาย Connection ระหว่าง Topics ได้ ซึ่งสำคัญมากสำหรับ Final Project Presentation และการเรียน Machine Learning ในอนาคต

---

## ภาพรวม: 4 CLO และ Journey ของรายวิชา

```
Week 01    Week 02-04         Week 05-07          Week 08-14
   │           │                  │                    │
   ▼           ▼                  ▼                    ▼
Vectors   Linear Algebra    Statistical          Models & Validation
          (Matrices, SVD,   Learning             (Regression,
          Eigenvalues,       Framework            Classification,
          PCA)               (Bias-Variance,      CV, Bootstrap)
                             EDA, Inference)
   └─────────── CLO1 ─────────┘└─ CLO2 ──┘└────── CLO3+4 ──────┘
```

---

## CLO1: เครื่องมือ Linear Algebra  *(Weeks 1–4)*

### Weeks 1–4: จาก Vector สู่ SVD

**สาระหลัก 4 สัปดาห์แรก**:

| Week | หัวข้อ | สิ่งที่เรียนรู้ | ใช้ใน ML ที่ไหน? |
|------|--------|--------------|---------------|
| 1 | Vectors | ข้อมูลแต่ละ row คือ Vector ใน Feature Space | Feature representation, Distance |
| 2 | Matrix & Gauss | Ax=b คือ linear system ทุก ML model แก้ | OLS Normal Equations, Neural Net weights |
| 3 | Subspaces & OLS | Column Space คือ space ของทุก prediction ที่เป็นไปได้ | Least Squares, Rank conditions |
| 4 | Eigenvalues, SVD, PCA | A = UΣVᵀ คือ x-ray ของ matrix ใดก็ได้ | PCA, Dimensionality Reduction, Recommender |

### Connection สำคัญ: Normal Equations

สูตร $\hat{\beta} = (X^TX)^{-1}X^Ty$ ปรากฏซ้ำในทุกที่:
- **Week 3**: Projection onto column space
- **Week 8-9**: OLS Regression coefficients  
- **Week 14**: Ridge Regression เปลี่ยนเป็น $(X^TX + \lambda I)^{-1}X^Ty$

### Connection สำคัญ: PCA และ Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ─── แสดง Connection ระหว่าง PCA (Week 4) และ Multicollinearity (Week 10) ─
# วัตถุประสงค์: PCA แก้ Multicollinearity ได้เพราะ PCs orthogonal กัน (VIF=1)
auto = pd.read_csv('Auto.csv', na_values='?').dropna()
X_auto = auto[['horsepower', 'weight', 'cylinders', 'displacement']]

# ตรวจสอบ Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_sc = StandardScaler().fit_transform(X_auto)
X_sm = pd.DataFrame(X_sc, columns=X_auto.columns)
X_sm_const = sm.add_constant(X_sm)

print("VIF before PCA:")
for i, col in enumerate(X_auto.columns):
    vif = variance_inflation_factor(X_sm_const.values, i + 1)
    print(f"  {col}: VIF = {vif:.2f}")

# PCA แก้ปัญหา
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_sc)
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(4)])
X_pca_const = sm.add_constant(X_pca_df)

print("\nVIF after PCA (should be ≈ 1 for all PCs):")
for i, col in X_pca_df.items():
    vif = variance_inflation_factor(X_pca_const.values, int(i[-1]))
    print(f"  PC{i[-1]}: VIF = {vif:.2f}")

print(f"\nExplained variance ratio: {pca.explained_variance_ratio_.round(3)}")
```

---

## CLO2: Statistical Learning Framework  *(Weeks 5–7)*

### Weeks 5–7: จาก Concept สู่ Inference

**Central Idea**: เราไม่รู้ $f(X)$ ที่แท้จริง เราแค่มี $Y = f(X) + \varepsilon$ และข้อมูล

| Week | หัวข้อ | สิ่งที่เรียนรู้ | ใช้ใน ML ที่ไหน? |
|------|--------|--------------|---------------|
| 5 | SL Overview | Supervised vs Unsupervised, Parametric vs Non-parametric | Framework ทุกอย่าง |
| 6 | Bias-Variance | $\text{MSE} = \text{Bias}^2 + \text{Var} + \sigma^2$ | เลือก Model, Regularization |
| 7 | EDA + Inference | Statistics เป็นภาษา, p-value มีความหมาย | Exploratory Analysis, A/B Testing |

### Central Theorem: Bias-Variance Decomposition

$$E\left[(y_0 - \hat{f}(x_0))^2\right] = \underbrace{\text{Var}(\hat{f}(x_0))}_{\text{Variance}} + \underbrace{[\text{Bias}(\hat{f}(x_0))]^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\varepsilon)}_{\text{Irreducible}}$$

**นี่คือ fundamental tradeoff ที่อยู่เบื้องหลังทุก decision ใน ML**:
- Model ง่ายเกินไป → High Bias (Underfitting)
- Model ซับซ้อนเกินไป → High Variance (Overfitting)
- Cross-Validation (Week 14) ช่วยหา sweet spot

```python
# ─── Visualize Bias-Variance Trade-Off ────────────────────────────────────
# วัตถุประสงค์: ทบทวน central concept ที่ใช้อธิบาย EVERY model decision
# เชื่อม Week 6 กับ Week 14 (CV เป็นวิธีค้นหา optimal complexity)
np.random.seed(0)
x = np.linspace(0, 1, 100)
true_f = np.sin(2 * np.pi * x)

complexities = np.array([1, 2, 3, 5, 8, 15])
n_sim = 50

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, degree in zip(axes.flatten(), complexities):
    predictions = []
    for _ in range(n_sim):
        x_sim = np.random.uniform(0, 1, 30).reshape(-1, 1)
        y_sim = np.sin(2 * np.pi * x_sim.ravel()) + np.random.normal(0, 0.3, 30)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        pipe = Pipeline([('poly', PolynomialFeatures(degree)), ('lr', LinearRegression())])
        pipe.fit(x_sim, y_sim)
        predictions.append(pipe.predict(x.reshape(-1, 1)))
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    
    ax.plot(x, true_f, 'k-', linewidth=2, label='True f(x)')
    for pred in predictions[:10]:
        ax.plot(x, pred, 'b-', alpha=0.1, linewidth=0.5)
    ax.plot(x, mean_pred, 'r-', linewidth=2, label='E[f̂(x)]')
    
    bias2 = np.mean((mean_pred - true_f) ** 2)
    variance = np.mean(predictions.var(axis=0))
    ax.set_title(f'Degree {degree}\nBias²={bias2:.3f}, Var={variance:.3f}')
    ax.set_ylim([-2, 2])
    ax.legend(fontsize=7)

plt.suptitle('Bias-Variance Decomposition Across Model Complexity', fontsize=13)
plt.tight_layout(); plt.show()
```

---

## CLO3: Regression & Classification  *(Weeks 8–13)*

### Weeks 8–13: Family ของ Models

**Progression**:

```
Week 8: SLR (1 predictor, continuous Y)
    ↓
Week 9: MLR (p predictors, continuous Y)
    ↓  
Week 10: Extensions (Qualitative X, Interactions, Poly) + Diagnostics
    ↓
Week 11: Logistic Regression (Binary Y)
    ↓
Week 12: LDA, QDA, Naive Bayes, KNN (Alternative Classifiers)
    ↓
Week 13: GLM (Unified framework) + ROC + Full Pipeline
```

### The Unified Story: Link Functions

| Model | Distribution | Link | Mean function |
|-------|-------------|------|--------------|
| Linear Regression | Normal | Identity | $\mu = X\beta$ |
| Logistic Regression | Binomial | Logit | $\mu = \sigma(X\beta)$ |
| Poisson Regression | Poisson | Log | $\mu = e^{X\beta}$ |
| LDA | Normal (same Σ) | — | Bayes-optimal boundary |
| KNN | Non-parametric | — | Local majority vote |

### 4 Common Questions Across All Models

ทุก model จาก Week 8–13 ตอบคำถามเดิม 4 ข้อ:

| Question | Linear Regression | Logistic Regression |
|----------|-------------------|---------------------|
| **Q1: Relationship exists?** | F-statistic | Chi-square test of deviance |
| **Q2: Which predictors matter?** | t-statistic, Adj R², AIC | z-statistic, AIC |
| **Q3: How well does it fit?** | R², RSE | Accuracy, AUC, Deviance |
| **Q4: Predict new obs?** | CI vs PI | P̂(Y=1|X), class label |

```python
# ─── สรุป Workflow ทุก Model ในโค้ดเดียว ─────────────────────────────────
# วัตถุประสงค์: เห็น pattern เดิมที่ใช้กับทุก model — syntax ต่างกันแต่ concept เดิม
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Load Default dataset
default_df = pd.read_csv('Default.csv')
default_df['default_bin'] = (default_df['default'] == 'Yes').astype(int)
default_df['student_bin'] = (default_df['student'] == 'Yes').astype(int)

X_d = default_df[['balance', 'income', 'student_bin']].values
y_d = default_df['default_bin'].values

X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)

# ─── ทุก classifier ใช้ pattern เดียวกัน: fit → predict_proba → AUC ──────
classifiers_review = {
    'Logistic Regression (Week 11)': LogisticRegression(max_iter=1000),
    'LDA (Week 12)': LinearDiscriminantAnalysis(),
    'KNN k=10 (Week 12)': KNeighborsClassifier(n_neighbors=10),
}

print("=== Model Comparison — Default Dataset ===")
for name, clf in classifiers_review.items():
    clf.fit(X_tr_sc, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te_sc)[:, 1])
    cv_auc = cross_val_score(clf, X_tr_sc, y_tr, cv=5, scoring='roc_auc').mean()
    print(f"{name:40s}: Test AUC = {auc:.4f}, CV AUC = {cv_auc:.4f}")
```

---

## CLO4: Validation & Model Selection  *(Week 14)*

### Week 14: ทำให้ทุกอย่างน่าเชื่อถือ

**สิ่งที่ Week 14 เพิ่มให้กับทุก Model**:

```
Model ใดก็ตาม (Week 8–13)
        │
        ├── ประเมินด้วย Single Test Set → High Variance estimate
        │
        └── ประเมินด้วย k-Fold CV → Reliable estimate
                │
                └── Bootstrap → SE ของ estimate
```

### Complete Pipeline Template

```python
# ─── Template Pipeline ที่ใช้ใน Final Project ────────────────────────────
# วัตถุประสงค์: สาธิต best practice ที่นักศึกษาควรใช้ใน Week 15 Project
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate

def complete_ml_pipeline(X, y, models_params):
    """
    Best practice pipeline สำหรับ Binary Classification
    Input: X (features), y (binary target), models_params (dict of model+params)
    Output: comparison table
    """
    # 1. Hold-out split ก่อน
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    
    results = []
    for model_name, (model, params) in models_params.items():
        # 2. Pipeline: Scale → Model
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        
        # 3. GridSearch + 10-Fold CV (inner)
        gs = GridSearchCV(pipe, params, cv=10, scoring='roc_auc',
                           return_train_score=True)
        gs.fit(X_tr, y_tr)
        
        # 4. Final evaluation
        test_auc = roc_auc_score(y_te, gs.predict_proba(X_te)[:, 1])
        
        results.append({
            'Model': model_name,
            'Best Params': gs.best_params_,
            'CV AUC': gs.best_score_,
            'Test AUC': test_auc
        })
    
    return pd.DataFrame(results)

# ตัวอย่างการใช้
models_and_params = {
    'Logistic': (LogisticRegression(max_iter=1000),
                  {'model__C': [0.01, 0.1, 1, 10]}),
    'KNN': (KNeighborsClassifier(),
             {'model__n_neighbors': [5, 10, 20, 30]}),
}

results = complete_ml_pipeline(X_d, y_d, models_and_params)
print(results[['Model', 'CV AUC', 'Test AUC']].round(4))
```

---

## สรุป: Mathematics ที่ซ่อนอยู่ใน ML

### Matrix Algebra ปรากฏที่ไหนบ้าง?

| ML Algorithm | Matrix Operation จาก Week 2–4 |
|-------------|------------------------------|
| Linear Regression | $\hat{\beta} = (X^TX)^{-1}X^Ty$ — Normal Equations |
| PCA | SVD: $A = U\Sigma V^T$ — dimension reduction |
| KNN | L2 distance = $\|x_i - x_j\|_2$ — Euclidean distance |
| Neural Networks | Chain rule ผ่าน matrix multiplication |
| Random Forest | Bootstrap sampling (Week 14) |
| Ridge Regression | $(X^TX + \lambda I)^{-1}X^Ty$ — regularized Normal Equations |

### Statistics ปรากฏที่ไหนบ้าง?

| ML Concept | Statistics จาก Week 6–7 |
|-----------|----------------------|
| Cross-Validation | SE ของ estimator (Week 7) |
| p-value ใน regression | t-test / F-test (Week 7) |
| Confidence Intervals | SE formula (Week 7) |
| Model Selection | Bias-Variance Trade-off (Week 6) |
| AUC interpretation | Probability theory (Week 6) |

---

## Roadmap: คณิตศาสตร์ในวิชาถัดไป

| ถัดไป | คณิตศาสตร์ที่ต้องการ | เรียนที่ Week |
|--------|-------------------|-------------|
| **Machine Learning** (Regularization, Ensemble) | Ridge: $(X^TX+\lambda I)^{-1}$; Bootstrap: Week 14 | 2, 14 |
| **Deep Learning** | Gradient Descent, Backprop (chain rule via matrices) | 2, 3 |
| **Dimensionality Reduction** | SVD, PCA, t-SNE | 4 |
| **Time Series** | Autocorrelation (Week 10 diagnostics), Stationarity | 10 |
| **Causal Inference** | Confounding (Week 9, 11), Regression adjustment | 9, 11 |
| **Bayesian ML** | Bayes Theorem (Week 6), Prior/Posterior | 6 |
| **NLP** | TF-IDF = matrix; Word2Vec = low-rank matrix | 2, 4 |

---

## เชื่อมกับสัปดาห์อื่น

ทุก Week เชื่อมกันในสัปดาห์นี้ — Note นี้เป็นสะพานย้อนกลับไปทุกที่

- **Week 02–04** → Foundation: Matrices, Eigenvalues, SVD, PCA
- **Week 05–07** → Framework: Statistical Learning, Bias-Variance, Inference  
- **Week 08–13** → Models: SLR, MLR, Diagnostics, Logistic, LDA/QDA/KNN, GLM, ROC
- **Week 14** → Validation: CV, Bootstrap, Model Selection
- **→ Final Project**: ประยุกต์ทั้งหมดบน Dataset จริงที่เลือก
