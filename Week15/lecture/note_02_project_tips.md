# Note 15-2: Tips การนำเสนอ Final Project & Data Storytelling
> Week 15 | CLO1–4 | Final Project Presentation Guide

Note สุดท้ายของรายวิชานี้จะเน้นทักษะที่สำคัญมากในอาชีพ Data Scientist แต่มักไม่ถูกสอนในห้องเรียน — **Data Storytelling** ความสามารถในการวิเคราะห์ข้อมูลด้วยคณิตศาสตร์ที่ซับซ้อนแต่นำเสนอผลลัพธ์ให้ผู้ฟังที่ไม่ใช่นักคณิตศาสตร์เข้าใจได้ เป้าหมายของ Note นี้คือให้นักศึกษาทราบ Structure ที่ดีของ Final Project Presentation, เรียนรู้วิธีเลือก Visualization ที่เหมาะสม, และหลีกเลี่ยง Common Mistakes ที่ทำให้ Presentation อ่อนแอ ทักษะเหล่านี้ใช้ได้ตลอดชีวิตการทำงาน ไม่ว่าจะใน Data Science, Research, Business Analysis หรือสาขาใดก็ตาม

---

## 15.1 Structure ของ Final Project ที่ดี

ในส่วนนี้เราจะเรียนรู้ Framework สำหรับจัดเรียง Project ให้มีเรื่องราวที่ชัดเจนและสม่ำเสมอ

### The 5-Part Story Arc

Project ที่ดีมีโครงสร้างเหมือน **เรื่องเล่า** — มีต้น กลาง ปลาย:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Problem & Data (2 นาที)                                     │
│     "เราพยายามตอบคำถามอะไร? ข้อมูลมาจากไหน?"                   │
│                                                                 │
│  2. EDA & Insights (2 นาที)                                     │
│     "ข้อมูลบอกอะไรก่อนสร้าง model?"                            │
│                                                                 │
│  3. Mathematical Analysis (5 นาที)                              │
│     "เราใช้ Mathematics อะไรจาก CLO1–4 บ้าง?"                  │
│                                                                 │
│  4. Results & Comparison (2 นาที)                               │
│     "Model ไหนดีที่สุด? ตัวเลขพิสูจน์อย่างไร?"                 │
│                                                                 │
│  5. Conclusion & Reflection (1 นาที)                           │
│     "เราเรียนรู้อะไร? ถ้าทำอีกครั้งจะเปลี่ยนอะไร?"             │
└─────────────────────────────────────────────────────────────────┘
```

### ตัวอย่าง: Presentation ที่ดี vs ไม่ดี

| ส่วน | ❌ อ่อนแอ | ✅ แข็งแกร่ง |
|------|---------|-----------|
| **Title** | "Machine Learning Project" | "วิเคราะห์ปัจจัยที่ส่งผลต่อราคาบ้านในกรุงเทพฯ ด้วย Multiple Regression" |
| **Problem** | "เราวิเคราะห์ข้อมูลบ้าน" | "สมมติคุณจะซื้อบ้านแต่ไม่รู้ว่าราคาแพงเกินไปหรือไม่ โมเดลของเราจะทำนายราคาที่ 'ยุติธรรม' ได้" |
| **Data** | "Dataset มี 1000 rows" | "ข้อมูล 1,247 ตัวอย่าง, 15 features จาก DDproperty.com, ช่วงราคา 1.2–85 ล้านบาท" |
| **Method** | "เราใช้ Regression" | "เราใช้ Multiple Linear Regression (Week 9) และตรวจสอบด้วย 6 Diagnostic Plots (Week 10) พบ log-transformation จำเป็น" |
| **Results** | "โมเดลดีมาก" | "R²_adj = 0.847, CV RMSE = 2.3 ล้าน บาท, VIF ทุกตัว < 5 → โมเดลแข็งแรง" |

---

## 15.2 Mathematical Analysis ที่ต้องครอบคลุม

ในส่วนนี้เราจะดูว่า CLO แต่ละข้อควรแสดงให้เห็นอย่างไรใน Final Project

### Checklist: CLO Coverage

#### CLO1 — Linear Algebra
- [ ] **Matrix Analysis**: แสดง Design Matrix X ขนาด n × p
- [ ] **Correlation Matrix**: visualize ด้วย heatmap (Week 7)
- [ ] **Multicollinearity Check**: VIF ทุก predictor (Week 10)
- [ ] **Bonus**: PCA หรือ SVD บน dataset (Week 4) — ถ้า dimensionality สูง

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# ─── Template: CLO1 Matrix Analysis Section ──────────────────────────────
# วัตถุประสงค์: แสดง matrix perspective ของ dataset — CLO1 requirement
def clo1_matrix_analysis(df, feature_cols, target_col):
    """Template สำหรับ CLO1 section ของ Final Project"""
    print("=" * 50)
    print("CLO1: Linear Algebra Analysis")
    print("=" * 50)
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Design Matrix
    print(f"\n1. Design Matrix X: {X.shape[0]} × {X.shape[1]}")
    print(f"   Rank(X) = {np.linalg.matrix_rank(X.values)}")
    
    # Correlation Matrix
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix (CLO1 — Week 7/10)')
    plt.tight_layout(); plt.show()
    
    # VIF
    X_sm = sm.add_constant(X)
    vif_df = pd.DataFrame({
        'Feature': feature_cols,
        'VIF': [variance_inflation_factor(X_sm.values, i+1) 
                for i in range(len(feature_cols))]
    }).sort_values('VIF', ascending=False)
    print("\n2. VIF (Multicollinearity Check):")
    print(vif_df.to_string(index=False))
    
    high_vif = vif_df[vif_df['VIF'] > 5]
    if len(high_vif) > 0:
        print(f"⚠️  High VIF (>5): {list(high_vif['Feature'])} → consider PCA or dropping one")
    else:
        print("✓ No Multicollinearity issues (all VIF < 5)")
    
    return vif_df
```

#### CLO2 — Statistical Learning Framework
- [ ] **EDA**: Distribution plots, outlier detection (IQR)
- [ ] **Bias-Variance**: อธิบายว่าโมเดลที่เลือกอยู่จุดไหน (Overfit? Underfit?)
- [ ] **Hypothesis Testing**: t-test หรือ correlation test บน features ที่สนใจ
- [ ] **Confidence Intervals**: CI ของ coefficients สำคัญ

```python
# ─── Template: CLO2 Statistical Learning Section ──────────────────────────
# วัตถุประสงค์: แสดงความเข้าใจ Statistical Framework — CLO2 requirement
def clo2_statistical_analysis(df, feature_cols, target_col):
    print("\n" + "=" * 50)
    print("CLO2: Statistical Learning Analysis")
    print("=" * 50)
    
    # 1. Distribution Analysis (EDA)
    fig, axes = plt.subplots(2, len(feature_cols)//2 + 1, figsize=(15, 6))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols + [target_col]):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, color='steelblue', alpha=0.7)
            axes[i].set_title(f'{col} distribution')
    plt.suptitle('EDA: Feature Distributions (CLO2 — Week 7)')
    plt.tight_layout(); plt.show()
    
    # 2. Outlier Detection
    print("\n2. Outlier Detection (IQR Method):")
    for col in feature_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        print(f"  {col}: {n_outliers} outliers")
    
    # 3. Correlation with target
    print("\n3. Pearson Correlation with Target:")
    correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    print(correlations.round(3))
    print("(Top features by correlation strength)")
```

#### CLO3 — Models
- [ ] **Regression** (Week 8–10): SLR → MLR → Diagnostics → เลือก best model
- [ ] **Classification** ถ้า response เป็น categorical (Week 11–13)
- [ ] **Interpretation**: ตีความ coefficients ด้วยภาษาที่ทุกคนเข้าใจ
- [ ] **Diagnostic Plots**: 4 standard plots ของ Regression (Week 10)

```python
# ─── Template: CLO3 Regression Analysis Section ───────────────────────────
# วัตถุประสงค์: แสดง complete regression pipeline — CLO3 requirement
def clo3_regression_analysis(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 50)
    print("CLO3: Regression Model Analysis")
    print("=" * 50)
    
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_te_sc = sc.transform(X_test)
    
    # Model 1: SLR (best single predictor)
    best_r2 = -np.inf
    best_feature = None
    for i, feat in enumerate(feature_names):
        lr = LinearRegression().fit(X_tr_sc[:, i:i+1], y_train)
        r2 = r2_score(y_test, lr.predict(X_te_sc[:, i:i+1]))
        if r2 > best_r2:
            best_r2, best_feature = r2, feat
    
    print(f"\nBest SLR feature: {best_feature} (Test R² = {best_r2:.4f})")
    
    # Model 2: MLR (all features)
    mlr = LinearRegression().fit(X_tr_sc, y_train)
    y_pred_mlr = mlr.predict(X_te_sc)
    r2_mlr = r2_score(y_test, y_pred_mlr)
    rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
    
    print(f"MLR (all features): Test R² = {r2_mlr:.4f}, RMSE = {rmse_mlr:.4f}")
    print(f"Improvement over best SLR: ΔR² = {r2_mlr - best_r2:+.4f}")
    
    return mlr
```

#### CLO4 — Cross-Validation & Bootstrap
- [ ] **k-Fold CV**: แสดง CV error สำหรับ model comparison
- [ ] **Optimal Hyperparameter**: CV curve แสดงการเลือก degree หรือ K
- [ ] **Bootstrap SE**: SE ของ coefficient สำคัญ

---

## 15.3 Visualization Tips

ในส่วนนี้เราจะเรียนรู้วิธีเลือก Plot ที่เหมาะสม เพราะ "A good chart replaces a paragraph"

### เลือก Plot อย่างไร?

| ต้องการแสดงอะไร | Plot ที่ดีที่สุด |
|----------------|---------------|
| Distribution ของ 1 variable | Histogram + KDE |
| Relationship X vs Y (continuous) | Scatter plot + Regression line |
| Comparison ระหว่าง groups | Box plot หรือ Violin plot |
| Correlation ระหว่างหลาย variables | Heatmap (Correlation Matrix) |
| Change over time | Line chart |
| Model Performance | ROC Curve, CV error curve |
| Feature Importance | Horizontal Bar chart (sorted) |
| Residuals | Scatter plot (eᵢ vs ŷᵢ) |

```python
# ─── Professional Visualization Template ──────────────────────────────────
# วัตถุประสงค์: สร้าง publication-quality plots สำหรับ presentation
# ใช้ consistent style ตลอด project

# กำหนด Style ให้สม่ำเสมอตลอด Project
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

# ─── Example: Comprehensive EDA Dashboard ───────────────────────────────
# วัตถุประสงค์: รวม EDA ทั้งหมดใน 1 figure — efficient use of space in presentation
def eda_dashboard(df, num_cols, target_col, title="EDA Dashboard"):
    n = len(num_cols)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    for i, col in enumerate(num_cols):
        # Row 1: Distribution
        axes[0, i].hist(df[col], bins=30, color=PALETTE[i % len(PALETTE)], 
                         alpha=0.7, edgecolor='white')
        axes[0, i].set_title(f'{col}\n(distribution)')
        
        # Row 2: Scatter vs Target
        axes[1, i].scatter(df[col], df[target_col], alpha=0.3, s=10,
                            color=PALETTE[i % len(PALETTE)])
        
        # Add regression line
        z = np.polyfit(df[col].dropna(), df[target_col].iloc[:len(df[col].dropna())], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        axes[1, i].plot(x_line, p(x_line), 'r-', linewidth=1.5)
        
        corr = df[[col, target_col]].corr().iloc[0, 1]
        axes[1, i].set_title(f'vs {target_col}\n(r = {corr:.3f})')
        axes[1, i].set_xlabel(col)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()
```

### ❌ Visualization Mistakes ที่ต้องหลีกเลี่ยง

```python
# ─── Common Mistakes Examples ─────────────────────────────────────────────
# วัตถุประสงค์: เรียนรู้จาก bad examples — สิ่งที่ผู้ฟังมองแล้วสับสน

# ❌ ผิด: ไม่มี labels
fig, ax = plt.subplots()
ax.scatter([1,2,3], [4,5,6])
# ต้องเพิ่ม: ax.set_xlabel(), ax.set_ylabel(), ax.set_title()

# ❌ ผิด: Pie chart กับ > 5 categories (ดูยาก)
# ✅ ถูก: ใช้ Horizontal Bar chart แทน

# ❌ ผิด: แสดง Accuracy อย่างเดียวสำหรับ Imbalanced Dataset
# ✅ ถูก: แสดง Precision, Recall, F1, AUC ด้วยเสมอ

# ❌ ผิด: table ตัวเลข 10 ทศนิยม
print(0.8471293847123)  # ✅ ควรเป็น 0.847 หรือ 84.7%

# ❌ ผิด: ไม่บอก baseline
print("Model Accuracy = 96%")  # ถ้า Majority class = 97% → model แย่กว่า random!
# ✅ ถูก: "Model Accuracy = 96% (Majority Baseline = 97%) → model ไม่ดี"
```

---

## 15.4 Common Mistakes ใน Analysis

### Mistakes ที่ลดคะแนน

| Mistake | ทำไมผิด | วิธีแก้ |
|---------|---------|-------|
| ไม่ตรวจ VIF | อาจมี Multicollinearity ทำให้ interpret ผิด | คำนวณ VIF ทุกครั้งที่ใช้ MLR |
| ใช้ Training R² เท่านั้น | Training R² ลดเสมอเมื่อเพิ่ม predictor | ใช้ Adjusted R² หรือ CV error |
| ไม่ทำ Diagnostic Plots | โมเดลอาจละเมิด assumptions | ทำ 4 Diagnostic Plots เสมอ (Week 10) |
| ลบ Outliers โดยไม่ตรวจสอบ | อาจลบ valid data | ตรวจก่อนว่าเป็น Data Entry Error หรือ Real |
| ไม่ normalize metric | Accuracy 96% ไม่ดีถ้า baseline 96% | เสนอ Lift over Baseline เสมอ |
| ใช้ same data ประเมิน model | Overfitting ที่ไม่ detected | Train/Test split หรือ k-Fold CV เสมอ |
| ไม่ scale features ก่อน KNN/LDA | Distance metric ถูก dominate โดย large scale variable | StandardScaler ก่อนเสมอ |

### Mistakes ใน Interpretation

```python
# ─── ตัวอย่าง: Interpretation ที่ถูกต้อง vs ผิด ──────────────────────────
# วัตถุประสงค์: แสดง common pitfalls ใน interpretation ของ regression coefficients

# สมมติ fit MLR แล้วได้:
# Sales = 2.94 + 0.046*TV + 0.189*Radio - 0.001*Newspaper

# ❌ ผิด: "TV ส่งผลต่อ Sales มากกว่า Radio เพราะ coefficient TV = 0.046 > 0"
# → ผิดเพราะ units ต่างกัน! ต้อง standardize ก่อน

# ✅ ถูก: "เมื่อ control for Radio และ Newspaper: TV เพิ่ม 1 หน่วย → Sales เพิ่ม 0.046 หน่วย"
# "เมื่อ control for TV และ Newspaper: Radio เพิ่ม 1 หน่วย → Sales เพิ่ม 0.189 หน่วย"
# "Newspaper: p-value = 0.86 → ไม่มีนัยสำคัญเมื่อ control for TV และ Radio"

# ❌ ผิด: "Correlation เท่ากับ Causation" (พบบ่อยมาก!)
# ✅ ถูก: "เราพบ Association ระหว่าง X และ Y" / "X เป็น predictor ที่ดีของ Y"
# → Causation ต้องการ Experimental Design หรือ Causal Inference methods

# ❌ ผิด: "p-value > 0.05 แสดงว่า H₀ เป็นความจริง"
# ✅ ถูก: "เราไม่มีหลักฐานเพียงพอที่จะ reject H₀ ที่ระดับ α = 0.05"
print("Key phrases for correct interpretation loaded")
```

---

## 15.5 Final Project Rubric Self-Check

ก่อน submit ตรวจสอบด้วย checklist นี้:

```python
# ─── Self-Check Rubric Template ──────────────────────────────────────────
# วัตถุประสงค์: ใช้เป็น checklist ก่อน submit final project

rubric_checklist = {
    "CLO1 — Linear Algebra (20 pts)": [
        "✓ แสดง Design Matrix dimensions และ rank",
        "✓ Correlation Matrix heatmap ครบ",
        "✓ VIF analysis สำหรับทุก predictor",
        "? PCA/SVD (Bonus หากมี dimensionality > 5)"
    ],
    "CLO2 — Statistical Framework (20 pts)": [
        "✓ EDA ครบ: distribution, outliers, skewness",
        "✓ อธิบาย Bias-Variance position ของโมเดลที่เลือก",
        "✓ Statistical test อย่างน้อย 1 (t-test หรือ F-test)",
        "✓ CI ของ coefficient สำคัญ"
    ],
    "CLO3 — Models (40 pts)": [
        "✓ SLR หรือ Logistic Regression baseline",
        "✓ MLR หรือ Multiple Classifier",
        "✓ Diagnostic Plots ครบ 4 plots (Regression) หรือ Confusion Matrix (Classification)",
        "✓ ตีความ coefficients อย่างถูกต้องด้วยภาษาที่เข้าใจได้",
        "✓ Compare ≥ 2 models"
    ],
    "CLO4 — Validation (20 pts)": [
        "✓ k-Fold CV (k ≥ 5) สำหรับทุก model",
        "✓ CV error ± SE รายงานครบ",
        "✓ Bootstrap SE ของ coefficient สำคัญ (Bonus)",
        "✓ Final model เลือกจาก CV ไม่ใช่แค่ Training Performance"
    ],
    "Presentation Quality": [
        "✓ Problem statement ชัดเจน (1-2 ประโยค)",
        "✓ Dataset description ครบ (n, p, source)",
        "✓ Visualizations มี labels ทุก axes",
        "✓ ไม่มี Correlation ≠ Causation mistakes",
        "✓ Conclusion ตอบ Problem Statement ตั้งแต่แรก"
    ]
}

for section, items in rubric_checklist.items():
    print(f"\n{section}:")
    for item in items:
        print(f"  {item}")
```

---

## สรุป: สิ่งที่สำคัญที่สุดจากรายวิชานี้

| ระดับ | สิ่งที่ได้เรียนรู้ |
|------|----------------|
| **Conceptual** | ข้อมูลคือ Matrix; Model คือ Function; Error คือ Noise + Bias |
| **Mathematical** | Normal Equations, Sigmoid, MLE, SVD, Bias-Variance — ทั้งหมดเชื่อมกัน |
| **Practical** | EDA → Fit → Diagnose → Validate → Select → Report |
| **Mindset** | "Correlation ≠ Causation" / "Training Error ≠ Test Error" / "Accuracy ≠ Performance" |

### Quote สำหรับจำ

> *"All models are wrong, but some are useful."* — George Box

> *"The goal of data analysis is not to find the 'true' model, but to find a model that is useful enough for the decision at hand."* — inspired by ISLP

---

## เชื่อมกับอนาคต

- **ต่อจาก CLO1**: Deep Learning ใช้ Matrix Multiplication ใน Forward Pass และ Backpropagation ทุก Layer
- **ต่อจาก CLO2**: Bayesian Learning ขยาย Bayes Theorem (Week 6) สู่ Prior/Posterior
- **ต่อจาก CLO3**: Regularization (Ridge, Lasso) ขยาย Normal Equations; Tree Models ขยาย Model Comparison
- **ต่อจาก CLO4**: Cross-Validation เป็น core ของ AutoML และ Hyperparameter Tuning ทุกประเภท

**ขอให้โชคดีกับ Final Project และการเรียนในอนาคต!**
