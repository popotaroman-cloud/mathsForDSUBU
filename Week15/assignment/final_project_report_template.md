# Template รายงาน Final Project
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | **Week 15**

**ชื่อกลุ่ม**: _________________________  
**สมาชิก**: 
1. ชื่อ _______ รหัส _______
2. ชื่อ _______ รหัส _______
3. ชื่อ _______ รหัส _______ (ถ้ามี)

**Dataset**: _________________________  
**แหล่งที่มา**: _________________________

---

## 1. บทนำ (Introduction)

### 1.1 Dataset Overview

*(อธิบาย dataset ที่เลือก — ที่มา ขนาด features และ target variable)*

| รายการ | ค่า |
|--------|-----|
| ชื่อ Dataset | |
| แหล่งที่มา | |
| จำนวน observations | |
| จำนวน features | |
| Target variable | |
| Type (Classification/Regression) | |

### 1.2 Problem Statement

*(อธิบาย 2–3 ประโยค: ปัญหาคืออะไร? ทำไมน่าสนใจ? จะตอบคำถามอะไร?)*

_______________

### 1.3 Methodology Overview

*(ใช้ tools อะไรบ้างจาก CLO1–4?)*

- CLO1: _______________
- CLO2: _______________
- CLO3: _______________
- CLO4: _______________

---

## 2. CLO1: Linear Algebra Analysis

### 2.1 Feature Covariance Matrix

*(คำนวณ Σ และ interpret)*

```python
# วาง code ที่นี่
```

**ผลลัพธ์:** *(อธิบาย covariance สูง/ต่ำระหว่าง features คู่ไหน)*

### 2.2 Eigenvalue Analysis

*(คำนวณ eigenvalues ของ Σ)*

| PC | Eigenvalue (λ) | Variance Explained (%) | Cumulative (%) |
|----|---------------|----------------------|----------------|
| PC1 | | | |
| PC2 | | | |
| ... | | | |

*(ต้องการ PC กี่ตัวเพื่ออธิบาย 90% variance?)*

### 2.3 PCA Visualization

*(Scatter plot PC1 vs PC2 colored by target variable)*

```python
# วาง code ที่นี่
```

**ผลลัพธ์:** *(classes แยกกันได้ไหมใน PCA space?)*

---

## 3. CLO2: Statistical Analysis

### 3.1 Descriptive Statistics

*(describe() และ histogram ของ numeric features)*

| Feature | Mean | Std | Min | Max | Skewness |
|---------|------|-----|-----|-----|---------|
| ... | | | | | |

### 3.2 Distribution Analysis

*(histogram + QQ plot ของ target variable)*

**สังเกต:** *(target มี distribution ปกติ? skewed? ต้องการ transformation?)*

### 3.3 Hypothesis Testing

*(ทดสอบ hypothesis ที่เหมาะสม)*

**H₀**: _______________  
**H₁**: _______________  
**Test**: _______________  
**Result**: t-stat = ___, p-value = ___  
**สรุป**: _______________

### 3.4 Correlation Analysis

*(Heatmap + feature ที่สำคัญ)*

**Features ที่ correlate กับ target มากที่สุด:**
1. _____ (r = _____)
2. _____ (r = _____)
3. _____ (r = _____)

---

## 4. CLO3: Machine Learning Models

### 4.1 Data Preparation

*(Split ratio, preprocessing steps)*

- Train/Test split: ___% / ___%
- Preprocessing: StandardScaler? Encoding?
- Class balance: ___

### 4.2 Model 1: _______________

*(ชื่อ method และ parameters)*

**Results:**

| Metric | ค่า |
|--------|-----|
| Accuracy/MSE | |
| Precision/R² | |
| Recall | |
| F1/RMSE | |

*(วาง confusion matrix หรือ residual plot)*

### 4.3 Model 2: _______________

*(เหมือน 4.2)*

### 4.4 Model 3: _______________

*(เหมือน 4.2)*

### 4.5 Comparison Table

| Model | Accuracy | F1 | AUC | หมายเหตุ |
|-------|----------|----|----|---------|
| Model 1 | | | | |
| Model 2 | | | | |
| Model 3 | | | | |

---

## 5. CLO4: Model Selection with CV

### 5.1 k-Fold CV Setup

- K = _____ folds
- CV Metric: _____
- Stratified: Yes / No

### 5.2 CV Results

| Model / Hyperparameter | CV Mean | CV SE |
|----------------------|---------|-------|
| ... | | |

*(วาง CV curve plot)*

### 5.3 Best Model Selection

**Best CV score**: _______________  
**One-SE Rule selection**: _______________  
**เหตุผลที่เลือก**: _______________

### 5.4 Final Test Evaluation

*(Evaluate ครั้งเดียวบน test set)*

| Metric | ค่า |
|--------|-----|
| Test Accuracy/MSE | |
| Test F1/R² | |
| Test AUC | |

*(CV score vs Test score ต่างกันมากไหม? เพราะอะไร?)*

---

## 6. Key Insights

*(สรุป 5–7 bullet ของสิ่งที่ค้นพบที่สำคัญที่สุด)*

1. _______________
2. _______________
3. _______________
4. _______________
5. _______________

---

## 7. Limitations & Future Work

*(อธิบาย 2–3 ข้อจำกัดของการวิเคราะห์นี้ และจะปรับปรุงอย่างไรในอนาคต)*

**Limitations:**
- _______________

**Future Work:**
- _______________

---

## 8. References

*(อ้างอิงแหล่งข้อมูลทุกแหล่งในรูปแบบ APA)*

1. _______________
2. _______________

---

*Template นี้ใช้กับ `final_project_guidelines.md` — ดูรายละเอียด rubric ใน `final_project_rubric.md`*
