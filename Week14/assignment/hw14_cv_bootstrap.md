# Homework 14: Cross-Validation & Bootstrap
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO4 | **LLo**: ใช้ Cross-Validation และ Bootstrap ประเมินและเลือก model ที่เหมาะสมได้  
**กำหนดส่ง**: สัปดาห์ที่ 15 (ก่อน Final Presentation)  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

เราถึง Week สุดท้ายของ content ใหม่ในรายวิชานี้แล้ว! สัปดาห์นี้เราเรียน Cross-Validation และ Bootstrap ซึ่งเป็นเครื่องมือที่ **ทำให้ model selection เป็น science ไม่ใช่ art** แทนที่จะเดา K หรือ degree ของ polynomial เราใช้ CV เพื่อหา optimal hyperparameter อย่าง principled ส่วน Bootstrap ช่วยประเมิน uncertainty ของ estimate ที่ซับซ้อนเกินกว่าจะมีสูตร ใน homework นี้คุณจะใช้ CV เพื่อเลือก model สำหรับทั้ง regression และ classification และใช้ Bootstrap เพื่อประเมิน confidence interval ของ statistic ที่ไม่มีสูตรปิด

---

## ข้อ 1 — k-Fold CV สำหรับ Polynomial Regression (3 คะแนน)

ในข้อนี้คุณจะใช้ k-Fold CV เพื่อเลือก **polynomial degree ที่เหมาะสม** สำหรับ Wage dataset โดยเลือก degree ที่ minimize test MSE ด้วย One-Standard-Error Rule ซึ่งเป็น best practice ใน industry

**Dataset**:
```python
import numpy as np, pandas as pd
np.random.seed(99)
n = 3000
age  = np.random.randint(18, 80, n)
edu  = np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])  # 0=HS, 1=College, 2=Advanced
# True: wage ~ poly(age,3) + edu
wage_true = 60 + 0.8*age - 0.015*age**2 + 0.0001*age**3 + 10*edu
wage = wage_true + np.random.normal(0, 15, n)
df_wage = pd.DataFrame({'wage': wage, 'age': age, 'edu': edu})
```

**สิ่งที่ต้องทำ**:
1. ใช้ feature `age` เท่านั้น (ยังไม่ใช้ edu)
2. ทดสอบ polynomial degree = 1, 2, 3, 4, 5, 6, 7, 8
3. สำหรับแต่ละ degree: คำนวณ **10-Fold CV MSE** + SE ของ fold scores
4. Plot: CV MSE ± 1 SE bars vs degree
5. เลือก degree ด้วย:
   - **Best CV score** (minimum MSE)
   - **One-SE Rule** (simpler model ใน 1 SE)
6. Fit final model ทั้ง 2 versions บน full data + แสดง polynomial curve
7. เพิ่ม feature `edu` เข้า model ที่เลือก → CV MSE เปลี่ยนไหม?

**Expected**: true degree = 3, one-SE rule อาจเลือก degree 2 หรือ 3

**Deliverable**: Code + CV plot + polynomial curves + คำอธิบาย 2–3 ประโยค

---

## ข้อ 2 — Bootstrap: Median และ Correlation (3 คะแนน)

ในข้อนี้คุณจะใช้ Bootstrap เพื่อประเมิน SE ของ statistics 2 ตัวที่ไม่มีสูตรปิดที่ง่าย: **Median** และ **Spearman Correlation** Bootstrap ช่วยให้เราสามารถสร้าง confidence interval สำหรับ statistic ใดๆ โดยไม่ต้องรู้ distribution เลย

**Dataset**: ใช้ `df_wage` จากข้อ 1

**สิ่งที่ต้องทำ**:

**Part A — Bootstrap SE of Median:**
1. คำนวณ sample median ของ `wage`
2. Bootstrap B=1000: resample, คำนวณ median แต่ละ sample
3. คำนวณ Bootstrap SE + 95% Percentile CI
4. Plot histogram ของ bootstrap medians + mark point estimate + CI
5. เปรียบเทียบกับ Normal approximation CI: median ± 1.96 × (1.253 × σ/√n)

**Part B — Bootstrap SE of Spearman Correlation:**
1. คำนวณ Spearman correlation ระหว่าง `age` และ `wage`
2. Bootstrap B=1000: resample, คำนวณ Spearman correlation แต่ละ sample
3. คำนวณ Bootstrap SE + 95% CI
4. Plot histogram ของ bootstrap correlations

**สูตร Normal approximation สำหรับ median SE:**
$$SE(\hat{m}) \approx \frac{1.253 \cdot \hat{\sigma}}{\sqrt{n}}$$

**Deliverable**: Code + 2 histogram plots + SE/CI ทั้งคู่ + comparison

---

## ข้อ 3 — CV-based Full Model Selection Pipeline (4 คะแนน)

ในข้อนี้คุณจะสร้าง **complete model selection pipeline** สำหรับ binary classification ที่ใช้ CV เป็น model selection criterion ทั้ง classifier type และ hyperparameter โดยไม่ touch test set จนกว่าจะเลือก model สุดท้ายแล้ว

**Dataset**: Heart Disease (synthetic)
```python
from sklearn.datasets import make_classification
from scipy.special import expit
np.random.seed(42); n_h = 918
X_heart, y_heart = make_classification(
    n_samples=n_h, n_features=13, n_informative=8,
    n_redundant=2, n_classes=2,
    weights=[0.55, 0.45], random_state=42)
features = [f'feature_{i+1}' for i in range(13)]
df_heart = pd.DataFrame(X_heart, columns=features)
df_heart['target'] = y_heart
```

**สิ่งที่ต้องทำ**:
1. Split: 20% test held-out, 80% train (stratify=y, random_state=42)
2. **Scaler fitting**: StandardScaler fit บน train เท่านั้น
3. **CV Grid Search** (ทำใน train set เท่านั้น):
   - KNN: K = 1, 3, 5, 7, 10, 15, 20, 30
   - Logistic Regression: C = 0.01, 0.1, 1, 10 (regularization)
   - ใช้ 5-Fold StratifiedKFold, scoring='roc_auc'
4. Plot: AUC vs K (KNN) และ AUC vs C (Logistic) — แสดง mean ± SE
5. เลือก:
   - Best KNN K
   - Best Logistic C  
   - (Bonus) LDA, QDA, GaussianNB ไม่มี hyperparameter → fit และ CV AUC เลย
6. Comparison table (CV AUC ของทุก method):

| Method | CV AUC (mean ± SE) |
|--------|------------------|
| KNN (K=best) | |
| Logistic (C=best) | |
| LDA | |
| ... | |

7. เลือก **best method** จาก CV AUC → fit บน full train set
8. **Final evaluation** บน test set (ครั้งเดียว): report Accuracy, Precision, Recall, F1, AUC
9. Plot ROC Curve บน test set
10. คำถาม: CV AUC กับ Test AUC ต่างกันมากไหม? เพราะอะไร?

**Deliverable**: Code + CV plots + comparison table + final test report + คำตอบ

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว
- ตั้งชื่อ: `hw14_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | k-Fold CV + One-SE Rule สำหรับ polynomial | 3 |
| 2 | Bootstrap SE ของ median + correlation | 3 |
| 3 | Full CV model selection pipeline | 4 |
| **รวม** | | **10** |
