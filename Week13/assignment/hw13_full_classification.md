# Homework 13: Full Classification Project
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO3 | **LLo**: ใช้ Generalized Linear Models, สร้าง ROC Curve/AUC และดำเนิน full classification pipeline บนชุดข้อมูลจริงได้  
**กำหนดส่ง**: สัปดาห์ที่ 14  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

เราเดินทางมาถึงจุดที่สามารถใช้ knowledge ของ classification ทั้งหมดจาก Week 11–13 ได้อย่างครบถ้วน ใน homework สุดท้ายของ CLO3 นี้ คุณจะสร้าง **full classification pipeline** ที่ครอบคลุมทุกขั้นตอน: (1) GLM สำหรับ count data, (2) ROC curve เปรียบเทียบ classifier, (3) full pipeline บน dataset จริง นี่คือทักษะที่ใช้ใน data science interview และ project จริงทุกงาน

---

## ข้อ 1 — Poisson GLM: Count Data Analysis (3 คะแนน)

ในข้อนี้คุณจะวิเคราะห์ว่า **ปัจจัยอะไรส่งผลต่อจำนวนอุบัติเหตุบนถนน** ซึ่งเป็น count data ที่ Poisson Regression เหมาะมาก Linear Regression บน count data อาจ predict ค่าลบได้ แต่ Poisson Regression guarantee ว่าค่าพยากรณ์เป็น non-negative เสมอ

**Dataset**: สร้างด้วย code ต่อไปนี้
```python
import numpy as np, pandas as pd
from scipy.special import expit
np.random.seed(123)
n = 500
speed_limit = np.random.choice([40, 60, 80, 100], n, p=[0.3,0.3,0.25,0.15])
rain = np.random.binomial(1, 0.25, n)
visibility = np.random.uniform(0.2, 1.0, n)  # 0=fog, 1=clear
log_lam = 1.0 - 0.005*speed_limit + 0.8*rain - 0.5*visibility
accidents = np.random.poisson(np.exp(log_lam))
df_acc = pd.DataFrame({'accidents': accidents, 'speed_limit': speed_limit,
                        'rain': rain, 'visibility': visibility})
```

**สิ่งที่ต้องทำ**:
1. Fit Poisson GLM: `accidents ~ speed_limit + rain + visibility`
2. แสดง summary + อ่านค่า coefficient ทุกตัว
3. คำนวณ IRR (Incidence Rate Ratio = e^β̂) สำหรับทุกตัวแปร
4. ตีความ IRR ของ `rain` และ `speed_limit` (2–3 ประโยค)
5. Fit Linear Regression บน dataset เดียวกัน
6. เปรียบเทียบ: ทั้ง 2 model predict อย่างไรสำหรับ speed=100 + rain + visibility=0.3?
   - Poisson prediction: ? (ต้องเป็น ≥ 0)
   - Linear prediction: ? (อาจเป็นลบหรือ unrealistic)

**Expected**: β̂_rain ≈ 0.8 → IRR ≈ 2.2 → ฝนตกเพิ่ม accident 2.2 เท่า

**Deliverable**: Code + summary output + IRR table + คำอธิบาย 3–4 ประโยค

---

## ข้อ 2 — ROC Curve + AUC: Multi-Classifier Comparison (3 คะแนน)

ในข้อนี้คุณจะเปรียบเทียบ classifier 5 ตัวบน **Pima Indians Diabetes dataset** ซึ่งเป็น classic medical classification ที่ต้องการ predict diabetes (Yes/No) AUC เป็น metric สำคัญในทางการแพทย์เพราะ threshold-independent และบอกว่า classifier สามารถ "rank" patients ได้ดีแค่ไหน

**Dataset**:
```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
# Synthetic Pima-like dataset
np.random.seed(42)
X_pima, y_pima = make_classification(
    n_samples=768, n_features=8, n_informative=6,
    n_redundant=1, weights=[0.65, 0.35],  # ~35% diabetes
    random_state=42)
feature_names = ['pregnancies','glucose','blood_pressure','skin_thickness',
                  'insulin','bmi','diabetes_pedigree','age']
X_pima = pd.DataFrame(X_pima, columns=feature_names)
```

**สิ่งที่ต้องทำ**:
1. Split: test_size=0.25, random_state=42, stratify=y
2. StandardScaler (fit on train only)
3. Fit: Logistic Regression, LDA, QDA, GaussianNB, KNN(K=7)
4. สร้าง full comparison table:

| Method | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|----|----|
| ... | | | | | |

5. Plot ROC Curves ของทุก 5 classifiers ในกราฟเดียว (พร้อม AUC legend)
6. หา **optimal threshold สำหรับ best model** ด้วย Youden's J statistic
7. ตอบ: สำหรับ diabetes screening ควรเลือก metric ไหนเป็นหลัก? เพราะอะไร? (2–3 ประโยค)

**Deliverable**: Code + comparison table + ROC plot + คำตอบ

---

## ข้อ 3 — Full Pipeline: Creditworthiness Prediction (4 คะแนน)

ในข้อนี้คุณจะสร้าง **end-to-end credit scoring pipeline** ซึ่งใกล้เคียงกับ real-world application ที่สุด ธนาคารต้องการ predict ว่าลูกค้าจะ repay loan ได้หรือไม่ (`repay=1`) ปัญหานี้มี class imbalance (ส่วนใหญ่ repay) และ FN มีราคาสูง (approve ลูกค้าที่ default)

**Dataset**:
```python
np.random.seed(7); n_credit = 1500
age        = np.random.randint(20, 70, n_credit)
income     = np.random.normal(40000, 20000, n_credit)
debt_ratio = np.random.beta(2, 5, n_credit)  # 0-1
credit_score = np.random.normal(650, 80, n_credit)
employment = np.random.binomial(1, 0.85, n_credit)
log_odds = (-5 + 0.02*age + 0.00004*income
            - 3*debt_ratio + 0.01*credit_score + 0.5*employment)
repay = np.random.binomial(1, expit(log_odds))
df_credit = pd.DataFrame({'repay':repay,'age':age,'income':income,
                           'debt_ratio':debt_ratio,'credit_score':credit_score,
                           'employment':employment})
print(f'Repay rate: {repay.mean():.2%}')
```

**สิ่งที่ต้องทำ**:
1. EDA (3 plots): distribution ของ repay, boxplot credit_score by repay, correlation heatmap
2. Split: train 80% / test 20%, stratify=y, random_state=42
3. Standardize features
4. Fit ทุก 5 classifiers + `class_weight='balanced'` สำหรับ Logistic Regression (2 versions)
5. Full comparison table (6 methods: LR, LR-balanced, LDA, QDA, NB, KNN(K=5))
6. Plot ROC Curve ของทุก 6 methods
7. **Threshold analysis สำหรับ LR-balanced**: ลอง threshold = 0.3, 0.4, 0.5, 0.6 แสดง Precision/Recall/F1
8. **Business recommendation**: 
   - เลือก classifier ไหน? เพราะอะไร?
   - ควร threshold เท่าไร ถ้า FN cost = 5× FP cost?
   - อธิบาย 3–5 ประโยค

**Deliverable**: Code + EDA plots + comparison tables + ROC curve + recommendation

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว
- ตั้งชื่อ: `hw13_XXXXXXXX.ipynb`
- ดู template ใน `hw13_code_submission.md`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | Poisson GLM + IRR interpretation | 3 |
| 2 | ROC + AUC comparison | 3 |
| 3 | Full Pipeline + Business recommendation | 4 |
| **รวม** | | **10** |
