# Slide Deck: Qualitative Predictors — Dummy Variables
> Week 10 | CLO3 | ISLP Ch.3.3.1 | 9 slides

---
## Slide 1 — Title
**Qualitative Predictors: Dummy Variables และ Reference Level**  
Week 10 | CLO3 | ISLP 3.3.1  
LLo: สร้าง dummy variable และตีความ regression coefficient ของ qualitative predictor ได้

---
## Slide 2 — Week Overview
**Key Message**: regression ไม่จำกัดแค่ numerical predictor — categorical variable ก็ใส่ได้ผ่าน dummy encoding

สัปดาห์นี้เราจะขยาย regression ให้รองรับ **qualitative predictor** (categorical), **interaction terms** และ **polynomial regression** รวมถึงเรียนรู้วิธีตรวจสอบ potential problems ของ regression model ผ่าน diagnostic plots ทักษะเหล่านี้จำเป็นสำหรับงาน Data Science จริงเพราะ dataset ส่วนใหญ่มีตัวแปรทั้ง numerical และ categorical ปะปนกัน ปัญหาเช่น non-linearity, outliers, และ multicollinearity เป็นสิ่งที่ต้องตรวจเสมอก่อนเชื่อผล regression สัปดาห์นี้จะเตรียมนักศึกษาให้สามารถทำ regression analysis แบบครบวงจรได้

---
## Slide 3 — ปัญหา: ใส่ Categorical ใน Regression ได้ไหม?
**Key Message**: ใส่ตรงไม่ได้ — ต้องแปลงเป็น 0/1 dummy variable ก่อน

**ตัวอย่าง**: Credit dataset — predictor: `Student` (Yes/No)

```
# ❌ ใส่ตรงไม่ได้
Sales ~ Student  ← "Yes"/"No" ไม่ใช่ตัวเลข
```

**วิธีแก้**: สร้าง dummy variable

$$x_i = \begin{cases} 1 & \text{ถ้า Student = Yes} \\ 0 & \text{ถ้า Student = No (reference level)} \end{cases}$$

**Model**:
$$\text{Balance} = \beta_0 + \beta_1 \cdot \text{Student\_Yes} + \varepsilon$$

- β₀: expected Balance ของ non-student
- β₀ + β₁: expected Balance ของ student
- β₁: **difference** ระหว่าง student และ non-student

---
## Slide 4 — Reference Level คืออะไร?
**Key Message**: reference level คือ baseline ที่ dummy = 0 — ทุก coefficient ตีความเทียบกับ baseline

**Binary predictor** (k=2): สร้าง **1** dummy
- Student = No → reference (dummy = 0)
- Student = Yes → dummy = 1, coefficient = difference from Non-student

**k-level predictor** (k > 2): สร้าง **k−1** dummies
- ตัวอย่าง: Ethnicity = {Asian, Caucasian, African American}
  - Reference: African American (หรือ alphabetically first)
  - Dummy 1: `Asian` (1 ถ้า Asian, 0 อื่น ๆ)
  - Dummy 2: `Caucasian` (1 ถ้า Caucasian, 0 อื่น ๆ)

**Dummy Variable Trap**: ถ้าสร้าง k dummies → perfect multicollinearity! (sum = 1 เสมอ)  
→ ต้องสร้างแค่ k−1

---
## Slide 5 — Python: Dummy Variables
**Key Message**: pandas และ statsmodels มี built-in สำหรับ dummy encoding

```python
import pandas as pd
import statsmodels.formula.api as smf

# ─── วิธีที่ 1: pd.get_dummies() ───────────────────────────
# วัตถุประสงค์: สร้าง k-1 dummies ด้วย drop_first=True
df_dummies = pd.get_dummies(df, columns=['Ethnicity'], drop_first=True)
# drop_first=True → หลีกเลี่ยง dummy variable trap

# ─── วิธีที่ 2: C() ใน statsmodels formula (แนะนำ) ─────────
# วัตถุประสงค์: สะดวก อ่านง่าย statsmodels จัดการ reference level ให้อัตโนมัติ
model = smf.ols('Balance ~ Income + Student + C(Ethnicity)', data=df).fit()
print(model.summary())

# ─── ดู reference level ──────────────────────────────────
# วัตถุประสงค์: รู้ว่า baseline คืออะไรก่อนตีความ coefficient
print(df['Ethnicity'].value_counts())
print(f"Reference level: {df['Ethnicity'].unique()[0]}")
```

---
## Slide 6 — ตีความ: Credit Dataset
**Key Message**: coefficient ของ dummy = ความต่างเฉลี่ยจาก reference group เมื่อ control ตัวแปรอื่น

```
OLS: Balance ~ Income + Student
==============================================
              coef    std err    t     P>|t|
----------------------------------------------
Intercept   211.14     32.46    6.50   0.000
Income        6.05      0.58   10.40   0.000
Student[T.Yes] 382.67  65.31   5.86   0.000
==============================================
```

**ตีความ**:
- β₀ = 211.14: credit balance ของ non-student ที่ income = 0
- β_Income = 6.05: income เพิ่ม $1k → balance เพิ่ม $6.05 (holding student status fixed)
- β_Student = 382.67: student มี balance สูงกว่า non-student **เฉลี่ย $382.67** (holding income fixed)
- p < 0.001 → student status มีผลต่อ balance อย่างมีนัยสำคัญ

---
## Slide 7 — k-level Categorical: Ethnicity
**Key Message**: ตีความแต่ละ dummy เทียบกับ reference — ไม่ใช่เทียบกัน

```
              coef    std err    t     P>|t|
--------------------------------------------
Ethnicity[T.Asian]        18.69   65.02  0.29  0.774
Ethnicity[T.Caucasian]   -12.50   56.68 -0.22  0.826
```

**ตีความ**:
- Reference: African American
- Asian: balance สูงกว่า African American เฉลี่ย $18.69 (p = 0.774 → NOT significant)
- Caucasian: balance ต่ำกว่า African American เฉลี่ย $12.50 (NOT significant)
- สรุป: ไม่มีหลักฐานว่า ethnicity มีผลต่อ credit balance

[FIGURE: boxplot แสดง Balance by Ethnicity — overlap มาก]

---
## Slide 8 — Parallel Regression Lines
**Key Message**: dummy predictor สร้าง parallel regression lines — intercept ต่างกันแต่ slope เหมือนกัน

[FIGURE: scatter plot + 2 fitted lines:
- Student = No (ล่าง, intercept = β₀)
- Student = Yes (บน, intercept = β₀ + β₁)
- Lines parallel — slope เหมือนกัน = β_Income]

**ข้อจำกัดของ model นี้**: slope เหมือนกัน → ถ้า effect ของ income ต่างกันระหว่าง student/non-student → ต้องใช้ **interaction term** (Slide 2)

---
## Slide 9 — Summary
**Key Message**: dummy variables แปลง categorical → numerical โดย reference level คือ baseline

**สิ่งที่เรียนรู้วันนี้**:
- **Binary predictor**: 1 dummy (0/1) — reference = 0
- **k-level predictor**: k−1 dummies — หลีกเลี่ยง dummy variable trap
- **Interpretation**: coefficient = difference from reference group
- **Python**: `C()` ใน formula หรือ `pd.get_dummies(drop_first=True)`

| Concept | Python |
|---------|--------|
| Auto dummy | `smf.ols('y ~ C(x)', data)` |
| Manual dummy | `pd.get_dummies(df, drop_first=True)` |
| Reference level | `C(x, Treatment(reference='X'))` |

**สัปดาห์ต่อไป — Slide 2**: Interaction terms + Polynomial Regression
