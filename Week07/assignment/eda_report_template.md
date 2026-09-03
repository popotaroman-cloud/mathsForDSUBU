# EDA Report Template — Week 07
**ชื่อ**: _____________________ **รหัสนักศึกษา**: ___________________  
**Dataset ที่เลือก**: _____________________ **Source**: _____________________  
**วันที่**: _____________________

---

## 1. Dataset Overview

**Description**: (อธิบาย dataset ใน 2–3 ประโยค: มาจากไหน ใช้ทำอะไร มีกี่ rows/columns)

| รายการ | ค่า |
|--------|-----|
| จำนวน rows | |
| จำนวน columns | |
| Target variable (ถ้ามี) | |
| Missing values | |

## 2. Descriptive Statistics

*(วาง output ของ `.describe()` ที่นี่)*

**การตีความ**: (อธิบาย 3 สิ่งที่สังเกตเห็นจาก descriptive stats)

1. 
2. 
3. 

## 3. Visualizations

### 3.1 Distribution Plots (Histogram / Boxplot)

*(แนบ plots)*

**การตีความ**: (อธิบาย distribution ของแต่ละตัวแปร: skewed? symmetric? outliers?)

### 3.2 Relationship Plot (Scatter / Heatmap)

*(แนบ correlation heatmap หรือ pairplot)*

**การตีความ**: (ตัวแปรใดมีความสัมพันธ์กัน? ทิศทาง? ขนาด?)

## 4. Outlier Detection

**Method**: IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR)

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | # Outliers |
|--------|----|----|-----|-------------|-------------|-----------|
| | | | | | | |

**Action**: (จะจัดการ outliers อย่างไร? remove? transform? keep?)

## 5. Confidence Interval

**Column ที่เลือก**: ___________________

| Parameter | 90% CI | 95% CI | 99% CI |
|-----------|--------|--------|--------|
| Mean ± Margin | | | |

**การตีความ**: (อธิบาย CI 95% ในภาษาที่ผู้ที่ไม่ใช่นักสถิติเข้าใจได้)

## 6. Hypothesis Test

**Hypothesis ที่ตั้ง**:
- H₀: 
- H₁: 

**Test ที่ใช้**: 

**ผลลัพธ์**:
- t-statistic / χ²: 
- p-value: 
- Conclusion: 

## 7. Key Insights

*(สรุป 3–5 insights ที่ได้จาก EDA นี้ ในแบบที่ผู้บริหารเข้าใจได้)*

1. 
2. 
3. 
4. 
5. 

## 8. Next Steps

*(ถ้าจะสร้าง model ต่อไป ควรทำอะไรต่อ? feature engineering? handle outliers? คุณแนะนำ predictor ตัวใดบ้าง?)
