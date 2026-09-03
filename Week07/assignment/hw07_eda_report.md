# Homework 07: EDA Report และ Statistical Inference
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO2 | **LLo**: คำนวณ descriptive statistics, สร้าง Confidence Interval และทดสอบสมมติฐานเบื้องต้นได้  
**กำหนดส่ง**: สัปดาห์ที่ 8 (ก่อนสอบกลางภาค)  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราได้เรียนรู้กระบวนการ **Exploratory Data Analysis (EDA)** และ **Statistical Inference** ซึ่งเป็นทักษะที่ Data Scientist ทุกคนต้องมีก่อนสร้าง model ใด ๆ การวิเคราะห์ข้อมูลเบื้องต้นช่วยให้เราเข้าใจ distribution ตรวจพบ outlier และทดสอบ hypothesis ก่อนที่จะ commit กับการสร้างโมเดล เป้าหมายของการบ้านนี้คือให้นักศึกษาฝึกทำ EDA แบบครบวงจรบน dataset จริงที่เลือกเอง พร้อมคำนวณ CI และทดสอบ t-test ซึ่งจะนำไปใช้โดยตรงใน Week 8–9 เมื่อเราอ่านผลลัพธ์จาก regression table ทักษะนี้ยังใช้ใน data science จริงทุกครั้งที่ต้องรายงานผลการวิเคราะห์ให้ผู้บริหารหรือลูกค้า

---

## ข้อ 1 — EDA บน Dataset ที่เลือก (4 คะแนน)

ในข้อนี้คุณจะทำ **Full EDA** บน dataset ที่คุณเลือก เพื่อแสดงให้เห็นว่าการสำรวจข้อมูลก่อน modeling มีความสำคัญอย่างไร และช่วยให้ตัดสินใจในขั้นต่อไปได้อย่างมีหลักการ

**Dataset**: เลือก 1 dataset จากรายการด้านล่าง (หรือ dataset ของตัวเองที่มี ≥ 200 rows, ≥ 4 columns):

| ตัวเลือก | Dataset | URL / แหล่ง |
|---------|---------|-------------|
| A | Default (ISLP) | `ISLP.load_data('Default')` |
| B | Boston Housing | `sklearn.datasets.load_boston()` |
| C | Iris | `sklearn.datasets.load_iris()` |
| D | WHO Health Indicators | WHO website |

**สิ่งที่ต้องทำ**:
1. **โหลด** dataset และแสดง `.info()`, `.describe()`, `.isnull().sum()`
2. **Univariate**: histogram หรือ boxplot สำหรับ ≥ 3 numerical columns
3. **Bivariate**: scatter plot matrix (`sns.pairplot`) หรือ correlation heatmap
4. **Outlier detection**: ใช้ IQR method ระบุ outliers ในอย่างน้อย 1 column
   - Outlier: ค่าที่ต่ำกว่า Q1 − 1.5×IQR หรือสูงกว่า Q3 + 1.5×IQR
5. **Summary**: สรุป 3 insights ที่ได้จาก EDA

**Deliverable**: Jupyter Notebook `.ipynb` ที่ run ผ่านแล้ว + คำอธิบายทุก plot

---

## ข้อ 2 — Confidence Interval (3 คะแนน)

ในข้อนี้คุณจะคำนวณ CI เพื่อแสดงให้เห็น uncertainty ในการประมาณค่า population parameter ซึ่งเป็น logic เดียวกับที่ใช้ใน regression coefficient CI ใน Week 8

**ใช้ dataset เดียวกับข้อ 1**

**สิ่งที่ต้องทำ**:
1. เลือก 1 numerical column และคำนวณ 95% CI สำหรับ **mean** แบบ manual:
   - คำนวณ x̄, s, SE = s/√n, t* = `stats.t.ppf(0.975, df=n-1)`
   - CI = x̄ ± t* × SE
2. Verify ด้วย `scipy.stats.t.interval(0.95, df=n-1, loc=x̄, scale=SE)`
3. Plot: histogram ของ column นั้น + ใส่ vertical lines แสดง x̄ และ CI bounds
4. ทดลองเปลี่ยน confidence level เป็น 90% และ 99% — อธิบายว่า CI width เปลี่ยนอย่างไร
5. **ตีความ**: CI (L, U) หมายความว่าอะไรในบริบทของ dataset นี้?

**Deliverable**: Code + 3 plots (95%, 90%, 99%) + คำอธิบาย 1 ย่อหน้า

---

## ข้อ 3 — Hypothesis Testing และ Case Study (3 คะแนน)

ในข้อนี้คุณจะทดสอบ hypothesis เพื่อตอบคำถามที่มีความหมายทางธุรกิจหรือวิทยาศาสตร์ เพื่อให้เชื่อมโยง statistical inference กับการตัดสินใจจริง

**สิ่งที่ต้องทำ**:
1. ตั้ง **hypothesis ที่น่าสนใจ** จาก dataset ของคุณ เช่น:
   - "กลุ่ม A มียอดขายมากกว่ากลุ่ม B" (two-sample t-test)
   - "ตัวแปร X และ Y มีความสัมพันธ์กัน" (correlation t-test)
2. ระบุ **H₀** และ **H₁** ชัดเจน
3. เลือก **test ที่เหมาะสม**: `ttest_1samp`, `ttest_ind`, `chi2_contingency`
4. รายงานผล: t-statistic (หรือ χ²), degrees of freedom, p-value, สรุป
5. **Case Study**: ยกตัวอย่างสถานการณ์จริงที่ใช้ hypothesis test นี้ใน industry (เช่น A/B testing, clinical trial) พร้อมอธิบาย 1 ย่อหน้า

**Deliverable**: Code + ผล test + สรุป 1 ย่อหน้า + Case Study 1 ย่อหน้า

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้วทุก cell
- ตั้งชื่อ: `hw07_XXXXXXXX.ipynb`
- ทุก plot ต้องมี title, axis label, legend ครบ

## เกณฑ์การให้คะแนน (ดูรายละเอียดใน hw07_rubric.md)

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | EDA Full Pipeline | 4 |
| 2 | Confidence Interval | 3 |
| 3 | Hypothesis Test + Case Study | 3 |
| **รวม** | | **10** |
