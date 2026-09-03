# Homework 08: Simple Linear Regression Analysis
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO3 | **LLo**: สร้าง SLR, ประเมิน SE/t-stat/p-value และวัด fit ด้วย R² ได้  
**กำหนดส่ง**: สัปดาห์ที่ 9  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราได้เรียนรู้ **Simple Linear Regression (SLR)** ซึ่งเป็นโมเดลพื้นฐานที่สุดใน supervised learning โดยอาศัยคณิตศาสตร์จาก Week 3 (Least Squares) มาประยุกต์กับ data จริง การอ่านผลลัพธ์จาก `statsmodels` summary table ได้ถูกต้องเป็นทักษะที่ Data Scientist ทุกคนต้องมี ในการบ้านนี้คุณจะฝึกสร้าง SLR ด้วยสามวิธี คำนวณและตีความ SE t-statistic p-value CI และ R² รวมถึงเขียนรายงานผลในแบบที่ผู้บริหารเข้าใจได้ ทักษะเหล่านี้จะนำไปใช้โดยตรงใน Week 9 (MLR) และตลอดการทำงานใน Data Science

---

## ข้อ 1 — SLR จาก Scratch และเปรียบเทียบ 3 วิธี (4 คะแนน)

ในข้อนี้คุณจะ implement SLR ด้วยมือก่อนแล้วจึงใช้ library เพื่อแสดงให้เห็นว่า math กับ code ให้ผลเหมือนกัน ทำให้เข้าใจว่า sklearn และ statsmodels ทำอะไรอยู่เบื้องหลัง

**Dataset**: Advertising.csv — Fit model: Sales ~ TV

**สิ่งที่ต้องทำ**:
1. **Manual**: คำนวณ β̂₁ และ β̂₀ จากสูตร Least Squares:
   $$\hat{\beta}_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$
2. **sklearn**: `LinearRegression().fit(X, y)` — verify ผลตรงกับ manual
3. **statsmodels**: `smf.ols('Sales ~ TV', data=df).fit()` — แสดง `.summary()`
4. Verify: β̂₀ ≈ 7.03, β̂₁ ≈ 0.0475 ทั้ง 3 วิธี
5. Plot: scatter + fitted line พร้อม annotation ของ equation

**Deliverable**: Code + output + เปรียบเทียบ β̂₀, β̂₁ จาก 3 วิธีในตาราง

---

## ข้อ 2 — อ่านและตีความ statsmodels Summary (3 คะแนน)

ในข้อนี้คุณจะฝึกอ่าน summary table ของ regression ซึ่งเป็นสิ่งที่ต้องทำทุกครั้งเมื่อสร้างโมเดล เพื่อให้ตอบได้ว่า predictor มีความสัมพันธ์กับ response จริงหรือไม่ และโมเดล fit ข้อมูลได้ดีแค่ไหน

**ใช้ result จาก statsmodels ข้อ 1**

**สิ่งที่ต้องทำ**:
1. อ่านค่าต่อไปนี้จาก summary และบอกความหมาย:
   - `coef` ของ TV: β̂₁ = _____ หมายความว่าอะไร?
   - `std err` ของ TV: SE(β̂₁) = _____
   - `t` ของ TV: t = β̂₁ / SE(β̂₁) = _____ คำนวณ verify
   - `P>|t|`: p-value = _____ แปลว่าอะไร?
   - `[0.025, 0.975]`: 95% CI = (_____, _____) — 0 อยู่ใน CI ไหม?
2. คำนวณ **RSE** = √(RSS/(n−2)) และ **R²** = 1 − RSS/TSS ด้วยตัวเอง
3. ตีความ R² = 0.612: หมายความว่าอะไร? อะไรอธิบาย 38.8% ที่เหลือ?

**Deliverable**: ตารางสรุปค่าทั้งหมด + คำอธิบาย 2–3 ประโยคต่อรายการ

---

## ข้อ 3 — Case Study: Radio vs Sales (3 คะแนน)

ในข้อนี้คุณจะ fit SLR ด้วยตัวเองสำหรับ predictor ตัวใหม่และเขียน business report เพื่อแสดงว่า regression analysis สามารถสนับสนุนการตัดสินใจทางธุรกิจได้อย่างไร

**Scenario**: ทีม marketing ต้องการรู้ว่าการเพิ่มงบ Radio ส่งผลต่อยอดขายหรือไม่ ก่อนตัดสินใจเพิ่ม budget

**สิ่งที่ต้องทำ**:
1. Fit `Sales ~ Radio` ด้วย statsmodels
2. รายงานค่า β̂₀, β̂₁, SE, t, p-value, R², RSE
3. Visualize: scatter + fitted line + residual plot
4. **เปรียบเทียบ** ผล Radio กับ TV จากข้อ 1 — ตัวใด fit ดีกว่า? (R² สูงกว่า)
5. **เขียน Business Report** 1 ย่อหน้า (5–7 ประโยค) สำหรับ Marketing Manager ที่ไม่ใช่นักสถิติ:
   - "เพิ่มงบ Radio 1,000 ดอลลาร์ → ยอดขายเพิ่มเฉลี่ย ___ หน่วย"
   - "ความเชื่อมั่น 95%: ผลนั้นอยู่ระหว่าง ___ ถึง ___ หน่วย"
   - "ทั้ง TV และ Radio มีผลต่อ Sales อย่างมีนัยสำคัญ — แต่จะใช้ทั้งสองพร้อมกันควรใช้ MLR (Week 9)"

**Deliverable**: Code + output + Business Report

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว
- ตั้งชื่อ: `hw08_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | SLR 3 วิธี | 4 |
| 2 | อ่าน Summary | 3 |
| 3 | Case Study + Business Report | 3 |
| **รวม** | | **10** |
