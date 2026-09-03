# Homework 09: Multiple Linear Regression Analysis
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO3 | **LLo**: สร้าง MLR, ตีความ F-statistic และเลือก predictor ที่สำคัญได้  
**กำหนดส่ง**: สัปดาห์ที่ 10  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราเรียนรู้ **Multiple Linear Regression (MLR)** ซึ่งเป็นการขยาย SLR ให้รองรับ predictor ได้หลายตัวพร้อมกัน จุดสำคัญที่ต้องเข้าใจคือ β̂ⱼ ใน MLR หมายถึงผลของ Xⱼ ต่อ Y **โดยยึด predictor ตัวอื่นคงที่** ซึ่งต่างจาก SLR ที่ไม่มีการ control เรายังเรียนเรื่อง F-statistic ที่ทดสอบว่า predictors ทั้งหมดมีความสัมพันธ์กับ Y หรือไม่ รวมถึง Adjusted R² และ AIC/BIC สำหรับเลือก model ที่ดีที่สุด ในการบ้านนี้คุณจะฝึกสร้าง MLR บน dataset จริง ตีความ F-statistic และ t-statistic พร้อมกัน ทำ variable selection และเขียน business report สรุปผล ทักษะเหล่านี้ใช้ตลอดในงาน Data Science ตั้งแต่การวิเคราะห์ปัจจัยที่ส่งผลต่อยอดขายไปจนถึงการพยากรณ์ราคาอสังหาริมทรัพย์

---

## ข้อ 1 — MLR: Normal Equations และ statsmodels (3 คะแนน)

ในข้อนี้คุณจะสร้าง MLR บน Advertising dataset ด้วยสองวิธีและ verify ว่าให้ผลเท่ากัน เพื่อให้เข้าใจว่า statsmodels ทำอะไรอยู่เบื้องหลัง

**Dataset**: Advertising.csv — Fit model: Sales ~ TV + Radio + Newspaper

**สิ่งที่ต้องทำ**:
1. **Normal Equations** (Week 3): สร้าง design matrix X (200×4, column แรกเป็น 1) และคำนวณ β̂ = (XᵀX)⁻¹Xᵀy ด้วย `np.linalg.lstsq()`
2. **statsmodels**: `smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit()` แสดง `.summary()`
3. **Verify**: β̂ จากทั้งสองวิธีตรงกัน (ใช้ `np.allclose`)
4. **ตาราง**:

| Coefficient | Normal Equations | statsmodels |
|-------------|-----------------|-------------|
| β̂₀ (Intercept) | | |
| β̂₁ (TV) | | |
| β̂₂ (Radio) | | |
| β̂₃ (Newspaper) | | |

5. เปรียบเทียบ β̂_TV จาก SLR (Sales ~ TV) กับ MLR — ทำไมค่าเปลี่ยน?

**Expected values**: β̂₀ ≈ 2.94, β̂_TV ≈ 0.046, β̂_Radio ≈ 0.189, β̂_News ≈ −0.001

**Deliverable**: Code + output + ตารางเปรียบเทียบ + คำอธิบาย 2–3 ประโยค

---

## ข้อ 2 — F-statistic และ t-statistic: ตีความร่วมกัน (3 คะแนน)

ในข้อนี้คุณจะฝึกตีความ F-statistic และ t-statistic พร้อมกัน เพื่อให้เห็นความต่างระหว่าง "model มีประโยชน์ไหม?" กับ "predictor ตัวนี้จำเป็นไหม?"

**ใช้ MLR model จากข้อ 1**

**สิ่งที่ต้องทำ**:
1. **F-statistic**:
   - บอกค่า F และ p-value จาก summary
   - คำนวณ verify ด้วยสูตร F = [(TSS−RSS)/p] / [RSS/(n−p−1)]
   - ตีความ: model มีประโยชน์ไหม?

2. **t-statistic ของแต่ละ predictor**:

| Predictor | β̂ | SE | t | p-value | Significant? |
|-----------|-----|-----|---|---------|-------------|
| Intercept | | | | | |
| TV | | | | | |
| Radio | | | | | |
| Newspaper | | | | | |

3. **ตอบคำถาม**:
   - F significant แต่ Newspaper p = 0.86 — ขัดแย้งกันไหม? อธิบาย
   - ถ้า F NOT significant แต่ t ของ TV significant — ควรเชื่อตัวใด? ทำไม?
   - ทำไม Newspaper ใน SLR มี p = 0.001 แต่ใน MLR p = 0.86?

**Deliverable**: Code + ตาราง + คำตอบ 2–3 ประโยคต่อคำถาม

---

## ข้อ 3 — Variable Selection และ Best Model (4 คะแนน)

ในข้อนี้คุณจะทำ variable selection เพื่อหา model ที่ดีที่สุดสำหรับ predict Sales โดยใช้ Adjusted R² และ AIC เป็นเกณฑ์ และเขียน business report สรุปผล

**สิ่งที่ต้องทำ**:
1. **Best Subset Selection**: ลอง model ทุก subset ที่เป็นไปได้ (2³−1 = 7 models) แสดงในตาราง:

| Model | p | R² | R²_adj | AIC | BIC |
|-------|---|-----|--------|-----|-----|
| TV | 1 | | | | |
| Radio | 1 | | | | |
| Newspaper | 1 | | | | |
| TV + Radio | 2 | | | | |
| TV + Newspaper | 2 | | | | |
| Radio + Newspaper | 2 | | | | |
| TV + Radio + Newspaper | 3 | | | | |

2. **เลือก Best Model** โดยใช้เกณฑ์ R²_adj และ AIC พร้อมอธิบาย

3. **Visualize**:
   - Scatter plot: actual vs predicted Sales (ใช้ best model)
   - Residual plot: residuals vs fitted values
   - Bar chart เปรียบเทียบ AIC ของทุก model

4. **Business Report** (5–7 ประโยค) สำหรับ CEO:
   - "Best model ใช้ X predictors ได้แก่ ___ อธิบาย Sales ได้ ___% (R² = ___)"
   - "งบโฆษณา TV ทุก $1,000 เพิ่ม → Sales เพิ่ม ___ หน่วย (95% CI: ___ ถึง ___)"
   - "งบโฆษณา Radio ทุก $1,000 เพิ่ม → Sales เพิ่ม ___ หน่วย"
   - "Newspaper ไม่มีผลต่อ Sales เมื่อ control TV และ Radio แล้ว — แนะนำให้ลดงบ Newspaper"
   - "limitation: model นี้เป็น linear — อาจต้องตรวจ non-linearity และ interaction (Week 10)"

**Deliverable**: Code + ตาราง model comparison + 3 plots + Business Report

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว
- ตั้งชื่อ: `hw09_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | MLR Normal Equations + statsmodels | 3 |
| 2 | F-statistic + t-statistic | 3 |
| 3 | Variable Selection + Business Report | 4 |
| **รวม** | | **10** |
