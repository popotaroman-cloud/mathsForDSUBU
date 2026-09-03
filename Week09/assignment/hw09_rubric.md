# Rubric — HW09: Multiple Linear Regression Analysis
**วิชา** 1145 201 | **CLO3**

---

## ข้อ 1 — MLR Normal Equations + statsmodels (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Normal Equations | สร้าง X ถูกต้อง (column 1 + 3 features), β̂ ถูกต้องและ allclose pass | สร้างได้แต่ไม่ verify | มีข้อผิดพลาดเล็กน้อย | ทำไม่ได้ | 1 |
| statsmodels summary | แสดง summary() ครบ อ่านค่า β̂ ทุกตัว | แสดงได้แต่ไม่อ่านค่า | fit ได้แต่ไม่แสดง | ทำไม่ได้ | 1 |
| Comparison + explanation | ตารางครบ อธิบาย confounding ว่าทำไม β̂_TV เปลี่ยน | ตารางครบแต่ไม่อธิบาย | ตารางไม่ครบ | ไม่ทำ | 1 |

---

## ข้อ 2 — F-statistic + t-statistic (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| F-statistic | อ่านค่า F ถูกต้อง คำนวณ verify ด้วยสูตร ตีความถูก | อ่านถูกแต่ไม่ verify | พยายามคำนวณ | ไม่คำนวณ | 1 |
| ตาราง t-statistic | ทุก row ถูกต้องครบ ระบุ significant/not | ถูก 3/4 | ถูก 2/4 | ผิดหมด | 1 |
| คำถาม 3 ข้อ | ตอบถูกทั้ง 3 อธิบาย multiple testing + confounding | ถูก 2/3 | ถูก 1/3 | ผิดหมด | 1 |

---

## ข้อ 3 — Variable Selection + Business Report (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Best subset table | ทุก 7 models ครบ R² R²_adj AIC BIC ถูก | 5/7 models | 3/7 models | ไม่ทำ | 1.5 |
| Visualization | actual vs predicted + residual plot + AIC bar chart ครบ 3 | 2/3 | 1/3 | ไม่ plot | 1 |
| เลือก best model | เลือกถูก (TV+Radio) อ้าง R²_adj และ AIC ชัดเจน | เลือกถูกแต่ไม่อ้างเหตุผล | เลือกผิด | ไม่เลือก | 0.5 |
| Business Report | ชัดเจน ไม่ใช้ jargon มีตัวเลข CI และ limitation | อธิบายได้แต่ jargon | อธิบายบางส่วน | ไม่เขียน | 1 |
