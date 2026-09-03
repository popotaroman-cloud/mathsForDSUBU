# Rubric — HW10: Regression Diagnostics and Extensions
**วิชา** 1145 201 | **CLO3**

---

## ข้อ 1 — Dummy Variable + Interaction (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Dummy variable | สร้าง TV_level ถูกต้อง fit ด้วย C() ตีความ coefficient ถูก | สร้างได้แต่ตีความผิด | สร้างได้บางส่วน | ทำไม่ได้ | 1 |
| Interaction model | fit TV:Radio ถูก เปรียบ R² ชัดเจน ตีความ β̂_TV:Radio ถูก | fit ได้แต่ตีความผิด | fit ได้ | ทำไม่ได้ | 1.5 |
| Hierarchy principle | อธิบายว่าต้องใส่ main effects ด้วยเมื่อมี interaction | อธิบายบางส่วน | ไม่อธิบาย | ผิด | 0.5 |

---

## ข้อ 2 — Regression Diagnostics (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| 4 Diagnostic plots | ครบทั้ง 4 plots อ่านผลถูกต้องทุก plot | ครบ 3/4 | ครบ 2/4 | ครบ 1/4 | 1.5 |
| อ่านผลและระบุปัญหา | ระบุ non-linearity ชัดเจน อธิบาย pattern ใน residual plot ถูก | ระบุได้ 1–2 ปัญหา | ระบุได้บางส่วน | ไม่อ่านผล | 1 |
| Polynomial fix | fit polynomial ถูก เปรียบ R² แสดง diagnostic ใหม่ที่ดีขึ้น | fit ได้แต่ไม่เปรียบ | fit polynomial | ทำไม่ได้ | 1 |
| Outlier identification | หา studentized residual > 2 ถูกต้อง ระบุ index และค่า | หาได้แต่ไม่ระบุ | พยายาม | ไม่ทำ | 0.5 |

---

## ข้อ 3 — VIF + Diagnostics Report (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Correlation heatmap + VIF table | heatmap + VIF ครบทุก predictor ระบุตัวที่มีปัญหา | VIF ถูก 3/4 | VIF ถูกบางส่วน | ไม่คำนวณ | 1 |
| Fix multicollinearity | ทำ fix ถูก (ลบ/PCA) เปรียบ AIC/R²_adj ก่อน-หลัง | fix ได้แต่ไม่เปรียบ | พยายาม | ไม่ทำ | 1 |
| Diagnostics Report | ชัดเจน ไม่ jargon ระบุปัญหา fix ผลหลัง fix และ limitation | อธิบายได้แต่ jargon | อธิบายบางส่วน | ไม่เขียน | 1 |
