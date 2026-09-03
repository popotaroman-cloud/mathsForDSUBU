# Rubric — HW03: Subspaces และ Least Squares
**วิชา** 1145 201 | **CLO1**

---

## ข้อ 1 — Four Fundamental Subspaces (3 คะแนน)

| เกณฑ์ | Excellent | Good | Pass | Fail | คะแนน |
|-------|----------|------|------|------|-------|
| Rank, Nullity, Rank-Nullity Theorem | ถูกต้องครบ verify theorem ด้วย code | ถูก 2/3 | ถูก 1/3 | ผิดหมด | 1 |
| SVD หา Null Space | ถูกต้อง verify A @ null_vec ≈ 0 | ทำได้แต่ไม่ verify | พยายาม SVD แต่ index ผิด | ทำไม่ได้ | 1 |
| Summary Table | ครบ 4 subspaces พร้อม dimension | ครบ 3/4 | ครบ 2/4 | ไม่ทำ | 1 |

---

## ข้อ 2 — Least Squares on Data (4 คะแนน)

| เกณฑ์ | Excellent | Good | Pass | Fail | คะแนน |
|-------|----------|------|------|------|-------|
| Normal Equations | แก้ถูกต้อง ผลตรงกับ lstsq | แก้ถูก ไม่ compare | แก้ได้แต่ข้อผิดพลาดเล็กน้อย | ผิดหรือไม่ทำ | 1 |
| Residuals, RSS, TSS, R² | คำนวณถูกครบ | ถูก 3/4 | ถูก 2/4 | ผิดหมด | 1 |
| Plot | scatter + line + residuals ครบ label ถูก | ครบ 2/3 | ครบ 1/3 | ไม่ plot | 1 |
| ตีความ β̂₀, β̂₁, R² | ตีความถูกต้องครบ 3 อย่าง | ถูก 2/3 | ถูก 1/3 | ไม่ตีความ | 1 |

---

## ข้อ 3 — Polynomial Regression (3 คะแนน)

| เกณฑ์ | Excellent | Good | Pass | Fail | คะแนน |
|-------|----------|------|------|------|-------|
| Fit 3 models | Linear, Quad, Cubic ถูกครบ ด้วย lstsq | ถูก 2/3 | ถูก 1/3 | ไม่ทำ | 1 |
| R² และ RSE ครบทุก model | ถูกครบ 3 model | ถูก 2/3 | ถูก 1/3 | ผิดหมด | 1 |
| Plot + การตีความ | plot ครบ 3 curves อธิบาย overfitting ได้ | plot ได้แต่ตีความไม่ครบ | plot บางส่วน | ไม่ plot | 1 |
