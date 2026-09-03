# Rubric — HW06: Bias-Variance Trade-Off and Model Assessment
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 6

---

## ข้อ 1 — MSE Decomposition โดยมือ (25 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **1a: คำนวณ β̂₁** | β̂₁_A และ β̂₁_B ถูกต้องทั้งคู่ พร้อมแสดงสูตร Σ(xy)/Σ(x²) | ถูก 1 ตัว ผิด 1 ตัว หรือถูกแต่ไม่แสดงสูตร | แสดงสูตรถูกต้องแต่คำนวณตัวเลขผิด | ไม่รู้จักสูตรหรือไม่มีคำตอบ | 8 |
| **1b-1c: Bias คำนวณ** | mean(predictions) ถูกต้อง, Bias = mean − f(3) ถูกต้อง | ผิดเล็กน้อยจาก error propagation ของ β̂₁ | คำนวณ mean prediction ผิดแต่ใช้สูตร Bias ถูก | ไม่เข้าใจความหมาย Bias | 7 |
| **1d: Variance คำนวณ** | Variance ของ {pred_A, pred_B} คำนวณถูกต้อง | ใช้ population variance แทน sample (หาร n แทน n-1) — รับได้ | คำนวณผิดแต่แสดงสูตรถูก | ไม่รู้จัก Variance ของ predictions | 5 |
| **1e: E[Test MSE]** | Bias² + Var + Irred คำนวณถูกต้อง ครบ 3 ส่วน | ลืม Var(ε) หรือคำนวณ Bias² ผิด 1 ส่วน | บวก 2 ส่วนแต่ขาด 1 | ไม่เข้าใจ decomposition | 5 |

**เฉลย:**
```
β̂₁_A = (1×2.3 + 2×4.8 + 3×6.1 + 4×8.5 + 5×9.9) / (1+4+9+16+25)
      = (2.3 + 9.6 + 18.3 + 34.0 + 49.5) / 55 = 113.7/55 ≈ 2.067

β̂₁_B = (1×1.7 + 2×3.2 + 3×7.4 + 4×7.6 + 5×10.8) / 55
      = (1.7 + 6.4 + 22.2 + 30.4 + 54.0) / 55 = 114.7/55 ≈ 2.085

pred_A(3) = 2.067 × 3 = 6.200
pred_B(3) = 2.085 × 3 = 6.255

mean_pred = (6.200 + 6.255) / 2 = 6.228
Bias = 6.228 − 6 = 0.228
Bias² = 0.052

Variance = ((6.200 − 6.228)² + (6.255 − 6.228)²) / 2 ≈ 0.0008
  (หมายเหตุ: variance ต่ำมากเพราะ 2 estimates ใกล้กัน)

E[Test MSE] ≈ 0.052 + 0.001 + 1.0 ≈ 1.053
```

**Note สำหรับ Instructor:** สูตร β̂₁ = Σ(xy)/Σ(x²) คือ OLS without intercept ไม่ใช่ standard (Σ(xy) - n·x̄·ȳ)/(Σ(x²) - n·x̄²) นักศึกษาต้องใช้สูตรที่โจทย์กำหนดเท่านั้น

---

## ข้อ 2 — Python Simulation: U-Curve (35 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **2a: Train/Test Split** | split ถูกต้อง random_state=99, reshape ถูก | random_state ต่างกันเล็กน้อย | split ผิดสัดส่วนแต่ทำ simulation ได้ | ไม่ทำ split หรือ test บน train set | 3 |
| **2b: U-curve plot** | plot มี: train MSE line, test MSE line, horizontal Var(ε) line, axis labels, title, legend | ขาด 1–2 element ใน plot | plot มีแค่ test MSE หรือ train MSE | ไม่มี plot หรือ code ผิดทั้งหมด | 12 |
| **2c: Optimal degree** | ระบุ degree ที่ min test MSE ถูกต้องจาก code ของตัวเอง | ระบุถูกแต่ไม่มี code ยืนยัน | ระบุผิดไป 1 degree | ไม่ระบุ | 5 |
| **2d: Bias-Variance decomp** | simulation loop ถูกต้อง คำนวณ mean_pred, bias_sq, variance ถูกต้อง, plot ครบ 5 degrees | loop ถูกแต่คำนวณ bias_sq ผิด (ใช้ mean absolute แทน square) | มี loop แต่ logic ผิดบางส่วน | ไม่มี simulation | 10 |
| **Code quality** | ทุก block มี comment อธิบาย, variable names ชัดเจน | ส่วนใหญ่มี comment | มี comment บางส่วน | ไม่มี comment เลย | 5 |

**Expected output สำหรับ 2c:**
Optimal degree ควรอยู่แถว 4–6 (ขึ้นกับ random split) — รับทุกค่าใน [3,7] ถ้า code ถูกต้อง

**Expected output สำหรับ 2d:**
```
Degree 1:  Bias² สูง (~1.5+), Variance ต่ำ (~0.02)  ← Underfit
Degree 5:  Bias² ต่ำ (~0.1), Variance ต่ำพอสมควร   ← Near optimal
Degree 14: Bias² ต่ำมาก (~0.05), Variance สูงมาก (~2+) ← Overfit
Total ≈ Bias² + Variance + 1.0 ทุก degree
```

---

## ข้อ 3 — Short Essay (25 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **3a: อธิบาย U-curve** | อธิบายได้ครบ: training MSE ลดเพราะ model memorize, test MSE เพิ่มเพราะ variance เพิ่มกว่า bias ที่ลด, เชื่อมกับ noise | อธิบายได้ 2 ใน 3 ส่วน | อธิบาย training MSE ถูกแต่ test MSE ไม่ชัด | อธิบายผิดหรือไม่ตอบ | 10 |
| **3b: noise เพิ่ม → U-curve เปลี่ยน** | ระบุว่า optimal degree เลื่อนซ้าย (simpler), Irreducible Error เพิ่ม, total E[MSE] เพิ่ม | ระบุว่า MSE เพิ่มแต่ไม่อธิบาย optimal degree | อธิบายผลบางส่วนถูก | ตอบผิดทิศทาง | 8 |
| **3c: "always" หมายความว่า** | อธิบายว่า decomposition เป็น mathematical result ใช้ได้กับ model ทุกชนิด (ไม่ขึ้นกับ algorithm) เพราะ derive จาก E[(Y-f̂)²] โดยตรง | ระบุว่า apply ได้กว้างแต่ไม่อธิบาย why | ระบุว่า apply กับ some models | อธิบายผิดหรือไม่ตอบ | 7 |

**เฉลยสำหรับ 3b:**
ถ้า noise_std เพิ่มจาก 1 → 3:
- Irreducible Error = 9 (เพิ่มจาก 1)
- Total E[MSE] เพิ่มทุก degree
- Optimal degree จะเลื่อนไปทาง simpler (ซ้าย) เพราะ signal-to-noise ratio ต่ำลง → ไม่คุ้มที่จะ fit complex model
- Bias² curve แทบไม่เปลี่ยน (ขึ้นกับ model form ไม่ใช่ noise)

---

## ข้อ 4 — Open-Ended Dataset (15 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| **Dataset & feature selection** | โหลดถูกต้อง, เลือก feature ด้วย correlation, อธิบายว่าทำไม | โหลดถูกแต่เลือก feature โดยไม่อธิบาย | ใช้ dataset แต่เลือก feature โดย random | Dataset ผิดประเภทหรือ target ไม่ใช่ regression | 3 |
| **U-curve plot** | plot ถูกต้อง มี horizontal irreducible error line | plot มี train/test MSE แต่ขาด elements บางอย่าง | plot อยู่แต่ผิดหรือ axis ผิด | ไม่มี plot | 7 |
| **วิเคราะห์** | ระบุ optimal degree + อธิบาย near-linear/nonlinear ถูกต้อง | ระบุ optimal degree แต่วิเคราะห์ลักษณะ data ไม่ชัด | มีคำตอบแต่ไม่มี evidence | ไม่มีการวิเคราะห์ | 5 |

---

## Bonus (สูงสุด 5 คะแนน)

- (+3) นำ Cross-Validation (ด้วย `cross_val_score`) มาแทนที่ single train/test split ในข้อ 2b และเปรียบเทียบผล
- (+2) Plot ภาพ decision boundary ของ KNN สำหรับ K_optimal บน classification dataset จาก Lab 06

---

## Notes สำหรับ Instructor

1. **ข้อ 1**: ยอมรับ Variance ที่ใช้ n หรือ n-1 — ทั้งคู่ถูกต้องขึ้นกับ convention
2. **ข้อ 2**: optimal degree อาจต่างกันในแต่ละ run ถ้า random_state ต่างกัน — ตรวจจาก code logic ไม่ใช่ค่าตัวเลข
3. **ข้อ 3b**: คำตอบหลักที่ต้องการคือ "optimal degree ลดลง" และ "irreducible error เพิ่ม" — ที่เหลือเป็น bonus explanation
4. **ข้อ 4 Option C**: ถ้านักศึกษาใช้ dataset ของตัวเอง ตรวจว่าเป็น regression problem จริง (Y continuous) และมี ≥500 rows ตาม requirement
