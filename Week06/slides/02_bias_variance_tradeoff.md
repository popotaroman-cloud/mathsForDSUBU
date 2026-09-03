# Slide Deck: Bias-Variance Trade-Off
> Week 6 | CLO2 | ISLP Ch.2.2.2 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 6**  
The Bias-Variance Trade-Off: ทำไม U-curve ถึงเกิด?  
CLO2: อธิบาย Bias-Variance Decomposition และ identify optimal model complexity ได้

---

## Slide 2 — The Core Question
**Key Message**: ทำไม Test MSE ถึงมี U-shape? — เพราะ 2 ส่วนของ error "ต่อสู้กัน" เมื่อ complexity เปลี่ยน

คำถามที่ Deck 1 ทิ้งไว้คือ "ทำไม Test MSE ถึงมี U-shape?" ไม่ได้เกิดขึ้นโดยบังเอิญ — มีคำอธิบายทางคณิตศาสตร์ที่ชัดเจน คือ Expected Test MSE ประกอบด้วย 3 ส่วนที่ทุกส่วนขึ้นกับ complexity แตกต่างกัน เมื่อ complexity เพิ่ม ส่วนหนึ่งลด อีกส่วนเพิ่ม จนได้ U-curve แนวคิดนี้เรียกว่า Bias-Variance Trade-Off และเป็น "กฎพื้นฐาน" ของ Statistical Learning ที่ไม่มีทางหลีกเลี่ยง

**Preview ของ Decomposition:**
```
E[(Y − f̂(x₀))²] = Bias²(f̂(x₀)) + Var(f̂(x₀)) + Var(ε)
                    ─────────────   ─────────────   ──────────
                    ↑ complexity    ↓ complexity    ลดไม่ได้
                    → ลด           → เพิ่ม
```

---

## Slide 3 — Decomposition: สูตรและความหมาย
**Key Message**: Expected Test MSE = Bias² + Variance + Irreducible Error — แต่ละส่วนมีความหมายทางกายภาพที่ชัดเจน

**สูตร (ISLP 2.2.2):**
```
E[(y₀ − f̂(x₀))²] = Var(f̂(x₀)) + [Bias(f̂(x₀))]² + Var(ε)
```

**ความหมายของแต่ละส่วน:**

**Variance(f̂(x₀)):**
```
= ความแปรปรวนของ f̂(x₀) เมื่อ training data เปลี่ยน
= ถ้า model sensitivity สูง → f̂ เปลี่ยนมากเมื่อ training data ต่างกัน
→ High flexibility = High Variance
```

**Bias(f̂(x₀)):**
```
= E[f̂(x₀)] − f(x₀)
= ความห่างระหว่าง "ค่าเฉลี่ยของ predictions" กับ "true f"
→ High simplicity (wrong assumption) = High Bias
```

**Var(ε):**
```
= noise ของ data (irreducible)
= ไม่สามารถลดได้ ไม่ว่า model จะดีแค่ไหน
```

---

## Slide 4 — Bias: เข้าใจด้วยรูปภาพ
**Key Message**: Bias คือ "systematic error" — model สมมติผิดทำให้ prediction ผิดทิศเสมอ ไม่ใช่แค่บางครั้ง

**[FIGURE: 3 ภาพซ้อนกัน
- True f = sine curve
- ทุกๆ training set (100 datasets) → fit linear model → 100 เส้นตรง
- Mean ของ 100 เส้นตรง = อยู่ห่างจาก sine curve อย่างชัดเจน
- Gap ระหว่าง mean prediction กับ true f = Bias]**

**ตัวอย่าง:**
```
True f(x) = sin(x)  ← curve
f̂(x) = β₀ + β₁x   ← linear (wrong assumption)

Bias = E[f̂(x₀)] − f(x₀) ≠ 0  ← ผิดทิศเสมอ
แม้ training data จะมีมากแค่ไหน → Bias ไม่ลด!
เพราะ linear model ไม่สามารถ capture sine curve ได้
```

---

## Slide 5 — Variance: เข้าใจด้วยรูปภาพ
**Key Message**: Variance คือ "instability" — model ที่ flexible เกินไปเปลี่ยนแปลงมากตาม training data ที่ต่างกัน

**[FIGURE: 3 ภาพซ้อนกัน
- True f = sine curve
- ทุกๆ training set → fit polynomial degree 15 → 100 curves ที่ wiggly
- Curves กระจายกันมาก ที่จุดเดียวกัน predictions ต่างกันมาก
- Variance = spread ของ predictions รอบ mean]**

**ตัวอย่าง:**
```
f̂_A(x₀) = 2.1   ← from training set A
f̂_B(x₀) = -0.5  ← from training set B  
f̂_C(x₀) = 3.7   ← from training set C

Variance = ความแปรปรวนของ {2.1, -0.5, 3.7, ...}
         = HIGH ← model ไม่ stable
```

**Key Insight:** Variance สูง ≠ Bias สูง — เราต้องพิจารณาทั้งคู่

---

## Slide 6 — Trade-Off: ทำไม 2 ส่วนนี้ต่อสู้กัน
**Key Message**: Flexibility ที่เพิ่มขึ้นลด Bias แต่เพิ่ม Variance — ไม่มีทางลดทั้งคู่พร้อมกัน

```
                    Bias²              Variance
                    ─────              ────────
Simple model    →   HIGH               LOW
  (linear)          (wrong form)       (stable)
  
Complex model   →   LOW                HIGH
  (degree 15)       (correct form)     (unstable)
```

**[FIGURE: 2 graphs ─ ซ้าย: Bias² ลดเมื่อ complexity เพิ่ม; ขวา: Variance เพิ่มเมื่อ complexity เพิ่ม]**

**ผลรวม: Total E[Test MSE] = Bias² + Variance + const.**
```
Low complexity:  HIGH Bias² + LOW Var = High Total (underfit region)
High complexity: LOW Bias² + HIGH Var = High Total (overfit region)
Middle:          Balanced Bias² + Var = Minimum Total ← Sweet Spot
```

---

## Slide 7 — Simulation: เห็นด้วยตาตัวเอง
**Key Message**: Simulation 200 training sets แสดงให้เห็น U-curve จริงๆ — Bias² + Variance + Irred ≈ Total E[MSE]

**[FIGURE: กราฟ Bias-Variance Decomposition]**
```
แกน X: Polynomial Degree (1 → 15)
แกน Y: Expected MSE

เส้น Bias² (น้ำเงิน):    ลดเรื่อยๆ ตาม degree
เส้น Variance (แดง):     เพิ่มเรื่อยๆ ตาม degree
เส้น Irreducible (เทา):  คงที่ = Var(ε)
เส้น Total (ดำ หนา):     U-shape → minimum ที่ optimal degree
```

**ยืนยันด้วยตัวเลข (ตัวอย่าง):**

| Degree | Bias² | Variance | Irred | Total |
|--------|-------|---------|-------|-------|
| 1 | 0.82 | 0.02 | 0.25 | 1.09 |
| 4 | 0.11 | 0.08 | 0.25 | 0.44 ← min |
| 10 | 0.03 | 0.52 | 0.25 | 0.80 |
| 15 | 0.02 | 1.85 | 0.25 | 2.12 |

→ Bias² + Variance + 0.25 ≈ Total ✓

---

## Slide 8 — ปัจจัยที่ส่งผลต่อ Optimal Complexity
**Key Message**: Optimal degree ไม่ใช่ค่าคงที่ — ขึ้นกับขนาด dataset และระดับ noise

**ปัจจัย 1 — n (จำนวน observations):**
```
n เล็ก → Variance สูงแม้ degree ต่ำ → optimal degree ต่ำ
n ใหญ่ → Variance ควบคุมได้ → optimal degree สูงขึ้น
→ "ข้อมูลมากขึ้น → ใช้ model ที่ complex กว่าได้อย่างปลอดภัย"
```

**ปัจจัย 2 — Var(ε) (ระดับ noise):**
```
noise สูง → irreducible error สูง → U-curve เลื่อนขึ้น
noise สูง → optimal ชอบ simpler model (ลด variance)
→ "data มี noise เยอะ → ใช้ model ง่ายๆ"
```

**ปัจจัย 3 — True complexity ของ f:**
```
f เกือบ linear → low degree เพียงพอ
f ซับซ้อน (หลาย wave) → ต้องการ degree สูง
→ "รู้จัก domain ของตัวเอง"
```

---

## Slide 9 — ใช้ Bias-Variance ในการเลือก Model
**Key Message**: Bias-Variance framework ช่วย diagnose ปัญหาของ model และเลือกวิธีแก้ที่ถูกต้อง

**Diagnostic Flowchart:**
```
Train MSE สูงมาก?
└─ Yes → High Bias (Underfitting) → เพิ่ม model complexity
                                  → เพิ่ม features
                                  → เปลี่ยน model form

Train MSE ต่ำ แต่ Test MSE สูง?
└─ Yes → High Variance (Overfitting) → เพิ่ม training data
                                     → ลด complexity (regularize)
                                     → Dropout (Neural Networks)

ทั้ง Train และ Test MSE ต่ำ?
└─ Yes → Good! ลองหา simpler model ที่ได้ผลใกล้เคียง
```

**"No Free Lunch" Theorem (informal):**  
ไม่มี model ที่ดีที่สุดสำหรับทุกปัญหา — optimal จะแตกต่างตาม dataset

---

## Slide 10 — Summary & Preview
**สิ่งที่เรียนรู้วันนี้**

```
E[(Y − f̂)²] = Bias²(f̂) + Var(f̂) + Var(ε)
                ↓ ด้วย flexibility  ↑ ด้วย flexibility  ลดไม่ได้
```

| ส่วน | ขึ้นกับ | แก้ด้วย |
|------|--------|--------|
| **Bias²** | Model form ผิด / ง่ายเกินไป | เพิ่ม complexity, features |
| **Variance** | Model ไม่ stable | เพิ่ม data, regularize, simplify |
| **Irreducible** | Noise ใน data | เก็บ features ที่ดีกว่า |

- **U-curve**: เกิดจาก Bias² ↓ + Variance ↑ เมื่อ complexity เพิ่ม
- **Optimal**: จุดที่ Bias² + Variance รวมกันต่ำที่สุด
- **n ใหญ่**: Variance ลด → ใช้ complex model ได้มากขึ้น

**สัปดาห์ถัดไป (Slide Deck 3):** Classification Setting — Error Rate, Bayes Classifier, KNN
