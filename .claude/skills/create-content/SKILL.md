---
description: สร้าง content สำหรับ Week ที่กำหนด — รับ argument เป็น "lecture|lab|assignment|slides|all week N"
argument-hint: <type> week <number>
allowed-tools: Read, Write, Glob, Bash
---

# Create Content: $ARGUMENTS

---

## Content Standards (ใช้กับทุก Week ทุก Stream)

---

### 🔴 กฎบังคับ 3 ข้อ (ใช้กับทุก Stream ทุก cell ทุกประโยค)

#### กฎ 1 — บทนำสัปดาห์ (Week Introduction)
**ทุก content ไม่ว่าจะเป็น lecture / lab / assignment / slides ต้องเริ่มด้วยบทนำสัปดาห์ 5–10 ประโยค** ที่ครอบคลุม 3 คำถามนี้เสมอ:

> 1. **สัปดาห์นี้เกี่ยวกับอะไร?** — อธิบาย concept หลักที่จะเรียนในสัปดาห์นี้
> 2. **เป้าหมายคืออะไร?** — นักศึกษาจะทำอะไรได้หลังเรียนจบ (ตรงกับ LLo ใน plan.md)
> 3. **นำไปใช้ทำอะไรได้?** — ยกตัวอย่าง use case จริงใน Data Science / อาชีพ

ตัวอย่างบทนำที่ดี:
```
สัปดาห์นี้เราจะเรียนรู้เรื่อง Multiple Linear Regression (MLR) ซึ่งเป็นการขยาย
Simple Linear Regression ให้รับ predictor ได้มากกว่า 1 ตัว แนวคิดหลักคือการ
อธิบายการเปลี่ยนแปลงของ Y ด้วย X หลายตัวพร้อมกัน โดยยึด predictor ตัวอื่น
คงที่ เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถสร้างโมเดล MLR อ่านผลลัพธ์
จาก F-statistic และ t-statistic ได้ รวมถึงเลือก predictor ที่สำคัญด้วย
Adjusted R² ได้อย่างถูกต้อง ในชีวิตจริง MLR ถูกใช้ในการวิเคราะห์ปัจจัยที่
ส่งผลต่อยอดขาย การพยากรณ์ราคาบ้าน และการวิเคราะห์ความเสี่ยงในภาคการเงิน
ทักษะนี้เป็นพื้นฐานสำคัญก่อนเรียน Machine Learning และ Feature Engineering
ในปีการศึกษาถัดไป
```

---

#### กฎ 2 — ทุก Code ต้องมีคำอธิบาย (Code Documentation)
**ทุกบรรทัด หรือทุก block ของ code ต้องมี comment อธิบายเสมอ** ทั้งใน lab notebook และ lecture note

รูปแบบ comment ที่ต้องใช้:

```python
# ─── [ทำอะไร] ───────────────────────────────────────────
# วัตถุประสงค์: [เพื่ออะไร — อธิบายในแง่ concept ไม่ใช่แค่ syntax]
code_here

# ตัวอย่างที่ถูกต้อง:
# ─── โหลด Advertising dataset ──────────────────────────
# วัตถุประสงค์: ใช้ข้อมูลยอดขายและงบโฆษณาเพื่อสร้าง regression model
df = pd.read_csv('Advertising.csv')

# แสดง 5 แถวแรกเพื่อตรวจสอบโครงสร้างข้อมูล
# วัตถุประสงค์: verify ว่า column names และ data types ถูกต้องก่อนนำไปใช้
df.head()

# สร้าง design matrix X โดยเพิ่ม column ค่า 1 สำหรับ intercept
# วัตถุประสงค์: β̂ = (XᵀX)⁻¹Xᵀy ต้องการ column แรกเป็น 1 แทน β₀
X = np.column_stack([np.ones(len(df)), df['TV'], df['Radio']])
```

**ข้อห้าม:**
```python
# ❌ ห้าม — comment ไม่บอกเหตุผล
df = pd.read_csv('Advertising.csv')  # read csv

# ❌ ห้าม — ไม่มี comment เลย
model = LinearRegression().fit(X, y)
r2 = model.score(X, y)
```

---

#### กฎ 4 — Case Study อย่างน้อย 1 เรื่องต่อ Week (Real-World Application)
**ทุก Week ต้องมี Case Study อย่างน้อย 1 เรื่อง** ที่แสดงให้เห็นว่า concept ของสัปดาห์นั้นถูกนำไปใช้งานจริงอย่างไร

**Case Study ที่ดีต้องประกอบด้วย 5 ส่วนนี้เสมอ:**

```
┌─────────────────────────────────────────────────────────┐
│  CASE STUDY: [ชื่อเรื่อง]                               │
│                                                         │
│  1. Scenario   — สถานการณ์จริงคืออะไร? ใครเป็นผู้ใช้?  │
│  2. Data       — ข้อมูลอะไร? มาจากไหน? มีกี่ตัวแปร?    │
│  3. Method     — ใช้ concept สัปดาห์นี้อย่างไร?         │
│  4. Result     — ผลลัพธ์ที่ได้คืออะไร? ตัวเลขจริง?     │
│  5. Insight    — ตัดสินใจ/สรุปอะไรจากผลนั้น?            │
└─────────────────────────────────────────────────────────┘
```

**ใส่ Case Study ที่ไหนในแต่ละ stream:**

| Stream | ตำแหน่ง | รูปแบบ |
|--------|---------|--------|
| **Lecture Note** | section `## Case Study` ก่อน Summary | เขียนเต็ม 5 ส่วน พร้อมเลขจริงและ Python snippet |
| **Lab Notebook** | Part สุดท้ายก่อน Reflection | นักศึกษาทำ Case Study ด้วยตัวเอง (guided) |
| **Assignment** | ข้อสุดท้าย (ข้อ open-ended) | โจทย์ที่ให้นักศึกษาวิเคราะห์ Case Study dataset จริง |
| **Slides** | slide คู่ก่อนท้าย (Case Study slide) | Scenario + ผลลัพธ์ + Insight ใน 3 bullets |

**แหล่ง Case Study ที่แนะนำตาม topic:**

| Week / Topic | Case Study แนะนำ |
|-------------|----------------|
| W2 (Matrix) | Google PageRank — adjacency matrix ของ web graph |
| W3 (Subspaces) | Recommendation system — matrix factorization (Netflix) |
| W4 (SVD/PCA) | Face recognition — Eigenfaces (AT&T dataset) |
| W5 (SL Overview) | Spam filter — supervised classification บน email dataset |
| W6 (Bias-Variance) | Medical diagnosis — overfitting บน small patient dataset |
| W7 (EDA) | Thailand GDP per capita — EDA ด้วย World Bank data |
| W8 (SLR) | Advertising spend vs sales — Advertising.csv (ISLP) |
| W9 (MLR) | House price prediction — Boston/Ames Housing dataset |
| W10 (Diagnostics) | Auto MPG — polynomial regression + residual analysis |
| W11 (Logistic) | Credit default prediction — Default dataset (ISLP) |
| W12 (LDA/QDA) | Stock market direction — Smarket dataset (ISLP) |
| W13 (GLM/ROC) | Bikeshare demand — Poisson regression (ISLP) |
| W14 (CV) | Optimal model selection — Auto dataset (ISLP) |
| W15 (Final) | นักศึกษาเลือก dataset จริงที่สนใจ |

**ตัวอย่าง Case Study ที่ดี (Week 8 — SLR):**
```markdown
## Case Study: วิเคราะห์ผลของงบโฆษณา TV ต่อยอดขาย

**Scenario**
บริษัท FMCG แห่งหนึ่งต้องการทราบว่างบโฆษณาทาง TV ส่งผลต่อยอดขายจริงหรือไม่
เพื่อตัดสินใจว่าควรเพิ่มงบ TV หรือควรลงทุนช่องทางอื่น

**Data**
- Advertising.csv: 200 ตลาด
- TV: งบโฆษณา TV (พันดอลลาร์), range 0.7–296.4
- Sales: ยอดขาย (พันหน่วย), range 1.6–27.0

**Method — Simple Linear Regression**
Sales ≈ β₀ + β₁ × TV

ผลจาก statsmodels:
β̂₀ = 7.03  (SE = 0.46, p < 0.001)
β̂₁ = 0.048 (SE = 0.003, p < 0.001)
R² = 0.612, RSE = 3.26

**Result**
ทุก ๆ งบ TV เพิ่มขึ้น 1,000 ดอลลาร์ → ยอดขายเพิ่มขึ้น 48 หน่วย
p-value < 0.001 แสดงว่า TV มีความสัมพันธ์กับ Sales อย่างมีนัยสำคัญ
แต่ R² = 0.612 หมายความว่ายังมี 38.8% ของ variance ที่ TV อธิบายไม่ได้

**Insight**
TV มีผลต่อยอดขายจริง แต่ไม่ใช่ปัจจัยเดียว
→ แนะนำให้เพิ่ม Radio และ Newspaper เข้าโมเดล (MLR ใน Week 9)
→ นี่คือ motivation ที่แท้จริงว่าทำไมเราต้องเรียน Multiple Regression
```

---

#### กฎ 5 — มาตรฐานการเขียนเอกสารคณิตศาสตร์ (Math Document Standards)
**สกัดจาก: เอกสาร "1.4 เมทริกซ์ผกผัน" — ตัวอย่างเอกสารคณิตศาสตร์ระดับมหาวิทยาลัยที่ดี**

ใช้กับ **lecture note** ทุกไฟล์ที่มีเนื้อหาทางคณิตศาสตร์

---

##### 5.1 โครงสร้างเนื้อหา (Content Structure)

ลำดับที่ถูกต้องภายในแต่ละ subtopic:
```
นิยาม (Definition)
  ↓
ตัวอย่างพิสูจน์/ยืนยันนิยาม (Illustrative Example)
  ↓
Note / ข้อสังเกต (Remarks)
  ↓
คุณสมบัติ / ทฤษฎีบท (Properties / Theorem)
  ↓
ตัวอย่างประยุกต์ (Applied Example + ตรวจสอบผลเฉลย)
  ↓
ข้อสังเกตต่อยอด (Further Remarks)
```

ย่อหน้าแรกของ section ต้องบอกว่าจะทำอะไร:
> "ในหัวข้อนี้ เราจะ[ทำอะไร] โดยการใช้[วิธีการ]"

---

##### 5.2 รูปแบบ Definition / Theorem / Example (Numbered Elements)

| ประเภท | Format | ตัวอย่าง |
|--------|--------|---------|
| **นิยาม** | `**นิยาม X.Y.Z**` (bold) + เงื่อนไข → ข้อสรุป | **นิยาม 1.4.1** ถ้า A เป็น... แล้ว B เป็น... |
| **ทฤษฎีบท** | `**ทฤษฎีบท X.Y.Z**` (bold) + ข้อความสมมูล (numbered list) | **ทฤษฎีบท 1.4.3** ข้อความต่อไปนี้สมมูลกัน: 1. ... 2. ... 3. ... |
| **คุณสมบัติ** | `**คุณสมบัติ X.Y.Z**` (bold) + numbered list | **คุณสมบัติ 1.4.3** 1. ถ้า A, B เป็น... แล้ว... |
| **ตัวอย่าง** | `**ตัวอย่าง X.Y.Z**` (bold) + โจทย์ + วิธีทำ + `#` ท้าย | **ตัวอย่าง 1.4.2** ให้ A = [...] จงแสดง... |
| **Note** | `**Note**` (bold) + numbered list ถ้ามีหลายข้อ | **Note** 1. ... 2. ... |
| **ข้อสังเกต** | `**ข้อสังเกต**` (bold) + numbered list | **ข้อสังเกต** 1. เมทริกซ์ที่... |

**กฎ**: ทุกตัวอย่างต้องลงท้ายด้วย `#` เพื่อบอกจุดสิ้นสุด

---

##### 5.3 ภาษาคณิตศาสตร์ภาษาไทยที่ถูกต้อง

| สถานการณ์ | วลีที่ต้องใช้ |
|-----------|-------------|
| แสดงผลลัพธ์ระหว่างทาง | "จะเห็นว่า" (ก่อนแสดงผลที่ได้) |
| สรุปผลสุดท้าย | "ดังนั้น" (ก่อนข้อสรุป) |
| บอกผลที่ต้องการพิสูจน์ | "จะแสดงว่า" |
| อ้างอิงทฤษฎีบท/นิยาม | "จากทฤษฎีบท X.Y.Z" / "จากนิยาม X.Y.Z" |
| ยืนยันผลลัพธ์ | "ตรวจสอบผลเฉลย:" (ตามด้วยการ verify) |
| ยกข้อยกเว้น | "อย่างที่ทราบว่า ไม่ใช่ทุก [X] ที่ [Y]" |
| อ้างอิงข้ามตัวอย่าง | "จากตัวอย่าง X.Y.Z" |

---

##### 5.4 การแสดง Step-by-Step (Worked Example Format)

สำหรับตัวอย่างที่มีหลายขั้นตอน:
```markdown
**ขั้นที่ 1** — [บอกว่าขั้นนี้ทำอะไร]
[แสดง matrix/equation ก่อนและหลัง]
[ระบุ row operation ที่ใช้: เช่น $-\frac{1}{4}R_2$]

จะเห็นว่า [แสดงผลยืนยัน]

**ขั้นที่ 2** — ...
```

สำหรับ augmented matrix [A | I] ให้แสดง **2 ฝั่งพร้อมกัน** เสมอ:
- ฝั่งซ้าย: การแปลง A → Iₙ  
- ฝั่งขวา: การแปลง Iₙ → A⁻¹
- ระบุ row operation กำกับลูกศร (~) เช่น `$\xrightarrow{-2R_2+R_1 \to R_1}$`

สูตรหรือกระบวนการสำคัญให้ใส่ **กรอบ (boxed)**:
```
$$\boxed{[A \mid I_n] \sim \cdots \sim [I_n \mid A^{-1}]}$$
```

---

##### 5.5 การ Verify และ Counterexample

**ทุกตัวอย่างที่คำนวณผลลัพธ์ต้องมี verification:**
```markdown
**ตรวจสอบผลเฉลย**
$$AA^{-1} = \begin{bmatrix}...\end{bmatrix}\begin{bmatrix}...\end{bmatrix} = I$$
และ $A^{-1}A = I$ ✓
```

**ทุก theorem ที่มี "ถ้า condition แล้ว result" ต้องมี counterexample** แสดงว่าถ้าไม่มี condition จะเกิดอะไร:
> "อย่างที่ทราบว่า ไม่ใช่ทุกเมทริกซ์ที่มีอินเวอร์ส พิจารณาตัวอย่างต่อไปนี้ [ตัวอย่าง singular matrix]"

---

##### 5.6 รูปแบบ Exercise ที่ดี

```markdown
## Exercise X.Y

1. [โจทย์หลัก — บอก method ที่ต้องใช้]
   1.1 [matrix/problem ย่อย 1]    1.2 [matrix/problem ย่อย 2]
   1.3 [matrix/problem ย่อย 3]    1.4 [matrix/problem ย่อย 4]

2. [โจทย์ inverse — กำหนด A⁻¹ หาย้อนกลับ]

3. [โจทย์เหมือนข้อ 1 แต่ใช้ method อื่น] พร้อมตรวจสอบว่า AA⁻¹ = A⁻¹A = Iₙ

4–5. [โจทย์ขนาดใหญ่ขึ้น 3×3, 4×4]

6–7. [โจทย์ประยุกต์ — แก้ระบบสมการด้วย inverse]

8. [โจทย์วิเคราะห์/พิจารณา — trivial solution, singular/non-singular]
```

หลักการออกแบบ exercise:
- **ระดับ easy**: สูตรสำเร็จ (เช่น สูตร 2×2 inverse)
- **ระดับ medium**: row operations + verify
- **ระดับ hard**: matrix ขนาดใหญ่, singular matrix, ประยุกต์กับระบบสมการ
- **ระดับ analysis**: พิจารณา existence, uniqueness, equivalence
- คำสั่ง "พร้อมตรวจสอบว่า..." ทุกข้อที่คำนวณ
- ระบุ Reference ท้าย exercise เสมอ

---

##### 5.7 Cross-reference และ Notation

- อ้างอิงทฤษฎีบทเสมอเมื่อสรุปผล: "จากทฤษฎีบท X.Y.Z สรุปได้ว่า..."
- เชื่อมโยงข้ามตัวอย่าง: "จากโจทย์ในตัวอย่าง X.Y.Z ในที่นี้เราจะ..."
- สมการที่อ้างอิงซ้ำต้องมีหมายเลข: (**), (***) หรือ (1), (2)
- ตัวแปรเมทริกซ์: ตัวพิมพ์ใหญ่ (A, B, I) — ตัวแปร scalar: ตัวเล็ก (a, b, x)
- อินเวอร์สเขียนเป็น $A^{-1}$ ไม่ใช่ $A^{(-1)}$

---

#### กฎ 3 — บอกเสมอว่าทำอะไร เพื่ออะไร (What & Why)
**ทุกขั้นตอนในทุก stream ต้องระบุทั้ง "ทำอะไร" และ "เพื่ออะไร" คู่กันเสมอ**

ใช้กับทุกที่ดังนี้:

| ที่ไหน | รูปแบบ "ทำอะไร เพื่ออะไร" |
|--------|--------------------------|
| Lecture — ก่อน section | ย่อหน้าเปิด: "ในส่วนนี้เราจะ [ทำอะไร] เพื่อ [เป้าหมาย] เพราะ [เหตุผล]" |
| Lab — ก่อนแต่ละ Part | markdown cell: "**Part นี้เราจะ [ทำอะไร]** เพื่อ [เป้าหมาย]" |
| Lab — ก่อนแต่ละ TODO | อธิบาย context ก่อน TODO ว่า "เราต้องการ [X] เพราะ [Y] ให้คุณเขียน code ที่..." |
| Assignment — ก่อนโจทย์แต่ละข้อ | บอก context: "ในข้อนี้คุณจะ [ทำอะไร] เพื่อแสดงให้เห็นว่า [ผลลัพธ์ที่คาดหวัง]" |
| Slides — Key Message ของแต่ละ slide | 1 ประโยคสรุปว่า slide นี้ต้องการสื่ออะไร |

---

### Lecture Notes (`.md`) — ปริมาณขั้นต่ำ
| ส่วน | ปริมาณขั้นต่ำ |
|------|--------------|
| **บทนำสัปดาห์** (ต้นไฟล์) | **5–10 ประโยค** ครอบคลุม: เกี่ยวกับอะไร + เป้าหมาย + นำไปใช้อะไร |
| แต่ละ subtopic section (`###`) | อย่างน้อย **3 ย่อหน้า**: นิยาม + อธิบาย + ตัวอย่าง |
| Worked Example | **1 ตัวอย่างเลขพร้อมเฉลยทุกขั้นตอน** ต่อ 1 subtopic หลัก |
| DS Connection | **2–3 ประโยค** อธิบายว่า concept นี้ใช้ใน Data Science อย่างไรและที่ไหน |
| Summary table | ครอบคลุมทุก concept (แนวคิด + สูตร + Python function) |
| ความยาวรวม | ไม่น้อยกว่า **800 คำ** ต่อ 1 ไฟล์ |

### Lab Notebooks (`.ipynb`) — ปริมาณขั้นต่ำ
| ส่วน | ปริมาณขั้นต่ำ |
|------|--------------|
| **บทนำสัปดาห์** (Cell markdown แรก) | **5–10 ประโยค** ครอบคลุม: เกี่ยวกับอะไร + เป้าหมาย + นำไปใช้อะไร |
| **Comment ใน code cell ทุก block** | ทุก block ต้องมี: `# ─── ทำอะไร ───` + `# วัตถุประสงค์: เพื่ออะไร` |
| Markdown instruction ก่อนแต่ละ Part | **3–5 ประโยค** อธิบาย: Part นี้ทำอะไร + เพื่ออะไร + ขั้นตอนที่จะทำ |
| Markdown instruction ก่อนแต่ละ TODO | **2–3 ประโยค**: context ของ TODO + สิ่งที่ต้องทำ + ผลลัพธ์ที่คาดหวัง |
| Exercise / TODO | อย่างน้อย **3 TODO** (ระดับ: easy → medium → hard) |
| Reflection question ท้าย lab | อย่างน้อย **2 คำถาม** ให้นักศึกษาคิดต่อ |

### Assignments (`.md`) — ปริมาณขั้นต่ำ
| ส่วน | ปริมาณขั้นต่ำ |
|------|--------------|
| **บทนำสัปดาห์** (ต้นไฟล์) | **5–10 ประโยค** ครอบคลุม: เกี่ยวกับอะไร + เป้าหมาย + นำไปใช้อะไร |
| คำอธิบาย context ก่อนแต่ละข้อ | **2–3 ประโยค** บอก: ทำอะไร เพื่ออะไร ในโจทย์ข้อนี้ |
| จำนวนข้อ | อย่างน้อย **3 ข้อ** ครอบคลุม LLo ที่กำหนด |
| Rubric | ทุกเกณฑ์ต้องมี **4 ระดับ** พร้อมคำอธิบายแต่ละระดับ |

### Slides Outline (`.md`) — ปริมาณขั้นต่ำ
| ส่วน | ปริมาณขั้นต่ำ |
|------|--------------|
| **Slide 2 — Week Overview** | **5–10 ประโยค** ครอบคลุม: เกี่ยวกับอะไร + เป้าหมาย + นำไปใช้อะไร |
| **Key Message ทุก slide** | **1 ประโยคสรุป** ว่า slide นี้ต้องการสื่ออะไร (what & why) |
| เนื้อหาต่อ slide | อย่างน้อย **3 bullet** หรือ 1 FIGURE description + 2 bullets |
| จำนวน slide ต่อ deck | อย่างน้อย **8 slides** |
| Slide สุดท้าย | Summary + preview สัปดาห์ถัดไป |

---

## Context ที่ต้องอ่านก่อนทำงาน

1. อ่าน `E:\2569\Math for DS\course_outline.md` — ดูว่า Week นั้นอยู่ใน CLO อะไร
2. อ่าน `E:\2569\Math for DS\Week[N]\plan.md` — เป็น source of truth ทั้งหมด:
   - subtopics ทั้งหมด
   - ชื่อไฟล์ที่ต้องสร้าง
   - dataset, library, tools ที่ใช้
   - LLo และ CLO
3. ตรวจสอบว่าไฟล์นั้นมีอยู่แล้วหรือไม่ก่อนสร้าง (ใช้ Glob)

---

## กรณี: `lecture week N`

สร้าง `lecture/note_01_xxx.md` และ `note_02_xxx.md` ตามชื่อที่ระบุใน plan.md

**Format ของ lecture note:**
```markdown
# Note [N]: [ชื่อหัวข้อ]
> Week [N] | CLO[X] | ISLP/Strang Reference: [ระบุ]

## [หัวข้อย่อย 1]  *(Reference)*
- นิยาม/แนวคิด (ภาษาไทย + English term)
- สูตร: $formula$ หรือ $$block formula$$
- ตัวอย่าง (worked example พร้อมเลข)
- **DS Connection**: เชื่อมกับ Data Science จริง ๆ

## [หัวข้อย่อย 2]
...

## สรุป (Summary)
| แนวคิด | สูตร/Method | Python Function |
|--------|------------|----------------|
| ...    | ...        | ...            |

## เชื่อมกับสัปดาห์อื่น
- ← Week X: [พื้นฐานที่ต้องมี]
- → Week Y: [ขยายต่อ]
```

---

## กรณี: `lab week N`

สร้าง `lab/labNN_xxx.ipynb` และ `labNN_solution.ipynb` ตามชื่อใน plan.md

**Format ของ lab notebook (JSON cells):**
- Cell 1 (markdown): `# Lab [N]: [ชื่อ]` + LLo + สิ่งที่จะเรียนรู้
- Cell 2 (code): imports ทั้งหมด
- Cell 3 (markdown): `## Part 1: [ชื่อ]` + คำอธิบาย
- Cell 4 (code): starter code พร้อม comment
- Cell 5 (code): `# TODO: นักศึกษาเติม code ที่นี่`
- ทำซ้ำ Part 2, 3, ...
- Cell สุดท้าย (markdown): สรุป + คำถาม reflection

**สำหรับ solution:** เติมทุก TODO ด้วย code ที่ถูกต้อง

**สำคัญ:** ใช้ dataset ตาม plan.md (เช่น Advertising.csv, iris, Default, Auto)

---

## กรณี: `assignment week N`

สร้าง `assignment/hwNN_xxx.md` และ `hwNN_rubric.md` ตามชื่อใน plan.md

**Format ของ HW:**
```markdown
# Homework [N]: [ชื่อ]
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO[X] | **LLo**: [ระบุ]  
**กำหนดส่ง**: [สัปดาห์ที่ N+1]

## ข้อ 1 — [ชื่อ] ([X] คะแนน)
[โจทย์ที่ชัดเจน สอดคล้องกับ LLo]

**Dataset**: [ระบุ]  
**Deliverable**: [code / รายงาน / graph]

## ข้อ 2 — ...
```

**Format ของ Rubric:**
```markdown
# Rubric — HW[N]
| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| [CLO] | ... | ... | ... | ... | X |
```

---

## กรณี: `slides week N`

สร้าง `slides/01_xxx.md` ถึง `04_xxx.md` เป็น slide outline (Markdown)

**Format:**
```markdown
# Slide Deck: [ชื่อ]
> Week [N] | [จำนวน] slides

---
## Slide 1 — Title
**[ชื่อวิชา] Week [N]**  
[หัวข้อ]  
CLO[X]: [ข้อความ LLo]

---
## Slide 2 — [หัวข้อ]
**Key Message**: [1 ประโยค]
- bullet 1
- bullet 2
- [FIGURE: คำอธิบาย diagram ที่ควรมี]
- [EXAMPLE: ตัวอย่างที่จะใช้]

---
## Slide N — Summary
**สิ่งที่เรียนรู้วันนี้**
- ...
**สัปดาห์ต่อไป**: [preview]
```

---

## กรณี: `all week N`

ทำตามลำดับ: **lab → lecture → assignment → slides**

หลังสร้างแต่ละ stream เสร็จ ให้รายงานไฟล์ที่สร้างและ path เต็ม

---

## หลังสร้างเสร็จทุกกรณี

อัปเดต `E:\2569\Math for DS\_status.md`:
- เปลี่ยน `⬜ ยังไม่เริ่ม` → `✅` สำหรับไฟล์ที่สร้างแล้ว
