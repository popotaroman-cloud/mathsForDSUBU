# Slide Deck: Final Project Rubric & Expectations
> Week 15 | CLO1+2+3+4 | Assessment | 9 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 15**  
Final Project: เกณฑ์ประเมิน, Expectations, และ Tips  
CLO1+2+3+4: 20% ของคะแนนทั้งหมด (Notebook 10% + Presentation 10%)

---

## Slide 2 — Overview: 20% ของเกรด
**Key Message**: Final Project คิดเป็น 20% ของเกรด — แบ่งเป็น Notebook 10% และ Presentation 10%

**น้ำหนักคะแนนทั้งหมดของวิชา:**
```
Weekly Assignments (HW01–14):   50%  ← งานรายสัปดาห์
Midterm Exam (Week 8):           15%
Lab Participation (Week 1–14):   15%
Final Project (Week 15):         20%  ← วันนี้
    ├── Jupyter Notebook:  10%
    └── Presentation:      10%
```

**โครงงานทำเป็น:**
- กลุ่ม 2–3 คน
- Dataset ที่กลุ่มเลือกเอง (≥500 rows, ≥5 features, regression หรือ classification)
- CLO1+2+3+4 ครบทุก CLO

---

## Slide 3 — Rubric: Jupyter Notebook (10%)
**Key Message**: Notebook จะถูกประเมินใน 5 ด้าน — ความถูกต้อง, ครบถ้วน, code quality, visualization, และ insight

| เกณฑ์ | Excellent (A) | Good (B) | Pass (C) | Fail |
|-------|-------------|---------|---------|------|
| **CLO1: LA Analysis** (2%) | PCA/SVD พร้อม interpretation ชัดเจน | PCA ถูกต้องแต่ interpretation น้อย | มี code แต่ไม่ interpret | ไม่มีหรือผิด |
| **CLO2: EDA + Stats** (2%) | EDA ครบ, Bias-Variance หรือ statistical test | EDA ครบแต่ขาด statistical analysis | EDA เบื้องต้น | ขาด CLO2 |
| **CLO3: Model** (3%) | Model ≥2 ชนิด, metrics ถูกต้อง, ตีความได้ | Model ถูกต้องแต่ตีความน้อย | มี 1 model | ผิดหรือขาด |
| **CLO4: CV** (2%) | CV ≥3 models, plot, สรุป best model | CV ถูกต้องแต่ plot น้อย | มี CV แต่ไม่สมบูรณ์ | ขาด CV |
| **Code Quality** (1%) | Comments ทุก block, ชื่อ variable ชัด | Comments ส่วนใหญ่มี | Comments บางส่วน | ไม่มี comment |

**Bonus (+1%):** Data storytelling — เล่าเรื่องจาก data ได้น่าสนใจ

---

## Slide 4 — Rubric: Presentation (10%)
**Key Message**: Presentation จะถูกประเมินใน 4 ด้าน — content ถูกต้อง, ชัดเจน, ครบ CLO, และ Q&A

| เกณฑ์ | Excellent (A) | Good (B) | Pass (C) | Fail |
|-------|-------------|---------|---------|------|
| **Mathematical Accuracy** (3%) | สูตรและผลถูกต้องสมบูรณ์ | ผิดเล็กน้อย 1–2 จุด | ผิดหลายจุดแต่ถูกบ้าง | ผิดส่วนใหญ่ |
| **Clarity & Visuals** (2%) | Slides ชัดเจน, graph อ่านออก, flow ดี | ชัดเจนแต่บาง slide รก | เข้าใจยากบางส่วน | ไม่ชัดเจนโดยรวม |
| **CLO Coverage** (3%) | ครบ CLO1–4 ทุกตัว | ครบ 3/4 CLO | ครบ 2/4 CLO | ≤1 CLO |
| **Q&A** (2%) | ตอบคำถามได้ทุกคน แสดง understanding ลึก | ตอบได้ส่วนใหญ่ | ตอบได้บางส่วน | ตอบไม่ได้ |

**Time Management:** 10 นาที ± 1 นาที (เกิน/ขาด เกิน 2 นาที → -0.5%)

---

## Slide 5 — Dataset Requirements
**Key Message**: Dataset ที่ดีเป็นครึ่งหนึ่งของโครงงาน — เลือกให้เหมาะกับ CLO ทั้ง 4

**ข้อกำหนดขั้นต่ำ:**
```
✅ ≥500 rows (observations)
✅ ≥5 features (predictors)
✅ Target variable ชัดเจน (regression: continuous, classification: categorical)
✅ ไม่ใช่ Iris, Titanic, หรือ MNIST (too simple, ทุกคนรู้จัก)
✅ ไม่ซ้ำกับ dataset ที่ใช้ใน lab/HW ของวิชา
```

**Dataset แนะนำ:**

| แหล่ง | ตัวอย่าง Dataset |
|------|---------------|
| data.go.th | เศรษฐกิจไทย, สาธารณสุข, การศึกษา |
| UCI ML Repository | Wine Quality, Heart Disease, Abalone |
| Kaggle | House Prices, Credit Risk, Bike Sharing |
| Our World in Data | CO₂, Life Expectancy, Education |
| WHO / World Bank | Health metrics by country/year |

---

## Slide 6 — Timeline & Deliverables
**Key Message**: ทำตาม timeline เพื่อไม่ให้งานกองในนาทีสุดท้าย

```
Week 13  → แจ้งชื่อกลุ่มและ Dataset ที่เลือก (approval)
Week 14  → ส่ง Jupyter Notebook draft ผ่าน LMS
Week 15  → นำเสนอ + ส่ง final Notebook + Peer Evaluation
```

**ไฟล์ที่ต้องส่ง (Week 15):**

| ไฟล์ | รายละเอียด |
|------|-----------|
| `final_[GroupID].ipynb` | Jupyter Notebook ที่รัน clean ได้ |
| `dataset.[csv/xlsx]` | Dataset ที่ใช้ (หรือ link ถ้าใหญ่เกิน) |
| `slides_[GroupID].pdf` | Slides ที่ใช้ Presentation |
| `peer_eval_[StudentID].md` | Peer Evaluation Form (แยกรายคน) |

**ZIP file ชื่อ:** `final_[GroupID].zip`

---

## Slide 7 — Common Mistakes to Avoid
**Key Message**: ข้อผิดพลาดที่พบบ่อยมากใน Final Project — หลีกเลี่ยงเหล่านี้จะได้คะแนนดีขึ้น

**❌ ข้อผิดพลาดที่พบบ่อย:**

```
1. ทำ EDA แต่ไม่อธิบาย → ใส่ graph แต่ไม่บอกว่า insight คืออะไร

2. Report Training MSE แทน Test MSE → ต้องแบ่ง train/test ก่อนเสมอ

3. ขาด CLO1 → ไม่มี PCA หรือ matrix analysis เลย

4. ไม่ใช้ Cross-Validation → ประเมิน model จาก single split เท่านั้น

5. Code ไม่มี comment → เหมือนส่งงาน TA ตรวจไม่ออก

6. Presentation เกิน 10 นาที → ฝึกซ้อมก่อน!

7. ไม่ standardize features ก่อน Logistic Regression/KNN
```

**✅ สิ่งที่จะทำให้ได้คะแนน Bonus:**
- แสดง connection ระหว่าง CLO ต่างๆ ใน notebook
- ใช้ Bootstrap estimate uncertainty ของ coefficient
- เล่าเรื่อง (data storytelling) ที่น่าสนใจ

---

## Slide 8 — Scoring Examples
**Key Message**: ตัวอย่าง output จาก Notebook ที่ได้ A vs C เพื่อให้เห็นภาพ

**Grade A — Notebook:**
```python
# ─── PCA with Interpretation ──────────────────────────────────────
# วัตถุประสงค์: ลด dimension เพื่อดู cluster ของ patients

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PC1 อธิบาย 43.2% — dominated by 'glucose' และ 'BMI'
# → สะท้อน metabolic syndrome risk factors
print(f'Explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}')
```

**Grade C — Notebook:**
```python
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1])
```

*Grade A อธิบาย why และ interpretation; Grade C มี code แต่ไม่อธิบาย*

---

## Slide 9 — Summary & Good Luck!
**เกณฑ์การผ่าน Final Project:**
- ต้องได้อย่างน้อย 50% ใน Notebook **และ** 50% ใน Presentation
- ต้องครอบคลุม CLO1, CLO2, CLO3, CLO4 ทุก CLO (ไม่ครบ = ไม่ผ่านเกณฑ์)

**Final Checklist ก่อนส่ง:**
```
☐ Notebook รัน clean ตั้งแต่ต้นจนจบ (Restart + Run All)
☐ ทุก code block มี comment
☐ ครบ CLO1 (LA), CLO2 (Stats/EDA), CLO3 (Model), CLO4 (CV)
☐ มี interpretation สำหรับทุก key result
☐ Slides ฝึกซ้อมแล้ว ≤ 10 นาที
☐ ส่ง Peer Evaluation Form แยกรายคน
```

**โชคดีกับ Final Project ทุกกลุ่ม!** 🎓
