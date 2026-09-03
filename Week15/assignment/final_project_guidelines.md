# Final Project Guidelines: Math for Data Science
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO1, CLO2, CLO3, CLO4  
**กำหนดนำเสนอ**: สัปดาห์ที่ 15 (ในชั้นเรียน)  
**คะแนนรวม**: 20 คะแนน (รายงาน 10% + Presentation 10%)

---

## ภาพรวมโครงงาน

Final Project เป็นโอกาสที่นักศึกษาจะนำความรู้ที่เรียนตลอด 14 สัปดาห์มาประยุกต์ใช้กับ **dataset จริงที่สนใจ** โดยต้องครอบคลุมทุก Learning Outcome ของรายวิชา งานนี้ทำเป็นกลุ่ม 2–3 คน และต้องแสดงให้เห็นว่าคณิตศาสตร์ที่เรียนมาช่วยแก้ปัญหาจริงได้อย่างไร

---

## ข้อกำหนด Dataset

| เกณฑ์ | ขั้นต่ำ |
|-------|--------|
| จำนวน observations | ≥ 500 rows |
| จำนวน features | ≥ 5 columns |
| ประเภทข้อมูล | ผสม: continuous + categorical |
| แหล่งที่มา | ต้องอ้างอิงแหล่งข้อมูลที่ถูกต้อง |

**Dataset แนะนำ:**

| ประเภท | แหล่ง | ตัวอย่าง Dataset |
|--------|-------|----------------|
| เศรษฐกิจ/สังคมไทย | data.go.th | GDP, ราคาสินค้า, อุบัติเหตุ |
| Healthcare | UCI ML Repository | Heart Disease, Breast Cancer |
| Finance | Kaggle | Credit Card Fraud, Stock |
| Environment | Our World in Data | CO₂, Climate, Energy |
| Sports | Kaggle | FIFA, NBA Stats |
| Transportation | data.go.th | Traffic, Accidents |

---

## สิ่งที่ต้องทำ (Deliverables)

### 1. Jupyter Notebook (ส่งก่อน Presentation 1 วัน)

Notebook ต้องครอบคลุม **4 ส่วน** ตาม CLO:

**ส่วน 1: CLO1 — Linear Algebra Analysis**
- Covariance matrix ของ features: Σ (คำนวณและ interpret)
- Eigenvalue analysis: λ₁, λ₂, ... (สัดส่วน variance ที่แต่ละ component อธิบายได้)
- PCA หรือ SVD: project data ลง 2D แล้ว visualize

**ส่วน 2: CLO2 — Statistical Analysis**
- Descriptive statistics: mean, median, std, skewness, kurtosis ของ numeric features
- Distribution analysis: histogram + QQ plot สำหรับ target variable
- Hypothesis testing: t-test หรือ chi-square test ที่สมเหตุสมผลกับ dataset
- Correlation analysis: heatmap + สรุป feature ที่ correlate กับ target

**ส่วน 3: CLO3 — Machine Learning Models**
- อย่างน้อย **3 classification หรือ regression models**
- สำหรับ Regression: Linear Regression + polynomial + อีก 1
- สำหรับ Classification: Logistic Regression + LDA/QDA + KNN
- Evaluation metrics ที่เหมาะสม (MSE/R² สำหรับ regression, Accuracy/F1/AUC สำหรับ classification)
- Confusion matrix หรือ residual plot

**ส่วน 4: CLO4 — Model Selection**
- ใช้ k-Fold CV เลือก best model (และ/หรือ hyperparameter)
- Report CV error + SE
- One-SE Rule: ระบุ simplest model ที่ดีพอ
- Final model evaluation บน held-out test set

### 2. Presentation (10 นาที + 2 นาที Q&A)

| ส่วน | เวลา | เนื้อหา |
|------|------|---------|
| Introduction | 1.5 นาที | Dataset + Problem Statement + ทำไมน่าสนใจ |
| Mathematical Analysis | 5 นาที | CLO1–4 highlights (ไม่ต้องทุก detail) |
| Key Insights | 2 นาที | ค้นพบอะไรสำคัญ? ผลลัพธ์ที่น่าสนใจ? |
| Conclusion | 1.5 นาที | สรุป + limitations + ขั้นตอนต่อไป |
| Q&A | 2 นาที | คำถามจากอาจารย์และนักศึกษา |

---

## รูปแบบการส่ง

| สิ่งที่ส่ง | รูปแบบ | deadline |
|---------|--------|---------|
| Jupyter Notebook | `.ipynb` + output ครบ | ก่อน Presentation 1 วัน (Email) |
| Presentation slides | `.pdf` หรือ PowerPoint | วันนำเสนอ |
| Peer Evaluation | กรอกแบบฟอร์ม | ในชั้นเรียน |

**ชื่อไฟล์ Notebook:** `final_XXXXXXXX_YYYYYYYY.ipynb` (รหัสนักศึกษาทุกคนในกลุ่ม)

---

## เกณฑ์การประเมิน (รายละเอียดใน `final_project_rubric.md`)

| Component | คะแนน |
|-----------|-------|
| CLO1: Linear Algebra | 4 |
| CLO2: Statistics | 3 |
| CLO3: Models | 5 |
| CLO4: CV + Model Selection | 3 |
| Code Quality + Documentation | 2 |
| Presentation + Q&A | 3 |
| **รวม** | **20** |

---

## Timeline

| วันที่ | กิจกรรม |
|-------|---------|
| สัปดาห์ที่ 14 (วันสุดท้าย) | ประกาศกลุ่ม + เลือก Dataset |
| ก่อนสัปดาห์ที่ 15 (5 วัน) | ส่ง Notebook ครั้งแรก (draft) — ขอ feedback จากอาจารย์ได้ |
| สัปดาห์ที่ 15 (วันก่อน) | ส่ง Notebook ฉบับสมบูรณ์ |
| สัปดาห์ที่ 15 (วัน Presentation) | นำเสนอ 10+2 นาทีต่อกลุ่ม |

---

## คำถามที่พบบ่อย

**Q: Dataset ต้องมี CLO ครบทุกข้อไหม?**  
A: ใช่ — ต้องครอบคลุม CLO1–4 ถ้า dataset ไม่เหมาะกับส่วนใดส่วนหนึ่ง ให้ปรึกษาอาจารย์ก่อน

**Q: ใช้ dataset ที่ใช้ใน homework ได้ไหม?**  
A: ไม่แนะนำ — ควรเลือก dataset ใหม่เพื่อแสดงว่าสามารถประยุกต์ใช้ความรู้กับ context ใหม่ได้

**Q: ต้องใช้ algorithm จาก Week ไหนบ้าง?**  
A: ไม่มีข้อบังคับ — ใช้ algorithm ที่เหมาะสมกับ dataset สำคัญคือ justify การเลือก

**Q: จะ cite ที่มา code ที่ copy จาก lab หรือ online ได้ไหม?**  
A: ได้ — ต้อง cite ทุกครั้ง แต่ต้องเข้าใจ code นั้นและปรับให้เข้ากับ context ของ dataset ตัวเอง
