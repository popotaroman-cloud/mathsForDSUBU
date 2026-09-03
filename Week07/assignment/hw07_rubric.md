# Rubric — HW07: EDA Report และ Statistical Inference
**วิชา** 1145 201 | **CLO2**

---

## ข้อ 1 — EDA Full Pipeline (4 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Data Loading & Inspection | โหลดถูกต้อง แสดง info/describe/null ครบ อธิบาย dataset ชัดเจน | ครบ 2/3 | ครบ 1/3 | ไม่ทำ | 1 |
| Univariate Visualization | ≥ 3 plots ครบ title/label อธิบาย distribution ได้ | 2 plots ครบ | 1 plot ครบ | ไม่ plot | 1 |
| Correlation Analysis | heatmap/pairplot ครบ อธิบาย correlation ได้ถูกต้อง | plot ได้แต่อธิบายไม่ครบ | plot ได้ | ไม่ทำ | 1 |
| Outlier Detection + Insights | IQR method ถูกต้อง ระบุ outliers ครบ สรุป 3 insights | IQR ถูกต้อง สรุปได้ 2/3 | IQR ถูกต้อง สรุปได้ 1/3 | ผิดหรือไม่ทำ | 1 |

---

## ข้อ 2 — Confidence Interval (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| คำนวณ CI manual | สูตรถูกต้อง ครบทุกขั้นตอน verify กับ scipy ผลตรงกัน | สูตรถูกแต่ขาดขั้นตอน | ผลถูกแต่ไม่แสดง working | ผิดหรือไม่ทำ | 1 |
| Plot 3 ระดับ CI | plot ครบ 3 ระดับ วาด vertical lines ครบ label ถูก | ครบ 2/3 | ครบ 1/3 | ไม่ plot | 1 |
| การตีความ | อธิบาย CI อย่างถูกต้องในบริบทของ dataset บอก trade-off 90/95/99 | อธิบายได้แต่ไม่ครบ | อธิบายได้บางส่วน | ผิดหรือไม่อธิบาย | 1 |

---

## ข้อ 3 — Hypothesis Test + Case Study (3 คะแนน)

| เกณฑ์ | Excellent (100%) | Good (75%) | Pass (50%) | Fail (<50%) | คะแนน |
|-------|-----------------|-----------|-----------|------------|-------|
| Hypothesis Setup | H₀ H₁ ชัดเจน เลือก test ถูกประเภท | H₀ H₁ ระบุ แต่ไม่ justify การเลือก test | H₀ H₁ ระบุ แต่ test ผิด | ไม่ระบุ | 0.5 |
| การคำนวณและ Report | t-stat/χ² df p-value ถูกต้องครบ สรุปถูก | ถูก 3/4 | ถูก 2/4 | ผิดหมด | 1 |
| Case Study | ยก use case จริงจาก industry ที่ชัดเจน อธิบายว่า test นั้น apply อย่างไร | ยก use case ได้แต่ไม่ชัด | ยก use case แต่ไม่เชื่อมกับ test | ไม่ทำ | 1.5 |

---

## หมายเหตุสำหรับผู้ตรวจ

- **Plot ที่ไม่มี title/axis label** → หัก 0.1 คะแนนต่อ plot
- **Code ที่ run ไม่ผ่าน** → หักคะแนนข้อนั้น 50%
- **Plagiarism** → 0 คะแนน และรายงานต่ออาจารย์
- **Dataset ที่เลือกเองนอกรายการ**: ผ่านถ้าตรงตาม requirement (≥200 rows, ≥4 columns)
