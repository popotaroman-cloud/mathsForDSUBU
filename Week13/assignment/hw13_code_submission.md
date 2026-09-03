# คำแนะนำการส่ง Code + Report — HW13
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | **Week 13**

---

## รูปแบบการส่ง

### ไฟล์ที่ต้องส่ง

| ไฟล์ | รูปแบบ | คำอธิบาย |
|------|--------|---------|
| `hw13_XXXXXXXX.ipynb` | Jupyter Notebook | Code + Output ทุก cell run ผ่านแล้ว |

**XXXXXXXX** = รหัสนักศึกษา 8 หลัก (เช่น `hw13_64010001.ipynb`)

---

## โครงสร้าง Notebook ที่คาดหวัง

Notebook ของคุณควรมีโครงสร้างดังนี้:

```
# Homework 13: Full Classification Project
## ชื่อ + รหัสนักศึกษา

## ข้อ 1: Poisson GLM
### 1.1 Dataset + EDA
### 1.2 Fit Poisson GLM
### 1.3 IRR Table + ตีความ
### 1.4 Poisson vs Linear Comparison

## ข้อ 2: ROC Curve + AUC
### 2.1 Dataset + Split + Scale
### 2.2 Fit All Classifiers
### 2.3 Comparison Table
### 2.4 ROC Curves
### 2.5 Optimal Threshold
### 2.6 สรุป

## ข้อ 3: Full Pipeline
### 3.1 EDA
### 3.2 Split + Scale
### 3.3 Fit + Evaluate
### 3.4 ROC Curves
### 3.5 Threshold Analysis
### 3.6 Business Recommendation
```

---

## มาตรฐาน Code ที่ต้องทำตาม

### ✅ ต้องมี: Code Comments
```python
# ─── ทำอะไร ────────────────────────────────────────────────────────────────────
# วัตถุประสงค์: เพื่ออะไร
code_here
```

### ✅ ต้องมี: Markdown Cell ก่อนแต่ละ section
- อธิบาย: ทำอะไร เพื่ออะไร ขั้นตอนนี้
- อย่างน้อย 2–3 ประโยค

### ✅ ต้องมี: Output ครบ
- ทุก cell ที่มี code ต้องมี output แสดง
- Plot ต้องมี title, xlabel, ylabel, legend
- Table ต้องมี column headers ครบ

### ❌ ห้าม:
- ส่ง notebook ที่ยังไม่ได้ run
- Code ที่ error ระหว่าง run
- Plot ที่ไม่มี label

---

## Checklist ก่อนส่ง

- [ ] ชื่อไฟล์ถูกต้อง: `hw13_XXXXXXXX.ipynb`
- [ ] Run all cells → Kernel → Restart & Run All
- [ ] ทุก cell มี output
- [ ] ข้อ 1: IRR table + คำอธิบาย
- [ ] ข้อ 2: ROC Curve plot + AUC ครบ 5 methods
- [ ] ข้อ 3: Comparison table 6 methods + Business recommendation
- [ ] Code มี comments ทุก block
- [ ] ไม่มี cell ที่ error

---

## วิธีเขียน Business Recommendation ที่ดี

Business recommendation ที่ดีต้องตอบ 3 คำถามนี้:

1. **เลือก classifier ไหน?**  
   "ฉันแนะนำ [method] เพราะ [metric] = [value] ซึ่งสูงกว่า methods อื่น"

2. **Threshold เท่าไร?**  
   "ควรใช้ threshold = [value] เพราะ FN cost = [X]× FP cost ดังนั้นต้องการ Recall ≥ [value]"

3. **Trade-off ที่ยอมรับได้?**  
   "การลด threshold จาก 0.5 เป็น [value] เพิ่ม Recall จาก [X] เป็น [Y] แต่ Precision ลดจาก [A] เป็น [B] ซึ่งยอมรับได้เพราะ..."

---

## ตัวอย่าง Business Recommendation ที่ดี

> "สำหรับ credit scoring pipeline ฉันแนะนำ Logistic Regression with class_weight='balanced' เพราะมี AUC = 0.92 ซึ่งสูงสุดในบรรดา classifiers ที่ทดสอบ และ interpretable ทำให้สามารถอธิบาย decision ต่อ regulator ได้
> 
> ควรใช้ threshold = 0.35 เนื่องจาก FN (approve ลูกค้าที่จะ default) มี cost = 5× FP (reject ลูกค้าที่ดี) ที่ threshold นี้ Recall = 0.83 (catch 83% ของผู้ที่จะ default) โดยยอมเสียสละ Precision จาก 0.78 (threshold=0.5) เป็น 0.61 ซึ่งยังอยู่ในระดับที่ธนาคารยอมรับได้"
