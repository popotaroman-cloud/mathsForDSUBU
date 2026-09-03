# Template รายงาน: Classification Methods Comparison
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | **Week 12**  
**ชื่อ**: _________________________ **รหัสนักศึกษา**: _________________________

---

## 1. Executive Summary (บทสรุป)

*(อธิบาย dataset ที่ใช้, เป้าหมาย, และ method ที่ดีที่สุดที่เลือก — 2–3 ประโยค)*

**Dataset**: _______________  
**Target variable**: _______________  
**Best method**: _______________  
**เหตุผล**: _______________

---

## 2. Dataset Description

| รายการ | ค่า |
|--------|-----|
| จำนวน observations (n) | |
| จำนวน features (p) | |
| จำนวน classes (K) | |
| Class names | |
| Class distribution (%) | |
| Train size | |
| Test size | |

**Feature ที่ใช้**: _______________

**Preprocessing**: ☐ StandardScaler  ☐ MinMaxScaler  ☐ ไม่มี  ☐ อื่นๆ: ___

---

## 3. Classifier Assumptions

| Classifier | Assumption | ถูกต้องกับ dataset? |
|-----------|-----------|-------------------|
| Logistic Regression | ไม่มี (discriminative) | ✓/✗/Partial |
| LDA | Gaussian, shared Σ | |
| QDA | Gaussian, per-class Σₖ | |
| Naive Bayes | Gaussian + independence | |
| KNN | ไม่มี (non-parametric) | |

**หมายเหตุ**: *(อธิบาย assumption ข้อใดที่น่ากังวลสำหรับ dataset ของคุณ)*

_______________

---

## 4. Performance Comparison Table

*(กรอกค่า metrics ทุกช่อง)*

| Method | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| Logistic Regression | | | | | |
| LDA | | | | | |
| QDA | | | | | |
| Naive Bayes | | | | | |
| KNN (K=___) | | | | | |

**Best Accuracy**: ___ (method: ___)  
**Best Recall**: ___ (method: ___)  
**Best AUC**: ___ (method: ___)

---

## 5. Confusion Matrix Analysis

*(วาง confusion matrix ของ best method ที่นี่)*

```
[Paste confusion matrix output or figure here]
```

| Metric | ค่า | ความหมาย |
|--------|-----|---------|
| True Positive (TP) | | |
| True Negative (TN) | | |
| False Positive (FP) | | FP คือ: |
| False Negative (FN) | | FN คือ: |

**ค่าใช้จ่ายของ FN vs FP**: *(อธิบายใน context ของ dataset)*

_______________

---

## 6. ROC Curve Analysis

*(วาง ROC curve plot ที่นี่)*

```
[Paste ROC curve figure here]
```

| Method | AUC |
|--------|-----|
| Logistic Regression | |
| LDA | |
| QDA | |
| Naive Bayes | |
| KNN | |

**สังเกต**: *(method ไหน hug มุมบนซ้ายได้ดีที่สุด?)*

_______________

---

## 7. Threshold Analysis

*(กรอกสำหรับ best method)*

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.2 | | | |
| 0.3 | | | |
| 0.5 | | | |

**Recommended threshold**: ___ **เหตุผล**: _______________

---

## 8. Method Selection Recommendation

**Classifier ที่แนะนำ**: _______________

**เหตุผล** (ครอบคลุม 3 มิตินี้):

1. **Performance**: *(metric ไหนสำคัญที่สุดสำหรับ problem นี้?)*  
   _______________

2. **Assumption**: *(assumption ของ method ที่เลือกสอดคล้องกับ data?)*  
   _______________

3. **Interpretability**: *(ต้องอธิบาย decision ให้ stakeholder ไหม? LDA/LR อธิบายง่ายกว่า KNN)*  
   _______________

---

## 9. Conclusions

*(สรุป 3–5 ประโยค: เรียนรู้อะไรจาก comparison นี้? ถ้าเปลี่ยน dataset จะเลือก method อื่นไหม?)*

_______________

---

*Template นี้ใช้ประกอบกับ `hw12_classification_comparison.md`*
