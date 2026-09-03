# Regression Diagnostics Report Template
**วิชา** 1145 201 | **Week 10** | **ชื่อ-นามสกุล**: ___________________  
**Student ID**: _________________ | **Dataset**: _________________  
**Model**: _________________ | **วันที่**: _________________

---

## 1. Executive Summary (1–2 ประโยค)

> สรุปผลสำคัญ: model นี้มีปัญหาอะไรบ้างและแก้ไขอย่างไร

_________________

---

## 2. Model Specification

**Full model formula**:
```
y ~ x1 + x2 + ... + xp
```

**Dataset overview**:
| รายการ | ค่า |
|--------|-----|
| n (observations) | |
| p (predictors) | |
| Response variable | |
| R² (full model) | |
| R²_adj | |
| F-statistic | |
| F p-value | |

---

## 3. Coefficient Table

| Predictor | β̂ | SE | t | p-value | Significant? |
|-----------|-----|-----|---|---------|-------------|
| Intercept | | | | | |
| x1 | | | | | |
| x2 | | | | | |

---

## 4. Diagnostic Plot Analysis

### 4.1 Residuals vs Fitted
**สังเกต**:
- [ ] สุ่มรอบ 0 (ปกติ)
- [ ] มี pattern (non-linearity)
- [ ] Fan shape (heteroscedasticity)

**สรุป**: ___________________

---

### 4.2 Normal Q-Q Plot
**สังเกต**:
- [ ] Points บน diagonal (normality OK)
- [ ] Heavy tails
- [ ] Skewed

**สรุป**: ___________________

---

### 4.3 Scale-Location
**สังเกต**:
- [ ] แบนราบ (homoscedasticity OK)
- [ ] Trend ขึ้น (variance เพิ่มตาม fitted)

**สรุป**: ___________________

---

### 4.4 Leverage / Cook's Distance
**High leverage points** (hᵢ > 2(p+1)/n):
- Observation IDs: ___________________
- ค่า x ที่ผิดปกติ: ___________________

**Influential points** (Cook's D > 4/n):
- Observation IDs: ___________________

**สรุป**: ___________________

---

## 5. Multicollinearity Check

### 5.1 Correlation Matrix (ดู pair ที่ |r| > 0.8)
| Pair | Correlation | ความรุนแรง |
|------|------------|-----------|
| | | |

### 5.2 VIF Table
| Predictor | VIF | ปัญหา? |
|-----------|-----|-------|
| x1 | | |
| x2 | | |

**Threshold**: VIF > 5 = moderate, VIF > 10 = severe

---

## 6. Problems Found และ Fixes

| ปัญหา | พบ? | Fix ที่เสนอ |
|-------|-----|-----------|
| Non-linearity | ☐ Yes ☐ No | polynomial / transformation |
| Heteroscedasticity | ☐ Yes ☐ No | log(y) / √y transformation |
| Non-normality | ☐ Yes ☐ No | robust regression / transformation |
| Outliers | ☐ Yes ☐ No | investigate / remove if error |
| High leverage | ☐ Yes ☐ No | investigate data collection |
| Multicollinearity | ☐ Yes ☐ No | remove one / PCA / Ridge |

---

## 7. Model After Fix (ถ้ามีการแก้ไข)

**Fix ที่ทำ**: ___________________

**Comparison**:
| Metric | Before Fix | After Fix |
|--------|-----------|----------|
| R² | | |
| R²_adj | | |
| AIC | | |
| RSE | | |

**Diagnostic plots หลัง fix**: ___________________

---

## 8. Conclusions (5–7 ประโยค)

> เขียนสรุปเป็น paragraph สำหรับผู้ที่ไม่ใช่นักสถิติ ระบุ:
> 1. ปัญหาที่พบ
> 2. วิธีแก้ไข
> 3. ผลหลังแก้ไข
> 4. ข้อจำกัดที่ยังเหลือ

___________________

---

*Template นี้ใช้สำหรับ HW10 และ Lab10 Week 10 | 1145 201*
