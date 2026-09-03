# Homework 10: Regression Diagnostics and Extensions
**วิชา** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล  
**CLO**: CLO3 | **LLo**: รวม qualitative predictor เข้า regression, ตรวจสอบ potential problems ด้วย diagnostic plots และแก้ปัญหาได้  
**กำหนดส่ง**: สัปดาห์ที่ 11  
**คะแนนเต็ม**: 10 คะแนน  

---

## บทนำ

สัปดาห์นี้เราขยาย regression ให้รองรับ qualitative predictor ผ่าน dummy variable, เพิ่ม interaction term จับ synergy ระหว่าง predictors และ polynomial feature สำหรับ non-linear relationship ที่สำคัญไม่แพ้กันคือการทำ **regression diagnostics** ซึ่งเป็นขั้นตอนที่นักสถิติและ Data Scientist ต้องทำทุกครั้งก่อนเชื่อผล regression เพราะ p-value น้อยและ R² สูง ไม่ได้รับประกันว่า model ถูกต้อง ในการบ้านนี้คุณจะฝึกเพิ่ม dummy variable, fit interaction model, ทำ diagnostic plots ครบ 4 ชุด และตรวจ VIF บน dataset จริง จากนั้นเขียนรายงานสรุปผลและเสนอแนะการแก้ปัญหา ทักษะเหล่านี้ใช้ในทุก regression project ก่อนเสนอผลให้ผู้บริหารหรือ publish

---

## ข้อ 1 — Dummy Variable และ Interaction (3 คะแนน)

ในข้อนี้คุณจะเพิ่ม categorical predictor เข้า regression และทดสอบว่า interaction term ช่วย model ไหม เพื่อให้เข้าใจว่า categorical data ถูกจัดการใน regression อย่างไร

**Dataset**: Advertising.csv + สร้าง categorical variable

**สิ่งที่ต้องทำ**:
1. สร้าง categorical variable `TV_level` จาก TV:
   - Low: TV < 100 (พัน$)
   - High: TV ≥ 100 (พัน$)
2. Fit model: `Sales ~ TV_level` ด้วย statsmodels (ใช้ `C()`)
3. ตีความ coefficient ของ TV_level: Sales ของ High-TV สูงกว่า Low-TV เฉลี่ยเท่าไร?
4. Fit interaction model: `Sales ~ TV + Radio + TV:Radio`
5. เปรียบเทียบ R² กับ additive model (Sales ~ TV + Radio)
6. ตีความ β̂_TV:Radio: หมายความว่าอะไร?

**Expected**: R² กระโดดจาก ~0.90 → ~0.97 เมื่อเพิ่ม interaction

**Deliverable**: Code + output + คำอธิบาย 2–3 ประโยคต่อข้อ

---

## ข้อ 2 — Regression Diagnostics (4 คะแนน)

ในข้อนี้คุณจะทำ diagnostic analysis ครบถ้วนบน regression model เพื่อตรวจสอบว่า assumptions ของ linear regression เป็นจริงหรือไม่

**Dataset**: Auto.csv (ถ้าไม่มี ดาวน์โหลดจาก ISLP)  
**Model**: mpg ~ horsepower

**สิ่งที่ต้องทำ**:
1. **4 Diagnostic Plots** (2×2 figure):
   - Residuals vs Fitted
   - Normal Q-Q Plot
   - Scale-Location
   - Leverage vs Residuals (Cook's Distance)

2. **อ่านผล**: สำหรับแต่ละ plot
   - เห็น pattern อะไร?
   - Assumption ข้อใดถูกละเมิด?
   - เสนอ fix อย่างไร?

3. **Polynomial Fix**: Fit `mpg ~ horsepower + I(horsepower**2)`
   - เปรียบเทียบ R² กับ linear model
   - สร้าง diagnostic plots ใหม่ — ดีขึ้นไหม?

4. **Outliers**: ระบุ observation ที่มี studentized residual |rᵢ| > 2
   - มีกี่ observations?
   - ดูค่า horsepower และ mpg ของ observations นั้น

**Deliverable**: Code + 4 plots (linear) + 4 plots (polynomial) + ตารางสรุป + คำอธิบาย

---

## ข้อ 3 — VIF และ Multicollinearity Report (3 คะแนน)

ในข้อนี้คุณจะตรวจสอบ multicollinearity บน dataset จริงและเขียนรายงานสรุปผลพร้อมเสนอแนวทางแก้ไข

**Dataset**: เลือก 1 จาก:
- Credit.csv: Balance ~ Rating + Limit + Income + Age + Education + Student
- Boston Housing: medv ~ lstat + rm + age + dis + tax + ptratio

**สิ่งที่ต้องทำ**:
1. Fit full MLR model
2. สร้าง correlation heatmap ของ predictors ทั้งหมด
3. คำนวณ VIF ของแต่ละ predictor
4. สรุปในตาราง:

| Predictor | Correlation กับตัวอื่น (max) | VIF | Multicollinearity? |
|-----------|---------------------------|-----|-------------------|
| Rating | | | |
| Limit | | | |
| ... | | | |

5. **แก้ปัญหา**: ถ้ามี VIF > 5
   - ลบ predictor ที่ collinear ออก → fit ใหม่ → เปรียบ AIC
   - หรือ PCA ถ้า VIF > 10

6. **Diagnostics Report** (paragraph 5–7 ประโยค):
   - "ตรวจพบปัญหา ___ จาก diagnostic plots: ..."
   - "VIF ของ ___ = ___ ซึ่งสูงกว่าเกณฑ์ 5 → แสดงว่า ___ correlated กับ ___"
   - "แก้ไขโดย ___ → R²_adj เปลี่ยนจาก ___ เป็น ___"
   - "หลังแก้ไข model ยังคง significant (F p-value = ___) และ diagnostics ดีขึ้น"

**Deliverable**: Code + tables + Diagnostics Report

---

## รูปแบบการส่งงาน

- ส่งเป็น **Jupyter Notebook** (`.ipynb`) ที่ run ผ่านแล้ว
- ตั้งชื่อ: `hw10_XXXXXXXX.ipynb`

## เกณฑ์การให้คะแนน

| ข้อ | เนื้อหา | คะแนน |
|-----|--------|-------|
| 1 | Dummy + Interaction | 3 |
| 2 | Regression Diagnostics | 4 |
| 3 | VIF + Report | 3 |
| **รวม** | | **10** |
