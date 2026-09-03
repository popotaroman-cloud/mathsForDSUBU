---
description: ตรวจสอบ QA อัตโนมัติสำหรับ Week ที่กำหนด — รัน qa_check.py แล้วสรุปผลและแนะนำวิธีแก้
argument-hint: <week_number>
allowed-tools: Bash, Read, Glob
---

# QA Check — Week $ARGUMENTS

## ขั้นตอน

### Step 1 — รัน qa_check.py
```bash
python "E:\2569\Math for DS\qa_check.py" $ARGUMENTS
```

อ่าน output ทั้งหมด บันทึก error และ warning ทุกข้อ

### Step 2 — ตรวจ Notebook ด้วย Glob
ค้นหาไฟล์ทุกชนิดใน `Week$ARGUMENTS/`:
- `*.ipynb` — ตรวจว่ามีหรือไม่ และ JSON valid
- `*.md` — ตรวจว่ามีเนื้อหา (ไม่ใช่ empty)
- `*.pdf` — ตรวจว่า slides มีหรือยัง

### Step 3 — ตรวจ CLO Coverage
อ่าน `Week$ARGUMENTS/plan.md` แล้วตรวจว่า:
- CLO ที่ระบุในสัปดาห์นี้คืออะไร
- มี assignment/rubric ที่วัด CLO นั้นหรือไม่
- LLo สอดคล้องกับ activity ที่กำหนดหรือไม่

### Step 4 — สรุปผล QA

รายงานในรูปแบบ:

```
## QA Report — Week [N]
รัน: [วันที่]

### ผล Auto-check (qa_check.py)
- Errors: [จำนวน]
- Warnings: [จำนวน]

### รายการที่ต้องแก้ไข
| # | ประเภท | ปัญหา | วิธีแก้ |
|---|--------|-------|--------|
| 1 | ERROR  | ...   | ...    |
| 2 | WARN   | ...   | ...    |

### ไฟล์ที่มีแล้ว ✅
- ...

### ไฟล์ที่ยังขาด ❌
- ...

### CLO Coverage
| CLO | วัดโดย | สถานะ |
|-----|-------|--------|
| CLO[X] | [ไฟล์ assignment] | ✅/❌ |

### สถานะรวม
⬜ ยังไม่เริ่ม / 🔄 บางส่วน / ✅ พร้อมใช้
```

### Step 5 — อัปเดต _status.md
หากพบว่าไฟล์ใด complete แล้ว ให้อัปเดต `E:\2569\Math for DS\_status.md`
เปลี่ยน `⬜` → `✅` สำหรับไฟล์ที่ผ่าน QA แล้ว
