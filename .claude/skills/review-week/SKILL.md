---
description: Review เนื้อหาของ Week ที่กำหนด — ตรวจ CLO alignment, ความถูกต้องของสูตร, ความสมบูรณ์ของ lab และ assignment
argument-hint: <week_number>
allowed-tools: Read, Glob, Grep, Bash
---

# Review Week $ARGUMENTS

## ขั้นตอนที่ต้องทำตามลำดับ

### Step 1 — อ่าน Plan และ Course Outline
อ่านไฟล์ต่อไปนี้เพื่อเข้าใจ context:
- `E:\2569\Math for DS\Week$ARGUMENTS\plan.md` — subtopics, CLO, LLo, รายการไฟล์ที่ควรมี
- `E:\2569\Math for DS\course_outline.md` — ตาราง 15 สัปดาห์ (ดู row ของ Week $ARGUMENTS)

### Step 2 — ตรวจ Structure
ใช้ Glob หา files ที่มีอยู่จริงใน:
- `E:\2569\Math for DS\Week$ARGUMENTS\lecture\`
- `E:\2569\Math for DS\Week$ARGUMENTS\lab\`
- `E:\2569\Math for DS\Week$ARGUMENTS\assignment\`
- `E:\2569\Math for DS\Week$ARGUMENTS\slides\`

เปรียบเทียบกับรายการใน plan.md — ไฟล์ไหนมีแล้ว ไฟล์ไหนยังขาด

### Step 3 — Review Lecture Notes
สำหรับทุก `.md` ใน `lecture/` ตรวจ **3 กฎบังคับ** ก่อน:

**กฎ 1 — บทนำสัปดาห์:**
- มีบทนำ 5–10 ประโยคที่ตอบครบ 3 คำถามหรือไม่:
  - [ ] บอกว่าสัปดาห์นี้เกี่ยวกับอะไร
  - [ ] บอกเป้าหมาย (นักศึกษาทำอะไรได้หลังเรียน)
  - [ ] บอกว่านำไปใช้ทำอะไรได้ใน DS จริง

**กฎ 2 — Code Documentation (ถ้ามี code block):**
- ทุก code block มี comment `# ─── ทำอะไร ───` หรือไม่
- ทุก block มี `# วัตถุประสงค์: เพื่ออะไร` หรือไม่
- ไม่มี code ที่ไม่มี comment เลย

**กฎ 3 — What & Why:**
- ทุก section มีย่อหน้าเปิดที่บอก "ในส่วนนี้เราจะ [ทำอะไร] เพื่อ [อะไร]" หรือไม่

**ตรวจเนื้อหา:**
- subtopics ใน plan.md ถูก cover ครบหรือไม่
- สูตร/สมการ: notation ถูกต้อง, มี worked example หรือไม่
- DS connection: มีอย่างน้อย 2–3 ประโยคหรือไม่

### Step 4 — Review Lab Notebook
สำหรับทุก `.ipynb` ใน `lab/` (ยกเว้น solution) ตรวจ **3 กฎบังคับ** ก่อน:

**กฎ 1 — บทนำสัปดาห์:**
- Cell markdown แรกมีบทนำ 5–10 ประโยคครบ 3 คำถามหรือไม่

**กฎ 2 — Code Documentation:**
- สุ่มตรวจ 3 code cells: ทุก block มี comment อธิบายทั้ง "ทำอะไร" และ "วัตถุประสงค์" หรือไม่
- มี code ที่ไม่มี comment เลยหรือไม่ (ถ้ามี = fail)

**กฎ 3 — What & Why:**
- ทุก Part มี markdown cell ที่บอก "Part นี้ทำอะไร เพื่ออะไร" หรือไม่
- ทุก TODO มี context อธิบายก่อนว่า "ต้องการ X เพราะ Y" หรือไม่

**ตรวจเนื้อหา:**
- import libraries ครบตาม plan.md
- dataset ตรงกับที่ระบุใน plan.md
- TODO ชัดเจนพอสำหรับนักศึกษา (ไม่กว้างเกิน / ไม่แคบเกิน)

### Step 5 — Review Assignments
สำหรับทุก `.md` ใน `assignment/`:
- ตรวจว่า HW โจทย์ครอบคลุม LLo ที่กำหนดใน plan.md
- ตรวจว่า rubric มีและ map กลับไป CLO
- ตรวจว่าระดับความยากเหมาะสมกับชั้นปีที่ 2

### Step 6 — CLO Alignment Check
ดู CLO ที่ระบุใน plan.md (CLO1/CLO2/CLO3/CLO4) แล้วตรวจว่า:
- เนื้อหาที่สอนจริงสนับสนุน CLO นั้นได้จริงหรือไม่
- LLo (Lesson Learning Outcome) วัดได้จาก assignment ที่มีหรือไม่

### Step 7 — สรุปรายงาน
รายงานในรูปแบบ:

```
## Review Report — Week [N]: [หัวข้อ]

### 🔴 3 กฎบังคับ
| กฎ | Lecture | Lab | Assignment | Slides |
|----|---------|-----|-----------|--------|
| กฎ 1: บทนำ 5–10 ประโยค (เกี่ยวกับอะไร / เป้าหมาย / นำไปใช้อะไร) | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ |
| กฎ 2: ทุก code มี comment (ทำอะไร + วัตถุประสงค์) | N/A | ✅/❌ | N/A | N/A |
| กฎ 3: ทุกขั้นตอนบอก What & Why | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ |

**กฎที่ไม่ผ่าน (ระบุตำแหน่งที่พบ):**
- ...

### ✅ จุดแข็ง
- ...

### ❌ ปัญหาที่ต้องแก้ไข (Critical)
- ...

### ⚠️ ข้อเสนอแนะ (Minor)
- ...

### 📋 ไฟล์ที่ขาด
| ไฟล์ | ประเภท | ความสำคัญ |
|------|--------|----------|
| ... | ... | ... |

### 🎯 CLO Alignment
| CLO | ครอบคลุมใน | สถานะ |
|-----|----------|--------|
| ... | ... | ✅/❌ |

### คะแนนภาพรวม: X/10
(หักคะแนนถ้า 3 กฎบังคับไม่ผ่าน: กฎละ -2 คะแนน)
```
