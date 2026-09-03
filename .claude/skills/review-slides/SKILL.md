---
description: ตรวจและแก้ไข Slide .md ให้สมบูรณ์ครบถ้วน — ประเมินความครบถ้วนของแต่ละ deck แล้วส่ง Agent แก้ไขเป็นไฟล์ๆ จากนั้น rebuild HTML
argument-hint: <week_N | all | path/to/slides/file.md | path/to/slides/>
allowed-tools: Read, Glob, Bash, Agent, Edit
---

# Review Slides

## สิ่งที่ Skill นี้ทำ

1. **Evaluate** — อ่านและตรวจสอบทุก slide .md ในขอบเขตที่กำหนดตาม Completeness Rubric
2. **Report** — แสดงรายงานสรุป: ปัญหาต่อ deck ต่อ slide
3. **Fix** — สำหรับทุก deck ที่มีปัญหา ส่ง Claude Agent 1 ตัวต่อ 1 ไฟล์ เพื่อแก้ไขใน .md โดยตรง
4. **Rebuild** — รัน `create_slides.py` เพื่อ rebuild HTML ที่แก้แล้ว

---

## ขั้นตอน

### Step 1 — ตีความ argument และหาไฟล์

| Input | ขอบเขต |
|-------|--------|
| `week 4` หรือ `4` | ทุก `.md` ใน `Week04/slides/` |
| `all` | ทุก `Week*/slides/*.md` |
| path ลงท้าย `.md` | ไฟล์นั้นเดียว |
| path เป็น folder | ทุก `.md` ใน folder นั้น |

ใช้ Glob หาไฟล์ทั้งหมดในขอบเขต

### Step 2 — อ่านและประเมิน (Evaluate Pass)

**สำหรับแต่ละ .md file**: ใช้ Read อ่านเนื้อหาทั้งหมด จากนั้น run Completeness Rubric (หัวข้อถัดไป)

**ตรวจ [FIGURE: ...] ว่า render เป็นกราฟจริงได้ไหม** โดยรัน Bash:

```bash
python -X utf8 -c "
import sys
sys.path.insert(0, 'E:\\\\2569\\\\Math for DS')
from create_slides import generate_figure, _HAS_MPL
descs = [
    ('Slide X', 'description 1'),
    ('Slide Y', 'description 2'),
]
for slide, desc in descs:
    r = generate_figure(desc) if _HAS_MPL else None
    status = 'CHART' if r else 'PLACEHOLDER'
    print(f'{status} | {slide} | {desc[:60]}')
"
```

ถ้าได้ `PLACEHOLDER` → flag เป็น 🟡 Minor `[FIGURE: ...] ยังไม่สร้างกราฟจริง`

รวบรวมผลเป็น list:
```
issues[filepath] = [
    {slide_num, type, severity, detail},
    ...
]
```

### Step 3 — รายงานสรุป

แสดงตารางต่อ deck:
```
## 📋 Review Report — [scope]

### [filename]  (N slides)
| Slide | ปัญหา | ระดับ |
|-------|-------|-------|
| Slide 2 | ไม่มี Key Message | 🔴 Critical |
| Slide 5 | เนื้อหาน้อยเกินไป (1 bullet) | 🟡 Minor |
| Slide 7 | [FIGURE: ...] คำอธิบายสั้นเกิน | 🟡 Minor |
| Deck    | ไม่มี Summary slide | 🔴 Critical |

สถานะ: [✅ ครบถ้วน / ⚠️ มีปัญหาเล็กน้อย / ❌ ต้องแก้ไข]
```

หลังรายงานครบทุก deck — แสดง **Summary**:
```
📊 สรุป: N deck | M ปัญหา Critical | K ปัญหา Minor
🔧 จะส่ง Agent แก้ไข X ไฟล์
```

### Step 4 — ส่ง Agent แก้ไข (Fix Pass)

**สำหรับทุก filepath ที่มี issues**: spawn Agent 1 ตัว ต่อ 1 ไฟล์

โดยใช้ **Agent Prompt Template** (ดูด้านล่าง)

- spawn แบบ **background=true** เพื่อทำงานพร้อมกัน (ถ้ามีหลายไฟล์)
- รอผลและรายงานไฟล์ที่แก้สำเร็จ

### Step 5 — Rebuild HTML

หลัง Agent ทุกตัวเสร็จ:
```bash
python "E:\2569\Math for DS\create_slides.py" <scope>
```

รายงาน .html ที่ rebuild สำเร็จ

---

## Completeness Rubric

### ✅ ระดับ Deck (ตรวจทั้ง deck)

| เกณฑ์ | ผ่าน | ระดับ |
|-------|------|-------|
| จำนวน slides | ≥ 8 slides | 🔴 Critical |
| มี Summary slide | Slide สุดท้ายมีคำว่า "สิ่งที่เรียนรู้" หรือ "Summary" หรือ "สรุป" | 🔴 Critical |
| มี Next Week preview | Slide สุดท้ายมี "สัปดาห์ต่อไป" หรือ "Next" | 🟡 Minor |
| Slide 2 เป็น Overview | Slide 2 มีเนื้อหา ≥ 300 chars | 🟡 Minor |

### ✅ ระดับ Slide (ตรวจทุก slide ยกเว้น Slide 1)

| เกณฑ์ | ผ่าน | ระดับ |
|-------|------|-------|
| Key Message | มี `**Key Message**: ...` ที่ยาว ≥ 15 chars | 🔴 Critical |
| Content Depth | ≥ 3 bullets หรือ ≥ 1 code block หรือ ≥ 1 table | 🟡 Minor |
| FIGURE description | `[FIGURE: ...]` ยาว ≥ 25 chars (ไม่ใช่แค่ "กราฟ" หรือ "chart") | 🟡 Minor |
| **FIGURE renders as chart** | `generate_figure(desc)` คืนค่า HTML (ไม่ใช่ None) — ตรวจด้วย Bash command ใน Step 2 | 🟡 Minor |
| ไม่มี slide ว่าง | มีเนื้อหา ≥ 1 บรรทัด | 🔴 Critical |

> **หมายเหตุ**: ถ้า `[FIGURE: ...]` ยังเป็น `PLACEHOLDER` หลัง generate — Agent ที่แก้ไขต้องทำอย่างใดอย่างหนึ่ง:
> 1. **ปรับ description** ให้มี keyword ที่ generator รู้จัก (เช่น เพิ่มคำว่า "element-wise", "scatter", "normal distribution" ฯลฯ)
> 2. **ระบุว่าเป็น structural diagram** ที่ไม่ควร auto-generate (เช่น flow chart, step-by-step algorithm) — ให้ comment ไว้ว่า `[FIGURE: ... <!-- structural, keep as placeholder -->]`

### ⚪ Slide 1 (Title Slide) — ตรวจต่างจากสไลด์อื่น

| เกณฑ์ | ผ่าน |
|-------|------|
| มีชื่อวิชา + Week | บรรทัดแรกมี "1145 201" หรือ "Week" |
| มี CLO ระบุ | มีคำว่า "CLO" |

---

## Agent Prompt Template

เมื่อพบไฟล์ที่ต้องแก้ไข ให้ spawn Agent ด้วย prompt ดังนี้ (แทน placeholder ด้วยข้อมูลจริง):

```
คุณเป็น slide content writer สำหรับรายวิชา 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล

งานของคุณ: แก้ไขไฟล์ slide .md ให้สมบูรณ์ตาม issues ที่ระบุ

## ไฟล์ที่ต้องแก้
Path: [FULL_PATH_TO_MD_FILE]

## Issues ที่พบ
[LIST_OF_ISSUES_FROM_EVALUATE_PASS]
ตัวอย่าง:
- Slide 3: ไม่มี Key Message → เพิ่ม **Key Message**: [ประโยคสรุป 1 ประโยค]
- Slide 5: เนื้อหาน้อย (1 bullet) → เพิ่ม bullets ให้ได้อย่างน้อย 3 ข้อ
- Slide 8: [FIGURE: กราฟ] คำอธิบายสั้นเกิน → เปลี่ยนเป็นคำอธิบายที่ระบุชนิดกราฟและข้อมูลที่แสดง
- Deck: ไม่มี Summary slide → เพิ่ม slide สุดท้ายที่มี "สิ่งที่เรียนรู้วันนี้" + preview Week ถัดไป

## context ของ Deck นี้
Week: [WEEK_NUMBER]
Deck topic: [TOPIC_FROM_FILENAME_OR_SLIDE1_TITLE]
CLO: [CLO_FROM_PLAN_OR_SLIDE1]

## กฎที่ต้องทำตามเสมอ

### Key Message
- 1 ประโยคสรุปว่า slide นี้ต้องการสื่ออะไร
- บอก "what" และ "why" คู่กัน
- ตัวอย่างที่ดี: "SVD ทำงานกับ matrix ทุกรูปร่าง ทำให้เป็น universal tool ที่ใช้ได้กับ data matrix จริงเสมอ"
- ตัวอย่างไม่ดี: "SVD คือการแยก matrix" (ไม่บอก why)

### Content Depth (ต่อ slide)
- ≥ 3 bullets ที่มีเนื้อหาจริง (ไม่ใช่แค่หัวข้อ)
- แต่ละ bullet ควรมี context หรือตัวอย่างประกอบ
- ถ้าเพิ่ม bullet ให้ต่อเนื่องกับ topic ของ slide

### FIGURE description — ต้องทำให้ generate กราฟจริงได้
- ระบุ: ชนิดกราฟ + ข้อมูลที่แสดง + สิ่งที่ต้องการให้เห็น
- ตัวอย่างที่ดี: "scatter plot ของ TV budget vs Sales พร้อม regression line แสดง positive correlation"
- ตัวอย่างไม่ดี: "กราฟ" หรือ "scatter plot"
- ใส่ keyword ที่ generator รู้จัก: `vector`, `scatter`, `regression`, `normal distribution`, `histogram`, `boxplot`, `bias variance tradeoff`, `logistic sigmoid`, `roc curve`, `confusion matrix`, `pca`, `element-wise`, `bar chart`, `learning curve`, `confidence interval`, `tss rss ess`
- ถ้าเป็น block diagram / structural diagram (flow chart, step-by-step) → เขียนว่า `[FIGURE: ... <!-- structural -->]` เพื่อบอกว่า placeholder เป็นสิ่งที่ตั้งใจไว้

### Summary Slide (ถ้าต้องเพิ่ม)
```markdown
## Slide N — Summary
**Key Message**: [ประโยคสรุป deck ทั้งหมด]

**สิ่งที่เรียนรู้วันนี้**
- [concept 1 จาก deck นี้]
- [concept 2]
- [concept 3]

**สัปดาห์ต่อไป**: [preview หัวข้อ Week ถัดไป ที่ต่อยอดจาก deck นี้]
```

## วิธีแก้ไขไฟล์

1. ใช้ Read อ่านไฟล์ทั้งหมดก่อน
2. ระบุ exact position ของแต่ละ issue
3. ใช้ Edit tool แก้ไขเฉพาะส่วนที่มีปัญหา (อย่าเขียนทับทั้งไฟล์)
4. ถ้าต้องเพิ่ม Summary slide ให้ append ต่อท้ายไฟล์
5. เมื่อแก้เสร็จ รายงานรายการที่แก้ไขแล้วกลับมา
```

---

## ตัวอย่างการใช้

```
/review-slides week 4
/review-slides Week08/slides/02_least_squares_estimation.md
/review-slides Week01/slides/
/review-slides all
```

---

## หมายเหตุสำคัญ

- **อย่า over-write**: Agent ต้องใช้ Edit tool แก้เฉพาะส่วนที่มีปัญหา ไม่เขียนทับทั้งไฟล์
- **คงสไตล์**: เนื้อหาที่เพิ่มต้องใช้ภาษาและรูปแบบเดียวกับ slides ที่มีอยู่แล้ว
- **Context-aware**: Agent ต้องอ่านทั้ง deck ก่อนแก้ เพื่อให้ content ต่อเนื่อง
- **Rebuild**: หลังแก้ไขเสร็จทุกไฟล์ ต้อง rebuild HTML เสมอ

---

## ไฟล์ที่เกี่ยวข้อง

- `E:\2569\Math for DS\create_slides.py` — rebuild HTML หลัง fix
- `E:\2569\Math for DS\Week[N]\plan.md` — context: CLO, LLo, หัวข้อ
- `E:\2569\Math for DS\course_outline.md` — Week progression สำหรับ next week preview
