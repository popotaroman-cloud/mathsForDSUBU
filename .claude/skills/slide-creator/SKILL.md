---
description: แปลง Slide Outline .md → HTML Presentation ตัวอักษรใหญ่เหมือน Google Slides — รองรับ LaTeX (KaTeX) และ auto-generate กราฟจริงจาก [FIGURE: ...]
argument-hint: <week_N | path/to/file.md | path/to/slides/ | all>
allowed-tools: Bash, Read, Glob
---

# Slide Page Creator

## สิ่งที่ Skill นี้ทำ

รัน `create_slides.py` เพื่อแปลง `.md` slide outline → `.html` presentation ที่:
- ตัวอักษรใหญ่ (Title ~50px, Content ~26px) ปรับ responsive อัตโนมัติ
- Slide 1 = Title Slide พื้นหลังไล่สี navy → blue
- Key Message = callout box สีฟ้า
- Support: bullet lists, numbered lists, code blocks, tables
- **LaTeX**: สูตรคณิตศาสตร์ผ่าน KaTeX 0.16.10 — `$...$` (inline) และ `$$...$$` (display)
- **Auto Figure**: `[FIGURE: description]` → กราฟ matplotlib จริงฝังเป็น base64 PNG (~70% match rate)
- Navigation: ↔ keyboard, Space, swipe, ปุ่ม ◀ ▶ + progress bar
- ต้องการ internet สำหรับ Google Fonts และ KaTeX CDN

---

## ขั้นตอน

### Step 1 — ตีความ argument

| Input | แปลเป็น |
|-------|--------|
| `week 4` หรือ `4` | `Week04/slides/` |
| `all` | ทุก `Week*/slides/` |
| path ลงท้าย `.md` | ไฟล์นั้นไฟล์เดียว |
| path เป็น folder | ทุก `.md` ใน folder |

### Step 2 — รัน create_slides.py

```bash
# สร้าง 1 ไฟล์
python "E:\2569\Math for DS\create_slides.py" "<md_path>"

# สร้าง 1 ไฟล์ ระบุ output
python "E:\2569\Math for DS\create_slides.py" "<md_path>" "<out.html>"

# สร้างทั้ง folder
python "E:\2569\Math for DS\create_slides.py" "<folder_path>"

# สร้างทั้ง week
python "E:\2569\Math for DS\create_slides.py" <N>

# สร้างทุก week
python "E:\2569\Math for DS\create_slides.py" all
```

### Step 3 — รายงานผล

แจ้งรายการ `.html` ที่สร้างพร้อม path เต็ม จำนวน slides ต่อ deck และขนาดไฟล์

---

## รูปแบบ .md ที่รองรับ

```markdown
# Slide Deck: ชื่อ Deck        ← ชื่อ presentation (ปรากฏใน <title>)
> Week N | CLO | X slides

---

## Slide 1 — Title             ← หัวข้อ slide
**Key Message**: ข้อความสำคัญ  ← callout box (optional)

เนื้อหา paragraph

- bullet item
- bullet item

---

## Slide 2 — Title
| Col 1 | Col 2 |
|-------|-------|
| data  | data  |

---

## Slide 3 — Code Example
\`\`\`python
x = np.array([1, 2, 3])
\`\`\`

---

## Slide 4 — LaTeX Math
**Key Message**: สูตรคณิตศาสตร์แสดงผ่าน KaTeX อัตโนมัติ

inline math: $\hat{\beta}_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$

display math (ขึ้นบรรทัดใหม่ จัดกลาง):

$$A = U \Sigma V^\top$$

- ใช้ `$...$` สำหรับสูตรในบรรทัด
- ใช้ `$$...$$` สำหรับสูตรแบบ block (centered)
- รองรับ: `\frac`, `\sum`, `\mathbf`, `\hat`, `\begin{pmatrix}...\end{pmatrix}` ฯลฯ
```

---

## LaTeX / KaTeX — รายละเอียด

| รูปแบบ | ตัวอย่างใน .md | ผลลัพธ์ใน HTML |
|--------|--------------|----------------|
| Inline | `$\beta_0 + \beta_1 x$` | สูตรแทรกในบรรทัด |
| Display | `$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$` | สูตร block จัดกลาง |
| Matrix | `$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$` | matrix หลายแถว |

**หมายเหตุสำคัญ:**
- สูตร LaTeX จะถูก **protect** ก่อน markdown processing — ดังนั้น `*` ใน `f^*(x)` จะไม่ถูกแปลงเป็น italic
- ต้องการ internet เพื่อโหลด KaTeX CDN (`cdn.jsdelivr.net`) ตอนเปิดไฟล์ HTML
- ถ้าไม่มี internet สูตรจะแสดงเป็น raw LaTeX source แต่ content อื่นยังทำงานปกติ

---

## ตัวอย่างการใช้

```
/slide-creator week 6
/slide-creator Week15/slides/01_course_summary.md
/slide-creator Week01/slides/
/slide-creator all
```

---

---

## Auto Figure Generation — `[FIGURE: description]`

`create_slides.py` จะพยายาม generate กราฟ matplotlib จาก description อัตโนมัติ
ถ้า match pattern → ฝัง PNG base64 ในไฟล์ HTML  
ถ้าไม่ match → แสดง dashed placeholder box เหมือนเดิม

### Pattern ที่รองรับ (keyword ใน description)

| Pattern | Keywords ที่ trigger | Generator |
|---------|---------------------|-----------|
| 2D Vector arrows | `vector`, `เวกเตอร์`, `2d plane`, `parallelogram`, `u+v`, `arrow` | เวกเตอร์ 2D พร้อม label |
| Orthogonal projection | `projection`, `orthogonal`, `ฉาย`, `perpendicular` | diagram proj_b a พร้อมมุมฉาก |
| Number line | `number line`, `scalar`, `สเกลาร์` | เส้นจำนวนพร้อม highlight |
| Scatter plot | `scatter`, `กระจาย`, `data point` | scatter พร้อม regression line ถ้ามีคำ |
| Regression line | `regression`, `best fit`, `ถดถอย`, `β₀`, `β₁`, `เส้นตรง` | scatter + best-fit line |
| Normal distribution | `normal`, `gaussian`, `bell curve`, `การแจกแจง`, `distribution` | bell curve (±1.96σ ถ้ามี CI) |
| Histogram | `histogram`, `ฮิสโทแกรม`, `frequency` | histogram + overlay curve |
| Boxplot | `boxplot`, `q1`, `q2`, `q3`, `quartile`, `iqr`, `whisker` | boxplot พร้อม annotation |
| Bias-Variance tradeoff | `bias`, `variance`, `tradeoff`, `overfit`, `underfit` | U-curve + sweet spot |
| Polynomial overfitting | `polynomial`, `degree`, `d=1`, `d=12` | 3-panel underfit/fit/overfit |
| Learning curve | `learning curve`, `train error`, `validation error` | train vs val error |
| Logistic sigmoid | `logistic`, `sigmoid`, `ล็อจิสติก`, `log-odds` | sigmoid + decision boundary |
| ROC curve | `roc`, `auc`, `fpr`, `tpr`, `false positive` | ROC + AUC shading |
| Confusion matrix | `confusion`, `heatmap` | heatmap พร้อมตัวเลข |
| Correlation heatmap | `correlation matrix` | 4×4 heatmap |
| PCA/SVD | `pca`, `principal component`, `svd`, `eigenvector` | scatter + PC arrows |
| Bar chart | `bar chart`, `bar plot`, `แท่ง`, `comparison`, `เปรียบเทียบ` | bar chart พร้อม label |
| LLN convergence | `convergence`, `x̄`, `lln`, `law of large`, `sample mean` | x̄ₙ → μ line plot |
| CI simulation | `ci interval`, `confidence interval`, `ครอบ true`, `95%` | 20 CI intervals |
| R² decomposition | `tss`, `rss`, `ess`, `r²`, `r2`, `explained` | stacked bar TSS=RSS+ESS |

### Fallback — ไม่ generate (ยังเป็น placeholder)

Diagram ที่ต้องวาด manually (ไม่เหมาะกับ matplotlib):
- Block diagrams เชิง conceptual (Strang's Big Picture, row×column rule)
- Step-by-step algorithm diagrams (Gram-Schmidt, matrix row operations)
- Index/mapping diagrams
- 3D projection diagrams (b = p + e)

### Dependency

```
pip install matplotlib numpy
```

ถ้าไม่ได้ install → ทุก `[FIGURE: ...]` จะ fallback เป็น placeholder อัตโนมัติ (ไม่ error)

---

## ไฟล์ที่เกี่ยวข้อง

- `E:\2569\Math for DS\create_slides.py` — Python converter script
- Output: `.html` ไฟล์เดียวกับ `.md` (เช่น `01_intro.md` → `01_intro.html`)
