# Note 1: บทบาทของคณิตศาสตร์ใน Data Science
> Week 1 | CLO1 | Strang Reference: Ch.1 (Introduction)

---

## บทนำ

สัปดาห์นี้เราจะเริ่มต้นการเดินทางสู่โลกของ **คณิตศาสตร์สำหรับวิทยาการข้อมูล (Mathematics for Data Science)** โดยตอบคำถามสำคัญว่า "ทำไม Data Scientist ต้องเรียนคณิตศาสตร์?" เป้าหมาย (LLo) ของสัปดาห์นี้คือให้นักศึกษาสามารถ **อธิบายความสำคัญของคณิตศาสตร์ในวิทยาการข้อมูล** และเข้าใจโครงสร้างพื้นฐานของทุกอย่างใน Machine Learning ซึ่งล้วนสร้างขึ้นจากคณิตศาสตร์สามสาขาหลัก ได้แก่ **Algebra เชิงเส้น (Linear Algebra)**, **แคลคูลัส (Calculus)** และ **ความน่าจะเป็น (Probability & Statistics)**

ในชีวิตจริง ทุกครั้งที่ Netflix แนะนำหนังให้คุณ, Google แปลภาษา, หรือโทรศัพท์จดจำใบหน้าของคุณ — ล้วนเกิดจากการคำนวณทางคณิตศาสตร์ทั้งสิ้น โดยเฉพาะ **Linear Algebra** ซึ่งเป็นหัวใจหลักของ Deep Learning ทุกระบบ ข้อมูล (Data) ทุกชนิดไม่ว่าจะเป็นรูปภาพ ข้อความ หรือตัวเลข ล้วนถูกแปลงเป็น **Vector** และ **Matrix** ก่อนที่คอมพิวเตอร์จะประมวลผลได้ สัปดาห์นี้เราจะปูพื้นฐานสำคัญนี้เพื่อให้การเรียนรู้ในสัปดาห์ต่อ ๆ ไปราบรื่นและมีความหมาย

---

## Section 1: ทำไม Data Scientist ต้องเรียนคณิตศาสตร์?  *(Strang Preface)*

ในส่วนนี้เราจะสำรวจว่า **คณิตศาสตร์เชื่อมกับ Machine Learning อย่างไร** เพื่อสร้างแรงจูงใจในการเรียนตลอดภาคการศึกษา

### 1.1 แผนที่คณิตศาสตร์สำหรับ DS

```
Data Science / Machine Learning
        │
        ├── Linear Algebra  ← สัปดาห์ที่ 1–4 (หัวข้อนี้!)
        │       ├── Vector, Matrix, Eigenvalues
        │       ├── SVD, PCA
        │       └── Least Squares (ฐานของ Regression)
        │
        ├── Calculus & Optimization  ← สัปดาห์ที่ 5–8
        │       ├── Gradient, Derivative
        │       └── Gradient Descent (หัวใจ Neural Net)
        │
        └── Probability & Statistics  ← สัปดาห์ที่ 9–12
                ├── Probability Distributions
                └── Bayesian Inference, MLE
```

**DS Connection**: โมเดล Machine Learning ทุกตัวที่ใช้อยู่จริง — ตั้งแต่ Linear Regression ง่าย ๆ ไปจนถึง GPT — ล้วนใช้ Linear Algebra เป็น "ภาษา" ในการสื่อสารระหว่างข้อมูลและคอมพิวเตอร์ หากเข้าใจ Linear Algebra จะเข้าใจว่า Neural Network "ทำอะไร" อยู่จริง ๆ

### 1.2 ตัวอย่างการใช้งานจริง

| งาน DS/ML | คณิตศาสตร์ที่ใช้ |
|-----------|----------------|
| Image Classification (CNN) | Matrix multiplication, Convolution |
| Recommendation System | Matrix Factorization, SVD |
| Natural Language Processing | Word Vectors, Cosine Similarity |
| PCA / Dimensionality Reduction | Eigenvalues, SVD |
| Linear Regression | Least Squares, Normal Equation |
| Neural Network Training | Gradient (Calculus), Backpropagation |

---

## Section 2: Scalar คืออะไร?  *(Strang 1.1)*

ในส่วนนี้เราจะนิยาม **Scalar** และแยกความแตกต่างจาก Vector เพื่อสร้างพื้นฐานการเข้าใจข้อมูลทางคณิตศาสตร์

**นิยาม (Scalar)**: จำนวนจริงตัวเดียว (real number) ไม่มีทิศทาง มีเพียงขนาด

**ตัวอย่าง Scalar ใน Data Science**:
- อุณหภูมิ: 36.5 °C
- ราคาหุ้น: 158.50 บาท
- คะแนนสอบ: 87.5 คะแนน
- Learning rate ใน Neural Network: α = 0.001

**สัญลักษณ์**: ใช้ตัวอักษรพิมพ์เล็ก เช่น $a, b, c, \alpha, \lambda$

**การดำเนินการบน Scalar**:
$$a + b, \quad a - b, \quad a \times b, \quad \frac{a}{b} \quad (b \neq 0)$$

**DS Connection**: ใน Neural Network แต่ละ "น้ำหนัก (weight)" เป็น scalar ตัวหนึ่ง แต่เครือข่ายที่มีล้านพารามิเตอร์จะรวม scalar เหล่านั้นเป็น vector และ matrix เพื่อให้คำนวณได้เร็วด้วย GPU

---

## Section 3: Vector คืออะไร?  *(Strang 1.1–1.2)*

ในส่วนนี้เราจะเรียนรู้ว่า **Vector** คืออะไร ทั้งในมุมมองเรขาคณิต (ลูกศร) และพีชคณิต (ลำดับตัวเลข) เพื่อให้เห็นว่า Vector คือหน่วยพื้นฐานที่สุดของข้อมูลใน ML

### 3.1 นิยามและสัญลักษณ์

**นิยาม (Vector)**: รายการของตัวเลข (ordered list of numbers) ที่มีทั้งขนาดและทิศทาง

**สัญลักษณ์**: ตัวอักษรพิมพ์เล็กตัวหนา $\mathbf{v}$ หรือมีลูกศร $\vec{v}$

**Column Vector** (มาตรฐานใน Linear Algebra):
$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} \in \mathbb{R}^n$$

**Row Vector**:
$$\mathbf{v}^T = \begin{pmatrix} v_1 & v_2 & \cdots & v_n \end{pmatrix}$$

### 3.2 ตัวอย่าง: Vector ใน Data Science

**ตัวอย่างที่ 1: Feature Vector ของบ้าน**
```
บ้านหลังหนึ่ง → v = [พื้นที่ (ตร.ม.), จำนวนห้องนอน, อายุ (ปี), ราคา (บาท)]
              → v = [120, 3, 5, 2500000]
```
Vector นี้อยู่ใน $\mathbb{R}^4$ (มี 4 มิติ)

**ตัวอย่างที่ 2: Word Vector (Word2Vec)**
```
คำว่า "กษัตริย์" → v = [0.8, -0.2, 0.5, ..., 0.1]  (300 มิติ)
คำว่า "ราชินี"  → u = [0.7, -0.1, 0.6, ..., 0.2]  (300 มิติ)
```
Vector ที่คล้ายกันหมายถึงคำที่มีความหมายใกล้เคียงกัน!

**ตัวอย่างที่ 3: Image as Vector**
```
ภาพขาวดำ 28×28 pixels (MNIST digit) →
flatten → v ∈ R^784 (784 ตัวเลข, แต่ละตัวคือความสว่าง 0-255)
```

### 3.3 การวาด Vector ใน 2D และ 3D

**ใน $\mathbb{R}^2$**: $\mathbf{v} = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$ คือลูกศรจาก origin (0,0) ไปยัง จุด (3, 2)

**ใน $\mathbb{R}^3$**: $\mathbf{w} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ คือลูกศรใน 3 มิติ

**DS Connection**: ใน Machine Learning "data point" แต่ละจุดคือ vector ใน high-dimensional space เช่น ภาพ 224×224 pixels (ImageNet) คือ vector ใน $\mathbb{R}^{150528}$ — เราต้องการ Linear Algebra เพื่อจัดการ space ขนาดใหญ่เหล่านี้ได้อย่างมีประสิทธิภาพ

---

## Section 4: การดำเนินการบน Vector  *(Strang 1.1)*

ในส่วนนี้เราจะเรียนการบวก Vector และคูณด้วย Scalar เพื่อเข้าใจ "การเคลื่อนที่" ใน feature space และ Linear Combination

### 4.1 การบวก Vector (Vector Addition)

**กฎ**: บวก component ต่อ component

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 \\ u_2 \end{pmatrix} + \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \end{pmatrix}$$

**ตัวอย่างตัวเลข**:
$$\begin{pmatrix} 2 \\ 3 \end{pmatrix} + \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$$

**การตีความเรขาคณิต**: กฎ Parallelogram — ต่อหัวลูกศรแรก ต่อท้ายลูกศรสอง

### 4.2 การคูณ Scalar (Scalar Multiplication)

$$c \cdot \mathbf{v} = c \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} c \cdot v_1 \\ c \cdot v_2 \end{pmatrix}$$

**ตัวอย่างตัวเลข**:
$$3 \cdot \begin{pmatrix} 2 \\ -1 \end{pmatrix} = \begin{pmatrix} 6 \\ -3 \end{pmatrix}$$

**การตีความ**: ยืดหรือหดลูกศร (c > 1: ยืด, 0 < c < 1: หด, c < 0: กลับทิศ)

### 4.3 Linear Combination

**นิยาม**: $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k$ เมื่อ $c_i$ เป็น scalar

**ตัวอย่างตัวเลข**:
$$2\begin{pmatrix}1\\0\end{pmatrix} + 3\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}2\\0\end{pmatrix} + \begin{pmatrix}0\\3\end{pmatrix} = \begin{pmatrix}2\\3\end{pmatrix}$$

**ความสำคัญ**: Span ของ vector ทั้งหมดที่สร้างได้จาก linear combination คือ "Column Space" ซึ่งเป็นแนวคิดสำคัญมากในสัปดาห์ที่ 3!

**DS Connection**: ใน Neural Network เลเยอร์แต่ละชั้นคือ linear combination ของ input vector ตามด้วย activation function — นี่คือหัวใจของ Deep Learning

---

## Section 5: Dot Product และ Cosine Similarity  *(Strang 1.2)*

ในส่วนนี้เราจะเรียน **Dot Product** ซึ่งเป็นการดำเนินการสำคัญที่สุดใน Linear Algebra และ **Cosine Similarity** ซึ่งใช้วัดความคล้ายกันของข้อมูล

### 5.1 Dot Product (ผลคูณจุด)

**สูตรพีชคณิต**:
$$\mathbf{v} \cdot \mathbf{w} = \mathbf{v}^T \mathbf{w} = \sum_{i=1}^{n} v_i w_i = v_1 w_1 + v_2 w_2 + \cdots + v_n w_n$$

**ตัวอย่างตัวเลข** (แบบ step-by-step):
$$\mathbf{v} = \begin{pmatrix}1\\2\\3\end{pmatrix}, \quad \mathbf{w} = \begin{pmatrix}4\\5\\6\end{pmatrix}$$
$$\mathbf{v} \cdot \mathbf{w} = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32$$

**สูตรเรขาคณิต**:
$$\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \|\mathbf{w}\| \cos(\theta)$$
โดย $\theta$ คือมุมระหว่าง vector ทั้งสอง

**สมบัติสำคัญ**:
- $\mathbf{v} \cdot \mathbf{w} = \mathbf{w} \cdot \mathbf{v}$ (สับเปลี่ยนได้)
- $\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2$
- ถ้า $\mathbf{v} \perp \mathbf{w}$ แล้ว $\mathbf{v} \cdot \mathbf{w} = 0$

### 5.2 Magnitude (Norm)

**สูตร**:
$$\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**ตัวอย่าง**:
$$\mathbf{v} = \begin{pmatrix}3\\4\end{pmatrix} \Rightarrow \|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9+16} = \sqrt{25} = 5$$

### 5.3 Unit Vector

**นิยาม**: $\hat{\mathbf{v}} = \dfrac{\mathbf{v}}{\|\mathbf{v}\|}$ โดยที่ $\|\hat{\mathbf{v}}\| = 1$

**ตัวอย่าง**:
$$\mathbf{v} = \begin{pmatrix}3\\4\end{pmatrix}, \quad \|\mathbf{v}\| = 5 \quad \Rightarrow \quad \hat{\mathbf{v}} = \begin{pmatrix}3/5\\4/5\end{pmatrix} = \begin{pmatrix}0.6\\0.8\end{pmatrix}$$

### 5.4 Cosine Similarity

**สูตร**:
$$\cos(\theta) = \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|}$$

| ค่า $\cos(\theta)$ | ความหมาย |
|-------------------|----------|
| 1.0 | สองทิศทางเดียวกันทุกประการ (identical direction) |
| 0.0 | ตั้งฉากกัน (orthogonal) |
| -1.0 | ตรงข้ามกันสมบูรณ์ (opposite direction) |

**ตัวอย่างตัวเลข** (step-by-step):
$$\mathbf{v} = \begin{pmatrix}1\\2\\3\end{pmatrix}, \quad \mathbf{w} = \begin{pmatrix}2\\4\\6\end{pmatrix}$$
$$\mathbf{v} \cdot \mathbf{w} = 2 + 8 + 18 = 28$$
$$\|\mathbf{v}\| = \sqrt{1+4+9} = \sqrt{14} \approx 3.742$$
$$\|\mathbf{w}\| = \sqrt{4+16+36} = \sqrt{56} \approx 7.483$$
$$\cos(\theta) = \frac{28}{\sqrt{14} \cdot \sqrt{56}} = \frac{28}{\sqrt{784}} = \frac{28}{28} = 1.0$$
ผลลัพธ์: $\mathbf{w} = 2\mathbf{v}$ จึงขนานกันพอดี → $\cos(\theta) = 1$

**DS Connection**: Cosine Similarity ใช้ใน NLP เพื่อวัดความคล้ายของคำหรือเอกสาร ในระบบค้นหาข้อความ (Search Engine) เมื่อผู้ใช้ค้นหาคำ ระบบจะเปลี่ยนคำค้นเป็น vector และหา document ที่มี cosine similarity สูงสุด

---

## Case Study: Word2Vec — ความหมายของคำในรูปแบบ Vector

**Scenario**: บริษัท Google ต้องการให้คอมพิวเตอร์ "เข้าใจ" ความหมายของคำและความสัมพันธ์ระหว่างคำ เช่น "กษัตริย์" กับ "ราชินี" ควรมีความหมายใกล้กัน แต่ "กษัตริย์" กับ "รถยนต์" ควรห่างกันมาก

**Data**: ข้อความภาษาอังกฤษหลายพันล้านประโยคจาก Wikipedia และ Google News คำแต่ละคำถูกแทนด้วย vector 300 มิติ (300-dimensional vector)

**Method**:
1. ฝึก Neural Network ขนาดเล็กให้ทำนายคำที่อยู่ใกล้เคียง (context words)
2. น้ำหนัก (weights) ของ Neural Network กลายเป็น word vectors
3. คำที่ปรากฏในบริบทเดียวกันบ่อย ๆ จะมี vector คล้ายกัน
4. วัดความคล้ายด้วย **Cosine Similarity**

**Result**:
```
king - man + woman ≈ queen   (คณิตศาสตร์ vector!)
Paris - France + Italy ≈ Rome
```
cosine_similarity(vector("กษัตริย์"), vector("ราชินี")) ≈ 0.85 (ใกล้มาก!)
cosine_similarity(vector("กษัตริย์"), vector("รถยนต์")) ≈ 0.02 (ห่างมาก!)

**Insight**: Linear Algebra บอกเราว่า "ความหมาย" ของภาษาสามารถแทนได้ด้วย vector ใน high-dimensional space และ Dot Product / Cosine Similarity คือเครื่องมือวัด "ระยะทางของความหมาย" — นี่คือพื้นฐานของ ChatGPT, Google Translate, และ Siri ทั้งหมด

```python
# ─── Word2Vec Cosine Similarity Demo ──────────────────────────────
# วัตถุประสงค์: แสดงการคำนวณ cosine similarity ระหว่าง word vectors
# เพื่อแสดงว่า dot product วัด "ความคล้าย" ได้อย่างไรใน NLP context

import numpy as np

# สมมติ word vectors ขนาดย่อ (3 มิติ เพื่อให้เห็นภาพ)
# ในความเป็นจริง Word2Vec ใช้ 300 มิติ
king  = np.array([0.8,  0.3,  0.7])
queen = np.array([0.7,  0.4,  0.8])
car   = np.array([-0.1, 0.9, -0.2])

def cosine_similarity(v, w):
    """คำนวณ cosine similarity ระหว่าง vector v และ w"""
    # dot product หารด้วยผลคูณของ norm
    return np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))

# คำนวณความคล้าย
sim_king_queen = cosine_similarity(king, queen)
sim_king_car   = cosine_similarity(king, car)

print(f"Cosine Similarity(king, queen) = {sim_king_queen:.4f}")  # ควรสูง
print(f"Cosine Similarity(king, car)   = {sim_king_car:.4f}")    # ควรต่ำ
```

---

## สรุป (Summary)

| แนวคิด | สูตร | Python Function |
|--------|------|----------------|
| Scalar | $a \in \mathbb{R}$ | `float`, `int` |
| Vector | $\mathbf{v} \in \mathbb{R}^n$ | `np.array([...])` |
| Vector Addition | $(\mathbf{u}+\mathbf{v})_i = u_i + v_i$ | `u + v` |
| Scalar Multiplication | $(c\mathbf{v})_i = c \cdot v_i$ | `c * v` |
| Dot Product | $\mathbf{v}^T\mathbf{w} = \sum v_i w_i$ | `np.dot(v, w)` |
| Norm (Magnitude) | $\|\mathbf{v}\| = \sqrt{\sum v_i^2}$ | `np.linalg.norm(v)` |
| Unit Vector | $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$ | `v / np.linalg.norm(v)` |
| Cosine Similarity | $\cos\theta = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}$ | `np.dot(v,w)/(norm(v)*norm(w))` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **ก่อนหน้า**: ไม่มี (สัปดาห์แรก) — ความรู้พื้นฐาน: เลขคณิต, พีชคณิต ม.ปลาย
- → **Week 2**: Vector เป็นส่วนประกอบของ Matrix — เราจะเรียน Matrix Operations, Elimination, และ LU Decomposition
- → **Week 3**: Dot Product นำไปสู่ Projection และ Least Squares
- → **Week 4**: Cosine Similarity เชื่อมกับ Eigenvalues และ PCA (วัดทิศทางความแปรปรวนใน data)
