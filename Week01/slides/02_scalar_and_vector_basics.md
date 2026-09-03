# Slide Deck: Scalar vs Vector — นิยามและตัวอย่าง
> Week 01 | CLO1 | 9 slides

---
## Slide 1 — Title
**Scalar และ Vector: หน่วยพื้นฐานของคณิตศาสตร์ DS**  
Week 1 | CLO1 | LLo: สร้าง Vector ได้และตีความในบริบท Data Science

---
## Slide 2 — Scalar คืออะไร?
**Key Message**: Scalar คือตัวเลขตัวเดียว — ขนาดโดยไม่มีทิศทาง

**นิยาม**: Scalar ∈ ℝ คือจำนวนจริงตัวเดียว ไม่มีทิศทาง

| Scalar | ตัวอย่างใน DS |
|--------|--------------|
| 0.92 | accuracy ของ model |
| 3.14 | learning rate × 10 |
| -2.5 | weight ของ neuron |
| 1000 | จำนวน training samples |

- สัญลักษณ์: ใช้ตัวอักษรพิมพ์เล็ก เช่น c, α, λ
- Python: `alpha = 0.01` → scalar
- [FIGURE: number line แสดง scalar ต่าง ๆ]

---
## Slide 3 — Vector คืออะไร?
**Key Message**: Vector คือลิสต์ของ scalar ที่มีทั้งขนาดและทิศทาง

**นิยาม**: Vector **v** ∈ ℝⁿ คือ array ของ n ตัวเลข

$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

- **Column vector** (default ใน math): n×1
- **Row vector**: 1×n
- **สัญลักษณ์**: ตัวหนา **v** หรือ $\vec{v}$

**Python:**
```python
v = np.array([1, 2, 3])   # 1D array = vector ใน NumPy
print(v.shape)             # (3,) — ไม่ใช่ (3,1)
```

---
## Slide 4 — Vector ใน Data Science
**Key Message**: ทุก data point คือ vector ใน feature space

[FIGURE: scatter plot ของ 2D data พร้อมลูกศร vector จาก origin]

**ตัวอย่าง**: บ้าน 1 หลัง → vector ∈ ℝ⁴

$$\mathbf{x}_{\text{house}} = [150, 3, 8, 5.5]$$

| ตำแหน่ง | Feature | ค่า |
|---------|---------|-----|
| x₁ | พื้นที่ (ตร.ม.) | 150 |
| x₂ | ห้องนอน | 3 |
| x₃ | อายุบ้าน (ปี) | 8 |
| x₄ | ระยะจากเมือง (กม.) | 5.5 |

→ Dataset 1000 บ้าน = Matrix ขนาด 1000 × 4

---
## Slide 5 — Vector Operations: Addition
**Key Message**: Vector addition = รวม feature โดย element-wise

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{pmatrix}$$

[FIGURE: Parallelogram rule — u, v, u+v]

**ใช้ใน ML**: Gradient Descent  
$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + (-\alpha \nabla L)$$

```python
u = np.array([2, 1])
v = np.array([1, 3])
print(u + v)  # [3, 4]
```

---
## Slide 6 — Vector Operations: Scalar Multiplication
**Key Message**: คูณ scalar กับ vector = ยืด/หด/กลับทิศทาง vector

$$c \cdot \mathbf{v} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \end{pmatrix}$$

[FIGURE: vector v, 2v, -v, 0.5v ใน 2D plane]

| c | ผล |
|---|----|
| c > 1 | ยืด vector |
| 0 < c < 1 | หด vector |
| c < 0 | กลับทิศ |
| c = 0 | zero vector |

---
## Slide 7 — Magnitude (Norm) ของ Vector
**Key Message**: Magnitude = ความยาวของ vector ใน feature space

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v}^T\mathbf{v}}$$

**Unit Vector**: $\hat{v} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$, $\|\hat{v}\| = 1$

```python
v = np.array([3, 4])
norm = np.linalg.norm(v)  # 5.0
unit = v / norm           # [0.6, 0.8]
```

**ใน DS**: L2 regularization ลงโทษ **||w||²** เพื่อ prevent overfitting  
**ใน NLP**: Unit vectors ทำให้เปรียบเทียบ word embeddings ได้

---
## Slide 8 — Dot Product เบื้องต้น
**Key Message**: Dot product วัด "ความเหมือน" ระหว่าง 2 vectors

$$\mathbf{u} \cdot \mathbf{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

```python
u = np.array([2, 3, 1])
v = np.array([1, -1, 4])
dot = np.dot(u, v)  # 2*1 + 3*(-1) + 1*4 = 3
```

**Cosine Similarity**:
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} \in [-1, 1]$$

- sim = 1 → ทิศทางเดียวกันทุกประการ
- sim = 0 → ตั้งฉากกัน (ไม่เกี่ยวข้องกัน)
- sim = -1 → ทิศทางตรงข้าม

---
## Slide 9 — Summary & Preview Lab
**Key Message**: Scalar และ Vector คือหน่วยพื้นฐานที่ข้อมูลทุกชนิดถูกแทน — เข้าใจแล้วจะอ่าน code และ paper ML ได้ทันที

**สิ่งที่เรียนรู้วันนี้**:
- Scalar = ตัวเลขเดียว, Vector = array ของตัวเลข
- Vector operations: addition, scalar multiply, magnitude, dot product
- ทุก data point ใน ML คือ vector ใน feature space

**Preview Lab 01**:
```python
# สิ่งที่จะทำใน Lab:
house = np.array([150, 3, 8, 5.5])    # สร้าง feature vector
norm  = np.linalg.norm(house)          # คำนวณ magnitude
unit  = house / norm                   # unit vector
sim   = np.dot(u, v) / (||u|| * ||v||) # cosine similarity
```

**สัปดาห์ต่อไป — Week 2**: Matrix คืออะไร และทำไม Linear System ถึงต้องแก้?
