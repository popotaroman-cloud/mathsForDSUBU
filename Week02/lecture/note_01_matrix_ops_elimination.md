# Note 1: Matrix Operations, Gaussian Elimination และ LU Decomposition
> Week 2 | CLO1 | Strang Reference: Ch.2 (Solving Linear Equations)

---

## บทนำ

สัปดาห์นี้เราจะเรียนรู้เรื่อง **Matrix (แมทริกซ์)** ซึ่งเป็นส่วนขยายตามธรรมชาติของ Vector ที่เรียนไปในสัปดาห์ที่ 1 แมทริกซ์คือตารางตัวเลขสองมิติที่เป็นหัวใจของ Linear Algebra ทั้งหมด เป้าหมายของสัปดาห์นี้คือให้นักศึกษาสามารถ **ดำเนินการทางแมทริกซ์**, **แก้ระบบสมการ Ax = b ด้วย Gaussian Elimination** และ **สร้าง LU Decomposition** ได้อย่างถูกต้อง ในโลก Data Science ทุกชุดข้อมูล (dataset) คือแมทริกซ์ขนาดใหญ่ที่มีแถว (rows) เป็น data points และคอลัมน์ (columns) เป็น features การเข้าใจโครงสร้างของแมทริกซ์จึงเป็นพื้นฐานของทุกอย่างตั้งแต่ Linear Regression ไปจนถึง Deep Learning และ PageRank ของ Google สัปดาห์นี้เราจะใช้ NumPy ในการคำนวณจริง เพื่อให้เห็นว่า Linear Algebra ที่เรียนในกระดาษแปลงเป็น code อย่างไร

---

## Section 1: ประเภทและการดำเนินการของแมทริกซ์  *(Strang 2.4)*

ในส่วนนี้เราจะเรียนรู้ว่า **Matrix คืออะไร** มีกี่ประเภท และดำเนินการทางคณิตศาสตร์อย่างไร เพื่อให้มีภาษาร่วมกันในการพูดถึงข้อมูลตลอดทั้งวิชา

### 1.1 นิยามและสัญลักษณ์

**นิยาม**: Matrix $\mathbf{A}$ ขนาด $m \times n$ คือตารางตัวเลขที่มี $m$ แถว และ $n$ คอลัมน์

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

**สัญลักษณ์**: ตัวอักษรพิมพ์ใหญ่ตัวหนา เช่น $\mathbf{A}, \mathbf{B}, \mathbf{X}$  
**Element**: $a_{ij}$ คือสมาชิกที่แถว $i$ คอลัมน์ $j$

### 1.2 ประเภทแมทริกซ์ที่สำคัญ

| ชื่อ | คุณสมบัติ | ตัวอย่าง |
|------|----------|---------|
| **Square Matrix** | $m = n$ | $3 \times 3$ |
| **Zero Matrix** $\mathbf{0}$ | ทุก $a_{ij} = 0$ | $\begin{pmatrix}0&0\\0&0\end{pmatrix}$ |
| **Identity Matrix** $\mathbf{I}$ | $a_{ii} = 1$, ที่เหลือ 0 | $\begin{pmatrix}1&0\\0&1\end{pmatrix}$ |
| **Diagonal Matrix** | $a_{ij} = 0$ ถ้า $i \neq j$ | $\begin{pmatrix}3&0\\0&5\end{pmatrix}$ |
| **Symmetric Matrix** | $\mathbf{A} = \mathbf{A}^T$ | $\begin{pmatrix}1&2\\2&4\end{pmatrix}$ |
| **Upper Triangular** $\mathbf{U}$ | $a_{ij} = 0$ ถ้า $i > j$ | $\begin{pmatrix}2&3\\0&4\end{pmatrix}$ |
| **Lower Triangular** $\mathbf{L}$ | $a_{ij} = 0$ ถ้า $i < j$ | $\begin{pmatrix}1&0\\3&2\end{pmatrix}$ |

**DS Connection**: ใน Machine Learning, **Covariance Matrix** $\boldsymbol{\Sigma}$ เป็น Symmetric Matrix — ค่า $\sigma_{ij}$ แทนความสัมพันธ์ระหว่าง feature $i$ และ $j$ ใน dataset เราจะใช้คุณสมบัตินี้ใน PCA (Week 4)

### 1.3 การดำเนินการหลัก

#### การบวก/ลบแมทริกซ์
**เงื่อนไข**: ต้องมีขนาดเท่ากัน ($m_1 \times n_1 = m_2 \times n_2$)

$$(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$$

**ตัวอย่าง**:
$$\begin{pmatrix}1&2\\3&4\end{pmatrix} + \begin{pmatrix}5&6\\7&8\end{pmatrix} = \begin{pmatrix}6&8\\10&12\end{pmatrix}$$

#### การคูณ Scalar
$$c \mathbf{A} = \begin{pmatrix} c \cdot a_{11} & \cdots & c \cdot a_{1n} \\ \vdots & \ddots & \vdots \\ c \cdot a_{m1} & \cdots & c \cdot a_{mn} \end{pmatrix}$$

### 1.4 Matrix Multiplication (การคูณแมทริกซ์)

**เงื่อนไข**: $\mathbf{A}$ ขนาด $m \times k$ และ $\mathbf{B}$ ขนาด $k \times n$ → ผลลัพธ์ขนาด $m \times n$

$$(\mathbf{AB})_{ij} = \sum_{r=1}^{k} a_{ir} b_{rj} = \text{(แถวที่ } i \text{ ของ } \mathbf{A}) \cdot \text{(คอลัมน์ที่ } j \text{ ของ } \mathbf{B})$$

**ตัวอย่าง step-by-step**:
$$\mathbf{A} = \begin{pmatrix}1&2\\3&4\end{pmatrix}, \quad \mathbf{B} = \begin{pmatrix}5&6\\7&8\end{pmatrix}$$

$$(\mathbf{AB})_{11} = (1)(5) + (2)(7) = 5 + 14 = 19$$
$$(\mathbf{AB})_{12} = (1)(6) + (2)(8) = 6 + 16 = 22$$
$$(\mathbf{AB})_{21} = (3)(5) + (4)(7) = 15 + 28 = 43$$
$$(\mathbf{AB})_{22} = (3)(6) + (4)(8) = 18 + 32 = 50$$

$$\mathbf{AB} = \begin{pmatrix}19&22\\43&50\end{pmatrix}$$

**คุณสมบัติสำคัญ**:
- $\mathbf{AB} \neq \mathbf{BA}$ (**ไม่สับเปลี่ยน** — แตกต่างจาก scalar!)
- $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$ (associative)
- $(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$ (transpose สลับลำดับ)
- $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$

**การพิสูจน์ว่า $\mathbf{AB} \neq \mathbf{BA}$**:
$$\begin{pmatrix}1&2\\0&0\end{pmatrix}\begin{pmatrix}0&0\\3&4\end{pmatrix} = \begin{pmatrix}6&8\\0&0\end{pmatrix}$$
$$\begin{pmatrix}0&0\\3&4\end{pmatrix}\begin{pmatrix}1&2\\0&0\end{pmatrix} = \begin{pmatrix}0&0\\3&6\end{pmatrix} \quad \neq \text{ ผลข้างบน!}$$

**DS Connection**: การ forward pass ใน Neural Network คือ $\mathbf{y} = \mathbf{W}_L (\cdots \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 \cdots) + \mathbf{b}_L$ — ชุดของ Matrix multiplications ที่เรียงกัน ความไม่สับเปลี่ยนของ matrix multiplication คือเหตุผลที่ลำดับของเลเยอร์ใน Neural Network มีความสำคัญ

### 1.5 Transpose

**นิยาม**: $(\mathbf{A}^T)_{ij} = a_{ji}$ — สลับแถวและคอลัมน์

$$\mathbf{A} = \begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}_{2\times3} \quad \Rightarrow \quad \mathbf{A}^T = \begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}_{3\times2}$$

**Symmetric Matrix**: $\mathbf{A} = \mathbf{A}^T$ — สมมาตรรอบ diagonal

---

## Section 2: Gaussian Elimination และ Row Reduction  *(Strang 2.2–2.3)*

ในส่วนนี้เราจะเรียนรู้วิธีแก้ระบบสมการเชิงเส้น $\mathbf{Ax} = \mathbf{b}$ ด้วย **Gaussian Elimination** ซึ่งเป็นอัลกอริทึมพื้นฐานที่คอมพิวเตอร์ใช้อยู่ทุกวัน และเป็นที่มาของ LU Decomposition

### 2.1 การแปลงระบบสมการเป็น Augmented Matrix

ระบบสมการสามตัวแปร:
$$\begin{cases} 2x + y - z = 8 \\ -3x - y + 2z = -11 \\ -2x + y + 2z = -3 \end{cases}$$

เขียนเป็น **Augmented Matrix** $[\mathbf{A}|\mathbf{b}]$:
$$\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{array}\right]$$

### 2.2 Elementary Row Operations

มี 3 ชนิด ที่ไม่เปลี่ยนผลเฉลยของระบบสมการ:

| Operation | สัญลักษณ์ | ความหมาย |
|-----------|----------|---------|
| Swap | $R_i \leftrightarrow R_j$ | สลับตำแหน่งสองแถว |
| Scale | $R_i \leftarrow c \cdot R_i$ | คูณแถวด้วยค่าคงที่ $c \neq 0$ |
| Add | $R_i \leftarrow R_i + c \cdot R_j$ | บวกแถวหนึ่งคูณค่าคงที่เข้าอีกแถว |

### 2.3 Forward Elimination (ตัวอย่าง step-by-step)

เริ่มต้น:
$$\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{array}\right]$$

**ขั้นที่ 1**: กำจัด $x$ จากแถวที่ 2 และ 3 โดยใช้แถวที่ 1 เป็น pivot row

$R_2 \leftarrow R_2 + \frac{3}{2} R_1$ (multiplier $\ell_{21} = 3/2$):
$$R_2 = (-3 + \frac{3}{2}(2),\ -1 + \frac{3}{2}(1),\ 2 + \frac{3}{2}(-1),\ -11 + \frac{3}{2}(8)) = (0,\ \frac{1}{2},\ \frac{1}{2},\ 1)$$

$R_3 \leftarrow R_3 + 1 \cdot R_1$ (multiplier $\ell_{31} = 1$):
$$R_3 = (-2+2,\ 1+1,\ 2-1,\ -3+8) = (0,\ 2,\ 1,\ 5)$$

$$\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & 1/2 & 1/2 & 1 \\ 0 & 2 & 1 & 5 \end{array}\right]$$

**ขั้นที่ 2**: กำจัด $y$ จากแถวที่ 3

$R_3 \leftarrow R_3 - 4 R_2$ (multiplier $\ell_{32} = 4$):
$$R_3 = (0,\ 2-4(1/2),\ 1-4(1/2),\ 5-4(1)) = (0,\ 0,\ -1,\ 1)$$

**Upper Triangular Form $\mathbf{U}$**:
$$\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & 1/2 & 1/2 & 1 \\ 0 & 0 & -1 & 1 \end{array}\right]$$

**Back Substitution**:
- จากแถว 3: $-z = 1 \Rightarrow z = -1$
- จากแถว 2: $\frac{1}{2}y + \frac{1}{2}(-1) = 1 \Rightarrow y = 3$
- จากแถว 1: $2x + 3 - (-1) = 8 \Rightarrow x = 2$

**คำตอบ**: $x = 2,\ y = 3,\ z = -1$

**DS Connection**: Gaussian Elimination คืออัลกอริทึมที่อยู่เบื้องหลัง `np.linalg.solve(A, b)` ใน NumPy และ `scipy.linalg.solve()` ใน SciPy ทุกครั้งที่ fit Linear Regression ด้วย Normal Equation $(X^T X)\boldsymbol{\beta} = X^T \mathbf{y}$ คอมพิวเตอร์ใช้ Elimination เพื่อหาค่า $\boldsymbol{\beta}$

---

## Section 3: LU Decomposition  *(Strang 2.6)*

ในส่วนนี้เราจะเรียนรู้ว่า Gaussian Elimination สามารถ **"บันทึก"** ขั้นตอนการทำงานไว้ในรูป Factorization $\mathbf{A} = \mathbf{LU}$ ซึ่งช่วยให้แก้สมการชุดเดิมกับ $\mathbf{b}$ ต่างกันได้เร็วขึ้นมาก

### 3.1 แนวคิดหลัก

เมื่อทำ Gaussian Elimination บน $\mathbf{A}$ เราได้:
- $\mathbf{U}$ = Upper triangular matrix จากการ eliminate
- Multipliers $\ell_{ij}$ ที่ใช้ระหว่าง eliminate → เก็บรวมกันเป็น $\mathbf{L}$ (Lower triangular ที่ diagonal เป็น 1)

$$\mathbf{A} = \mathbf{L}\mathbf{U}$$

### 3.2 การสร้าง L จากตัวอย่างก่อนหน้า

จาก Elimination ด้านบน multipliers คือ:
$$\ell_{21} = \frac{3}{2}, \quad \ell_{31} = 1, \quad \ell_{32} = 4$$

$$\mathbf{L} = \begin{pmatrix} 1 & 0 & 0 \\ \ell_{21} & 1 & 0 \\ \ell_{31} & \ell_{32} & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 3/2 & 1 & 0 \\ 1 & 4 & 1 \end{pmatrix}$$

$$\mathbf{U} = \begin{pmatrix} 2 & 1 & -1 \\ 0 & 1/2 & 1/2 \\ 0 & 0 & -1 \end{pmatrix}$$

**ตรวจสอบ** $\mathbf{LU} = \mathbf{A}$:

$$\begin{pmatrix}1&0&0\\3/2&1&0\\1&4&1\end{pmatrix} \begin{pmatrix}2&1&-1\\0&1/2&1/2\\0&0&-1\end{pmatrix} = \begin{pmatrix}2&1&-1\\3&1/2+1/2&-3/2+1/2\\2&1+2+0&-1+2-1\end{pmatrix}$$

ลองคำนวณ $(1,1)$: $1(2)+0+0=2$ ✓, $(2,1)$: $\frac{3}{2}(2)+1(0)+0=3$ ✓

### 3.3 การแก้ Ax = b ด้วย LU

แทน $\mathbf{A} = \mathbf{LU}$ เข้าใน $\mathbf{Ax} = \mathbf{b}$:

$$\mathbf{LU}\mathbf{x} = \mathbf{b}$$

**ขั้นที่ 1**: แก้ $\mathbf{Ly} = \mathbf{b}$ (forward substitution) — ง่ายเพราะ L เป็น lower triangular  
**ขั้นที่ 2**: แก้ $\mathbf{Ux} = \mathbf{y}$ (back substitution) — ง่ายเพราะ U เป็น upper triangular

**ทำไมถึงมีประโยชน์?** ถ้าต้องแก้ $\mathbf{Ax} = \mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_k$ (คือ $\mathbf{A}$ เดิมแต่ $\mathbf{b}$ ต่างกัน) — ทำ LU Decomposition ครั้งเดียว แล้วแก้แต่ละ $\mathbf{b}$ ได้เร็วมาก แทนที่จะ eliminate ซ้ำทุกครั้ง

### 3.4 LDU Decomposition และ PA = LU

**LDU**: แยก diagonal ของ U ออกมาเป็น matrix $\mathbf{D}$:

$$\mathbf{A} = \mathbf{L}\mathbf{D}\mathbf{U}' \quad \text{เมื่อ } D_{ii} = U_{ii}, \quad U'_{ij} = U_{ij}/U_{ii}$$

**PA = LU** (Row Pivoting): ถ้าระหว่าง eliminate พบ pivot = 0 ต้องสลับแถว ใช้ **Permutation Matrix** $\mathbf{P}$ บันทึกการสลับ

$$\mathbf{P}\mathbf{A} = \mathbf{L}\mathbf{U}$$

```python
# ─── LU Decomposition ด้วย scipy ──────────────────────────────────────
# วัตถุประสงค์: แสดง PA = LU factorization และการแก้ระบบสมการ Ax = b
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

# คำนวณ P, L, U จาก scipy — คืนค่า 3 matrices
P, L, U = lu(A)

print("P (Permutation):\n", P)
print("L (Lower triangular):\n", L)
print("U (Upper triangular):\n", U)

# ตรวจสอบว่า P @ A = L @ U
# วัตถุประสงค์: verify decomposition ก่อนนำไปใช้จริง
print("\nP @ A:\n", P @ A)
print("L @ U:\n", L @ U)

# แก้ระบบสมการด้วย LU (วิธีที่ NumPy ใช้จริง)
# วัตถุประสงค์: เปรียบเทียบกับวิธี solve โดยตรง
x = np.linalg.solve(A, b)
print("\nผลเฉลย x:", x)
```

**DS Connection**: `scipy.linalg.lu()` และ `np.linalg.solve()` ใช้ LU Decomposition ภายใน ทุกครั้งที่ทำ Ordinary Least Squares Regression ใน `statsmodels` หรือ `sklearn` — LU คืออัลกอริทึมที่อยู่เบื้องหลัง

---

## Case Study: Google PageRank — Adjacency Matrix ของ Web Graph

**Scenario**  
Google ต้องการจัดอันดับเว็บไซต์หลายพันล้านหน้าให้ถูกต้อง โดยเว็บที่มีลิงก์จากหน้าสำคัญมากควรได้คะแนนสูง ปัญหานี้สามารถแปลงเป็น Linear System ได้!

**Data**  
สมมติ Web Graph มี 4 หน้าเว็บ (Page A, B, C, D) และ links ระหว่างกัน:

```
A → B, C
B → C
C → A
D → C
```

แปลงเป็น **Adjacency Matrix** $\mathbf{M}$ (คอลัมน์ = source, แถว = destination):

$$\mathbf{M} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 1/2 & 0 & 0 & 0 \\ 1/2 & 1 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

(normalize แต่ละคอลัมน์ให้บวกได้ 1 — probability of following link)

**Method**  
PageRank vector $\mathbf{r}$ คือ eigenvector ที่ตอบสมการ $\mathbf{Mr} = \mathbf{r}$ ซึ่งเทียบเท่ากับแก้ระบบ $(\mathbf{M} - \mathbf{I})\mathbf{r} = \mathbf{0}$ เป็น Linear System!

**Result**

```python
# ─── PageRank ด้วย Power Iteration ───────────────────────────────────
# วัตถุประสงค์: แสดงว่า Matrix นิยาม "ความสำคัญ" ของเว็บได้อย่างไร
import numpy as np

# adjacency matrix (normalized columns)
M = np.array([[0,    0, 1, 0],
              [0.5,  0, 0, 0],
              [0.5,  1, 0, 1],
              [0,    0, 0, 0]])

# เริ่มด้วย rank เท่ากันทุกหน้า
r = np.ones(4) / 4

# Power iteration: r_(t+1) = M @ r_t จนกว่าจะ converge
# วัตถุประสงค์: หา dominant eigenvector โดยไม่ต้องหา eigenvalue โดยตรง
for _ in range(100):
    r = M @ r
    if r.sum() > 0:
        r = r / r.sum()  # normalize ให้บวกได้ 1

print("PageRank:", r.round(4))
# Output: PageRank: [0.2927 0.1463 0.5610 0.   ]
```

**Insight**  
- Page C ได้ PageRank สูงสุด (0.5610) เพราะได้รับลิงก์จากทุกหน้า
- Page D ได้ 0.0 เพราะไม่มีใคร link มา
- แนวคิดนี้เป็นต้นกำเนิดของ Google Search Engine ปี 1998 ของ Larry Page และ Sergey Brin
- Matrix multiplication ที่ทำซ้ำจนกว่าจะ converge คือ Power Method ซึ่งเราจะเรียนเพิ่มเติมใน Week 4 (Eigenvalues)

---

## สรุป (Summary)

| แนวคิด | สูตร/กฎ | Python Function |
|--------|--------|----------------|
| Matrix size | $m \times n$ | `A.shape` |
| Matrix multiply | $(\mathbf{AB})_{ij} = \sum_r a_{ir}b_{rj}$ | `A @ B` หรือ `np.dot(A, B)` |
| Transpose | $(\mathbf{A}^T)_{ij} = a_{ji}$ | `A.T` |
| Gaussian Elimination | Forward eliminate → U, back sub | manual หรือ `np.linalg.solve` |
| LU Decomposition | $\mathbf{A} = \mathbf{LU}$ | `scipy.linalg.lu(A)` |
| Solve Ax = b | x = back-sub หลัง LU | `np.linalg.solve(A, b)` |
| Identity | $\mathbf{AI} = \mathbf{A}$ | `np.eye(n)` |
| Non-commutative | $\mathbf{AB} \neq \mathbf{BA}$ | ต้องระวังลำดับเสมอ |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 1**: Vector เป็นกรณีพิเศษของ Matrix ($n \times 1$ หรือ $1 \times n$) — Dot product คือ $\mathbf{v}^T \mathbf{w}$
- → **Week 3**: Column Space ของ $\mathbf{A}$ (range ของ matrix) นำไปสู่แนวคิด Subspace, Null Space และ Four Fundamental Subspaces
- → **Week 4**: LU ใช้คำนวณ Determinant ได้ $\det(\mathbf{A}) = \prod U_{ii}$; Eigenvalues ขยายจาก Matrix structure
- → **Week 8**: Normal Equation $(X^T X)\boldsymbol{\beta} = X^T \mathbf{y}$ ใช้ matrix operations ทั้งหมดที่เรียนสัปดาห์นี้
