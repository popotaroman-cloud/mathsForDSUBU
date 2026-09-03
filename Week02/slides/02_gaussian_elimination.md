# Slide Deck: Gaussian Elimination และ Row Reduction
> Week 02 | CLO1 | Strang 2.2–2.3 | 9 slides

---
## Slide 1 — Title
**Gaussian Elimination: แก้ระบบสมการ Ax = b**  
Week 2 | CLO1 | Algorithm ที่ NumPy ใช้เบื้องหลัง `linalg.solve()`

---
## Slide 2 — ปัญหา: แก้ระบบสมการเชิงเส้น
**Key Message**: ทุก ML model กำลังแก้ระบบสมการอยู่เบื้องหลัง

**ระบบสมการ 3 ตัวแปร**:
```
2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3
```

ในรูป Matrix: **Ax = b**

$$\begin{pmatrix} 2 & 1 & -1 \\ -3 & -1 & 2 \\ -2 & 1 & 2 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} 8 \\ -11 \\ -3 \end{pmatrix}$$

**เป้าหมาย**: หา x, y, z ที่ทำให้ทุกสมการเป็นจริง

---
## Slide 3 — Augmented Matrix
**Key Message**: รวม A และ b เป็น [A|b] เพื่อทำ row operations พร้อมกัน

$$[\mathbf{A}|\mathbf{b}] = \left(\begin{array}{rrr|r} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{array}\right)$$

```python
A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

# Augmented matrix
aug = np.column_stack([A, b])
print(aug)
```

**Elementary Row Operations** (ไม่เปลี่ยน solution):
1. Swap row i ↔ row j
2. Scale row i: cRᵢ → Rᵢ
3. Add multiple: Rⱼ → Rⱼ − mRᵢ

---
## Slide 4 — Forward Elimination Step-by-Step
**Key Message**: ทำให้ elements ใต้ diagonal เป็น 0 ทีละ pivot

**Step 1**: Eliminate column 1 (use row 1 as pivot)

$$R_2 \leftarrow R_2 + \frac{3}{2}R_1, \quad R_3 \leftarrow R_3 + R_1$$

$$\left(\begin{array}{rrr|r} 2 & 1 & -1 & 8 \\ 0 & 0.5 & 0.5 & 1 \\ 0 & 2 & 1 & 5 \end{array}\right)$$

**Step 2**: Eliminate column 2 (use row 2 as pivot)

$$R_3 \leftarrow R_3 - 4R_2$$

$$\left(\begin{array}{rrr|r} 2 & 1 & -1 & 8 \\ 0 & 0.5 & 0.5 & 1 \\ 0 & 0 & -1 & 1 \end{array}\right) = [\mathbf{U}|\mathbf{c}]$$

---
## Slide 5 — Back Substitution
**Key Message**: จาก Upper Triangular ถอยหาค่า x จาก row ล่างขึ้นบน

$$\begin{pmatrix} 2 & 1 & -1 \\ 0 & 0.5 & 0.5 \\ 0 & 0 & -1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} 8 \\ 1 \\ 1 \end{pmatrix}$$

**Row 3**: $-z = 1 \Rightarrow z = -1$

**Row 2**: $0.5y + 0.5(-1) = 1 \Rightarrow y = 3$

**Row 1**: $2x + 3 - (-1) = 8 \Rightarrow x = 2$

**Solution**: x = 2, y = 3, z = -1

```python
x = np.linalg.solve(A, b)
print(x)  # [2, 3, -1]
```

---
## Slide 6 — Pivot Variables vs Free Variables
**Key Message**: Pivot columns → unique solution; free columns → infinite solutions

**Full rank** (pivots ครบทุก column): Unique solution
```
[1 0 0 | a]
[0 1 0 | b]  → x = (a, b, c)
[0 0 1 | c]
```

**Rank deficient** (missing pivot): Infinite solutions หรือ no solution
```
[1 2 0 | 3]
[0 0 1 | 2]  → x₂ = free parameter (ค่าอะไรก็ได้)
[0 0 0 | 0]
```

**ใน DS**: Rank deficient XᵀX → multicollinearity → cannot invert → Ridge Regression ช่วยได้

---
## Slide 7 — Row Echelon Form (REF) vs RREF
**Key Message**: REF ใช้ Back Substitution; RREF ได้คำตอบตรง

**REF** (Staircase form): pivot = 1 ที่ leading position, zeros below
$$\begin{pmatrix} 1 & * & * \\ 0 & 1 & * \\ 0 & 0 & 1 \end{pmatrix}$$

**RREF** (Reduced REF): zeros ทั้ง above และ below each pivot
$$\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

```python
from sympy import Matrix
A_sym = Matrix([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
print(A_sym.rref())  # Sympy: reduced row echelon form
```

---
## Slide 8 — Numerical Stability: Partial Pivoting
**Key Message**: เลือก pivot ที่มีค่าสัมบูรณ์ใหญ่สุดเพื่อลด round-off error

**ปัญหา**: ถ้า pivot เล็กมาก (เช่น 0.0001) → การหาร multiplier ทำให้ error ขยาย

**Partial Pivoting**: swap rows เพื่อให้ largest element เป็น pivot ก่อน

```python
# NumPy ทำ partial pivoting อัตโนมัติ
# สังเกตใน scipy.linalg.lu(A):
P, L, U = linalg.lu(A)
# P คือ permutation matrix จาก row swaps
```

**ทำไมสำคัญ**: ML datasets มักมี features ต่าง scale → numerical issues ถ้าไม่ pivot

---
## Slide 9 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- Augmented matrix [A|b] รวมระบบสมการเป็นรูป Matrix
- Forward elimination → Upper Triangular U
- Back substitution → solution x
- Rank deficient → no unique solution → multicollinearity

**Algorithm ใน Python**:
```
np.linalg.solve(A, b)  ← ใช้ LAPACK's _gesv (LU + pivot)
```

**สัปดาห์ต่อไป**: LU Decomposition — ทำไม A = LU ช่วยในการแก้หลาย b ได้เร็ว
