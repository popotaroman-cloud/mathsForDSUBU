# Slide Deck: Supervised vs Unsupervised & Regression vs Classification
> Week 5 | CLO2 | ISLP Ch.2.1.4–2.1.5 | 10 slides

---

## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล | Week 5**  
Taxonomy ของ Statistical Learning: 4 ประเภทที่ต้องรู้จัก  
CLO2: จำแนก ML problems เป็น Supervised/Unsupervised และ Regression/Classification ได้

---

## Slide 2 — The Big Picture
**Key Message**: ML problems ทั้งหมดแบ่งได้ด้วย 2 คำถาม: "มี Y หรือไม่?" และ "Y เป็นตัวเลขหรือ category?"

ก่อนสร้าง ML model ทุกครั้ง นักวิทยาการข้อมูลต้องจำแนกปัญหาให้ถูกต้องก่อน เพราะการเลือก algorithm ผิดประเภททำให้ผลลัพธ์ไม่มีความหมาย การแบ่งประเภทใช้ 2 คำถามหลัก: หนึ่ง มี labeled response variable (Y) ในข้อมูล training หรือไม่? สอง ถ้ามี Y นั้นเป็น quantitative (ตัวเลขต่อเนื่อง) หรือ qualitative (category/class)? จากนี้ได้ 3 ประเภทหลัก: Supervised Regression, Supervised Classification, และ Unsupervised เป้าหมายของ deck นี้คือให้นักศึกษาเห็น examples จริงของแต่ละประเภทและตอบได้อย่างรวดเร็วว่าปัญหาที่พบคือประเภทไหน

```
มี Y หรือไม่?
├─ Yes → Supervised Learning
│         Y เป็นตัวเลขต่อเนื่อง? → Regression
│         Y เป็น category?       → Classification
└─ No  → Unsupervised Learning
          → Clustering, PCA, Anomaly Detection
```

---

## Slide 3 — Supervised Learning
**Key Message**: Supervised = เรียนจากตัวอย่างที่มี "เฉลย" (label) — training data มีทั้ง X และ Y

**นิยาม:**
```
Supervised Learning:
Training data: {(X₁, Y₁), (X₂, Y₂), ..., (Xₙ, Yₙ)}
Goal: เรียนรู้ f̂ ที่สามารถ predict Y สำหรับ X ใหม่ได้
```

**ตัวอย่าง:**
- Email spam detection: X=email content, Y=spam/not (labeled)
- House price: X=features, Y=price (known from sold houses)
- Cancer diagnosis: X=biopsy results, Y=malignant/benign (doctor labeled)

**กระบวนการ:**
```
Training:    {(X₁,Y₁), ..., (Xₙ,Yₙ)} → สร้าง f̂
Prediction:  X_new → f̂(X_new) = Ŷ_new
Evaluation:  เปรียบเทียบ Ŷ กับ Y_true บน test set
```

**[FIGURE: แสดง training data points (X,Y) → model → prediction บน new X]**

---

## Slide 4 — Regression: Quantitative Response
**Key Message**: Regression ใช้เมื่อ Y เป็น continuous number — MSE เป็น loss function ที่พบบ่อยที่สุด

**นิยาม:** Y เป็น quantitative variable (real number)

**ตัวอย่าง Regression Problems:**
| ตัวอย่าง | X (inputs) | Y (output) |
|---------|-----------|------------|
| ราคาบ้าน | พื้นที่, ทำเล, ห้อง, ปีสร้าง | ราคา ($) |
| อุณหภูมิวันพรุ่งนี้ | ปัจจัย 20 อย่าง | อุณหภูมิ (°C) |
| Salary | การศึกษา, ประสบการณ์, ตำแหน่ง | เงินเดือน ($) |
| Stock price | past prices, volume, news sentiment | price ($) |
| Energy consumption | weather, time, building type | kWh |

**Loss function — Mean Squared Error:**
```
MSE = (1/n) Σᵢ (Yᵢ - Ŷᵢ)²
```

**Algorithm ที่ใช้บ่อย:** Linear Regression, Ridge, Lasso, SVR, Neural Networks

---

## Slide 5 — Classification: Qualitative Response
**Key Message**: Classification ใช้เมื่อ Y เป็น class/category — accuracy และ confusion matrix เป็น metrics หลัก

**นิยาม:** Y เป็น qualitative variable (category, class label)

**ตัวอย่าง Classification Problems:**
| ตัวอย่าง | X (inputs) | Y (classes) |
|---------|-----------|------------|
| Email spam | word frequency | spam / not spam (2 classes) |
| Digit recognition | pixel values | 0–9 (10 classes) |
| Cancer diagnosis | biopsy + imaging | benign / malignant (2 classes) |
| Sentiment analysis | review text | positive / negative / neutral |
| Language ID | character frequencies | Thai/English/Chinese/... |

**Metrics:**
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**Algorithm ที่ใช้บ่อย:** Logistic Regression, LDA, KNN, Decision Tree, Neural Network

---

## Slide 6 — Regression vs Classification: A Critical Distinction
**Key Message**: ข้อผิดพลาดที่พบบ่อยมากคือใช้ Regression กับ categorical Y — ต้องจำแนกให้ถูก

**คำถามสำคัญ: Y เป็น quantitative หรือ qualitative?**

| Y type | ตัวอย่าง | Method |
|--------|---------|--------|
| Continuous | 3.5, 7.2, 150.0 | Regression |
| Ordinal | low/medium/high | Ordinal logistic / treat as continuous (careful) |
| Binary | 0/1, Yes/No | Logistic Regression, LDA |
| Multi-class | A/B/C/D/E | Multinomial Logistic, LDA, SVM |

**กรณีที่ต้องระวัง:**
```
❌ ผิด: ใช้ Linear Regression กับ Y = {0=Fail, 1=Pass}
       → อาจ predict Ŷ = -0.3 หรือ 1.7 ซึ่งไม่มีความหมาย

✓ ถูก: ใช้ Logistic Regression → Ŷ = P(Y=1|X) ∈ [0,1]
```

**[FIGURE: Linear Reg กับ binary Y — extrapolation problem; Logistic Reg — sigmoid stays in [0,1]]**

---

## Slide 7 — Unsupervised Learning
**Key Message**: Unsupervised = หา "ความเป็นระเบียบ" ใน data โดยไม่มี Y label — ยากกว่า supervised เพราะไม่มีเฉลย

**นิยาม:**
```
Unsupervised Learning:
Training data: {X₁, X₂, ..., Xₙ}  ← ไม่มี Y!
Goal: หา structure, pattern, grouping ใน X
```

**3 task หลักของ Unsupervised:**
- **Clustering**: จัดกลุ่ม observations ที่คล้ายกัน (K-means, Hierarchical)
- **Dimensionality Reduction**: ลด dimension ในขณะที่รักษา information (PCA! ← Week 4)
- **Anomaly Detection**: หา outliers ที่ผิดปกติ (Isolation Forest, Autoencoder)

**ตัวอย่าง:**
- แบ่งกลุ่มลูกค้าตาม purchasing behavior (ไม่มี predefined groups)
- หา "ยีนที่ทำงานคล้ายกัน" จาก expression data
- ตรวจจับ network intrusion ที่ผิดปกติ

**[FIGURE: scatter ก่อน clustering → หลัง clustering มีสีแยกกลุ่ม, ไม่มี labeled Y]**

---

## Slide 8 — PCA เชื่อมกับ Week 4
**Key Message**: PCA ที่เรียนใน Week 4 คือ Unsupervised Learning — เราลด dimension โดยไม่ใช้ Y เลย

**PCA = Unsupervised Dimensionality Reduction:**
```python
from sklearn.decomposition import PCA

# ไม่ต้องการ y เลย! — Unsupervised
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # ← ไม่มี y parameter

# แต่ visualize ด้วย color = y label (ถ้ามี)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
```

**ทำไม PCA เป็น Unsupervised:**
- ไม่ใช้ Y ในการ fit (หาแค่ directions of max variance)
- เป็นไปได้ที่จะทำ PCA แม้ไม่รู้ label เลย
- label (y) ใช้แค่ visualize ผลเท่านั้น

**เชื่อมกับ Supervised:**
- PCA → Reduce dimension → ส่งต่อเป็น features ให้ classifier
- เรียกว่า "PCA preprocessing" ก่อน supervised model

---

## Slide 9 — Case Study: Spam Filter Classification
**Key Message**: Spam filter เป็น supervised classification ที่ใช้จริงทุกวัน — ตัวอย่างที่เชื่อมโยงทุก concept

**1. Scenario:**  
Gmail, Yahoo, Outlook ต้องกรอง spam ออก โดยอัตโนมัติ — ผู้ใช้กว่า 1 พันล้านคน ต้องการ precision สูง (false positive ทำลาย UX)

**2. Data:**  
- 50,000+ emails (Enron spam dataset)
- X: word frequency (TF-IDF), sender reputation, links count
- Y: spam (1) / not spam (0) — labeled by users

**3. Method:**  
Naive Bayes: P(spam|words) ∝ P(words|spam) × P(spam)  
Logistic Regression: log(P/1-P) = β₀ + β₁ word_freq + ...

**4. Result:**  
Logistic Regression: accuracy ≈ 97%, precision ≈ 99%  
False positive rate < 0.1% (important: ham ≠ spam)

**5. Insight:**  
- Precision สำคัญกว่า recall สำหรับ spam (ผิดพลาดใน ham = user หัวร้อน)
- Feature: "FREE", "CLICK HERE", "WINNER" เป็น strong predictors
- เชื่อมกับ Week 11: Logistic Regression ในเชิงลึก

---

## Slide 10 — Summary: ML Taxonomy
**สิ่งที่เรียนรู้วันนี้**

```
Statistical Learning
├── Supervised (มี Y label)
│   ├── Regression (Y = quantitative)
│   │   Examples: house price, temperature, salary
│   │   Metrics: MSE, RMSE, R²
│   │   Methods: Linear Reg, SVR, Neural Net
│   │
│   └── Classification (Y = qualitative)
│       Examples: spam, cancer, digit
│       Metrics: Accuracy, Precision, Recall, AUC
│       Methods: Logistic, LDA, KNN, SVM, Neural Net
│
└── Unsupervised (ไม่มี Y)
    ├── Clustering: K-means, DBSCAN
    ├── Dimensionality Reduction: PCA (Week 4!)
    └── Anomaly Detection: Isolation Forest
```

**สัปดาห์ถัดไป (Week 6):** Bias-Variance Trade-off — อธิบาย U-shape ของ test error อย่างเป็น mathematical ว่าเกิดจาก Bias² + Variance + Irreducible Error (ISLP Ch.2.2)
