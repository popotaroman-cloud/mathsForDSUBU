# Slide Deck: Descriptive Statistics
> Week 07 | CLO2 | ISLP Ch.2.3 | 10 slides

---
## Slide 1 — Title
**1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล — Week 7**  
Descriptive Statistics: Mean, Variance, Skewness, Correlation  
CLO2: คำนวณ descriptive statistics และสร้าง CI ได้

---
## Slide 2 — Week Overview
**Key Message**: EDA ไม่ใช่ optional — เป็น mandatory ก่อน modeling ทุกครั้ง

สัปดาห์นี้เราจะเรียนรู้ทักษะ **EDA (Exploratory Data Analysis)** และ **Statistical Inference** ซึ่งเป็นสะพานเชื่อมระหว่าง raw data กับ regression model ที่เราจะสร้างใน Week 8 ก่อนที่จะ fit โมเดลใด ๆ Data Scientist ที่ดีจะต้องเข้าใจ distribution ของข้อมูล ตรวจหา outlier และทดสอบ hypothesis ก่อนเสมอ เป้าหมายคือให้นักศึกษาอ่านผลจาก `.describe()` ตีความ correlation และทำ t-test ได้อย่างถูกต้อง ทักษะเหล่านี้ใช้ใน Data Science ทุกวัน ตั้งแต่ A/B testing ใน tech companies ไปจนถึงการวิเคราะห์ผลการทดลองทางการแพทย์

---
## Slide 3 — Measures of Central Tendency
**Key Message**: Mean ได้รับอิทธิพลจาก outlier — Median robust กว่าสำหรับ skewed data

| Measure | สูตร | ใช้เมื่อ |
|---------|------|---------|
| **Mean** $\bar{X}$ | $\frac{1}{n}\sum x_i$ | Symmetric distribution |
| **Median** | ค่ากลางเมื่อเรียงลำดับ | Skewed distribution, outliers |
| **Mode** | ค่าที่พบบ่อยที่สุด | Categorical หรือ discrete data |

```python
df['Sales'].mean()    # 14.02
df['Sales'].median()  # 12.90
df['Sales'].mode()    # most frequent
```

**ตัวอย่าง**: เงินเดือนในบริษัท → median เหมาะกว่า mean เพราะ CEO เงินเดือนสูงมาก

---
## Slide 4 — Measures of Dispersion
**Key Message**: Variance วัดการกระจาย — SD อยู่ใน unit เดียวกับ data → interpret ได้ตรง

$$\text{Var}(X) = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2, \quad \text{SD} = \sqrt{\text{Var}(X)}$$
$$\text{IQR} = Q_3 - Q_1$$

```python
df['TV'].var()   # sample variance (ddof=1)
df['TV'].std()   # sample SD
df['TV'].quantile([0.25, 0.75])
iqr = df['TV'].quantile(0.75) - df['TV'].quantile(0.25)
```

**Outlier rule**: ค่า < Q₁ − 1.5×IQR หรือ > Q₃ + 1.5×IQR  
[FIGURE: boxplot แสดง Q1, Q2, Q3, whiskers, outliers]

---
## Slide 5 — Skewness และ Kurtosis
**Key Message**: Skewness บอกทิศทางหาง; Kurtosis บอกความหนักของหาง

[FIGURE: 3 distributions: symmetric, right-skewed, left-skewed]

| Shape | Skewness | ตัวอย่าง |
|-------|---------|---------|
| Symmetric | ≈ 0 | IQ scores, Height |
| Right-skewed | > 0 | Income, House price |
| Left-skewed | < 0 | Age at death (modern) |

```python
df.skew()      # skewness ของทุก column
df.kurtosis()  # excess kurtosis (normal = 0)
```

**ใน ML**: Right-skewed features ควร log-transform ก่อน regression

---
## Slide 6 — Covariance และ Pearson Correlation
**Key Message**: Correlation = standardized covariance → compare ระหว่าง variable pairs ได้

$$\text{Cov}(X,Y) = \frac{1}{n-1}\sum(x_i-\bar{x})(y_i-\bar{y})$$
$$r = \frac{\text{Cov}(X,Y)}{s_X s_Y} \in [-1, 1]$$

```python
df.corr()          # Pearson correlation matrix
df.cov()           # Covariance matrix
df.corr(method='spearman')  # Spearman (rank-based, robust)
```

**ตีความ**: |r| < 0.3 = weak, 0.3–0.7 = moderate, > 0.7 = strong  
**ระวัง**: correlation ≠ causation!

---
## Slide 7 — Correlation Heatmap
**Key Message**: Heatmap ช่วยเห็น patterns ของ correlation ทุกคู่ตัวแปรในภาพเดียว

[FIGURE: correlation heatmap ของ Advertising dataset — TV สีแดงเข้ม]

```python
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Matrix')
```

**จาก Advertising dataset**:
- TV–Sales: r = 0.78 (strong positive)
- Radio–Sales: r = 0.58 (moderate)
- Newspaper–Sales: r = 0.23 (weak)

---
## Slide 8 — Pandas .describe() และ EDA Workflow
**Key Message**: `.describe()` ใน 1 บรรทัดให้ข้อมูลครบที่สุด — เริ่ม EDA ทุกครั้งจากนี้

```python
df.describe()
# Output: count, mean, std, min, 25%, 50%, 75%, max
```

**EDA Workflow**:
1. `df.info()` → dtypes, missing values
2. `df.describe()` → central tendency, dispersion
3. `df.hist()` → distribution shapes
4. `df.boxplot()` → outliers
5. `sns.heatmap(df.corr())` → relationships
6. `sns.pairplot(df)` → all pairs

---
## Slide 9 — Case Study: Thailand GDP (Descriptive Analysis)
**Key Message**: Descriptive stats บน real data → meaningful business insight

**Scenario**: Economic analyst รายงาน GDP per capita ของ ASEAN

**Data**: 10 ประเทศ ASEAN, GDP per capita (USD) ปี 2022

**Descriptive Results**:
- Mean: $17,301 | Median: $4,172
- SD: $26,450 | Skewness: +2.1 (right-skewed)
- Singapore: outlier ($82,808 >> others)

**Insight**:
- Median เหมาะกว่า mean เพราะ Singapore เป็น outlier
- Thailand GDP = $7,232 อยู่ที่ percentile ที่ 50–60 ของภูมิภาค
- ควรใช้ log(GDP) สำหรับ regression เพราะ right-skewed

---
## Slide 10 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- Mean vs Median: เลือกตาม skewness
- SD, IQR: วัดการกระจาย; IQR robust ต่อ outliers
- Pearson r: correlation [−1, 1]; ≠ causation
- `.describe()`, `.corr()`, `sns.heatmap()`: เครื่องมือ EDA หลัก

**สัปดาห์ต่อไป — ในชุดนี้**: EDA Visualization ด้วย seaborn
