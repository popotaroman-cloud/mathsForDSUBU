# Note 1: Descriptive Statistics และ Exploratory Data Analysis (EDA)
> Week 7 | CLO2 | ISLP Reference: Ch.2.3 Lab

---

## บทนำ

สัปดาห์นี้เราจะเรียนรู้ **Exploratory Data Analysis (EDA)** ซึ่งเป็นขั้นตอนแรกและสำคัญที่สุดของทุก Data Science project ก่อนที่จะ fit model ใด ๆ เราต้องเข้าใจข้อมูลก่อน: มีกี่ตัวแปร? มีค่าผิดปกติหรือเปล่า? features มีความสัมพันธ์กันอย่างไร? เป้าหมายคือให้นักศึกษาสามารถคำนวณ descriptive statistics ด้วยมือและ pandas, สร้าง visualizations ที่มีความหมาย และระบุปัญหาในข้อมูลก่อน modeling ทักษะ EDA ที่ดีทำให้ประหยัดเวลาและหลีกเลี่ยงข้อผิดพลาดที่พบบ่อยใน machine learning ในชีวิตจริง Data Scientists ใช้เวลา 60–80% ของโปรเจกต์อยู่ที่ขั้นตอนนี้

---

## Section 1: Descriptive Statistics  *(ISLP 2.3.9)*

ในส่วนนี้เราจะเรียนรู้ **descriptive statistics** ที่สำคัญ ทั้งสูตรคณิตศาสตร์และ Python implementation เพื่อสร้างพื้นฐานการวิเคราะห์ข้อมูลทุกชุด

### 1.1 Measures of Central Tendency (การวัดแนวโน้มสู่ศูนย์กลาง)

**Mean (ค่าเฉลี่ย)**:
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Median (มัธยฐาน)**: ค่ากลางเมื่อเรียงลำดับ  
- $n$ คี่: ค่าตำแหน่งที่ $\frac{n+1}{2}$
- $n$ คู่: ค่าเฉลี่ยของสองตำแหน่งกลาง

**Mode (ฐานนิยม)**: ค่าที่ปรากฏบ่อยที่สุด

**เมื่อไหรใช้อะไร**:
- Mean: ข้อมูล symmetric, ไม่มี outliers
- Median: ข้อมูล skewed (เช่น income, house price)
- Mode: categorical data

**ตัวอย่าง**: ราคาบ้าน (ล้านบาท): [2, 3, 3, 4, 5, 100]  
Mean = 19.5 (ผิดปกติเพราะ 100), Median = 3.5 (ดีกว่า)

### 1.2 Measures of Dispersion (การวัดการกระจาย)

**Variance (ความแปรปรวน)**:
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

(หาร $n-1$ สำหรับ sample variance — Bessel's correction)

**Standard Deviation (ส่วนเบี่ยงเบนมาตรฐาน)**:
$$s = \sqrt{s^2}$$

**หน่วยเดียวกับข้อมูล** — ใช้ interpret ได้ตรงกว่า variance

**Interquartile Range (IQR)**:
$$\text{IQR} = Q_3 - Q_1$$

ข้อดี: robust ต่อ outliers (ไม่ขึ้นกับค่าสุดขีด)

**Z-score** (standard score):
$$z_i = \frac{x_i - \bar{x}}{s}$$

ความหมาย: ค่า $x_i$ อยู่ห่างจาก mean กี่ standard deviations

### 1.3 Shape Statistics

**Skewness** (ความเบ้):
$$\text{Skew} = \frac{\frac{1}{n}\sum(x_i-\bar{x})^3}{s^3}$$

| ค่า Skewness | ความหมาย |
|-------------|---------|
| ≈ 0 | Symmetric |
| > 0 | Right-skewed (tail ยาวขวา) |
| < 0 | Left-skewed (tail ยาวซ้าย) |

**Kurtosis** (ความแหลม): วัดความหนาของ tails เทียบกับ Normal distribution

### 1.4 Covariance และ Correlation

**Covariance** วัดทิศทางความสัมพันธ์:
$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Pearson Correlation** — ปรับ scale ให้อยู่ใน $[-1, 1]$:
$$r_{XY} = \frac{\text{Cov}(X,Y)}{s_X s_Y}$$

| $r$ | ความหมาย |
|-----|---------|
| 1.0 | Perfect positive linear |
| 0.7 | Strong positive |
| 0.3 | Weak positive |
| 0.0 | No linear relationship |
| -0.7 | Strong negative |
| -1.0 | Perfect negative linear |

**Spearman Correlation**: ใช้ rank แทนค่าจริง — robust ต่อ outliers และ monotonic relationships ที่ไม่เป็น linear

```python
# ─── Descriptive Statistics ด้วย NumPy และ Pandas ──────────────────────
# วัตถุประสงค์: สาธิตทุก descriptive statistics ที่สำคัญในทางปฏิบัติ
import numpy as np
import pandas as pd
from scipy import stats

# สร้าง dataset สมมติ: salary ของพนักงาน 20 คน (หน่วย: พัน บาท/เดือน)
np.random.seed(42)
salary = np.array([25, 30, 28, 35, 45, 40, 55, 32, 29, 48,
                   38, 42, 27, 33, 200, 36, 44, 31, 39, 50])

# ─── คำนวณด้วยมือ ──────────────────────────────────────────────────────
# วัตถุประสงค์: เข้าใจสูตรก่อนใช้ library
mean = salary.mean()
median = np.median(salary)
std = salary.std(ddof=1)       # ddof=1 = sample std (Bessel's correction)
variance = salary.var(ddof=1)
q1, q3 = np.percentile(salary, [25, 75])
iqr = q3 - q1

print("=== Manual Calculation ===")
print(f"Mean:    {mean:.2f}")
print(f"Median:  {median:.2f}")
print(f"Std:     {std:.2f}")
print(f"Var:     {variance:.2f}")
print(f"Q1:      {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"Skewness: {stats.skew(salary):.4f}")

# ─── ด้วย Pandas ──────────────────────────────────────────────────────────
# วัตถุประสงค์: df.describe() เป็น quick summary ที่ใช้ได้จริง
df = pd.DataFrame({'salary': salary,
                   'age': np.random.randint(25, 55, 20),
                   'years_exp': np.random.randint(1, 20, 20)})
print("\n=== df.describe() ===")
print(df.describe().round(2))

# ─── Correlation Matrix ──────────────────────────────────────────────────
# วัตถุประสงค์: เข้าใจความสัมพันธ์ระหว่าง features ก่อน regression
print("\n=== Correlation Matrix ===")
print(df.corr(method='pearson').round(3))
print("\n=== Spearman Correlation ===")
print(df.corr(method='spearman').round(3))
```

**DS Connection**: `df.describe()` และ `.corr()` เป็นสิ่งแรกที่ทำใน EDA เสมอ ค่า correlation สูงระหว่าง features (>0.8) เป็น signal ของ multicollinearity ที่ต้องแก้ก่อน fit regression model

---

## Section 2: Exploratory Data Analysis (EDA)  *(ISLP 2.3.4, 2.3.9)*

ในส่วนนี้เราจะเรียนรู้ **EDA workflow** อย่างเป็นระบบ เพื่อให้การสำรวจข้อมูลมีประสิทธิภาพและครบถ้วนก่อน modeling ทุกครั้ง

### 2.1 EDA Workflow

```
EDA Pipeline:
1. Load & Inspect    → shape, dtypes, head/tail, info
2. Missing Values    → isnull().sum(), heatmap
3. Univariate        → histogram, boxplot per feature
4. Bivariate         → scatter, correlation heatmap
5. Multivariate      → pair plot, grouped analysis
6. Outlier Detection → IQR method, Z-score
7. Document          → findings, decisions
```

### 2.2 Visualizations สำคัญ

**Histogram**: distribution ของ feature เดียว  
**Boxplot**: median, IQR, whiskers, outliers  
**Scatter Plot**: ความสัมพันธ์ระหว่าง 2 variables  
**Correlation Heatmap**: overview ของ pairwise correlations

### 2.3 Outlier Detection

**IQR Method**:
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

ค่าที่อยู่นอก fences = outlier

**Z-score Method**: $|z| > 3$ = outlier (สำหรับ approximately normal data)

```python
# ─── Full EDA บน Advertising Dataset (Synthetic) ──────────────────────
# วัตถุประสงค์: แสดง complete EDA workflow ที่ใช้ได้จริง
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 200

# สร้าง synthetic Advertising data
TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
Newspaper = np.random.uniform(0.3, 114.0, n)
Sales = 2.9 + 0.046*TV + 0.189*Radio - 0.001*Newspaper + np.random.normal(0, 1.5, n)
df = pd.DataFrame({'TV': TV, 'Radio': Radio, 'Newspaper': Newspaper, 'Sales': Sales})

# ─── Step 1: Inspect ─────────────────────────────────────────────────
print(f"Shape: {df.shape}")
print(df.head(3))
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ─── Step 2: Descriptive Stats ───────────────────────────────────────
print("\nDescriptive Stats:\n", df.describe().round(2))

# ─── Step 3: Univariate — Histograms ─────────────────────────────────
# วัตถุประสงค์: ดู distribution shape ของแต่ละ feature
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, col in zip(axes, df.columns):
    ax.hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel('Value')
plt.suptitle('Histograms: Advertising Data')
plt.tight_layout()
plt.savefig("eda_histograms.png", dpi=100, bbox_inches='tight')

# ─── Step 4: Correlation Heatmap ─────────────────────────────────────
# วัตถุประสงค์: ระบุ features ที่สัมพันธ์กับ Sales และสัมพันธ์ระหว่างกัน
fig, ax = plt.subplots(figsize=(6, 4))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("eda_heatmap.png", dpi=100, bbox_inches='tight')

# ─── Step 5: Pair Plot ────────────────────────────────────────────────
# วัตถุประสงค์: ดูทุก pairwise scatter plots พร้อมกัน
g = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.4})
g.fig.suptitle('Pair Plot: Advertising', y=1.02)
plt.savefig("eda_pairplot.png", dpi=100, bbox_inches='tight')

# ─── Step 6: Outlier Detection ──────────────────────────────────────
# วัตถุประสงค์: ระบุ outliers ด้วย IQR method
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {n_outliers} outliers (fence: [{lower:.2f}, {upper:.2f}])")
```

**DS Connection**: correlation heatmap ใน Advertising data แสดงว่า TV มี correlation สูงกับ Sales (r ≈ 0.78) แต่ Newspaper ต่ำ (r ≈ 0.23) — นี่คือข้อมูลสำคัญที่จะใช้ออกแบบ model ใน Week 8 และ 9

---

## Case Study: EDA บน Thailand Economic Data

**Scenario**  
นักวิเคราะห์ต้องการเข้าใจ GDP per capita ของประเทศในภูมิภาค ASEAN ก่อนสร้าง forecasting model

**Data** (สมมติ): 10 ประเทศ × ตัวแปร 4 ตัว (GDP, Population, Trade/GDP, Life Expectancy)

**Method**

```python
# ─── EDA บน Economic Data ──────────────────────────────────────────────
# วัตถุประสงค์: แสดง EDA สำหรับ cross-sectional data
import numpy as np
import pandas as pd

# ASEAN data (สมมติ, หน่วย: GDP = USD หมื่นล้าน, Pop = ล้านคน)
asean = pd.DataFrame({
    'country': ['Thailand','Vietnam','Indonesia','Philippines','Malaysia',
                'Singapore','Myanmar','Cambodia','Laos','Brunei'],
    'gdp_bn': [495, 363, 1319, 394, 337, 397, 65, 27, 19, 14],
    'pop_m': [70, 97, 274, 114, 33, 6, 54, 17, 7, 0.4],
    'trade_pct': [109, 210, 43, 67, 132, 320, 41, 93, 87, 98],
    'life_exp': [78, 73, 72, 71, 76, 83, 65, 70, 68, 78]
})

# Compute GDP per capita
asean['gdp_per_cap'] = (asean['gdp_bn'] * 1e9 / (asean['pop_m'] * 1e6)).round(0)

print(asean[['country', 'gdp_per_cap', 'trade_pct', 'life_exp']].to_string(index=False))
print("\nDescriptive Stats (GDP per capita):")
print(asean['gdp_per_cap'].describe())
print(f"\nSkewness: {asean['gdp_per_cap'].skew():.2f} (right-skewed เพราะ Singapore)")
```

**Result**  
Singapore ($gdp\_per\_cap \approx$ $66,000) เป็น outlier ชัดเจน — skewness > 2

**Insight**  
- Mean GDP per capita ≈ $10,000 แต่ Median ≈ $5,700 — ใช้ Median เป็น representative มากกว่า
- ต้องทำ log transformation ก่อน regression เพราะ right-skewed
- Singapore เป็น influential observation ที่ต้องพิจารณาว่าจะ include หรือ exclude ใน model

---

## สรุป (Summary)

| Statistic | สูตร | Python |
|-----------|------|--------|
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | `np.mean(x)` / `df.mean()` |
| Variance | $s^2 = \frac{1}{n-1}\sum(x_i-\bar{x})^2$ | `np.var(x, ddof=1)` |
| Std Dev | $s = \sqrt{s^2}$ | `np.std(x, ddof=1)` |
| IQR | $Q_3 - Q_1$ | `np.percentile(x,[25,75])` |
| Correlation | $r = \text{Cov}(X,Y)/(s_X s_Y)$ | `df.corr()` |
| Summary | ทุกอย่างพร้อมกัน | `df.describe()` |
| Heatmap | pairwise correlations | `seaborn.heatmap(df.corr())` |

---

## เชื่อมกับสัปดาห์อื่น

- ← **Week 5–6**: Statistical Learning framework ต้องการ EDA เป็น first step เสมอ
- → **Week 7 (note 2)**: ต่อจาก descriptive → inferential statistics (CI, t-test)
- → **Week 8**: Advertising dataset ที่ทำ EDA วันนี้จะใช้ใน Simple Linear Regression
- → **Week 9**: Correlation analysis ระบุ features สำหรับ Multiple Regression
