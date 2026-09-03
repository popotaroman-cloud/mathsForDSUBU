# Slide Deck: EDA Visualization
> Week 07 | CLO2 | ISLP 2.3.4, 2.3.9 | 9 slides

---
## Slide 1 — Title
**EDA Visualization: "Never Model Before You Visualize"**  
Week 7 | CLO2 | Tools: pandas, matplotlib, seaborn

---
## Slide 2 — EDA Workflow
**Key Message**: EDA ไม่ได้มีแค่ plot — มี systematic workflow ที่ต้องทำทีละขั้น

```
┌─────────────────────────────────────────────────────┐
│              EDA Workflow                            │
│                                                     │
│  1. Load     → df = pd.read_csv(...)                │
│  2. Inspect  → .info(), .describe(), .isnull()      │
│  3. Univariate  → histogram, boxplot, density      │
│  4. Bivariate   → scatter, heatmap                 │
│  5. Multivariate→ pairplot, grouped plots          │
│  6. Clean    → handle outliers, missing values     │
│  7. Document → insights, hypotheses                │
└─────────────────────────────────────────────────────┘
```

---
## Slide 3 — Histogram และ Distribution
**Key Message**: Histogram บอก shape ของ distribution — อ่านก่อน modeling เสมอ

[FIGURE: 4 histograms ของ TV, Radio, Newspaper, Sales จาก Advertising]

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flatten(), df.columns):
    ax.hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution of {col}')
    ax.axvline(df[col].mean(), color='red', ls='--', label='Mean')
    ax.legend()
plt.tight_layout()
```

**ระวัง**: TV distribution ≈ uniform (งบกระจายสม่ำเสมอ)

---
## Slide 4 — Boxplot: Outlier Detection
**Key Message**: Boxplot แสดง Q1, Q2, Q3, whiskers, และ outlier ในภาพเดียว

[FIGURE: Annotated boxplot แสดง Q1, Median, Q3, IQR, whiskers, outlier points]

```python
fig, ax = plt.subplots(figsize=(10, 5))
df.boxplot(ax=ax, grid=True)
ax.set_title('Boxplot — Advertising Dataset')
```

**อ่าน boxplot**:
- กล่อง = IQR (Q1 ถึง Q3)
- เส้นกลาง = Median
- Whiskers = Q1 − 1.5×IQR ถึง Q3 + 1.5×IQR
- จุดนอก whiskers = **outliers**

---
## Slide 5 — Scatter Plot และ Bivariate Analysis
**Key Message**: Scatter plot เผย relationship, direction, strength, และ non-linearity

```python
# Simple scatter
plt.scatter(df['TV'], df['Sales'], alpha=0.5)

# With regression line
sns.regplot(data=df, x='TV', y='Sales',
            scatter_kws={'alpha': 0.4})

# Residual pattern (ดูว่า linear assumption ถูกต้องไหม)
# รูปแบบ fan/curve → violation ของ assumptions
```

[FIGURE: scatter TV vs Sales + fitted line]

---
## Slide 6 — Pair Plot (Scatter Matrix)
**Key Message**: pairplot เห็น all bivariate relationships ใน 1 ภาพ

```python
sns.pairplot(df, diag_kind='kde',
             plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot — Advertising Dataset', y=1.02)
```

[FIGURE: 4×4 pairplot ของ Advertising dataset]

**อ่าน pairplot**:
- **Diagonal**: distribution ของแต่ละ feature (KDE)
- **Off-diagonal**: scatter ของทุกคู่
- ดู pattern: linear? non-linear? clusters?

---
## Slide 7 — Violin Plot และ Grouped Analysis
**Key Message**: Violin plot = boxplot + density → เห็น multimodal distribution ได้

```python
# Default dataset: balance distribution by default status
default_df = ISLP.load_data('Default')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Boxplot
sns.boxplot(data=default_df, x='default', y='balance', ax=axes[0])
# Violin plot
sns.violinplot(data=default_df, x='default', y='balance', ax=axes[1])
```

[FIGURE: boxplot vs violin plot ของ balance by default status]

---
## Slide 8 — Detecting Issues: Outliers และ Missing Values
**Key Message**: ตรวจสอบ data quality ก่อน modeling — garbage in, garbage out

**Missing Values**:
```python
df.isnull().sum()           # count missing per column
df.isnull().mean() * 100    # % missing
# Heatmap ของ missing:
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
```

**Outliers (Z-score method)**:
```python
z_scores = np.abs(stats.zscore(df))
outlier_mask = (z_scores > 3).any(axis=1)  # Z > 3σ
print(f'Outliers: {outlier_mask.sum()} rows')
df_clean = df[~outlier_mask]
```

---
## Slide 9 — Summary
**สิ่งที่เรียนรู้วันนี้**:
- EDA Workflow: Load → Inspect → Visualize → Clean → Document
- Histogram: shape, skewness
- Boxplot: Q1, Q3, IQR, outliers
- Scatter/Heatmap: bivariate relationships
- Pairplot: all pairs at once

**Quick Reference**:
```python
df.hist()        # all histograms
df.boxplot()     # all boxplots
df.corr()        # correlation matrix
sns.pairplot(df) # all scatter pairs
sns.heatmap(df.corr(), annot=True)  # heatmap
```

**สัปดาห์ต่อไป — ในชุดนี้**: Sampling Distributions และ Standard Error
