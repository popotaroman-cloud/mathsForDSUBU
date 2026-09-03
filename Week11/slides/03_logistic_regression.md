# Slide Deck: Logistic Regression — Sigmoid, Log-odds, MLE
> Week 11 | CLO3 | ISLP Ch.4.3.1–4.3.3 | 10 slides

---
## Slide 1 — Title
**Logistic Regression: Sigmoid Function, Log-odds, และ MLE**  
Week 11 | CLO3 | ISLP 4.3.1–4.3.3  
LLo: สร้าง Logistic Regression, ตีความ coefficient และ predict probability ได้

---
## Slide 2 — Sigmoid Function
**Key Message**: sigmoid ทำให้ linear combination ของ X map ไปใน [0,1] ได้อย่าง smooth

**Logistic (Sigmoid) Function**:
$$p(X) = \Pr(Y=1|X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$

[FIGURE: sigmoid curve S-shape:
- ซ้ายสุด → 0 (asymptote)
- กลาง → 0.5 (เมื่อ β₀ + β₁X = 0)
- ขวาสุด → 1 (asymptote)]

**Properties**:
- p(X) ∈ (0,1) เสมอ (ไม่ถึง 0 และ 1)
- Monotone increasing ถ้า β₁ > 0
- Symmetric around inflection point

---
## Slide 3 — Log-odds (Logit)
**Key Message**: log-odds เป็นเส้นตรงใน X — นี่คือทำให้ Logistic Regression ยัง "linear" ใน sense หนึ่ง

**Odds**:
$$\text{Odds} = \frac{p(X)}{1-p(X)} \in [0,\infty)$$

- Odds = 1: p = 0.5 (equally likely)
- Odds = 4: p = 0.8 (4 ต่อ 1)
- Odds = 0.25: p = 0.2 (1 ต่อ 4)

**Log-odds (Logit)**:
$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X$$

**สำคัญ**: log-odds เป็นฟังก์ชันเส้นตรงของ X → ทำให้ inference คล้าย linear regression

---
## Slide 4 — ตีความ β₁ ใน Logistic Regression
**Key Message**: β₁ = เพิ่ม 1 หน่วย X → log-odds เพิ่ม β₁ → odds คูณ e^β₁

**ตีความ β₁**:
- เพิ่ม X ขึ้น 1 หน่วย → log-odds เพิ่ม β₁
- → Odds คูณด้วย **e^β₁** (Odds Ratio)

**ตัวอย่าง (Default ~ Balance)**:
- β̂₁ = 0.0055
- ทุก $1 เพิ่มใน balance → log-odds ของ default เพิ่ม 0.0055
- → odds คูณ e^0.0055 = 1.0055 (เพิ่ม 0.55%)
- ทุก $100 เพิ่ม → odds คูณ e^0.55 = 1.73 (เพิ่ม 73%)

**ข้อสำคัญ**: β₁ ไม่ใช่ "ผล X ต่อ p โดยตรง" เพราะ slope ของ sigmoid ไม่คงที่ตลอด X  
ต้องระบุ X ที่เฉพาะเจาะจงถ้าต้องการ marginal effect

---
## Slide 5 — Maximum Likelihood Estimation
**Key Message**: MLE หา β ที่ทำให้ likelihood ของ data ที่เห็นสูงที่สุด

**ทำไมไม่ใช้ OLS?**
- RSS = Σ(yᵢ − p(xᵢ))² ไม่มี closed form solution
- Non-convex ถ้าใช้กับ 0/1 response

**MLE — Log-likelihood**:
$$\ell(\beta_0, \beta_1) = \sum_{i=1}^{n} \left[y_i \log p(x_i) + (1-y_i)\log(1-p(x_i))\right]$$

**Intuition**: maximize ℓ → ทำให้ p̂(xᵢ) → 1 เมื่อ yᵢ = 1 และ p̂(xᵢ) → 0 เมื่อ yᵢ = 0

**Optimization**: Newton-Raphson, gradient ascent — ทำ numerically  
Python (`statsmodels`, `sklearn`) handle ให้อัตโนมัติ

---
## Slide 6 — z-statistic (แทน t-statistic)
**Key Message**: z-statistic ใน Logistic Regression = t-statistic ใน Linear Regression — logic เดียวกัน

**z-statistic**:
$$z = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} \sim N(0,1) \quad \text{(ภายใต้ } H_0: \beta_j = 0\text{)}$$

- ใช้ Normal distribution (แทน t) เพราะ MLE → asymptotically normal
- Large sample: z ≈ t → ผลคล้ายกัน

**Logistic Summary (Default ~ Balance)**:
```
             coef     std err    z       P>|z|  [0.025  0.975]
Intercept  -10.651      0.361  -29.49   0.000   -11.358  -9.944
balance      0.005      0.000   24.95   0.000     0.005   0.006
```

**p-value < 0.001**: balance มีผลต่อ P(default) อย่างมีนัยสำคัญ

---
## Slide 7 — Making Predictions: Decision Boundary
**Key Message**: predict class = compare p̂ กับ threshold (default = 0.5)

**ขั้นตอน**:
1. Compute p̂(X) = sigmoid(β̂₀ + β̂₁X)
2. เปรียบกับ threshold τ = 0.5
3. ถ้า p̂ ≥ 0.5 → predict Y = 1 (Default = Yes)

**Decision Boundary** (เมื่อ τ = 0.5):
$$\beta_0 + \beta_1 X = 0 \Rightarrow X^* = -\frac{\beta_0}{\beta_1}$$

**ตัวอย่าง**: β̂₀ = -10.651, β̂₁ = 0.0055 → X* = 10.651/0.0055 ≈ **1936 ดอลลาร์**  
ลูกค้าที่มี balance > $1,936 → predict Default = Yes

**Custom threshold**: ถ้า FP cost ≠ FN cost → ปรับ τ (Week 13 ROC curve)

---
## Slide 8 — Python: Logistic Regression
**Key Message**: statsmodels ให้ full inference; sklearn ให้ predict_proba — ใช้ตามจุดประสงค์

```python
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

# ─── statsmodels: full inference (z, p, CI) ──────────────
# วัตถุประสงค์: ได้ z-stat, p-value, CI เหมือน regression summary
sm_logit = smf.logit('default_num ~ balance', data=default).fit()
print(sm_logit.summary())

# ─── sklearn: predict probability ────────────────────────
# วัตถุประสงค์: ใช้ใน pipeline สำหรับ prediction
sk_logit = LogisticRegression(max_iter=1000)
sk_logit.fit(default[['balance']], default['default_num'])

# predict class
y_pred  = sk_logit.predict(default[['balance']])
# predict probability
y_proba = sk_logit.predict_proba(default[['balance']])[:, 1]

print(f'β̂₀ = {sk_logit.intercept_[0]:.4f}')
print(f'β̂₁ = {sk_logit.coef_[0][0]:.6f}')
```

---
## Slide 9 — Case Study: Default Dataset
**Key Message**: Logistic Regression บอกว่าลูกค้าที่มี balance สูงมีโอกาส default สูงกว่าอย่างมีนัยสำคัญ

```python
# ─── Sigmoid curve บน Default data ────────────────────────
bal_range = np.linspace(0, 3000, 200)
prob_range = sm_logit.predict(pd.DataFrame({'balance': bal_range}))

plt.figure(figsize=(8,4))
plt.scatter(default['balance'], default['default_num'],
            alpha=0.05, c=['red' if y else 'blue' for y in default['default_num']], s=5)
plt.plot(bal_range, prob_range, 'k-', lw=2, label='P(default=Yes|balance)')
plt.axhline(0.5, color='orange', lw=1.5, linestyle='--', label='τ=0.5')
plt.axvline(1936, color='green', lw=1.5, linestyle='--', label='Decision boundary')
plt.xlabel('Balance ($)'); plt.ylabel('P(Default=Yes)')
plt.title('Logistic Regression: Default vs Balance')
plt.legend(); plt.show()
```

**ผลลัพธ์**: P(default | balance=2000) = 0.586 → predict Default=Yes

---
## Slide 10 — Summary
**Key Message**: Logistic Regression = linear model กับ sigmoid transformation — ตีความผ่าน log-odds

**สิ่งที่เรียนรู้วันนี้**:
- **Sigmoid**: p(X) = 1/(1+e⁻ᶻ) — ∈ (0,1) เสมอ
- **Log-odds**: log(p/(1-p)) = β₀ + β₁X — linear ใน X
- **β₁ interpretation**: X เพิ่ม 1 → log-odds เพิ่ม β₁ → odds × eβ₁
- **MLE**: optimize log-likelihood แทน RSS
- **z-statistic**: β̂ⱼ/SE ∼ N(0,1) — logic เดียวกับ t-statistic

| Concept | สูตร | Python |
|---------|------|--------|
| Sigmoid | 1/(1+e⁻ᶻ) | `scipy.special.expit(z)` |
| Log-likelihood | Σ[y log p + (1-y)log(1-p)] | `smf.logit().fit()` |
| Predict prob | — | `model.predict_proba(X)[:,1]` |
| z-statistic | β̂/SE | `model.tvalues` (statsmodels) |

**สัปดาห์ต่อไป — Slide 4**: Multiple Logistic Regression + Multinomial
