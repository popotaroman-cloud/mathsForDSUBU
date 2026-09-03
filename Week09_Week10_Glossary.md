# Glossary: Week 9–10 — Multiple Linear Regression & Regression Diagnostics

> ครอบคลุม Week 9 (Multiple Linear Regression, ISLP Ch.3.2) และ Week 10 (Qualitative Extensions & Regression Diagnostics, ISLP Ch.3.3)
> รวมคำย่อ + ศัพท์เฉพาะทั้งหมดที่ใช้ในเอกสารทั้งสองสัปดาห์ พร้อมนิยาม สูตรคำนวณ และลิงก์ไปยังไฟล์ต้นทาง

**สัญลักษณ์ในคอลัมน์อ้างอิง**: `note1`/`note2` = lecture note · `slideN` = slide deck · `lab` = lab notebook · `hw` = assignment

---

## สารบัญ

- [Week 9 — คำย่อ (Abbreviations)](#week-9--คำย่อ-abbreviations)
- [Week 9 — ศัพท์เฉพาะ (Terms)](#week-9--ศัพท์เฉพาะ-terms)
- [Week 10 — คำย่อ (Abbreviations)](#week-10--คำย่อ-abbreviations)
- [Week 10 — ศัพท์เฉพาะ (Terms)](#week-10--ศัพท์เฉพาะ-terms)

---

## Week 9 — คำย่อ (Abbreviations)

| คำย่อ | ชื่อเต็ม | นิยาม | สูตร | อ้างอิง |
|---|---|---|---|---|
| **SLR** | Simple Linear Regression | การถดถอยเชิงเส้นที่มี predictor ตัวเดียว ใช้เทียบกับ MLR | $Y = \beta_0 + \beta_1 X + \varepsilon$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide1](Week09/slides/01_mlr_model.md) |
| **MLR** | Multiple Linear Regression | การถดถอยเชิงเส้นที่รับ predictor หลายตัวพร้อมกัน โดยประมาณค่าสัมประสิทธิ์ของแต่ละตัวขณะยึดตัวอื่นให้คงที่ | $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [note2](Week09/lecture/note_02_model_selection_fit.md) · [lab](Week09/lab/lab09_multiple_regression.ipynb) · [hw](Week09/assignment/hw09_mlr_analysis.md) |
| **OLS** | Ordinary Least Squares | วิธีประมาณค่าพารามิเตอร์โดยลด RSS ให้น้อยที่สุด | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [lab](Week09/lab/lab09_multiple_regression.ipynb) |
| **RSS** | Residual Sum of Squares | ผลรวมกำลังสองของ residual วัดความคลาดเคลื่อนที่โมเดลอธิบายไม่ได้ | $\text{RSS} = \sum_i(y_i - \hat{y}_i)^2$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide2](Week09/slides/02_f_statistic.md) |
| **TSS** | Total Sum of Squares | ผลรวมกำลังสองของความแปรปรวนทั้งหมดของ Y รอบค่าเฉลี่ย | $\text{TSS} = \sum_i(y_i - \bar{y})^2$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **RSE** | Residual Standard Error | ค่าประมาณ irreducible error (σ̂) ของโมเดล | $\text{RSE} = \sqrt{\text{RSS}/(n-p-1)}$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [hw](Week09/assignment/hw09_mlr_analysis.md) |
| **R²** | R-squared (Coefficient of Determination) | สัดส่วนความแปรปรวนของ Y ที่โมเดลอธิบายได้ — เพิ่มขึ้นเสมอเมื่อเพิ่ม predictor | $R^2 = 1 - \dfrac{\text{RSS}}{\text{TSS}}$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [note2](Week09/lecture/note_02_model_selection_fit.md) · [lab](Week09/lab/lab09_multiple_regression.ipynb) |
| **Adj. R² / R²_adj** | Adjusted R-squared | R² ที่ปรับลงโทษจำนวน predictor เพื่อเปรียบเทียบโมเดลต่างขนาดได้ยุติธรรม | $R^2_{\text{adj}} = 1-(1-R^2)\dfrac{n-1}{n-p-1}$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide1](Week09/slides/01_mlr_model.md) · [slide3](Week09/slides/03_variable_selection.md) |
| **F / F-statistic** | F-statistic (Overall Significance Test) | ทดสอบว่า predictor ทั้งหมดร่วมกันมีความสัมพันธ์กับ Y หรือไม่ (H₀: β₁=…=βₚ=0) | $F = \dfrac{(\text{TSS}-\text{RSS})/p}{\text{RSS}/(n-p-1)} = \dfrac{MSR}{MSE}$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide2](Week09/slides/02_f_statistic.md) · [hw](Week09/assignment/hw09_mlr_analysis.md) |
| **MSR** | Mean Regression Sum of Squares | ความแปรปรวนเฉลี่ยที่โมเดลอธิบายได้ต่อ 1 parameter | $MSR = (\text{TSS}-\text{RSS})/p$ | [slide2](Week09/slides/02_f_statistic.md) |
| **MSE** | Mean Squared Error (residual) | ความแปรปรวนเฉลี่ยที่โมเดลอธิบายไม่ได้ (residual variance) | $MSE = \text{RSS}/(n-p-1)$ | [slide2](Week09/slides/02_f_statistic.md) |
| **AIC** | Akaike Information Criterion | เกณฑ์เลือกโมเดลโดยลงโทษความซับซ้อน — เลือกโมเดลที่ค่าต่ำสุด | $\text{AIC} = n\ln(\text{RSS}/n) + 2(p+1)$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide3](Week09/slides/03_variable_selection.md) |
| **BIC** | Bayesian Information Criterion | คล้าย AIC แต่ลงโทษความซับซ้อนมากกว่า (โดยเฉพาะเมื่อ n มาก) ทำให้เลือกโมเดลเล็กกว่า | $\text{BIC} = n\ln(\text{RSS}/n) + \ln(n)(p+1)$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide3](Week09/slides/03_variable_selection.md) |
| **Cp (Mallow's Cp)** | Mallow's Cp | เกณฑ์เลือกโมเดล ใช้ minimize เพื่อสมดุลระหว่างความพอดี (fit) กับความซับซ้อน | $C_p = \dfrac{\text{RSS}+2p\hat{\sigma}^2}{\hat{\sigma}^2}$ | [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **CI** | Confidence Interval | ช่วงความเชื่อมั่นสำหรับค่าเฉลี่ยของ response (mean response) ที่ X=x₀ | $\hat{y}_0 \pm t^* \cdot SE_{mean}$, $\;SE_{mean}=\hat{\sigma}\sqrt{\tfrac{1}{n}+\tfrac{(x_0-\bar x)^2}{\sum(x_i-\bar x)^2}}$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide4](Week09/slides/04_confidence_prediction_intervals.md) · [hw](Week09/assignment/hw09_mlr_analysis.md) |
| **PI** | Prediction Interval | ช่วงพยากรณ์สำหรับค่าสังเกตใหม่รายตัว (individual response) — กว้างกว่า CI เสมอ | $\hat{y}_0 \pm t^* \cdot SE_{pred}$, $\;SE_{pred}=\hat{\sigma}\sqrt{1+\tfrac{1}{n}+\tfrac{(x_0-\bar x)^2}{\sum(x_i-\bar x)^2}}$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide4](Week09/slides/04_confidence_prediction_intervals.md) · [hw](Week09/assignment/hw09_mlr_analysis.md) |
| **t / t-statistic** | t-statistic | สถิติทดสอบนัยสำคัญของ predictor แต่ละตัวทีละตัว | $t = \hat{\beta}_j/SE(\hat{\beta}_j)$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide2](Week09/slides/02_f_statistic.md) |
| **p-value** | Probability Value | ความน่าจะเป็นของการได้ผลลัพธ์สุดขั้วเท่านี้หรือมากกว่า ถ้า H₀ เป็นจริง | — | [note1](Week09/lecture/note_01_mlr_estimation.md) · [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **df** | Degrees of Freedom | องศาความเป็นอิสระที่กำหนด distribution ของสถิติทดสอบ | $F \sim F(p,\, n-p-1)$, $\;t \sim t(n-p-1)$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide2](Week09/slides/02_f_statistic.md) |
| **coef** | Coefficient (β̂) | ค่าประมาณของพารามิเตอร์ที่แสดงในตาราง regression output | $\hat{\beta}_j$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide1](Week09/slides/01_mlr_model.md) |
| **SE** | Standard Error | ค่าความคลาดเคลื่อนมาตรฐานของค่าประมาณพารามิเตอร์ | $SE(\hat{\beta}_j)$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [note2](Week09/lecture/note_02_model_selection_fit.md) |

## Week 9 — ศัพท์เฉพาะ (Terms)

| ศัพท์ | นิยาม | สูตร (ถ้ามี) | อ้างอิง |
|---|---|---|---|
| **Confounding** | เมื่อตัวแปรหลายตัวมี correlation กัน SLR ที่ fit ทีละตัวจะให้ coefficient คลาดเคลื่อน เพราะ "ดูดซับ" ผลของตัวแปรที่ไม่ได้รวมในโมเดล | — | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide1](Week09/slides/01_mlr_model.md) |
| **Design Matrix** | เมทริกซ์ขนาด $n\times(p+1)$ ที่คอลัมน์แรกเป็น 1 (สำหรับ intercept) คอลัมน์ที่เหลือคือค่า predictor | $\mathbf{X} = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide1](Week09/slides/01_mlr_model.md) |
| **Normal Equations** | ระบบสมการที่แก้เพื่อหา β̂ ใน OLS (ต่อยอดจาก Week 3) | $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$ | [note1](Week09/lecture/note_01_mlr_estimation.md) · [slide1](Week09/slides/01_mlr_model.md) |
| **Partial Effect** | ผลของ Xⱼ ต่อ Y เมื่อควบคุมตัวแปรอื่นให้คงที่ ("holding others fixed") — ต่างจาก marginal effect ของ SLR | $\partial Y/\partial X_j = \beta_j$ | [note1](Week09/lecture/note_01_mlr_estimation.md) |
| **Marginal Effect** | ผลของ X₁ ใน SLR ซึ่งรวม indirect effect ที่ผ่านตัวแปรอื่นที่ correlated ด้วย | — | [note1](Week09/lecture/note_01_mlr_estimation.md) |
| **Model / Variable Selection** | กระบวนการเลือก predictor ที่เหมาะสมเข้าโมเดล | — | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide3](Week09/slides/03_variable_selection.md) |
| **Best Subset Selection** | ลองทุก combination ของ predictor ($2^p$ โมเดล) แล้วเลือกที่ดีที่สุด — ครบถ้วนแต่หนักมากเมื่อ p ใหญ่ | $2^p$ models | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide3](Week09/slides/03_variable_selection.md) |
| **Forward Stepwise Selection** | เริ่มจาก null model แล้วเพิ่ม predictor ทีละตัวที่ improve criterion มากที่สุด | complexity $O(p^2)$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [lab](Week09/lab/lab09_multiple_regression.ipynb) |
| **Backward Stepwise / Elimination** | เริ่มจาก full model แล้วลบ predictor ที่มี p-value สูงสุดออกทีละตัว | — | [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **Null Model** | โมเดลที่ไม่มี predictor เลย (มีแค่ intercept) — จุดเริ่มต้นของ forward selection | $p=0$ | [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **Full Model** | โมเดลที่มี predictor ครบทุกตัว — จุดเริ่มต้นของ backward elimination | — | [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **Overfitting** | การเลือกโมเดลที่ fit training data ดีเกินไปจนอาจไม่ generalize กับข้อมูลใหม่ | — | [slide3](Week09/slides/03_variable_selection.md) |
| **Parsimony** | หลักการเลือกโมเดลที่เรียบง่ายที่สุดเท่าที่ยังมีประสิทธิภาพเพียงพอ | — | [note2](Week09/lecture/note_02_model_selection_fit.md) |
| **Mean Response** | ค่าเฉลี่ยของ Y ที่ประชากรทั้งหมดซึ่งมีค่า X=x₀ (สิ่งที่ CI ประมาณ) | $\mathbb{E}[Y\mid X=x_0]$ | [note2](Week09/lecture/note_02_model_selection_fit.md) · [slide4](Week09/slides/04_confidence_prediction_intervals.md) |
| **Individual Response** | ค่าสังเกตใหม่รายตัวเดียว (สิ่งที่ PI ประมาณ) | $Y_{\text{new}}\mid X=x_0$ | [slide4](Week09/slides/04_confidence_prediction_intervals.md) |
| **Irreducible Error** | ความไม่แน่นอนที่มีอยู่เสมอแม้โมเดลจะสมบูรณ์แค่ไหน — ทำให้ PI กว้างกว่า CI เสมอ | $\varepsilon$ | [slide4](Week09/slides/04_confidence_prediction_intervals.md) |

---

## Week 10 — คำย่อ (Abbreviations)

| คำย่อ | ชื่อเต็ม | นิยาม | สูตร | อ้างอิง |
|---|---|---|---|---|
| **VIF** | Variance Inflation Factor | วัดว่า variance ของ β̂ⱼ พองขึ้นกี่เท่าจาก multicollinearity (VIF > 5–10 ถือว่าเป็นปัญหา) | $\text{VIF}(\hat{\beta}_j) = \dfrac{1}{1-R^2_{X_j\mid X_{-j}}}$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide4](Week10/slides/04_multicollinearity_vif.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **DW** | Durbin-Watson Statistic | ทดสอบ lag-1 autocorrelation ของ residuals (DW≈2 = ปกติ, DW<1 = positive autocorrelation) | `durbin_watson(model.resid)` | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **BP** | Breusch-Pagan Test | การทดสอบอย่างเป็นทางการสำหรับ Heteroscedasticity (H₀: variance คงที่) | `het_breuschpagan()` | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **HC3** | Heteroscedasticity-Consistent SE (type 3) | ตัวปรับ standard error ให้ทนทานต่อ Heteroscedasticity โดยไม่ต้องแก้โมเดล | `model.get_robustcov_results('HC3')` | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **GLS** | Generalized Least Squares | วิธี regression ที่คำนึงถึงโครงสร้าง correlation ของ error — ใช้แก้ correlated errors | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **WLS** | Weighted Least Squares | วิธีแก้ Heteroscedasticity เมื่อทราบโครงสร้างของ variance | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **PCA** | Principal Component Analysis | ใช้รวมตัวแปรที่ collinear เป็น component เดียว (ต่อยอดจาก Week 4) เพื่อช่วยแก้ multicollinearity | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide4](Week10/slides/04_multicollinearity_vif.md) |
| **PCR** | Principal Component Regression | Regression ที่ fit บน principal components แทน predictors เดิม เพื่อแก้ multicollinearity | — | [slide4](Week10/slides/04_multicollinearity_vif.md) |
| **R² / Adj. R²** | R-squared / Adjusted R-squared | ใช้เทียบโมเดล additive vs interaction/polynomial และเทียบก่อน-หลังแก้ multicollinearity — นิยามและสูตรดูที่ Week 9 | ดู [ตาราง Week 9](#week-9--คำย่อ-abbreviations) | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [note2](Week10/lecture/note_02_regression_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **RSE** | Residual Standard Error | ใช้เทียบ additive vs interaction model (เช่น RSE ลดจาก 1.68 เหลือ 1.00 เมื่อเพิ่ม interaction term) — นิยามดูที่ Week 9 | ดู [ตาราง Week 9](#week-9--คำย่อ-abbreviations) | [note1](Week10/lecture/note_01_qualitative_extensions.md) |
| **AIC** | Akaike Information Criterion | ใช้เลือกโมเดลหลังแก้ปัญหา multicollinearity เทียบก่อน-หลัง — นิยามดูที่ Week 9 | ดู [ตาราง Week 9](#week-9--คำย่อ-abbreviations) | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |

## Week 10 — ศัพท์เฉพาะ (Terms)

| ศัพท์ | นิยาม | สูตร (ถ้ามี) | อ้างอิง |
|---|---|---|---|
| **Qualitative / Categorical Predictor** | ตัวแปรจัดประเภท (เช่น เพศ, ภูมิภาค) ที่ต้องแปลงเป็นตัวเลขก่อนใส่ใน regression | — | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide1](Week10/slides/01_qualitative_predictors.md) |
| **Dummy Variable (Indicator Variable)** | ตัวแปร 0/1 ที่แทน category เพื่อใช้ใน regression | $x_i = \begin{cases}1 & \text{female}\\0 & \text{male (reference)}\end{cases}$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide1](Week10/slides/01_qualitative_predictors.md) · [lab](Week10/lab/lab10_regression_diagnostics.ipynb) |
| **Reference Level (Baseline)** | Category ที่ dummy = 0 ทุกตัว ใช้เป็น baseline สำหรับตีความ coefficient ของ category อื่น | — | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide1](Week10/slides/01_qualitative_predictors.md) |
| **k−1 Dummy Coding** | สำหรับ predictor ที่มี k categories ต้องสร้างแค่ k−1 dummies เพื่อไม่ให้เกิด Dummy Variable Trap | $x_{i,1}=\mathbf{1}[\text{Asian}],\;x_{i,2}=\mathbf{1}[\text{Caucasian}]$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) |
| **Dummy Variable Trap** | การสร้าง dummy ครบทุก category (k ตัว) ทำให้เกิด perfect multicollinearity กับ intercept ทำให้ $(X^TX)$ เป็น singular | $\text{Cat}_1+\text{Cat}_2+\text{Cat}_3=1$ เสมอ | [note1](Week10/lecture/note_01_qualitative_extensions.md) |
| **Interaction Term** | เทอมที่จับว่าผลของ predictor หนึ่งขึ้นอยู่กับค่าของอีก predictor หนึ่ง | $Y=\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_1X_2+\varepsilon$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide2](Week10/slides/02_interaction_polynomial.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Additive Assumption / Additive Model** | สมมติฐานว่าผลของ X₁ ต่อ Y ไม่ขึ้นกับค่าของ X₂ | $\partial Y/\partial X_1=\beta_1$ (คงที่) | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide2](Week10/slides/02_interaction_polynomial.md) |
| **Non-additive (Interaction) Model** | โมเดลที่ผลของ X₁ เปลี่ยนไปตามค่าของ X₂ | $\partial Y/\partial X_1 = \beta_1+\beta_3X_2$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) |
| **Synergistic Effect** | เมื่อ interaction coefficient เป็นบวก แสดงว่าตัวแปรสองตัวเสริมฤทธิ์กัน | $\beta_3>0$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide2](Week10/slides/02_interaction_polynomial.md) |
| **Antagonistic Effect** | เมื่อ interaction coefficient เป็นลบ แสดงว่าตัวแปรสองตัวขัดกัน | $\beta_3<0$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide2](Week10/slides/02_interaction_polynomial.md) |
| **Hierarchical Principle** | ถ้าใส่ interaction term $X_1X_2$ ต้องใส่ main effects $X_1$, $X_2$ ด้วยเสมอ แม้ p-value ของ main effect จะสูง | — | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide2](Week10/slides/02_interaction_polynomial.md) · [lab](Week10/lab/lab10_regression_diagnostics.ipynb) |
| **Polynomial Regression** | ขยาย linear model ให้ fit ความสัมพันธ์ไม่เป็นเส้นตรง โดยยังคง "linear in parameters" | $Y=\beta_0+\beta_1X+\beta_2X^2+\cdots+\beta_dX^d+\varepsilon$ | [note1](Week10/lecture/note_01_qualitative_extensions.md) · [slide2](Week10/slides/02_interaction_polynomial.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Feature Engineering** | การสร้าง feature ใหม่ (เช่น X², interaction) จาก feature เดิม เพื่อปรับปรุงโมเดล | — | [note1](Week10/lecture/note_01_qualitative_extensions.md) |
| **Parallel Regression Lines** | เมื่อใส่ dummy predictor เพียงอย่างเดียว (ไม่มี interaction) จะได้เส้น regression ที่ intercept ต่างกันแต่ slope เท่ากัน | — | [slide1](Week10/slides/01_qualitative_predictors.md) |
| **Non-linearity** | ปัญหาเมื่อความสัมพันธ์จริงระหว่าง X และ Y ไม่เป็นเส้นตรง แต่ใช้ linear model ทำให้เกิด systematic bias | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Correlation of Error Terms** | สมมติฐานว่า error terms เป็นอิสระต่อกัน หากละเมิด (เช่น autocorrelation) จะทำให้ SE ต่ำกว่าจริง | $\text{Cov}(\varepsilon_i,\varepsilon_j)=0$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **Autocorrelation** | ความสัมพันธ์ระหว่าง error ที่เวลาต่างกัน พบบ่อยใน time series data | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **Homoscedasticity** | สมมติฐานว่าความแปรปรวนของ error คงที่ตลอดช่วงของ X | $\text{Var}(\varepsilon_i)=\sigma^2$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) |
| **Heteroscedasticity** | การละเมิด homoscedasticity — variance ของ error ไม่คงที่ (เช่น เพิ่มตาม fitted value) | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [slide4](Week10/slides/04_multicollinearity_vif.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Residual Plot (Residuals vs Fitted)** | กราฟ residual บนแกน Y เทียบกับ fitted value บนแกน X ใช้ตรวจ non-linearity และ heteroscedasticity | $e_i=y_i-\hat{y}_i$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Scale-Location Plot** | กราฟตรวจ homoscedasticity — ควรมีเส้น horizontal สม่ำเสมอ | $\sqrt{\lvert\text{Standardized Residuals}\rvert}$ vs Fitted | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) |
| **Q-Q Plot (Normal Q-Q Plot)** | กราฟตรวจว่า residuals มีการกระจายแบบ Normal หรือไม่ | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) |
| **Outlier** | Observation ที่มี $y_i$ ห่างจาก $\hat{y}_i$ มากผิดปกติ (residual ใหญ่) | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Studentized Residual** | Residual ที่ปรับ scale ด้วย SE โดยคำนึงถึงผลของการลบ observation นั้นออก ใช้หา outlier ($\lvert r_i\rvert>3$ หรือ $>2$) | $r_i = \dfrac{e_i}{\hat{\sigma}_{-i}\sqrt{1-h_i}}$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **High Leverage Point** | Observation ที่มีค่า X ผิดปกติ และมีอิทธิพลต่อ β̂ มากกว่าปกติ | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [lab](Week10/lab/lab10_regression_diagnostics.ipynb) |
| **Leverage Statistic** | วัดว่า observation หนึ่งมีอิทธิพลต่อ regression line มากแค่ไหน ($0\le h_i\le 1$) | $h_i = x_i^T(X^TX)^{-1}x_i$, threshold $h_i>\dfrac{2(p+1)}{n}$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Influential Observation** | จุดที่ทั้ง leverage สูงและ residual ใหญ่ ส่งผลกระทบมากที่สุดต่อ β̂ | High Leverage + Large Residual | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) |
| **Cook's Distance** | วัดผลกระทบรวมของ observation หนึ่งต่อ β̂ ทั้งหมด ($D_i>4/n$ หรือ $>1$ ถือว่า influential) | $D_i = \dfrac{r_i^2}{p+1}\cdot\dfrac{h_i}{1-h_i}$ | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Collinearity / Multicollinearity** | ปัญหาที่ predictors บางคู่มี correlation สูง ทำให้ SE(β̂) พองตัวและตีความยาก | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide4](Week10/slides/04_multicollinearity_vif.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **Correlation Matrix** | ตารางแสดง pairwise correlation ระหว่าง predictors — ขั้นแรกตรวจ multicollinearity แต่จับ multiway collinearity ไม่ได้ | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide4](Week10/slides/04_multicollinearity_vif.md) |
| **Data Entry Error** | สาเหตุหนึ่งของ outlier ที่เกิดจากข้อผิดพลาดในการบันทึกข้อมูล | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **Measurement Error** | สาเหตุหนึ่งของ outlier ที่เกิดจากความผิดพลาดในการวัด | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) |
| **Ridge Regression** | วิธี regularization ที่เพิ่ม penalty term เพื่อลด variance ของ β̂ เมื่อมี collinearity (แลกกับ bias) | — | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide4](Week10/slides/04_multicollinearity_vif.md) |
| **4 Standard Diagnostic Plots** | ชุด 4 กราฟมาตรฐาน (เทียบเท่า `plot(lm)` ใน R) ใช้ตรวจสอบ regression assumptions ครบวงจร | Residuals vs Fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) · [hw](Week10/assignment/hw10_regression_diagnostics.md) |
| **6 Potential Problems (ISLP 3.3.3)** | รายการปัญหา 6 ประเภทที่ ISLP ระบุว่าพบบ่อยใน Linear Regression | Non-linearity, Correlated Errors, Heteroscedasticity, Outliers, High Leverage, Multicollinearity | [note2](Week10/lecture/note_02_regression_diagnostics.md) · [slide3](Week10/slides/03_residual_diagnostics.md) |

---

*อ้างอิงหลัก: ISLP — James, Witten, Hastie, Tibshirani, Taylor, "An Introduction to Statistical Learning with Applications in Python", Ch.3.2–3.3*
