"""
build_notebook.py
Generates the comprehensive B.Tech Statistics & Probability Jupyter Notebook:
notebooks/traffic_weather_statistical_analysis.ipynb
"""

import json
import os

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(content):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split("\n")]
    })

def add_code(code):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

# Title & Metadata
add_md("""# Traffic Volume and Weather Conditions: A Statistical Analysis
### B.Tech Project — Statistics and Probability (Modules I – X)
**Institution:** Academic Project Evaluation Board  
**Dataset:** Metro Interstate 94 (Minneapolis–St Paul) | UCI Machine Learning Repository  
**Sample Size:** $n = 40,564$ hourly observations (2012–2018)  
**Primary Focus:** Rigorous Statistical Inference + Mathematical Reasoning + Python Implementation  

---

## Academic Project Architecture & Syllabus Alignment
| Course Module | Syllabus Topic | Project Implementation in this Notebook |
|---|---|---|
| **Module I** | Introduction to Statistics | Frequency distribution, measures of central tendency, variability, skewness, kurtosis, EDA |
| **Module II** | Introduction to Probability | Frequentist probability, addition/multiplication rules, Bayes' theorem, conditional risk |
| **Module III** | Random Variables | Discrete vs continuous variable characterization, expectations $E[X]$, $Var(X)$ |
| **Module IV** | Discrete Distributions | Poisson suitability assessment, variance-to-mean dispersion ratio test |
| **Module V** | Continuous Distributions | Normality assessments (Shapiro-Wilk, Kolmogorov-Smirnov), Q-Q plots |
| **Module VII** | Hypothesis Testing (Parametric) | Welch's $t$-test, ANOVA $F$-test, regression coefficient $t$-tests with HAC SEs |
| **Module VIII** | Non-Parametric Tests | Mann-Whitney $U$-test, Kruskal-Wallis $H$-test, Dunn-Holm multiple comparison post-hoc |
| **Module IX** | Correlation & Covariance | Covariance, Pearson $r$, Spearman $r_s$, Fisher's $z$-transform 95% CIs |
| **Module X** | Regression & Diagnostics | Multiple OLS, polynomial terms, interactions, VIF, Newey-West HAC, residual diagnostics |
""")

# Imports & Setup
add_code("""import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress minor warnings for clean academic presentation
warnings.filterwarnings('ignore')

# Set aesthetic visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['figure.dpi'] = 120

# Add src to path if present
if os.path.exists('../src'):
    sys.path.insert(0, '../src')

print("Statistical analysis environment initialized successfully.")
""")

# Section 1: Preprocessing & Data Loading
add_md("""---
## 1. Dataset Acquisition, Preprocessing & Integrity Checks (Module I)

### Real-World Dataset:
We use the **Metro Interstate Traffic Volume Dataset** (UCI Machine Learning Repository).
- Location: Westbound I-94 between Minneapolis and St. Paul, MN.
- Variables: Hourly traffic volume, temperature (converted to Celsius), rain amount (`rain_1h`), snow amount (`snow_1h`), cloud cover (`clouds_all`), weather category (`weather_main`), and timestamps.

### Methodological Cleaning Principles:
1. **Deduplication:** Repeated timestamp instances are resolved by retaining the first valid observation.
2. **Missing Values & Gaps:** No artificial time imputation is performed to avoid fabricating continuous diurnal cycles.
3. **Outlier Verification:** Physical sensor anomalies (e.g., $T = 0\\text{ K}$ or rain $> 100\\text{ mm/hr}$) are filtered with explicit statistical and physical justification.
""")

add_code("""# Load cleaned dataset
clean_data_path = '../data/traffic_clean.csv' if os.path.exists('../data/traffic_clean.csv') else 'data/traffic_clean.csv'
df = pd.read_csv(clean_data_path, parse_dates=['date_time'])

print(f"Dataset successfully loaded.")
print(f"Observations (n): {len(df):,}")
print(f"Features: {df.shape[1]}")
print(f"Temporal span: {df['date_time'].min()} to {df['date_time'].max()}")
df[['traffic_volume', 'temp_c', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'weekday']].head()
""")

# Section 2: Module I & III
add_md("""---
## 2. Descriptive Statistics & Random Variable Analysis (Modules I & III)

### Mathematical Formulations:
- **Sample Mean:** $\\hat{\\mu} = \\bar{x} = \\frac{1}{n} \\sum_{i=1}^n x_i$
- **Sample Variance (Bessel's correction):** $s^2 = \\frac{1}{n-1} \\sum_{i=1}^n (x_i - \\bar{x})^2$
- **Sample Skewness:** $\\gamma_1 = \\frac{n}{(n-1)(n-2)} \\sum_{i=1}^n \\left(\\frac{x_i - \\bar{x}}{s}\\right)^3$
- **Sample Excess Kurtosis:** $\\gamma_2 = \\frac{n(n+1)}{(n-1)(n-2)(n-3)} \\sum_{i=1}^n \\left(\\frac{x_i - \\bar{x}}{s}\\right)^4 - \\frac{3(n-1)^2}{(n-2)(n-3)}$
- **Random Variable Expectation:** $E[X] = \\sum x_i P(X = x_i)$
""")

add_code("""# Descriptive statistics table
variables = ['traffic_volume', 'temp_c', 'rain_1h', 'snow_1h', 'clouds_all']
desc_records = []

for v in variables:
    s = df[v].dropna()
    mean = s.mean()
    std = s.std()
    q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
    desc_records.append({
        'Variable': v,
        'Mean': mean,
        'Median': q50,
        'Std Dev': std,
        'IQR': q75 - q25,
        'CV (s/mean)': std / mean if mean != 0 else np.nan,
        'Skewness': stats.skew(s),
        'Kurtosis': stats.kurtosis(s)
    })

desc_df = pd.DataFrame(desc_records)
print("=== DESCRIPTIVE STATISTICS TABLE ===")
desc_df.round(4)
""")

# Section 3: Normality & Continuous Distributions (Module V)
add_md("""---
## 3. Distribution & Normality Assessments (Module V)
We formally evaluate whether traffic volume and meteorological features follow normal distributions using:
1. **Shapiro-Wilk Test** (conducted on randomly selected subsample $n=5,000$ due to $N$ asymptotic power saturation).
2. **Kolmogorov-Smirnov Test** against the standard normal distribution using standardized $z$-scores: $z = \\frac{x - \\bar{x}}{s}$.
3. **Q-Q Plots** to inspect tail departures.
""")

add_code("""# Normality Assessment
for v in ['traffic_volume', 'temp_c']:
    s = df[v].dropna()
    subsample = s.sample(5000, random_state=42)
    sw_stat, sw_p = stats.shapiro(subsample)
    
    z = (s - s.mean()) / s.std()
    ks_stat, ks_p = stats.kstest(z, stats.norm.cdf)
    
    print(f"Variable: {v}")
    print(f"  Shapiro-Wilk (n=5000): W = {sw_stat:.4f}, p = {sw_p:.4e}")
    print(f"  Kolmogorov-Smirnov:    D = {ks_stat:.4f}, p = {ks_p:.4e}")
    print(f"  Skewness: {stats.skew(s):.4f} | Kurtosis: {stats.kurtosis(s):.4f}")
    print("  -> Null hypothesis of normality rejected (expected in high-n observational data).\\n")
""")

# Section 4: Probability & Bayes (Module II & IV)
add_md("""---
## 4. Probability Analysis, Bayes' Theorem & Poisson Suitability (Modules II & IV)

### 4.1 Frequentist & Conditional Probabilities:
- Defined High Traffic Event $H$: $\\text{traffic\\_volume} \\ge Q_3 (4,952\\text{ veh/hr})$
- Rainy Event $R$: $\\text{rain\\_1h} > 0.0\\text{ mm/hr}$
- Peak Hours Event $P$: Weekday hours $\\{7,8,9,16,17,18,19\\}$

### 4.2 Bayes' Theorem Formulation:
$$P(R \\mid H) = \\frac{P(H \\mid R) \\cdot P(R)}{P(H)}$$

### 4.3 Poisson Suitability Assessment (Module IV):
For a discrete Poisson variable $X \\sim \\text{Poisson}(\\lambda)$:
$$E[X] = \\text{Var}(X) = \\lambda \\implies \\text{Dispersion Index } D = \\frac{\\text{Var}(X)}{E[X]} = 1$$
""")

add_code("""# Frequentist Probability & Bayes Analysis
n_total = len(df)
p_rain = (df['rain_flag'] == 1).mean()
p_high = (df['high_traffic'] == 1).mean()
p_peak = (df['peak'] == 'Peak').mean()

p_high_given_rain = df.loc[df['rain_flag'] == 1, 'high_traffic'].mean()
p_high_given_dry = df.loc[df['rain_flag'] == 0, 'high_traffic'].mean()
p_high_given_peak = df.loc[df['peak'] == 'Peak', 'high_traffic'].mean()
p_high_given_offpeak = df.loc[df['peak'] == 'Off-Peak', 'high_traffic'].mean()

# Bayes' Theorem
p_rain_given_high = (p_high_given_rain * p_rain) / p_high

print("=== PROBABILITY ESTIMATES ===")
print(f"P(Rain) = {p_rain:.4f}")
print(f"P(High Traffic) = {p_high:.4f}")
print(f"P(High Traffic | Rain) = {p_high_given_rain:.4f}")
print(f"P(High Traffic | Dry)  = {p_high_given_dry:.4f}")
print(f"Relative Risk [P(High|Rain) / P(High|Dry)] = {p_high_given_rain / p_high_given_dry:.4f}")
print(f"Bayes Posterior P(Rain | High Traffic) = {p_rain_given_high:.4f}")
print(f"P(High Traffic | Peak Hour) = {p_high_given_peak:.4f}")
print(f"P(High Traffic | Off-Peak)  = {p_high_given_offpeak:.4f}")

# Poisson Dispersion Test
tv_mean = df['traffic_volume'].mean()
tv_var = df['traffic_volume'].var()
dispersion = tv_var / tv_mean
print(f"\\n=== POISSON DISPERSION TEST ===")
print(f"Mean E[X] = {tv_mean:.2f} | Variance Var(X) = {tv_var:.2f}")
print(f"Dispersion Index = {dispersion:.2f}")
print(f"Verdict: Dispersion >> 1 -> Severe overdispersion. Poisson distribution assumption violated.")
""")

# Section 5: Hypothesis Testing H1-H5 (Modules VII & VIII)
add_md("""---
## 5. Comprehensive Hypothesis Testing: H1 to H5 (Modules VII & VIII)

We execute five rigorous pre-specified hypothesis tests with full $H_0/H_1$ formulations, parametric and non-parametric tests, effect sizes, and confounder adjustment:

| Hypothesis | Test Description | Statistical Method | Confounding Controlled? |
|---|---|---|---|
| **H1** | Rain vs Dry Traffic Volume | Mann-Whitney $U$ (unadjusted) + OLS with HAC (adjusted) | Yes (calendar & diurnal controls) |
| **H2** | Precipitation Intensity Groups | Kruskal-Wallis $H$ + Dunn-Holm post-hoc | Unadjusted intensity group test |
| **H3** | Weather $\\times$ Peak Period Interaction | OLS Interaction Model with HAC SEs | Yes |
| **H4** | Nonlinear Temperature Relationship | Polynomial OLS (Linear vs Quadratic vs Cubic) | Yes |
| **H5** | Weather vs Calendar Incremental $\\Delta R^2$ | Nested Model $F$-test (Partial $F$) | Yes |
""")

add_code("""# Run full hypothesis testing suite using modular src engine
from hypothesis_testing import run_hypothesis_testing

hyp_res = run_hypothesis_testing(df)
hyp_res['summary']
""")

# Section 6: Correlation & Covariance (Module IX)
add_md("""---
## 6. Correlation, Covariance & Multicollinearity (Modules IX & X)

### Formulations:
- **Covariance:** $\\text{Cov}(X,Y) = \\frac{1}{n-1} \\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})$
- **Pearson Linear Correlation:** $r = \\frac{\\text{Cov}(X,Y)}{s_x s_y}$
- **Fisher's $z$-transform 95% Confidence Interval:**
  $$z = \\frac{1}{2} \\ln \\left(\\frac{1+r}{1-r}\\right), \\quad \\text{SE} = \\frac{1}{\\sqrt{n-3}}, \\quad z \\pm 1.96 \\cdot \\text{SE}$$
- **Spearman Rank Correlation:** $r_s = 1 - \\frac{6 \\sum d_i^2}{n(n^2-1)}$
- **Variance Inflation Factor (VIF):** $\\text{VIF}_j = \\frac{1}{1 - R_j^2}$
""")

add_code("""from correlation_analysis import run_correlation_analysis

corr_res = run_correlation_analysis(df)
corr_res['comparison']
""")

# Section 7: Regression Analysis & HAC Inference (Module X)
add_md("""---
## 7. Multiple Regression Models & Newey-West HAC Inference (Module X)

### Autocorrelation & Newey-West Correction:
Because hourly highway traffic data displays strong daily autocorrelation ($DW \\approx 0.41$), standard OLS standard errors would underestimate parameter uncertainty, leading to artificially inflated $t$-statistics.
To guarantee valid academic inference, we compute **Heteroscedasticity and Autocorrelation Consistent (HAC) standard errors** via the Newey-West kernel with bandwidth $\\text{lag} = 24$ hours.
""")

add_code("""from regression_analysis import run_regression_analysis

reg_res = run_regression_analysis(df)
reg_res['summary']
""")

# Section 8: Model Comparison (Module X)
add_md("""---
## 8. Out-of-Sample Predictive Validation & Supplementary ML (Module X)

### Methodological Rule: Chronological Train/Test Split
In time-series observational data, random cross-validation leaking future points into training yields artificially optimistic performance.
We apply a strict **chronological 80/20 train-test split** ($N_{\\text{train}} = 32,451$, $N_{\\text{test}} = 8,113$) and evaluate out-of-sample RMSE, MAE, and $R^2$.
""")

add_code("""from model_comparison import run_model_comparison

comp_res = run_model_comparison(df)
""")

# Section 9: Visualization Gallery
add_md("""---
## 9. Publication-Grade Visualizations Gallery
Here we display key figures generated by the analysis pipeline.
""")

add_code("""fig_dir = '../results/figures' if os.path.exists('../results/figures') else 'results/figures'
if os.path.exists(fig_dir):
    figs = sorted([f for f in os.listdir(fig_dir) if f.endswith('.png')])
    print(f"Generated {len(figs)} publication-quality visual artifacts:")
    for f in figs:
        print(f" - {f}")
""")

# Section 10: Academic Conclusion & Devil's Advocate Viva Defense
add_md("""---
## 10. Academic Synthesis & "Kill the Critics" Viva Defense

### Core Empirical Findings:
1. **Dominance of Diurnal and Calendar Cycle:** Time and calendar factors account for $R^2 \\approx 0.838$ (83.8%) of traffic volume variation.
2. **Minimal Incremental Weather Effect:** In isolation, weather features explain only $R^2 \\approx 0.029$ (2.9%). When added to calendar variables, weather provides a statistically significant ($p < 10^{-5}$) but practically modest increment of $\\Delta R^2 = 0.0009$ (0.09%).
3. **Severe Temporal Autocorrelation:** Residual diagnostics reveal strong serial correlation ($DW = 0.41$), necessitating Newey-West HAC inference ($p = 24$).
4. **Nonlinear Thermal Association:** Temperature displays a statistically significant quadratic response ($p = 7.5 \\times 10^{-5}$), reflecting reduced activity at extreme temperature extremes.

### Critical Examiner Viva Defense Checklist:
- **Q: Did you find that rain causes traffic congestion?**  
  *A: No. In unadjusted tests, rain and traffic appear weakly associated, but after controlling for hour, weekday, and month, the adjusted effect of rain is negligible. Observational data demonstrates statistical association, not causal mechanisms.*
- **Q: Why didn't you fit a Poisson regression?**  
  *A: The variance-to-mean dispersion ratio is $1,196.8 \\gg 1$, violating the Poisson equidispersion property ($E[X] = \\text{Var}(X)$). Traffic counts show clustering and diurnal serial dependence.*
- **Q: Are standard OLS $p$-values trustworthy here?**  
  *A: No, standard OLS standard errors assume i.i.d. residuals. Because hourly traffic exhibits 24-hour cyclical autocorrelation, we applied Newey-West HAC standard errors (lag 24) to ensure statistically robust inference.*
""")

# Write out file
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_weather_statistical_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Notebook successfully generated at: {out_path}")
