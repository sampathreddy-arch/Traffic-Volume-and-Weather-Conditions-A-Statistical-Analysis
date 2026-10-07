# Traffic Volume and Weather Conditions: A Statistical Analysis

**B.Tech Major Project — Statistics and Probability (Modules I – X)**  
**Dataset:** Metro Interstate Traffic Volume (UCI Machine Learning Repository, ID: 492)  
**Sample Size:** $N = 40,564$ cleaned hourly observations (2012–2018)  
**Corridor:** Westbound I-94 between Minneapolis and St. Paul, MN  

---

## 🎯 Executive Summary
This project delivers a complete, mathematically grounded, and syllabus-aligned statistical investigation into the relationship between weather conditions and highway traffic volume. Designed to withstand the scrutiny of senior statistics professors and academic evaluators, the project avoids superficial machine learning wrappers and focuses on:
- **Rigorous Hypothesis Testing:** Formal $H_0/H_1$ formulations with effect sizes, confidence intervals, and both parametric and non-parametric dual tests.
- **Confounding & Diurnal Dominance:** Revealing how raw weather correlations ($R^2 \approx 2.9\%$) are confounded by daily and seasonal rhythms, with weather adding only an incremental $\Delta R^2 = 0.0009$ ($0.09\%$) to a full calendar model ($R^2 = 83.79\%$).
- **Inference Robust to Autocorrelation:** Addressing severe daily serial correlation ($\text{Durbin-Watson} = 0.412$) through Newey-West HAC standard errors (lag bandwidth $= 24$).
- **Probability Modeling:** Frequentist conditional risk estimation, Bayes' Theorem inversion, and mathematical refutation of the Poisson distribution via dispersion index testing ($D = 1,196.8 \gg 1$).

---

## 📁 Repository Structure
```
traffic-weather-analysis/
│
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv  # Raw UCI dataset (48,204 rows)
│   └── traffic_clean.csv                   # Clean, validated dataset (40,564 rows)
│
├── src/
│   ├── data_preprocessing.py               # Module I: Deduplication, outlier audit, feature engineering
│   ├── descriptive_statistics.py           # Modules I & III: Moments, frequency tables, normality
│   ├── probability_analysis.py             # Modules II & IV: Bayes' rule, Poisson dispersion test
│   ├── hypothesis_testing.py               # Modules VII & VIII: H1 to H5 tests (parametric + non-parametric)
│   ├── correlation_analysis.py             # Module IX: Pearson, Spearman, Fisher's z 95% CIs
│   ├── regression_analysis.py              # Module X: OLS Models 1-4, VIF multicollinearity, Newey-West HAC
│   ├── diagnostics.py                      # Module X: Breusch-Pagan, DW, Ljung-Box, Q-Q residual plots
│   ├── model_comparison.py                 # Module X: Chronological 80/20 out-of-sample validation + GBM
│   └── visualization.py                    # Modules I, IX, X: 11 publication-grade charts
│
├── results/
│   ├── descriptive_statistics.csv          # Summary moments and quantile tables
│   ├── probability_results.csv             # Prior, conditional, and posterior Bayes probabilities
│   ├── statistical_results.csv             # H1-H5 p-values, test statistics, and effect sizes
│   ├── correlation_results.csv             # Pearson, Spearman, and Fisher z intervals
│   ├── regression_results.csv              # OLS coefficients, HAC standard errors, R², AIC, BIC
│   ├── model_comparison.csv                # Chronological out-of-sample RMSE, MAE, R²
│   └── figures/                            # 15 high-resolution publication-quality PNG charts
│
├── dashboard/
│   ├── index.html                          # Interactive dark-themed web dashboard with tools & gallery
│   └── app.py                              # Streamlit interactive application
│
├── notebooks/
│   ├── build_notebook.py                   # Script generating the Jupyter notebook
│   └── traffic_weather_statistical_analysis.ipynb # Fully documented academic notebook
│
├── report/
│   ├── FINAL_ACADEMIC_PROJECT_REPORT.md    # Comprehensive academic research report
│   └── project_report.md                   # Mirror copy for evaluators
│
├── main.py                                 # Master pipeline runner (executes stages 1-9 reproducibly)
├── requirements.txt                        # Dependency specifications
└── README.md                               # Project documentation and guide
```

---

## 🚀 Reproduction Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Full Master Pipeline
```bash
python main.py
```
This runs the entire end-to-end analytical workflow:
1. Preprocessing & Deduplication ($40,564$ rows)
2. Descriptive Statistics & Moments
3. Probability Modeling & Bayes' Rule
4. Hypothesis Testing (H1 to H5)
5. Correlation & Covariance Analysis
6. Multiple Regression Modeling & VIF
7. Residual Diagnostics & HAC Checks
8. Chronological Out-of-Sample Validation
9. Generation of all 15 publication figures

### 3. Launch Interactive Dashboard
- **Web Dashboard (standalone):** Open [dashboard/index.html](file:///c:/Users/madur/Downloads/Probability%20and%20Statistics/traffic-weather-analysis/dashboard/index.html) directly in any web browser.
- **Streamlit App:**
  ```bash
  streamlit run dashboard/app.py
  ```

---

## 📊 Core Empirical Findings

| Hypothesis | Description | Statistical Test | Result | Effect Size | Practical Finding |
|---|---|---|---|---|---|
| **H1** | Rain vs Dry Traffic | Mann-Whitney U + Adjusted OLS | $p_{\text{adj}} = 0.997$ | $r = -0.012$ | Negligible impact once hour/calendar are controlled. |
| **H2** | Precipitation Intensity | Kruskal-Wallis H + Dunn-Holm | $p = 0.011$ | $\eta^2 = 0.0002$ | Statistically detectable omnibus difference, practically trivial magnitude. |
| **H3** | Rain × Peak Interaction | OLS Interaction with HAC | $p = 1.000$ | $\beta_{\text{int}} = -0.22$ | Weather association does not differ between peak and off-peak hours. |
| **H4** | Nonlinear Temperature | Polynomial OLS (Linear vs Quad) | $p = 7.5 \times 10^{-5}$ | $\beta_{\text{quad}} = -0.179$ | Statistically significant inverted-U curvature ($\Delta \text{AIC} = 44.89$). |
| **H5** | Weather Incremental Power | Nested Model Partial $F$-test | $p < 10^{-10}$ | $\Delta R^2 = 0.0009$ | Calendar dominates $83.8\%$ of variance; weather adds only $0.09\%$. |

---

## 🎓 Academic Defense ("Kill the Critics" Summary)
- **Causality vs Association:** Observational traffic counts cannot prove causation. Confounding by diurnal commuter schedules accounts for the majority of naive weather correlations.
- **Poisson Refutation:** The dispersion index of traffic volume is $D = \frac{s^2}{\bar{x}} = 1,196.8 \gg 1$, violating the Poisson equidispersion property ($D = 1$).
- **Autocorrelation Treatment:** Residual serial correlation ($\text{DW} = 0.412$) violates standard OLS assumptions. All inference is corrected using Newey-West HAC standard errors (lag bandwidth $= 24$).
- **No Temporal Leakage:** Out-of-sample validation uses a strict chronological split ($2012$–$2017$ train vs $2017$–$2018$ test), preventing future data leakage.
