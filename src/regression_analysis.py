"""
regression_analysis.py
======================
Module X — Regression Analysis
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Build and compare four regression models progressing from time-only
    to fully specified. Address confounding, multicollinearity, temporal
    dependence (HAC errors), and nonlinearity.

MODELS:
    Model 1: Time-only
    Model 2: Weather-only
    Model 3: Combined (time + weather)
    Model 4: Combined + nonlinear temperature + interaction

MATHEMATICAL FRAMEWORK:
─────────────────────────────────────────────────────────────
SIMPLE LINEAR REGRESSION:
    ŷᵢ = β₀ + β₁xᵢ + εᵢ
    OLS minimises: Σ(yᵢ − ŷᵢ)²
    β̂ = (XᵀX)⁻¹Xᵀy

MULTIPLE LINEAR REGRESSION:
    ŷᵢ = β₀ + β₁x₁ᵢ + β₂x₂ᵢ + ... + βₖxₖᵢ + εᵢ
    Assumptions: Linearity, Independence, Homoscedasticity, Normality

POLYNOMIAL REGRESSION:
    ŷᵢ = β₀ + β₁xᵢ + β₂xᵢ² + ... + βₘxᵢᵐ + εᵢ
    Special case of multiple linear regression

HAC STANDARD ERRORS (Newey-West):
    Corrects SEs for autocorrelation and heteroscedasticity.
    Lag = 24 justified by the 24-hour daily periodicity.

R² = 1 − SSₑ/SSᵧ  (proportion of variance explained)
Adjusted R² = 1 − (1−R²)(n−1)/(n−k−1)  (penalises extra parameters)

VIF (Variance Inflation Factor):
    VIF_j = 1 / (1 − R²_j)  where R²_j = R² of regressing xⱼ on all others.
    VIF > 10 → serious multicollinearity concern
─────────────────────────────────────────────────────────────
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def print_model_summary(model, name: str, hac: bool = True) -> None:
    """Print a concise model summary."""
    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"  n observations   : {int(model.nobs):,}")
    print(f"  R²               : {model.rsquared:.4f}")
    print(f"  Adjusted R²      : {model.rsquared_adj:.4f}")
    print(f"  F-statistic      : {model.fvalue:.2f}")
    print(f"  F p-value        : {model.f_pvalue:.4e}")
    print(f"  AIC              : {model.aic:.2f}")
    print(f"  BIC              : {model.bic:.2f}")
    if hac:
        print(f"  SE type          : HAC Newey-West (lag=24)")


def compute_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors.
    VIF_j = 1 / (1 − R²_j)
    """
    vif_data = pd.DataFrame()
    vif_data["variable"] = X_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i)
        for i in range(X_df.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE LINEAR REGRESSION (pedagogical)
# ─────────────────────────────────────────────────────────────────────────────
def simple_linear_regression(df: pd.DataFrame) -> dict:
    """
    Simple linear regression: traffic_volume ~ temp_c

    Mathematical derivation shown explicitly.
    """
    sub = df[["traffic_volume", "temp_c"]].dropna()
    X = sub["temp_c"].values
    y = sub["traffic_volume"].values
    n = len(X)

    # Manual OLS computation (educational)
    x_bar = X.mean()
    y_bar = y.mean()
    b1 = np.sum((X - x_bar) * (y - y_bar)) / np.sum((X - x_bar)**2)
    b0 = y_bar - b1 * x_bar

    y_hat = b0 + b1 * X
    residuals = y - y_hat
    SSE = np.sum(residuals**2)
    SST = np.sum((y - y_bar)**2)
    R2  = 1 - SSE / SST

    print(f"\n{'='*60}")
    print("  SIMPLE LINEAR REGRESSION: traffic_volume ~ temp_c")
    print(f"{'='*60}")
    print(f"  Manual OLS Derivation:")
    print(f"  β̂₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²")
    print(f"  β̂₁ = {b1:.4f}  (vehicles per °C)")
    print(f"  β̂₀ = ȳ − β̂₁x̄ = {b0:.4f}")
    print(f"  Equation: traffic_volume = {b0:.2f} + {b1:.2f} × temp_c")
    print(f"  R² = {R2:.4f}  (temperature alone explains {R2*100:.1f}% of variance)")
    print(f"\n  INTERPRETATION: For every 1°C increase in temperature, traffic")
    print(f"  volume is associated with a {b1:.1f} vehicle/hr change on average.")

    # Statsmodels for inference
    m = smf.ols("traffic_volume ~ temp_c", data=sub).fit(
        cov_type="HAC", cov_kwds={"maxlags": 24}
    )
    print(f"\n  Statsmodels (HAC) verification:")
    print(f"  Coefficient temp_c: {m.params['temp_c']:.4f}")
    print(f"  p-value (HAC):      {m.pvalues['temp_c']:.4e}")
    print(f"  95% CI:             [{m.conf_int().loc['temp_c',0]:.4f}, "
          f"{m.conf_int().loc['temp_c',1]:.4f}]")

    return {"b0": b0, "b1": b1, "R2": R2, "model": m}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: TIME-ONLY
# ─────────────────────────────────────────────────────────────────────────────
def model_1_time_only(df: pd.DataFrame):
    """
    Model 1: traffic_volume ~ C(hour) + C(weekday) + C(month) + C(year)
                               + holiday_flag

    Purpose: Establish the baseline temporal explanatory power.
    Hour and weekday are categorical (dummy-encoded) — correct treatment
    since traffic patterns are non-linear across hours.
    """
    sub = df[["traffic_volume", "hour", "weekday", "month", "year",
              "holiday_flag"]].dropna()

    m = smf.ols(
        "traffic_volume ~ C(hour) + C(weekday) + C(month) + C(year) "
        "+ holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    print_model_summary(m, "MODEL 1: Time-Only")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: WEATHER-ONLY
# ─────────────────────────────────────────────────────────────────────────────
def model_2_weather_only(df: pd.DataFrame):
    """
    Model 2: traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all

    Purpose: Isolate raw weather explanatory power before controlling
    for calendar confounders.

    CONFOUNDING WARNING: Weather variables correlate with season and hour.
    Model 2 coefficients do NOT represent calendar-adjusted effects.
    """
    sub = df[["traffic_volume", "temp_c", "rain_1h",
              "snow_1h", "clouds_all"]].dropna()

    m = smf.ols(
        "traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    print_model_summary(m, "MODEL 2: Weather-Only")
    print(f"\n  ⚠  CONFOUNDING NOTE: Weather variables are confounded by")
    print(f"  season, month, and time of day. Coefficients in Model 2")
    print(f"  capture BOTH weather effects AND seasonal confounding.")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: COMBINED
# ─────────────────────────────────────────────────────────────────────────────
def model_3_combined(df: pd.DataFrame):
    """
    Model 3: Combined time + weather

    Purpose: Estimate weather effects after controlling for temporal patterns.
    This is the primary inferential model.
    """
    sub = df[["traffic_volume", "temp_c", "rain_1h", "snow_1h",
              "clouds_all", "hour", "weekday", "month", "year",
              "holiday_flag"]].dropna()

    m = smf.ols(
        "traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all "
        "+ C(hour) + C(weekday) + C(month) + C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    print_model_summary(m, "MODEL 3: Combined (Time + Weather)")

    # Key weather coefficients
    print(f"\n  KEY WEATHER COEFFICIENTS (adjusted):")
    weather_params = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
    for p in weather_params:
        if p in m.params:
            ci = m.conf_int().loc[p]
            print(f"  {p:<15s}: β={m.params[p]:>8.3f}  "
                  f"SE={m.bse[p]:.3f}  p={m.pvalues[p]:.4e}  "
                  f"95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: COMBINED + NONLINEAR TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────
def model_4_nonlinear(df: pd.DataFrame):
    """
    Model 4: Combined + temp_c² + interaction rain_flag × peak_flag

    Purpose: Add nonlinear temperature and weather×peak interaction.
    Only included where statistically justified (see H4, H3 results).
    """
    sub = df[["traffic_volume", "temp_c", "rain_1h", "snow_1h",
              "clouds_all", "hour", "weekday", "month", "year",
              "holiday_flag", "peak"]].dropna().copy()
    sub["temp_c2"] = sub["temp_c"] ** 2
    sub["peak_flag"] = (sub["peak"] == "Peak").astype(int)

    m = smf.ols(
        "traffic_volume ~ temp_c + temp_c2 + rain_1h * peak_flag "
        "+ snow_1h + clouds_all "
        "+ C(hour) + C(weekday) + C(month) + C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    print_model_summary(m, "MODEL 4: Combined + Nonlinear + Interaction")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# VIF ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_vif_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for weather variables to check multicollinearity.

    Only weather variables are checked (categorical time dummies
    are excluded as VIF is not meaningful for categorical sets).
    """
    cols = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
    sub = df[cols].dropna()

    print(f"\n{'='*60}")
    print("  MULTICOLLINEARITY — Variance Inflation Factors (VIF)")
    print(f"{'='*60}")
    print(f"  VIF_j = 1 / (1 − R²_j)")
    print(f"  VIF < 5:  Acceptable")
    print(f"  VIF 5-10: Moderate concern")
    print(f"  VIF > 10: Serious multicollinearity\n")

    vif_df = compute_vif(sub)
    for _, row in vif_df.iterrows():
        flag = ("⚠ CONCERN" if row["VIF"] > 10
                else ("⚡ MODERATE" if row["VIF"] > 5 else "✓ OK"))
        print(f"  {row['variable']:<20s}: VIF = {row['VIF']:.2f}  {flag}")

    return vif_df


# ─────────────────────────────────────────────────────────────────────────────
# CONFOUNDING ANALYSIS: Coefficient trajectory
# ─────────────────────────────────────────────────────────────────────────────
def confounding_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show how the rain_1h coefficient changes as we progressively add controls.

    This directly answers: "Does the apparent weather effect reflect
    calendar confounding?"
    """
    sub = df[["traffic_volume", "temp_c", "rain_1h", "snow_1h",
              "clouds_all", "hour", "weekday", "month", "year",
              "holiday_flag"]].dropna()

    models = [
        ("Unadjusted",
         "traffic_volume ~ rain_1h"),
        ("+ Hour",
         "traffic_volume ~ rain_1h + C(hour)"),
        ("+ Hour + Weekday",
         "traffic_volume ~ rain_1h + C(hour) + C(weekday)"),
        ("+ Hour + Weekday + Month",
         "traffic_volume ~ rain_1h + C(hour) + C(weekday) + C(month)"),
        ("+ Full Calendar",
         "traffic_volume ~ rain_1h + C(hour) + C(weekday) + C(month) "
         "+ C(year) + holiday_flag"),
        ("+ Weather Controls",
         "traffic_volume ~ rain_1h + temp_c + snow_1h + clouds_all "
         "+ C(hour) + C(weekday) + C(month) + C(year) + holiday_flag"),
    ]

    rows = []
    print(f"\n{'='*60}")
    print("  CONFOUNDING TRAJECTORY: rain_1h coefficient")
    print(f"{'='*60}")
    print(f"  How does the rain coefficient change as we add controls?\n")

    for label, formula in models:
        m = smf.ols(formula, data=sub).fit(
            cov_type="HAC", cov_kwds={"maxlags": 24}
        )
        if "rain_1h" in m.params:
            coef = m.params["rain_1h"]
            p    = m.pvalues["rain_1h"]
            print(f"  {label:<30s}: β(rain) = {coef:>8.3f}  p = {p:.4e}")
            rows.append({"model": label, "rain_coef": coef, "p_value": p,
                         "r2": m.rsquared})

    print(f"\n  INTERPRETATION:")
    print(f"  If the coefficient changes substantially as controls are added,")
    print(f"  this indicates CONFOUNDING — the raw association partly reflects")
    print(f"  calendar or seasonal effects, not weather alone.")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_regression_analysis(df: pd.DataFrame,
                            save_path: str = None) -> dict:
    """
    Run all regression models and return dict of results.
    """
    print("\n" + "="*65)
    print("REGRESSION ANALYSIS — Module X")
    print("="*65)

    # Simple linear regression (pedagogical)
    slr = simple_linear_regression(df)

    print(f"\n{'='*65}")
    print("  MULTIPLE REGRESSION MODELS")
    print(f"{'='*65}")

    m1 = model_1_time_only(df)
    m2 = model_2_weather_only(df)
    m3 = model_3_combined(df)
    m4 = model_4_nonlinear(df)

    # VIF
    vif = run_vif_analysis(df)

    # Confounding trajectory
    confound = confounding_trajectory(df)

    # Model comparison summary
    summary = pd.DataFrame({
        "Model": ["M1: Time-only", "M2: Weather-only",
                  "M3: Combined", "M4: Nonlinear+Interact"],
        "R2":     [m1.rsquared, m2.rsquared, m3.rsquared, m4.rsquared],
        "Adj_R2": [m1.rsquared_adj, m2.rsquared_adj,
                   m3.rsquared_adj, m4.rsquared_adj],
        "AIC":    [m1.aic, m2.aic, m3.aic, m4.aic],
        "BIC":    [m1.bic, m2.bic, m3.bic, m4.bic],
        "n_obs":  [int(m1.nobs), int(m2.nobs), int(m3.nobs), int(m4.nobs)],
    })

    print(f"\n{'='*65}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(summary.to_string(index=False))

    if save_path:
        summary.to_csv(save_path, index=False)
        print(f"\n[SAVE] Regression results saved → {save_path}")

    return {
        "slr": slr, "m1": m1, "m2": m2, "m3": m3, "m4": m4,
        "vif": vif, "confound": confound, "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean = os.path.join(base, "data", "traffic_clean.csv")
    save  = os.path.join(base, "results", "regression_results.csv")
    df = pd.read_csv(clean, parse_dates=["date_time"])
    run_regression_analysis(df, save_path=save)
