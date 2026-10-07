"""
hypothesis_testing.py
=====================
Modules VII, VIII — Testing of Hypotheses I & II
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Implement and interpret all five research hypotheses using
    statistically appropriate tests. Every test includes:
    - Mathematical justification
    - H₀ and H₁
    - Assumptions check
    - Test statistic and p-value
    - Effect size
    - Type I and Type II error discussion
    - Clear interpretation without causal language

CRITICAL RULES:
    1. Mann-Whitney U is used ONLY as unadjusted exploratory evidence for H1.
    2. The main H1 conclusion uses regression-based adjusted evidence.
    3. Multiple comparisons in H2 use Holm correction.
    4. H3 uses interaction term in regression.
    5. H4 uses formal model comparison for nonlinearity.
    6. H5 uses nested model comparison (ΔR²).

SIGNIFICANCE LEVEL: α = 0.05 throughout.

TYPE I ERROR (α):  Probability of rejecting H₀ when it is true.
                   Accepted risk: α = 0.05 (5%).
TYPE II ERROR (β): Probability of failing to reject H₀ when H₁ is true.
                   Not formally computed here; large sample sizes give
                   high power for detecting even small effects.
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ALPHA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Print hypothesis header
# ─────────────────────────────────────────────────────────────────────────────
def hypothesis_header(code: str, h0: str, h1: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {code}")
    print(f"  H₀: {h0}")
    print(f"  H₁: {h1}")
    print(f"  α  = {ALPHA}")
    print(f"{'='*65}")


def significance_decision(p: float, alpha: float = ALPHA) -> str:
    if p < alpha:
        return f"REJECT H₀  (p = {p:.4e} < α = {alpha})"
    else:
        return f"FAIL TO REJECT H₀  (p = {p:.4f} ≥ α = {alpha})"


def rank_biserial_r(U: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation effect size for Mann-Whitney U.
        r = 1 − (2U / (n1·n2))
    Range: [-1, 1]. |r| < 0.1 small, 0.1-0.3 medium, > 0.3 large.
    """
    return 1 - (2 * U) / (n1 * n2)


def eta_squared_kruskal(H: float, k: int, n: int) -> float:
    """
    η² (eta-squared) for Kruskal-Wallis:
        η² = (H − k + 1) / (n − k)
    """
    return (H - k + 1) / (n - k)


# ─────────────────────────────────────────────────────────────────────────────
# H1: RAIN vs DRY TRAFFIC VOLUME
# ─────────────────────────────────────────────────────────────────────────────
def test_h1_rain_vs_dry(df: pd.DataFrame) -> dict:
    """
    H1: Traffic volume differs between rainy and dry periods.
    H₀: Rain has no statistically significant association with traffic volume.

    STEP 1 (Unadjusted): Mann-Whitney U test — exploratory only.
        Does NOT control for hour, weekday, month, or year.
        Cannot distinguish weather effect from calendar confounding.

    STEP 2 (Adjusted): OLS regression with calendar controls.
        traffic_volume ~ rain_flag + C(hour) + C(weekday) + C(month) + C(year)
        The coefficient on rain_flag is the adjusted association.

    Mann-Whitney U:
        Tests whether one distribution is stochastically greater than another.
        H₀: P(X_rain > X_dry) = 0.5
        Does NOT require normality.
        Effect size: Rank-biserial correlation r = 1 − 2U/(n₁n₂)
    """
    hypothesis_header(
        "H1: RAIN vs DRY TRAFFIC VOLUME",
        h0="Rain has no statistically significant adjusted association with "
           "traffic volume.",
        h1="Traffic volume differs between rainy and dry periods after "
           "accounting for major calendar effects."
    )

    rain_vol = df.loc[df["rain_flag"] == 1, "traffic_volume"].dropna()
    dry_vol  = df.loc[df["rain_flag"] == 0, "traffic_volume"].dropna()

    # ── STEP 1: Unadjusted Mann-Whitney U ────────────────────────────────
    print("\n  STEP 1 — UNADJUSTED EVIDENCE (Mann-Whitney U)")
    print("  ⚠ This test does NOT control for confounders.\n")

    U, p_mw = stats.mannwhitneyu(rain_vol, dry_vol, alternative="two-sided")
    r_rb    = rank_biserial_r(U, len(rain_vol), len(dry_vol))

    print(f"  n(Rain)   = {len(rain_vol):,}  |  Median = {rain_vol.median():.1f}")
    print(f"  n(Dry)    = {len(dry_vol):,}  |  Median = {dry_vol.median():.1f}")
    print(f"  U statistic = {U:.0f}")
    print(f"  p-value     = {p_mw:.4e}")
    print(f"  Effect size (rank-biserial r) = {r_rb:.4f}")
    print(f"\n  Decision: {significance_decision(p_mw)}")
    print(f"\n  CAUTION: This result is confounded by hour, weekday, month,")
    print(f"  and year. Rain is not randomly assigned; it occurs more in")
    print(f"  certain months/hours. DO NOT interpret as causal evidence.")

    # ── STEP 2: Adjusted OLS Regression ──────────────────────────────────
    print(f"\n  STEP 2 — ADJUSTED EVIDENCE (OLS Regression with Controls)")
    print(f"  Model: traffic_volume ~ rain_flag + C(hour) + C(weekday) "
          f"+ C(month) + C(year)")

    sub = df[["traffic_volume", "rain_flag", "hour", "weekday",
              "month", "year", "holiday_flag"]].dropna()
    # Use HAC-robust standard errors (Newey-West, lag=24 = 1 daily cycle)
    model = smf.ols(
        "traffic_volume ~ rain_flag + C(hour) + C(weekday) + C(month) "
        "+ C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    rain_coef = model.params["rain_flag"]
    rain_se   = model.bse["rain_flag"]
    rain_t    = model.tvalues["rain_flag"]
    rain_p    = model.pvalues["rain_flag"]
    rain_ci   = model.conf_int().loc["rain_flag"]

    print(f"\n  rain_flag coefficient (adjusted): {rain_coef:.2f}")
    print(f"  HAC Std Error:                     {rain_se:.2f}")
    print(f"  t-statistic:                       {rain_t:.3f}")
    print(f"  p-value (HAC):                     {rain_p:.4e}")
    print(f"  95% CI:                            [{rain_ci[0]:.2f}, {rain_ci[1]:.2f}]")
    print(f"  Model R²:                          {model.rsquared:.4f}")
    print(f"\n  Decision (adjusted): {significance_decision(rain_p)}")
    print(f"\n  INTERPRETATION:")
    if rain_p < ALPHA:
        direction = "lower" if rain_coef < 0 else "higher"
        print(f"  After controlling for hour, weekday, month, year, and holiday,")
        print(f"  rainy hours are associated with {abs(rain_coef):.0f} vehicles/hr {direction}")
        print(f"  traffic volume on average (95% CI: [{rain_ci[0]:.0f}, {rain_ci[1]:.0f}]).")
        print(f"  This association is statistically significant at α = {ALPHA}.")
    else:
        print(f"  After controlling for calendar variables, the association")
        print(f"  between rain and traffic volume is not statistically")
        print(f"  significant at α = {ALPHA}.")
    print(f"\n  Type I Error Risk: α = {ALPHA} — 5% chance of falsely")
    print(f"  rejecting H₀ if it is true.")
    print(f"  Type II Error: With n = {len(sub):,}, statistical power is very")
    print(f"  high; even small effects are likely detected.")

    return {
        "U_stat": U, "p_mannwhitney": p_mw, "r_rank_biserial": r_rb,
        "rain_adj_coef": rain_coef, "rain_adj_se": rain_se,
        "rain_adj_p": rain_p, "rain_adj_ci": rain_ci.tolist(),
        "model_r2": model.rsquared,
        "n_rain": len(rain_vol), "n_dry": len(dry_vol),
    }


# ─────────────────────────────────────────────────────────────────────────────
# H2: PRECIPITATION INTENSITY — KRUSKAL-WALLIS + DUNN
# ─────────────────────────────────────────────────────────────────────────────
def test_h2_precipitation_groups(df: pd.DataFrame) -> dict:
    """
    H2: Traffic volume differs across precipitation intensity groups.
    H₀: No statistically significant difference across groups.

    Test: Kruskal-Wallis H-test
        Extension of Mann-Whitney to k ≥ 3 groups.
        H₀: All groups have the same population distribution.
        Test statistic:
            H = [12/(n(n+1))] Σ[nⱼ · (R̄ⱼ − (n+1)/2)²]
        Asymptotically χ²(k-1) under H₀.

    Post-hoc: Dunn's test with Holm correction
        If H is significant, Dunn pairwise tests identify which groups differ.
        Holm procedure controls family-wise error rate (FWER) ≤ α.
    """
    hypothesis_header(
        "H2: TRAFFIC VOLUME ACROSS PRECIPITATION INTENSITY GROUPS",
        h0="No statistically significant difference in traffic volume "
           "across precipitation groups.",
        h1="Traffic volume differs across precipitation/weather intensity "
           "conditions (Dry, Light, Moderate, Heavy)."
    )

    # Group summaries
    groups = ["Dry", "Light", "Moderate", "Heavy"]
    group_data = {}
    print(f"\n  Group summary:")
    for g in groups:
        mask = df["precipitation_intensity"] == g
        vol  = df.loc[mask, "traffic_volume"].dropna()
        if len(vol) > 0:
            group_data[g] = vol
            print(f"  {g:<12s}: n={len(vol):6,}  median={vol.median():7.1f}  "
                  f"mean={vol.mean():7.1f}")

    arrays = [v.values for v in group_data.values() if len(v) > 0]
    if len(arrays) < 2:
        print("  Insufficient groups for Kruskal-Wallis.")
        return {}

    # Kruskal-Wallis
    H, p_kw = stats.kruskal(*arrays)
    k = len(arrays)
    n = sum(len(a) for a in arrays)
    eta2 = eta_squared_kruskal(H, k, n)

    print(f"\n  Kruskal-Wallis H-test:")
    print(f"  H statistic = {H:.4f}")
    print(f"  df          = {k-1}")
    print(f"  p-value     = {p_kw:.4e}")
    print(f"  η² (eta-squared) = {eta2:.4f}  "
          f"({'small' if eta2 < 0.06 else 'medium' if eta2 < 0.14 else 'large'})")
    print(f"\n  Decision: {significance_decision(p_kw)}")

    # Post-hoc Dunn
    if p_kw < ALPHA:
        print(f"\n  POST-HOC: Dunn's test with Holm correction")
        print(f"  (Holm controls family-wise error rate ≤ α = {ALPHA})")

        all_vals = []
        all_labels = []
        for g, v in group_data.items():
            all_vals.extend(v.tolist())
            all_labels.extend([g] * len(v))

        dunn_df = sp.posthoc_dunn(
            pd.DataFrame({"volume": all_vals, "group": all_labels}),
            val_col="volume", group_col="group", p_adjust="holm"
        )
        print(f"\n  Dunn p-values (Holm-corrected):")
        print(dunn_df.round(4).to_string())

        print(f"\n  INTERPRETATION:")
        print(f"  Pairs with p < {ALPHA} show statistically significant differences.")
        print(f"  IMPORTANT: Kruskal-Wallis + Dunn tests the unadjusted")
        print(f"  difference; results may be confounded by seasonal patterns.")
        print(f"  Claim 'snow is worse than rain' requires explicit pairwise")
        print(f"  testing with appropriate corrections — shown above.")

    # Explicit snow vs rain comparison
    print(f"\n  EXPLICIT SNOW vs RAIN COMPARISON:")
    rain_only = df.loc[(df["rain_flag"] == 1) & (df["snow_flag"] == 0),
                       "traffic_volume"].dropna()
    snow_only = df.loc[(df["snow_flag"] == 1) & (df["rain_flag"] == 0),
                       "traffic_volume"].dropna()
    if len(rain_only) > 10 and len(snow_only) > 10:
        U_sr, p_sr = stats.mannwhitneyu(snow_only, rain_only, alternative="two-sided")
        r_sr = rank_biserial_r(U_sr, len(snow_only), len(rain_only))
        print(f"  n(Rain only) = {len(rain_only):,}  median = {rain_only.median():.1f}")
        print(f"  n(Snow only) = {len(snow_only):,}  median = {snow_only.median():.1f}")
        print(f"  U = {U_sr:.0f},  p = {p_sr:.4e},  r = {r_sr:.4f}")
        print(f"  Decision: {significance_decision(p_sr)}")
    else:
        print(f"  Insufficient snow-only observations for separate comparison.")

    return {
        "H_stat": H, "p_kruskal": p_kw, "eta_squared": eta2,
        "groups": {g: {"n": len(v), "median": v.median()}
                   for g, v in group_data.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# H3: WEATHER × PEAK/OFF-PEAK INTERACTION
# ─────────────────────────────────────────────────────────────────────────────
def test_h3_interaction(df: pd.DataFrame) -> dict:
    """
    H3: The association between weather and traffic differs between
        peak and off-peak hours.
    H₀: No weather × peak/off-peak interaction.

    Method: OLS regression with interaction term.
        Model: traffic_volume ~ rain_flag * peak_indicator + controls

    The interaction coefficient measures whether the rain–traffic
    association differs in magnitude between peak and off-peak periods.
    """
    hypothesis_header(
        "H3: WEATHER × PEAK/OFF-PEAK INTERACTION",
        h0="There is no weather × peak/off-peak interaction effect on "
           "traffic volume.",
        h1="The association between weather and traffic volume differs "
           "between peak and off-peak periods."
    )

    sub = df[["traffic_volume", "rain_flag", "peak", "snow_flag",
              "hour", "weekday", "month", "year", "holiday_flag"]].dropna()
    sub = sub.copy()
    sub["peak_flag"] = (sub["peak"] == "Peak").astype(int)

    # Model without interaction (restricted)
    m_restricted = smf.ols(
        "traffic_volume ~ rain_flag + peak_flag + C(hour) + C(weekday) "
        "+ C(month) + C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # Model with interaction (unrestricted)
    m_full = smf.ols(
        "traffic_volume ~ rain_flag * peak_flag + C(hour) + C(weekday) "
        "+ C(month) + C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # F-test for interaction term improvement
    interact_coef = m_full.params.get("rain_flag:peak_flag", np.nan)
    interact_p    = m_full.pvalues.get("rain_flag:peak_flag", np.nan)
    interact_ci   = m_full.conf_int().loc["rain_flag:peak_flag"] \
                    if "rain_flag:peak_flag" in m_full.conf_int().index \
                    else [np.nan, np.nan]

    print(f"\n  Interaction coefficient (rain_flag × peak_flag): "
          f"{interact_coef:.2f}")
    print(f"  p-value (HAC): {interact_p:.4e}")
    print(f"  95% CI: [{interact_ci[0]:.2f}, {interact_ci[1]:.2f}]")
    print(f"\n  Decision: {significance_decision(interact_p)}")
    print(f"\n  INTERPRETATION:")
    print(f"  The interaction term measures whether the rain-traffic")
    print(f"  association differs in peak vs off-peak periods.")
    if interact_p < ALPHA:
        print(f"  Significant interaction: the association of rain with")
        print(f"  traffic volume is statistically different during peak hours")
        print(f"  compared to off-peak hours.")
    else:
        print(f"  No significant interaction: rain's association with traffic")
        print(f"  volume does not statistically differ across time periods.")

    return {
        "interaction_coef": interact_coef,
        "interaction_p": interact_p,
        "interaction_ci": list(interact_ci),
        "restricted_r2": m_restricted.rsquared,
        "full_r2": m_full.rsquared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H4: NONLINEAR TEMPERATURE EFFECT
# ─────────────────────────────────────────────────────────────────────────────
def test_h4_nonlinear_temperature(df: pd.DataFrame) -> dict:
    """
    H4: Temperature has a nonlinear association with traffic volume.
    H₀: A linear temperature term adequately represents the relationship.

    Method: Compare linear vs polynomial regression using:
    1. F-test (partial F) for quadratic term significance
    2. AIC / BIC model comparison
    3. Adjusted R² improvement

    Formal test — NOT just visual inspection.
    """
    hypothesis_header(
        "H4: NONLINEAR TEMPERATURE-TRAFFIC RELATIONSHIP",
        h0="The linear temperature term adequately represents the "
           "temperature-traffic relationship.",
        h1="Temperature has a nonlinear (polynomial) association with "
           "traffic volume."
    )

    sub = df[["traffic_volume", "temp_c", "hour", "weekday",
              "month", "year", "holiday_flag", "rain_flag"]].dropna()

    # Model A: Linear temperature
    mA = smf.ols(
        "traffic_volume ~ temp_c + C(hour) + C(weekday) + C(month) "
        "+ C(year) + holiday_flag + rain_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # Model B: Quadratic temperature
    sub = sub.copy()
    sub["temp_c2"] = sub["temp_c"] ** 2

    mB = smf.ols(
        "traffic_volume ~ temp_c + temp_c2 + C(hour) + C(weekday) "
        "+ C(month) + C(year) + holiday_flag + rain_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # Model C: Cubic temperature
    sub["temp_c3"] = sub["temp_c"] ** 3
    mC = smf.ols(
        "traffic_volume ~ temp_c + temp_c2 + temp_c3 + C(hour) "
        "+ C(weekday) + C(month) + C(year) + holiday_flag + rain_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # Results
    quad_coef = mB.params.get("temp_c2", np.nan)
    quad_p    = mB.pvalues.get("temp_c2", np.nan)
    quad_ci   = mB.conf_int().loc["temp_c2"] \
                if "temp_c2" in mB.conf_int().index else [np.nan, np.nan]

    print(f"\n  MODEL COMPARISON:")
    print(f"  {'Model':<20s}  {'R²':>7s}  {'Adj R²':>8s}  {'AIC':>12s}  {'BIC':>12s}")
    for m, name in [(mA, "A: Linear"), (mB, "B: Quadratic"), (mC, "C: Cubic")]:
        print(f"  {name:<20s}  {m.rsquared:.4f}  {m.rsquared_adj:.6f}  "
              f"{m.aic:>12.2f}  {m.bic:>12.2f}")

    print(f"\n  Quadratic term (temp_c²) coefficient: {quad_coef:.4f}")
    print(f"  p-value: {quad_p:.4e}")
    print(f"  95% CI:  [{quad_ci[0]:.4f}, {quad_ci[1]:.4f}]")
    print(f"\n  Decision: {significance_decision(quad_p)}")
    print(f"\n  AIC difference (A→B): {mA.aic - mB.aic:.2f}  "
          f"(positive = B is better)")
    print(f"\n  INTERPRETATION:")
    if quad_p < ALPHA:
        print(f"  The quadratic temperature term is statistically significant.")
        print(f"  Temperature has a nonlinear (inverted-U or U-shaped)")
        print(f"  association with traffic volume, after controlling for")
        print(f"  calendar and other weather variables.")
    else:
        print(f"  The quadratic term is not statistically significant.")
        print(f"  We fail to reject H₀: a linear term is adequate.")

    return {
        "linear_r2": mA.rsquared, "quad_r2": mB.rsquared,
        "quad_coef": quad_coef, "quad_p": quad_p,
        "quad_ci": list(quad_ci),
        "aic_linear": mA.aic, "aic_quad": mB.aic,
        "bic_linear": mA.bic, "bic_quad": mB.bic,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H5: TIME vs WEATHER EXPLANATORY POWER
# ─────────────────────────────────────────────────────────────────────────────
def test_h5_explanatory_power(df: pd.DataFrame) -> dict:
    """
    H5: Time/calendar variables explain more variation than weather variables.
    H₀: Weather variables do not add meaningful explanatory power beyond time.

    Method: Nested model comparison using ΔR² (increment in R²).

    Models:
        M1 (Time only):    traffic_volume ~ C(hour) + C(weekday) + C(month)
                           + C(year) + holiday_flag
        M2 (Weather only): traffic_volume ~ temp_c + rain_1h + snow_1h
                           + clouds_all
        M3 (Combined):     M1 + M2

    ΔR²(M1→M3) = R²(M3) − R²(M1)
        → Incremental explanatory power of weather AFTER time.

    F-test for ΔR² significance:
        F = [(R²_full − R²_reduced) / q] / [(1 − R²_full) / (n − p − 1)]
        where q = number of added predictors.
    """
    hypothesis_header(
        "H5: CALENDAR vs WEATHER EXPLANATORY POWER",
        h0="Weather variables do not add meaningful explanatory power "
           "beyond time/calendar variables.",
        h1="Time/calendar variables explain more variation in traffic "
           "volume than weather variables alone; weather adds incremental "
           "but smaller explanatory power."
    )

    sub = df[["traffic_volume", "temp_c", "rain_1h", "snow_1h",
              "clouds_all", "hour", "weekday", "month", "year",
              "holiday_flag"]].dropna()
    n = len(sub)

    # M1: Time only
    m1 = smf.ols(
        "traffic_volume ~ C(hour) + C(weekday) + C(month) "
        "+ C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # M2: Weather only
    m2 = smf.ols(
        "traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # M3: Combined
    m3 = smf.ols(
        "traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all "
        "+ C(hour) + C(weekday) + C(month) + C(year) + holiday_flag",
        data=sub
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    # ΔR² computations
    delta_r2_time_to_combined = m3.rsquared - m1.rsquared
    delta_r2_weather_to_combined = m3.rsquared - m2.rsquared

    # F-test: does adding weather to M1 significantly improve fit?
    q = 4   # temp_c, rain_1h, snow_1h, clouds_all
    p_params = m3.df_model   # number of regressors
    F = ((m3.rsquared - m1.rsquared) / q) / \
        ((1 - m3.rsquared) / (n - p_params - 1))
    p_F = 1 - stats.f.cdf(F, q, n - p_params - 1)

    print(f"\n  {'Model':<25s}  {'R²':>7s}  {'Adj R²':>8s}  {'AIC':>12s}")
    for m, name in [(m1, "M1 (Time only)"),
                    (m2, "M2 (Weather only)"),
                    (m3, "M3 (Combined)")]:
        print(f"  {name:<25s}  {m.rsquared:.4f}  {m.rsquared_adj:.6f}  "
              f"{m.aic:>12.2f}")

    print(f"\n  ΔR² (Time → Combined):    {delta_r2_time_to_combined:.4f}  "
          f"(weather adds this much beyond time)")
    print(f"  ΔR² (Weather → Combined): {delta_r2_weather_to_combined:.4f}  "
          f"(time adds this much beyond weather)")
    print(f"\n  F-test (weather added to M1): F = {F:.4f},  p = {p_F:.4e}")
    print(f"  Decision: {significance_decision(p_F)}")
    print(f"\n  INTERPRETATION:")
    print(f"  M1 (time only) explains {m1.rsquared:.1%} of traffic volume variance.")
    print(f"  M2 (weather only) explains {m2.rsquared:.1%}.")
    print(f"  M3 (combined) explains {m3.rsquared:.1%}.")
    print(f"  Weather adds ΔR² = {delta_r2_time_to_combined:.4f} beyond time variables.")
    if delta_r2_time_to_combined < 0.05:
        print(f"  This increment is modest — time/calendar variables are the")
        print(f"  dominant predictors of traffic volume.")
    else:
        print(f"  Weather contributes non-trivially to explanatory power.")
    print(f"\n  NOTE: Statistical significance of ΔR² (p = {p_F:.4e}) does")
    print(f"  not imply practical importance. Even a small but significant")
    print(f"  weather effect may have real-world operational relevance.")

    return {
        "r2_time": m1.rsquared,   "adj_r2_time": m1.rsquared_adj,
        "r2_weather": m2.rsquared, "adj_r2_weather": m2.rsquared_adj,
        "r2_combined": m3.rsquared, "adj_r2_combined": m3.rsquared_adj,
        "delta_r2_weather_contribution": delta_r2_time_to_combined,
        "delta_r2_time_contribution": delta_r2_weather_to_combined,
        "F_stat": F, "p_F": p_F,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_hypothesis_testing(df: pd.DataFrame,
                           save_path: str = None) -> dict:
    """
    Execute all five hypothesis tests and return results dictionary.
    """
    print("\n" + "="*65)
    print("HYPOTHESIS TESTING — Modules VII & VIII")
    print("="*65)

    h1 = test_h1_rain_vs_dry(df)
    h2 = test_h2_precipitation_groups(df)
    h3 = test_h3_interaction(df)
    h4 = test_h4_nonlinear_temperature(df)
    h5 = test_h5_explanatory_power(df)

    # Compile summary table
    summary = [
        {
            "Hypothesis": "H1",
            "Test": "Mann-Whitney U (unadj) + OLS (adj)",
            "p_value": h1.get("rain_adj_p", np.nan),
            "Effect_size": h1.get("r_rank_biserial", np.nan),
            "Significant": h1.get("rain_adj_p", 1) < ALPHA,
        },
        {
            "Hypothesis": "H2",
            "Test": "Kruskal-Wallis + Dunn-Holm",
            "p_value": h2.get("p_kruskal", np.nan),
            "Effect_size": h2.get("eta_squared", np.nan),
            "Significant": h2.get("p_kruskal", 1) < ALPHA,
        },
        {
            "Hypothesis": "H3",
            "Test": "OLS Interaction Term",
            "p_value": h3.get("interaction_p", np.nan),
            "Effect_size": h3.get("interaction_coef", np.nan),
            "Significant": h3.get("interaction_p", 1) < ALPHA,
        },
        {
            "Hypothesis": "H4",
            "Test": "Polynomial OLS (quadratic term F-test)",
            "p_value": h4.get("quad_p", np.nan),
            "Effect_size": h4.get("quad_coef", np.nan),
            "Significant": h4.get("quad_p", 1) < ALPHA,
        },
        {
            "Hypothesis": "H5",
            "Test": "Nested Model ΔR² + F-test",
            "p_value": h5.get("p_F", np.nan),
            "Effect_size": h5.get("delta_r2_weather_contribution", np.nan),
            "Significant": h5.get("p_F", 1) < ALPHA,
        },
    ]

    summary_df = pd.DataFrame(summary)
    print(f"\n{'='*65}")
    print("  HYPOTHESIS TESTING SUMMARY")
    print(f"{'='*65}")
    print(summary_df.to_string(index=False))

    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"\n[SAVE] Results saved → {save_path}")

    return {
        "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5,
        "summary": summary_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean = os.path.join(base, "data", "traffic_clean.csv")
    save  = os.path.join(base, "results", "statistical_results.csv")
    df = pd.read_csv(clean, parse_dates=["date_time"])
    run_hypothesis_testing(df, save_path=save)
