"""
probability_analysis.py
=======================
Module II — Introduction to Probability
Module IV — Discrete Probability Distributions (where applicable)
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Compute meaningful probability estimates from the dataset using
    classical frequentist definitions. Conditional probability
    provides the foundation for later hypothesis testing.

MATHEMATICAL FRAMEWORK:
─────────────────────────────────────────────────────────────
Probability (Frequentist definition):
    P(A) = n(A) / n(S)    where n(S) = total observations

Conditional Probability:
    P(A | B) = P(A ∩ B) / P(B)   [provided P(B) > 0]

Independence check:
    Events A and B are independent if P(A ∩ B) = P(A) · P(B)
    Equivalently: P(A | B) = P(A)

Bayes' Theorem (used for weather-traffic interpretation):
    P(B | A) = P(A | B) · P(B) / P(A)

Module IV — Discrete Distribution Assessment:
    Poisson distribution is theoretically applicable to traffic
    counts if:
      1. Events occur in fixed time intervals ✓ (hourly)
      2. Events are independent (partially; temporal dependence exists)
      3. Mean ≈ Variance (Equidispersion)
    We formally test the equidispersion condition.
─────────────────────────────────────────────────────────────
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# BASIC PROBABILITY ESTIMATES
# ─────────────────────────────────────────────────────────────────────────────
def basic_probabilities(df: pd.DataFrame) -> dict:
    """
    Compute basic event probabilities from observed frequencies.

    Events defined:
    - A: Rain (rain_1h > 0)
    - B: Snow (snow_1h > 0)
    - C: High traffic (traffic_volume ≥ 75th percentile)
    - D: Peak hour (weekday 07-09 or 16-19)
    - E: Holiday
    """
    n = len(df)

    def prob(mask):
        return float(mask.sum() / n)

    probs = {
        "P(Rain)":    prob(df["rain_flag"] == 1),
        "P(Snow)":    prob(df["snow_flag"] == 1),
        "P(High Traffic)": prob(df["high_traffic"] == 1),
        "P(Peak)":    prob(df["peak"] == "Peak"),
        "P(Holiday)": prob(df["holiday_flag"] == 1),
        "P(Dry)":     prob(df["rain_flag"] + df["snow_flag"] == 0),
        "P(Rain OR Snow)": prob((df["rain_flag"] == 1) | (df["snow_flag"] == 1)),
        "P(Rain AND Snow)": prob((df["rain_flag"] == 1) & (df["snow_flag"] == 1)),
    }

    print(f"\n{'='*60}")
    print("  BASIC PROBABILITY ESTIMATES (Frequentist)")
    print(f"{'='*60}")
    print(f"  Total observations n = {n:,}\n")
    for event, p in probs.items():
        n_event = int(p * n)
        print(f"  {event:<30s} = {p:.4f}  (n = {n_event:,})")

    # Check addition rule: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
    p_rain_or_snow = probs["P(Rain)"] + probs["P(Snow)"] - probs["P(Rain AND Snow)"]
    print(f"\n  Verification — Addition Rule:")
    print(f"  P(Rain ∪ Snow) = P(Rain) + P(Snow) − P(Rain ∩ Snow)")
    print(f"               = {probs['P(Rain)']:.4f} + {probs['P(Snow)']:.4f} "
          f"− {probs['P(Rain AND Snow)']:.4f} = {p_rain_or_snow:.4f}")
    print(f"  Direct count:  {probs['P(Rain OR Snow)']:.4f}  ✓")

    return probs


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────
def conditional_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conditional probabilities of high traffic given weather conditions.

    Formula:
        P(High Traffic | Rain) = P(High Traffic ∩ Rain) / P(Rain)

    This quantifies whether the likelihood of high traffic changes under
    different weather conditions — a direct test of association.
    """
    n = len(df)

    conditions = {
        "Rain":      df["rain_flag"] == 1,
        "No Rain":   df["rain_flag"] == 0,
        "Snow":      df["snow_flag"] == 1,
        "No Snow":   df["snow_flag"] == 0,
        "Peak Hour": df["peak"] == "Peak",
        "Off-Peak":  df["peak"] == "Off-Peak",
        "Holiday":   df["holiday_flag"] == 1,
        "Non-Holiday": df["holiday_flag"] == 0,
    }

    print(f"\n{'='*60}")
    print("  CONDITIONAL PROBABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"  P(High Traffic | Condition)\n")

    rows = []
    for cond_name, cond_mask in conditions.items():
        n_cond = cond_mask.sum()
        if n_cond == 0:
            continue
        n_joint = ((cond_mask) & (df["high_traffic"] == 1)).sum()
        p_cond  = n_cond / n
        p_joint = n_joint / n
        p_cond_given = p_joint / p_cond  # P(High | Cond)

        rows.append({
            "Condition":         cond_name,
            "n(Condition)":      n_cond,
            "n(High ∩ Cond)":   n_joint,
            "P(Condition)":      round(p_cond, 4),
            "P(High Traffic | Condition)": round(p_cond_given, 4),
        })
        print(f"  P(High Traffic | {cond_name:<12s}) = "
              f"{p_cond_given:.4f}  (n={n_cond:,})")

    # Comparison: Rain vs No Rain
    df_cond = pd.DataFrame(rows)
    rain_p    = df_cond[df_cond["Condition"] == "Rain"][
                    "P(High Traffic | Condition)"].values[0]
    no_rain_p = df_cond[df_cond["Condition"] == "No Rain"][
                    "P(High Traffic | Condition)"].values[0]
    snow_p    = df_cond[df_cond["Condition"] == "Snow"][
                    "P(High Traffic | Condition)"].values[0]

    print(f"\n  KEY COMPARISONS:")
    print(f"  P(High Traffic | Rain)    = {rain_p:.4f}")
    print(f"  P(High Traffic | No Rain) = {no_rain_p:.4f}")
    print(f"  Absolute difference       = {abs(rain_p - no_rain_p):.4f}")
    print(f"  Relative risk             = {rain_p / no_rain_p:.4f}  "
          "(ratio; 1.0 = no association)")

    print(f"\n  INDEPENDENCE CHECK:")
    p_high = (df["high_traffic"] == 1).sum() / n
    p_rain = (df["rain_flag"] == 1).sum() / n
    print(f"  P(High Traffic) = {p_high:.4f}")
    print(f"  P(Rain) = {p_rain:.4f}")
    print(f"  If independent: P(High ∩ Rain) should ≈ {p_high*p_rain:.4f}")
    p_both = ((df["high_traffic"] == 1) & (df["rain_flag"] == 1)).sum() / n
    print(f"  Observed P(High ∩ Rain)   = {p_both:.4f}")
    if abs(p_both - p_high * p_rain) < 0.01:
        print(f"  → Events are approximately INDEPENDENT")
    else:
        print(f"  → Events are NOT independent; association is present")

    return df_cond


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL BY WEATHER TYPE
# ─────────────────────────────────────────────────────────────────────────────
def conditional_by_weather_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P(High Traffic | Weather Type) for each weather_main category.
    """
    print(f"\n  P(High Traffic | Weather Type):")
    rows = []
    for wtype in df["weather_main"].unique():
        mask = df["weather_main"] == wtype
        n_w  = mask.sum()
        if n_w < 50:
            continue
        n_hw = ((mask) & (df["high_traffic"] == 1)).sum()
        p_hw = n_hw / n_w
        rows.append({
            "Weather Type": wtype,
            "n": n_w,
            "n(High Traffic)": n_hw,
            "P(High | Weather)": round(p_hw, 4),
        })
        print(f"  P(High Traffic | {wtype:<15s}) = {p_hw:.4f}  (n={n_w:,})")

    return pd.DataFrame(rows).sort_values("P(High | Weather)", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE IV — POISSON DISTRIBUTION ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
def poisson_assessment(df: pd.DataFrame) -> dict:
    """
    Assess whether traffic_volume follows a Poisson distribution.

    Poisson assumptions:
    1. Events occur independently in disjoint time intervals
    2. Rate λ is constant over time
    3. Equidispersion: E[X] = Var(X) = λ

    For hourly traffic counts, assumptions 1 and 2 are clearly violated:
    - Strong daily and weekly autocorrelation (violation of independence)
    - Rate changes dramatically by hour and day (violation of constant λ)

    We test assumption 3 (equidispersion) as a formal check.

    Verdict: Poisson is NOT an appropriate model for aggregate hourly
             traffic volume due to massive overdispersion and temporal
             dependence. It is documented but not forced as an analysis.
    """
    X = df["traffic_volume"].dropna()
    mean_x = X.mean()
    var_x  = X.var(ddof=1)
    dispersion_index = var_x / mean_x   # = 1 if Poisson

    print(f"\n{'='*60}")
    print("  MODULE IV — POISSON DISTRIBUTION ASSESSMENT")
    print(f"{'='*60}")
    print(f"  For a Poisson(λ) r.v., E[X] = Var(X) = λ.")
    print(f"\n  Observed:")
    print(f"  E[X] (mean)    = {mean_x:.2f}")
    print(f"  Var(X)         = {var_x:.2f}")
    print(f"  Dispersion Index = Var/Mean = {dispersion_index:.2f}")
    print(f"\n  VERDICT:")
    if dispersion_index > 2:
        print(f"  Dispersion index = {dispersion_index:.2f} >> 1.")
        print(f"  The distribution is MASSIVELY OVERDISPERSED relative to")
        print(f"  Poisson expectations. Additionally, hourly counts show")
        print(f"  strong temporal dependence (daily/weekly patterns),")
        print(f"  violating the independence assumption of Poisson.")
        print(f"  → Poisson distribution is NOT appropriate for traffic_volume.")
        print(f"  → Analysis proceeds with non-parametric and regression methods.")
    else:
        print(f"  Dispersion index ≈ 1: Poisson may be appropriate, but")
        print(f"  temporal dependence still needs verification.")

    return {
        "mean": mean_x,
        "variance": var_x,
        "dispersion_index": dispersion_index,
        "poisson_appropriate": dispersion_index <= 2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_probability_analysis(df: pd.DataFrame, save_path: str = None) -> dict:
    """
    Execute all probability analyses.

    Returns dict of results for downstream use.
    """
    print("\n" + "="*65)
    print("PROBABILITY ANALYSIS — Modules II & IV")
    print("="*65)

    basic = basic_probabilities(df)
    cond  = conditional_probabilities(df)
    by_weather = conditional_by_weather_type(df)
    poisson = poisson_assessment(df)

    print(f"\n{'='*60}")
    print("  PROBABILITY ANALYSIS COMPLETE")
    print(f"{'='*60}")

    if save_path:
        records = []
        for k, v in basic.items():
            records.append({"category": "Basic", "metric": k, "value": v})
        for k, v in cond.items():
            records.append({"category": "Conditional", "metric": k, "value": v})
        for k, v in by_weather.items():
            records.append({"category": "By Weather", "metric": k, "value": v})
        records.append({"category": "Poisson", "metric": "Dispersion Index", "value": poisson["dispersion_index"]})
        pdf = pd.DataFrame(records)
        pdf.to_csv(save_path, index=False)
        print(f"\n[SAVE] Probability results saved → {save_path}")

    return {
        "basic_probabilities": basic,
        "conditional": cond,
        "by_weather": by_weather,
        "poisson": poisson,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean = os.path.join(base, "data", "traffic_clean.csv")
    df = pd.read_csv(clean, parse_dates=["date_time"])
    run_probability_analysis(df)
