"""
correlation_analysis.py
=======================
Module IX — Correlation and Covariance
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Compute and interpret covariance, Pearson correlation, and Spearman
    rank correlation between traffic volume and weather variables.
    All coefficients include p-values, confidence intervals, and
    plain-language interpretations.

MATHEMATICAL FRAMEWORK:
─────────────────────────────────────────────────────────────
COVARIANCE:
    Cov(X, Y) = (1/(n-1)) Σ (xᵢ − x̄)(yᵢ − ȳ)

    - Positive Cov: X and Y tend to move together
    - Negative Cov: X and Y tend to move in opposite directions
    - Units: product of X and Y units; not scale-invariant

PEARSON CORRELATION:
    r = Cov(X,Y) / (sₓ · sᵧ)
      = Σ (xᵢ−x̄)(yᵢ−ȳ) / √[Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²]

    Range: [−1, 1]; scale-invariant
    Assumptions:
      (i)  Both variables are continuous
      (ii) Relationship is approximately linear
      (iii) Both variables are approximately normally distributed
      (iv) Homoscedasticity

    t-statistic for H₀: ρ = 0
        t = r√(n-2) / √(1-r²)  ~  t(n-2) under H₀

SPEARMAN RANK CORRELATION:
    rₛ = 1 − (6 Σdᵢ²) / (n(n²-1))   [for no tied ranks]
    (scipy implementation handles ties via Pearson on ranks)

    Advantages over Pearson:
    - Does not require linearity
    - Robust to outliers
    - Appropriate for skewed distributions

CRITICAL CAUTION:
    Correlation ≠ Causation.
    A high correlation coefficient describes the strength of a
    statistical association; it does not establish that one
    variable influences the other.
─────────────────────────────────────────────────────────────
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import json

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def interpret_r(r: float) -> str:
    """Return qualitative interpretation of correlation magnitude."""
    a = abs(r)
    if a < 0.10:
        return "negligible"
    elif a < 0.30:
        return "weak"
    elif a < 0.50:
        return "moderate"
    elif a < 0.70:
        return "moderately strong"
    else:
        return "strong"


def pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    """
    Fisher z-transformation confidence interval for Pearson r.

        z = arctanh(r) = 0.5 · ln((1+r)/(1-r))
        SE(z) = 1 / √(n-3)
        95% CI for z: z ± z_α/2 · SE(z)
        Transform back: r_ci = tanh(z ± ...)
    """
    z     = np.arctanh(r)
    se    = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo    = np.tanh(z - z_crit * se)
    hi    = np.tanh(z + z_crit * se)
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# COVARIANCE
# ─────────────────────────────────────────────────────────────────────────────
def compute_covariance(df: pd.DataFrame,
                       target: str = "traffic_volume") -> pd.Series:
    """
    Compute sample covariance between target and all numeric weather variables.

        Cov(X,Y) = (1/(n-1)) Σ (xᵢ−x̄)(yᵢ−ȳ)
    """
    weather_vars = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
    sub = df[[target] + weather_vars].dropna()

    cov_matrix = sub.cov()                 # pandas uses ddof=1
    cov_series = cov_matrix[target].drop(target)

    print(f"\n{'='*60}")
    print("  COVARIANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Cov(X,Y) = (1/(n-1)) Σ (xᵢ−x̄)(yᵢ−ȳ)")
    print(f"  n = {len(sub):,}\n")

    for var, cov_val in cov_series.items():
        direction = "positive" if cov_val > 0 else "negative"
        print(f"  Cov(traffic_volume, {var:<15s}) = {cov_val:>12.4f}  "
              f"[{direction}]")

    print(f"\n  NOTE: Covariance units are the product of both variable units.")
    print(f"  Scale-invariant correlation is more interpretable.")

    return cov_series


# ─────────────────────────────────────────────────────────────────────────────
# PEARSON CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_pearson(df: pd.DataFrame,
                    target: str = "traffic_volume") -> pd.DataFrame:
    """
    Compute Pearson r, t-statistic, p-value, and 95% CI for each pair.
    """
    weather_vars = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
    rows = []

    print(f"\n{'='*60}")
    print("  PEARSON CORRELATION (r)")
    print(f"{'='*60}")
    print(f"  r = Cov(X,Y) / (sₓ·sᵧ)")
    print(f"  H₀: ρ = 0  |  H₁: ρ ≠ 0")
    print(f"  t-statistic: t = r√(n-2)/√(1-r²), df = n-2\n")

    for var in weather_vars:
        sub = df[[target, var]].dropna()
        n   = len(sub)
        r, p = stats.pearsonr(sub[target], sub[var])
        ci_lo, ci_hi = pearson_ci(r, n)
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)

        direction = "positive" if r > 0 else "negative"
        strength  = interpret_r(r)

        print(f"  {var}:")
        print(f"    r = {r:.4f}  ({direction}, {strength})")
        print(f"    t = {t_stat:.4f},  df = {n-2},  p = {p:.4e}")
        print(f"    95% CI (Fisher z): [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    Interpretation: {strength.capitalize()} {direction} linear "
              f"association with traffic volume.")
        if p < 0.05:
            print(f"    → Statistically significant (p < 0.05)")
        else:
            print(f"    → Not statistically significant (p ≥ 0.05)")
        print()

        rows.append({
            "variable": var,
            "pearson_r": r,
            "t_stat": t_stat,
            "p_value": p,
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "direction": direction,
            "strength": strength,
            "n": n,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SPEARMAN CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_spearman(df: pd.DataFrame,
                     target: str = "traffic_volume") -> pd.DataFrame:
    """
    Compute Spearman rank correlation rₛ for each pair.

    Why Spearman in addition to Pearson?
    - Traffic volume and precipitation are heavily skewed
    - Rain/snow distributions are highly non-normal (mostly zeros)
    - Spearman is robust to outliers and does not require linearity
    - Comparing Pearson vs Spearman reveals whether any Pearson
      result is driven by outliers or non-linearity
    """
    weather_vars = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
    rows = []

    print(f"\n{'='*60}")
    print("  SPEARMAN RANK CORRELATION (rₛ)")
    print(f"{'='*60}")
    print(f"  rₛ computed on ranks; robust to outliers and non-linearity.")
    print(f"  Comparison with Pearson r reveals non-linearity influence.\n")

    for var in weather_vars:
        sub = df[[target, var]].dropna()
        n   = len(sub)
        rs, p = stats.spearmanr(sub[target], sub[var])
        strength = interpret_r(rs)
        direction = "positive" if rs > 0 else "negative"

        print(f"  {var}:")
        print(f"    rₛ = {rs:.4f}  ({direction}, {strength}),  p = {p:.4e}")

        rows.append({
            "variable": var,
            "spearman_rs": rs,
            "p_value": p,
            "direction": direction,
            "strength": strength,
            "n": n,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE: PEARSON vs SPEARMAN
# ─────────────────────────────────────────────────────────────────────────────
def compare_correlations(pearson_df: pd.DataFrame,
                         spearman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Pearson and Spearman results side by side.

    A large difference between r and rₛ suggests the Pearson result is
    influenced by non-linearity, outliers, or distributional violations.
    """
    merged = pearson_df[["variable", "pearson_r", "p_value"]].merge(
        spearman_df[["variable", "spearman_rs",
                      "p_value"]].rename(columns={"p_value": "p_spearman"}),
        on="variable"
    )
    merged["difference"] = (merged["pearson_r"] - merged["spearman_rs"]).round(4)
    merged["interpretation"] = merged["difference"].apply(
        lambda d: "Consistent (linear)" if abs(d) < 0.05
        else ("Non-linearity/outlier influence likely" if abs(d) < 0.15
              else "Strong non-linearity or outlier influence")
    )

    print(f"\n{'='*60}")
    print("  PEARSON vs SPEARMAN COMPARISON")
    print(f"{'='*60}")
    print(merged.to_string(index=False))
    print(f"\n  NOTE: Where r ≈ rₛ, the relationship is approximately linear.")
    print(f"  Where |r - rₛ| is large, non-linearity or outliers influence r.")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and display full Pearson correlation matrix for numeric variables.
    """
    cols = ["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all",
            "hour", "weekday", "month"]
    sub  = df[cols].dropna()
    cmat = sub.corr(method="pearson").round(4)

    print(f"\n{'='*60}")
    print("  PEARSON CORRELATION MATRIX")
    print(f"{'='*60}")
    print(cmat.to_string())

    return cmat


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_correlation_analysis(df: pd.DataFrame,
                             save_path: str = None) -> dict:
    """
    Execute all correlation analyses.

    Returns dict with all results.
    """
    print("\n" + "="*65)
    print("CORRELATION AND COVARIANCE ANALYSIS — Module IX")
    print("="*65)

    cov_series = compute_covariance(df)
    pearson_df = compute_pearson(df)
    spearman_df = compute_spearman(df)
    comparison  = compare_correlations(pearson_df, spearman_df)
    cmat        = correlation_matrix(df)

    print(f"\n{'='*60}")
    print("  CRITICAL CAUTION")
    print(f"{'='*60}")
    print("  The correlations reported above describe the strength and")
    print("  direction of STATISTICAL ASSOCIATIONS between variables.")
    print("  They do NOT establish causation. Weather conditions may")
    print("  correlate with traffic volume because both share common")
    print("  confounders (season, time of day, etc.).")

    if save_path:
        pearson_df.to_csv(save_path, index=False)
        print(f"\n[SAVE] Pearson results saved → {save_path}")

    return {
        "covariance":   cov_series,
        "pearson":      pearson_df,
        "spearman":     spearman_df,
        "comparison":   comparison,
        "corr_matrix":  cmat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean = os.path.join(base, "data", "traffic_clean.csv")
    save  = os.path.join(base, "results", "correlation_results.csv")
    df = pd.read_csv(clean, parse_dates=["date_time"])
    run_correlation_analysis(df, save_path=save)
