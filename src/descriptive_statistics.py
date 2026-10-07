"""
descriptive_statistics.py
==========================
Modules I, III — Descriptive Statistics and Random Variable Analysis
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Compute and interpret all descriptive statistics for key variables.
    For every statistic, provide the mathematical formula, computed value,
    and plain-language interpretation. Traffic volume is treated as a
    quantitative random variable X; its expected value, variance, and
    standard deviation are computed from the empirical distribution.

MATHEMATICAL FRAMEWORK:
    For a sample {x₁, x₂, ..., xₙ}:

    Mean (Expected Value):  μ̂ = (1/n) Σ xᵢ
    Variance (sample):      s² = (1/(n-1)) Σ (xᵢ − μ̂)²
    Std deviation:          s  = √s²
    Skewness (Pearson's):   γ₁ = [n/((n-1)(n-2))] Σ((xᵢ−μ̂)/s)³
    IQR:                    IQR = Q₃ − Q₁
    CV:                     CV  = s / μ̂ (coefficient of variation)
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CORE STATISTICS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def compute_statistics(series: pd.Series, name: str) -> dict:
    """
    Compute full descriptive statistics for a numeric series.

    Parameters
    ----------
    series : pd.Series — the variable
    name   : str       — variable name for display

    Returns
    -------
    dict with all statistical measures
    """
    s = series.dropna()
    n = len(s)

    # Central tendency
    mean_val   = float(s.mean())
    median_val = float(s.median())
    try:
        mode_val = float(s.mode().iloc[0])
    except Exception:
        mode_val = np.nan

    # Variability
    var_val  = float(s.var(ddof=1))      # sample variance (ddof=1)
    std_val  = float(s.std(ddof=1))      # sample std dev
    min_val  = float(s.min())
    max_val  = float(s.max())
    range_val = max_val - min_val
    q1_val   = float(s.quantile(0.25))
    q3_val   = float(s.quantile(0.75))
    iqr_val  = q3_val - q1_val
    cv_val   = std_val / mean_val if mean_val != 0 else np.nan

    # Skewness (using scipy for Fisher's definition, adjusted for bias)
    skew_val = float(stats.skew(s, bias=False))

    # Kurtosis (excess, using scipy)
    kurt_val = float(stats.kurtosis(s, bias=False))

    result = {
        "variable": name,
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "variance": var_val,
        "std_dev": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "Q1": q1_val,
        "Q3": q3_val,
        "IQR": iqr_val,
        "CV": cv_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }
    return result


def print_statistics(stats_dict: dict) -> None:
    """Pretty-print a statistics dictionary with interpretation."""
    name = stats_dict["variable"]
    print(f"\n{'─'*60}")
    print(f"  DESCRIPTIVE STATISTICS: {name.upper()}")
    print(f"{'─'*60}")
    print(f"  n (sample size)     : {stats_dict['n']:,}")
    print(f"  Mean  (μ̂)           : {stats_dict['mean']:.4f}")
    print(f"  Median              : {stats_dict['median']:.4f}")
    print(f"  Mode                : {stats_dict['mode']:.4f}")
    print(f"  Variance (s²)       : {stats_dict['variance']:.4f}")
    print(f"  Std Dev (s)         : {stats_dict['std_dev']:.4f}")
    print(f"  CV (s/μ̂)            : {stats_dict['CV']:.4f}")
    print(f"  Min                 : {stats_dict['min']:.4f}")
    print(f"  Max                 : {stats_dict['max']:.4f}")
    print(f"  Range               : {stats_dict['range']:.4f}")
    print(f"  Q1 (25th pct)       : {stats_dict['Q1']:.4f}")
    print(f"  Q3 (75th pct)       : {stats_dict['Q3']:.4f}")
    print(f"  IQR                 : {stats_dict['IQR']:.4f}")
    print(f"  Skewness (γ₁)       : {stats_dict['skewness']:.4f}")
    print(f"  Excess Kurtosis     : {stats_dict['kurtosis']:.4f}")

    # Interpretation
    sk = stats_dict["skewness"]
    skew_desc = ("approximately symmetric" if abs(sk) < 0.5
                 else ("moderately right-skewed" if sk > 0
                       else "moderately left-skewed"))
    if abs(sk) > 1:
        skew_desc = "right-skewed" if sk > 0 else "left-skewed"

    print(f"\n  INTERPRETATION:")
    print(f"  Distribution is {skew_desc} (γ₁ = {sk:.3f}).")
    if stats_dict["mean"] > stats_dict["median"]:
        print(f"  Mean ({stats_dict['mean']:.2f}) > Median ({stats_dict['median']:.2f}): "
              "right tail pulls the mean upward.")
    elif stats_dict["mean"] < stats_dict["median"]:
        print(f"  Mean ({stats_dict['mean']:.2f}) < Median ({stats_dict['median']:.2f}): "
              "left tail pulls the mean downward.")
    else:
        print("  Mean ≈ Median: consistent with near-symmetry.")

    cv = stats_dict["CV"]
    if not np.isnan(cv):
        if cv < 0.1:
            print(f"  CV = {cv:.3f}: Low variability relative to the mean.")
        elif cv < 0.3:
            print(f"  CV = {cv:.3f}: Moderate variability relative to the mean.")
        else:
            print(f"  CV = {cv:.3f}: High variability relative to the mean.")


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM VARIABLE ANALYSIS (Module III)
# ─────────────────────────────────────────────────────────────────────────────
def random_variable_analysis(df: pd.DataFrame) -> dict:
    """
    Treat traffic_volume as a discrete quantitative random variable X.

    Mathematical treatment:
    -----------------------
    The empirical distribution of X is constructed from observed frequencies.
    For a discrete random variable:

        E[X] = Σ xᵢ · P(X = xᵢ)   (Expected Value)
        Var(X) = E[X²] − (E[X])²   (Variance of X)
        SD(X)  = √Var(X)

    Since traffic_volume takes many integer values, we use the sample
    estimates as proxies:
        E[X] ≈ x̄  (sample mean)
        Var(X) ≈ s² (sample variance)
    """
    X = df["traffic_volume"].dropna()
    n = len(X)

    # Empirical PMF approximation (grouped into 100 bins)
    probs, edges = np.histogram(X, bins=100, density=False)
    probs = probs / n   # relative frequency = empirical probability

    midpoints = 0.5 * (edges[:-1] + edges[1:])
    E_X  = float(np.sum(midpoints * probs))             # E[X]
    E_X2 = float(np.sum(midpoints**2 * probs))          # E[X²]
    Var_X = E_X2 - E_X**2                               # Var(X) = E[X²] - E[X]²
    SD_X  = np.sqrt(Var_X)

    print(f"\n{'='*60}")
    print("  RANDOM VARIABLE ANALYSIS — Traffic Volume (X)")
    print(f"{'='*60}")
    print(f"  Treatment: Traffic volume is a discrete quantitative r.v.")
    print(f"             Observations are treated as realisations of X.")
    print(f"\n  Using empirical distribution (100-bin histogram):")
    print(f"  E[X]   = Σ xᵢ · P(X=xᵢ) ≈ {E_X:,.1f} vehicles/hour")
    print(f"  E[X²]  = Σ xᵢ² · P(X=xᵢ) ≈ {E_X2:,.1f}")
    print(f"  Var(X) = E[X²] − (E[X])² ≈ {Var_X:,.1f}")
    print(f"  SD(X)  = √Var(X) ≈ {SD_X:,.1f} vehicles/hour")
    print(f"\n  Sample estimates (ddof=1):")
    print(f"  x̄ = {X.mean():.2f}  |  s² = {X.var(ddof=1):.2f}  |  s = {X.std(ddof=1):.2f}")
    print(f"\n  INTERPRETATION:")
    print(f"  On average, approximately {X.mean():.0f} vehicles pass per hour.")
    print(f"  A standard deviation of {X.std():.0f} vehicles/hr reflects")
    print(f"  substantial variability driven by peak/off-peak cycles.")

    return {"E_X": E_X, "Var_X": Var_X, "SD_X": SD_X}


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL DISTRIBUTION ASSESSMENT (Module V)
# ─────────────────────────────────────────────────────────────────────────────
def assess_normality(df: pd.DataFrame, variable: str) -> dict:
    """
    Assess whether a variable is approximately normally distributed.

    Method: Do NOT rely on a single normality test. Use:
    1. Shapiro-Wilk (n ≤ 5000 subsample, due to test limitations)
    2. Kolmogorov-Smirnov test
    3. Skewness / Kurtosis inspection
    4. Q-Q plot (generated in visualization.py)

    CRITICAL NOTE: With n > 40,000, normality tests have very high power
    and will almost always reject H₀ even for trivial departures from
    normality. Graphical inspection and skewness/kurtosis are therefore
    the primary evidence.
    """
    series = df[variable].dropna()
    n = len(series)

    # Shapiro-Wilk on subsample (max 5000)
    subsample = series.sample(min(5000, n), random_state=42)
    sw_stat, sw_p = stats.shapiro(subsample)

    # KS test
    z = (series - series.mean()) / series.std()
    ks_stat, ks_p = stats.kstest(z, stats.norm.cdf)

    skew_val = float(stats.skew(series, bias=False))
    kurt_val = float(stats.kurtosis(series, bias=False))

    print(f"\n  NORMALITY ASSESSMENT: {variable}")
    print(f"  Shapiro-Wilk (n=5000 subsample): W={sw_stat:.4f}, p={sw_p:.4e}")
    print(f"  Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_p:.4e}")
    print(f"  Skewness γ₁ = {skew_val:.4f}  |  Excess kurtosis = {kurt_val:.4f}")
    print(f"\n  INTERPRETATION:")
    print(f"  With n={n:,}, normality tests have near-perfect power.")
    print(f"  A significant p-value does NOT mean the departure is")
    print(f"  practically important. Skewness = {skew_val:.3f} suggests")
    if abs(skew_val) < 0.5:
        print(f"  the distribution is approximately symmetric. The normal")
        print(f"  distribution is a reasonable working assumption for this variable.")
    else:
        print(f"  the distribution departs from normality. Non-parametric")
        print(f"  methods or transformations are preferred.")

    return {
        "variable": variable,
        "sw_stat": sw_stat, "sw_p": sw_p,
        "ks_stat": ks_stat, "ks_p": ks_p,
        "skewness": skew_val, "kurtosis": kurt_val,
        "n": n
    }


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY TABLE (Module I)
# ─────────────────────────────────────────────────────────────────────────────
def frequency_table(df: pd.DataFrame, variable: str,
                    bins: int = 10) -> pd.DataFrame:
    """
    Create a frequency distribution table for a numeric variable.

    Columns: Class, Frequency, Relative Frequency, Cumulative Freq.
    """
    s = df[variable].dropna()
    freq, edges = np.histogram(s, bins=bins)
    rel_freq = freq / freq.sum()
    cum_freq  = np.cumsum(rel_freq)
    labels = [f"[{edges[i]:.1f}, {edges[i+1]:.1f})" for i in range(len(edges)-1)]

    table = pd.DataFrame({
        "Class interval": labels,
        "Frequency":       freq,
        "Relative Freq":  rel_freq.round(4),
        "Cumulative Freq": cum_freq.round(4),
    })
    print(f"\n  FREQUENCY TABLE: {variable} ({bins} classes)")
    print(table.to_string(index=False))
    return table


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_descriptive_statistics(df: pd.DataFrame,
                               save_path: str = None) -> pd.DataFrame:
    """
    Run all descriptive statistics and return summary DataFrame.

    Parameters
    ----------
    df        : cleaned DataFrame
    save_path : optional path to save CSV results

    Returns
    -------
    pd.DataFrame — summary table of all statistics
    """
    print("\n" + "="*65)
    print("DESCRIPTIVE STATISTICS & RANDOM VARIABLE ANALYSIS")
    print("="*65)

    variables = {
        "traffic_volume": "Traffic Volume (vehicles/hr)",
        "temp_c":         "Temperature (°C)",
        "rain_1h":        "Rainfall (mm/hr)",
        "snow_1h":        "Snowfall (mm/hr)",
        "clouds_all":     "Cloud Coverage (%)",
    }

    results = []
    for col, label in variables.items():
        s = compute_statistics(df[col], label)
        print_statistics(s)
        results.append(s)

    # Random variable treatment of traffic volume
    rv_stats = random_variable_analysis(df)

    # Normality assessment
    print(f"\n{'='*65}")
    print("NORMALITY ASSESSMENT (Module V — Continuous Distributions)")
    print("="*65)
    for col in ["traffic_volume", "temp_c"]:
        assess_normality(df, col)

    # Frequency tables
    print(f"\n{'='*65}")
    print("FREQUENCY DISTRIBUTION TABLES (Module I)")
    print("="*65)
    frequency_table(df, "traffic_volume", bins=12)

    summary_df = pd.DataFrame(results)
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"\n[SAVE] Descriptive statistics saved → {save_path}")

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean = os.path.join(base, "data", "traffic_clean.csv")
    save  = os.path.join(base, "results", "descriptive_statistics.csv")

    df = pd.read_csv(clean, parse_dates=["date_time"])
    run_descriptive_statistics(df, save_path=save)
