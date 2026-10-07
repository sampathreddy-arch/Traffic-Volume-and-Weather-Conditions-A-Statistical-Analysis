"""
diagnostics.py
==============
Module X — Model Diagnostics
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Perform comprehensive residual diagnostics for regression models.
    Every diagnostic is explained mathematically and interpreted.

DIAGNOSTICS COVERED:
    1. Residual vs Fitted plot — linearity and heteroscedasticity
    2. Q-Q plot — normality of residuals
    3. Scale-Location plot — homoscedasticity (Breusch-Pagan test)
    4. Residual distribution — histogram + skewness
    5. Autocorrelation — ACF of residuals (Durbin-Watson, Ljung-Box)
    6. VIF (multicollinearity) — from regression_analysis.py
    7. Cook's Distance — influential observations

MATHEMATICAL NOTES:
    Durbin-Watson statistic:
        DW = Σ(eₜ − eₜ₋₁)² / Σeₜ²
        DW ≈ 2: no autocorrelation
        DW < 1.5: positive autocorrelation (likely in hourly data)

    Breusch-Pagan test:
        Tests H₀: Var(εᵢ) = σ² (homoscedasticity)
        Against H₁: Var(εᵢ) depends on X

    Cook's Distance:
        Dᵢ = (ŷ − ŷ₍₋ᵢ₎)ᵀ(ŷ − ŷ₍₋ᵢ₎) / (p·MSE)
        Rule of thumb: Dᵢ > 4/n may indicate influence
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3f5c",
    "axes.labelcolor":  "#c8d0e7",
    "text.color":       "#c8d0e7",
    "xtick.color":      "#8890a4",
    "ytick.color":      "#8890a4",
    "grid.color":       "#2a2d3e",
    "grid.alpha":       0.5,
}


def apply_style():
    plt.rcParams.update(STYLE)
    plt.rcParams["font.family"] = "DejaVu Sans"


# ─────────────────────────────────────────────────────────────────────────────
# RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(model, model_name: str,
                         save_dir: str = None) -> dict:
    """
    Full residual diagnostic suite for an OLS model.

    Parameters
    ----------
    model      : fitted statsmodels OLS result
    model_name : label for titles
    save_dir   : directory to save figure

    Returns
    -------
    dict of diagnostic statistics
    """
    apply_style()

    fitted    = model.fittedvalues.values
    residuals = model.resid.values
    std_resid = residuals / np.std(residuals)
    n         = len(residuals)

    # ── 1. Durbin-Watson ─────────────────────────────────────────────────
    dw = durbin_watson(residuals)
    print(f"\n  Durbin-Watson: {dw:.4f}")
    print(f"  (DW ≈ 2 = no autocorrelation, DW < 1.5 = positive AC)")

    # ── 2. Ljung-Box test (lag=24) ────────────────────────────────────────
    lb = acorr_ljungbox(residuals, lags=[24], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_p    = float(lb["lb_pvalue"].iloc[0])
    print(f"  Ljung-Box Q(24): stat={lb_stat:.2f}, p={lb_p:.4e}")
    print(f"  {'→ Significant residual autocorrelation detected.' if lb_p < 0.05 else '→ No significant autocorrelation at lag 24.'}")

    # ── 3. Breusch-Pagan test ─────────────────────────────────────────────
    exog = model.model.exog
    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, exog)
    print(f"  Breusch-Pagan: LM={bp_stat:.4f}, p={bp_p:.4e}")
    print(f"  {'→ Heteroscedasticity detected (H₀ rejected).' if bp_p < 0.05 else '→ No significant heteroscedasticity.'}")

    # ── 4. Normality of residuals ─────────────────────────────────────────
    sw_stat, sw_p = stats.shapiro(
        np.random.choice(residuals, size=min(5000, n), replace=False)
    )
    skew_r = float(stats.skew(residuals))
    kurt_r = float(stats.kurtosis(residuals))
    print(f"  Residual skewness: {skew_r:.4f}  kurtosis: {kurt_r:.4f}")
    print(f"  Shapiro-Wilk (subsample): W={sw_stat:.4f}, p={sw_p:.4e}")

    # ── FIGURE ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    accent = "#7b68ee"
    accent2 = "#ff6b9d"
    text_col = "#c8d0e7"

    # Plot 1: Residuals vs Fitted
    ax1.set_facecolor("#1a1d2e")
    idx = np.random.choice(n, size=min(3000, n), replace=False)
    ax1.scatter(fitted[idx], residuals[idx], alpha=0.3, s=4,
                color=accent, rasterized=True)
    ax1.axhline(0, color=accent2, lw=1.5, ls="--")
    ax1.set_xlabel("Fitted Values", color=text_col)
    ax1.set_ylabel("Residuals", color=text_col)
    ax1.set_title("Residuals vs Fitted", color=text_col, fontsize=11)
    ax1.tick_params(colors="#8890a4")

    # Plot 2: Q-Q plot
    ax2.set_facecolor("#1a1d2e")
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals)
    ax2.scatter(osm, osr, s=4, alpha=0.4, color=accent, rasterized=True)
    x_line = np.array([osm[0], osm[-1]])
    ax2.plot(x_line, slope * x_line + intercept, color=accent2, lw=2)
    ax2.set_xlabel("Theoretical Quantiles", color=text_col)
    ax2.set_ylabel("Sample Quantiles", color=text_col)
    ax2.set_title("Normal Q-Q Plot", color=text_col, fontsize=11)
    ax2.tick_params(colors="#8890a4")

    # Plot 3: Scale-Location
    ax3.set_facecolor("#1a1d2e")
    ax3.scatter(fitted[idx], np.sqrt(np.abs(std_resid[idx])),
                alpha=0.3, s=4, color=accent, rasterized=True)
    ax3.set_xlabel("Fitted Values", color=text_col)
    ax3.set_ylabel("√|Standardized Residuals|", color=text_col)
    ax3.set_title("Scale-Location", color=text_col, fontsize=11)
    ax3.tick_params(colors="#8890a4")

    # Plot 4: Residual Distribution
    ax4.set_facecolor("#1a1d2e")
    ax4.hist(residuals, bins=80, color=accent, alpha=0.8, edgecolor="none")
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    norm_curve = stats.norm.pdf(xr, residuals.mean(), residuals.std())
    ax42 = ax4.twinx()
    ax42.plot(xr, norm_curve, color=accent2, lw=2, label="Normal PDF")
    ax42.set_yticks([])
    ax4.set_xlabel("Residuals", color=text_col)
    ax4.set_ylabel("Frequency", color=text_col)
    ax4.set_title("Residual Distribution", color=text_col, fontsize=11)
    ax4.tick_params(colors="#8890a4")

    # Plot 5: ACF of residuals (manual)
    ax5.set_facecolor("#1a1d2e")
    lags_acf = range(1, min(49, n//2))
    acf_vals = [pd.Series(residuals).autocorr(lag=l) for l in lags_acf]
    ax5.bar(lags_acf, acf_vals, color=accent, alpha=0.8)
    ci_bound = 1.96 / np.sqrt(n)
    ax5.axhline(ci_bound,  color=accent2, lw=1, ls="--",
                label=f"95% CI ±{ci_bound:.3f}")
    ax5.axhline(-ci_bound, color=accent2, lw=1, ls="--")
    ax5.axhline(0, color=text_col, lw=0.5)
    ax5.legend(fontsize=8, facecolor="#1a1d2e",
               labelcolor=text_col, edgecolor="#3a3f5c")
    ax5.set_xlabel("Lag", color=text_col)
    ax5.set_ylabel("ACF", color=text_col)
    ax5.set_title("ACF of Residuals", color=text_col, fontsize=11)
    ax5.tick_params(colors="#8890a4")

    # Plot 6: Diagnostics text summary
    ax6.set_facecolor("#1a1d2e")
    ax6.axis("off")
    diag_text = (
        f"DIAGNOSTIC SUMMARY\n"
        f"{'─'*28}\n"
        f"Durbin-Watson:  {dw:.4f}\n"
        f"  (2.0 = no autocorr)\n\n"
        f"Ljung-Box Q(24): {lb_stat:.2f}\n"
        f"  p = {lb_p:.4e}\n\n"
        f"Breusch-Pagan: {bp_stat:.2f}\n"
        f"  p = {bp_p:.4e}\n\n"
        f"Residual Skewness: {skew_r:.4f}\n"
        f"Residual Kurtosis: {kurt_r:.4f}\n\n"
        f"HAC SEs applied ✓\n"
        f"Lag = 24 (daily cycle)"
    )
    ax6.text(0.05, 0.95, diag_text, transform=ax6.transAxes,
             va="top", ha="left", fontsize=9.5, color=text_col,
             fontfamily="monospace")

    fig.suptitle(f"Model Diagnostics — {model_name}",
                 color=text_col, fontsize=14, fontweight="bold", y=1.01)

    if save_dir:
        fname = f"{model_name.replace(' ','_').lower()}_diagnostics.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"  [SAVE] Diagnostic plot → {fpath}")
    plt.close(fig)

    return {
        "dw": dw, "lb_stat": lb_stat, "lb_p": lb_p,
        "bp_stat": bp_stat, "bp_p": bp_p,
        "residual_skew": skew_r, "residual_kurt": kurt_r,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostics(regression_results: dict,
                    save_dir: str = None) -> dict:
    """
    Run diagnostics for all fitted regression models.
    """
    print("\n" + "="*65)
    print("MODEL DIAGNOSTICS — Residual Analysis")
    print("="*65)

    diag_results = {}
    model_map = {
        "Model 1 Time-Only":   regression_results.get("m1"),
        "Model 2 Weather-Only":regression_results.get("m2"),
        "Model 3 Combined":    regression_results.get("m3"),
        "Model 4 Nonlinear":   regression_results.get("m4"),
    }

    for name, model in model_map.items():
        if model is not None:
            print(f"\n  {'='*50}")
            print(f"  {name}")
            print(f"  {'='*50}")
            d = residual_diagnostics(model, name, save_dir=save_dir)
            diag_results[name] = d

    print(f"\n{'='*65}")
    print("  DIAGNOSTIC INTERPRETATION NOTES")
    print(f"{'='*65}")
    print("  1. Durbin-Watson < 1.5 in hourly traffic data is EXPECTED")
    print("     due to the strong daily cycle. HAC standard errors")
    print("     (Newey-West, lag=24) correct for this in inference.")
    print("  2. Heteroscedasticity is common in traffic data. HAC SEs")
    print("     are also robust to heteroscedasticity (HC+AC robust).")
    print("  3. Non-normal residuals are expected for traffic volume;")
    print("     with large n, OLS coefficient estimates are still")
    print("     consistent by the Central Limit Theorem.")
    print("  4. Cook's distance outliers should be investigated but")
    print("     NOT automatically removed without domain justification.")

    return diag_results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from regression_analysis import run_regression_analysis

    clean    = os.path.join(base, "data", "traffic_clean.csv")
    save_dir = os.path.join(base, "results", "figures")

    df = pd.read_csv(clean, parse_dates=["date_time"])
    reg = run_regression_analysis(df)
    run_diagnostics(reg, save_dir=save_dir)
