"""
model_comparison.py
===================
Module X — Model Comparison and Selection
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Compare all regression models using AIC, BIC, R², adjusted R²,
    ΔR², RMSE, MAE, and chronological predictive validation.
    Optional: Random Forest supplementary ML comparison.

CRITICAL RULES:
    - Use chronological train/test split (NOT random shuffle)
    - ML is supplementary; statistical inference is primary
    - Report RMSE and MAE alongside R² for predictive comparison
    - Do NOT declare ML "better" without addressing interpretability
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def chronological_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Split dataset chronologically (last 20% = test).

    Why NOT random split:
        Traffic volume is time-ordered. Random split allows future
        information to "leak" into training data, inflating performance
        estimates. A chronological split provides honest out-of-sample
        evaluation.
    """
    df_sorted = df.sort_values("date_time").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:cutoff]
    test  = df_sorted.iloc[cutoff:]
    print(f"\n[SPLIT] Chronological 80/20 split:")
    print(f"  Train: {len(train):,} rows | {train['date_time'].min()} → "
          f"{train['date_time'].max()}")
    print(f"  Test:  {len(test):,} rows  | {test['date_time'].min()} → "
          f"{test['date_time'].max()}")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# OLS PREDICTIVE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def ols_predictive(formula: str, train: pd.DataFrame,
                   test: pd.DataFrame, name: str) -> dict:
    """
    Fit OLS on train, predict on test, compute RMSE/MAE/R².
    Uses OLS (not HAC) for prediction (HAC is for inference SEs only).
    """
    m = smf.ols(formula, data=train).fit()
    y_pred = m.predict(test)
    y_true = test["traffic_volume"]

    # Align indices
    valid = y_true.notna() & y_pred.notna()
    y_t = y_true[valid].values
    y_p = y_pred[valid].values

    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae  = mean_absolute_error(y_t, y_p)
    r2   = r2_score(y_t, y_p)

    print(f"  {name:<35s}: RMSE={rmse:7.2f}  MAE={mae:7.2f}  R²={r2:.4f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2_test": r2,
            "R2_train": m.rsquared}


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT BOOSTING (Supplementary ML)
# ─────────────────────────────────────────────────────────────────────────────
def gradient_boosting_comparison(train: pd.DataFrame,
                                 test: pd.DataFrame) -> dict:
    """
    Optional supplementary ML comparison using Gradient Boosting.

    IMPORTANT DISCLAIMER:
        GBM provides a non-parametric benchmark for predictive accuracy.
        It does NOT provide:
        - Interpretable coefficients
        - Confidence intervals
        - Hypothesis tests
        - Causal inference

        This analysis is SECONDARY. Statistical models remain primary.

    Feature sets:
        A: Time features only
        B: Weather features only
        C: Time + Weather
    """
    feature_sets = {
        "A: Time-only":    ["hour", "weekday", "month", "year",
                            "holiday_flag"],
        "B: Weather-only": ["temp_c", "rain_1h", "snow_1h", "clouds_all"],
        "C: Time+Weather": ["hour", "weekday", "month", "year",
                            "holiday_flag", "temp_c", "rain_1h",
                            "snow_1h", "clouds_all"],
    }

    target = "traffic_volume"
    results = []

    print(f"\n{'='*60}")
    print("  SUPPLEMENTARY ML: Gradient Boosting (Chronological Split)")
    print(f"{'='*60}")
    print(f"  ⚠  ML is supplementary. Statistical inference is primary.\n")
    print(f"  {'Feature Set':<25s}: {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}")

    for name, feats in feature_sets.items():
        avail_feats = [f for f in feats if f in train.columns]
        Xtr = train[avail_feats].fillna(train[avail_feats].median())
        Xte = test[avail_feats].fillna(train[avail_feats].median())
        ytr = train[target]
        yte = test[target]

        gb = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        gb.fit(Xtr, ytr)
        y_pred = gb.predict(Xte)

        rmse = np.sqrt(mean_squared_error(yte, y_pred))
        mae  = mean_absolute_error(yte, y_pred)
        r2   = r2_score(yte, y_pred)

        print(f"  {name:<25s}: {rmse:>8.2f}  {mae:>8.2f}  {r2:>8.4f}")
        results.append({
            "model": name, "RMSE": rmse, "MAE": mae, "R2": r2
        })

    print(f"\n  ML INTERPRETATION NOTE:")
    print(f"  GBM captures non-linearities and interactions automatically.")
    print(f"  A large GBM vs OLS R² gap suggests unmodelled non-linearity.")
    print(f"  GBM results do NOT replace statistical hypothesis testing.")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_model_comparison(df: pd.DataFrame,
                         save_path: str = None) -> dict:
    """
    Run all model comparisons and return results.
    """
    print("\n" + "="*65)
    print("MODEL COMPARISON — Chronological Predictive Validation")
    print("="*65)

    train, test = chronological_split(df)

    formulas = {
        "M1: Time-only": (
            "traffic_volume ~ C(hour) + C(weekday) + C(month) "
            "+ year + holiday_flag"
        ),
        "M2: Weather-only": (
            "traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all"
        ),
        "M3: Combined": (
            "traffic_volume ~ temp_c + rain_1h + snow_1h + clouds_all "
            "+ C(hour) + C(weekday) + C(month) + year + holiday_flag"
        ),
    }

    print(f"\n  {'Model':<35s}: {'RMSE':>8s}  {'MAE':>8s}  {'R²_test':>8s}")
    ols_results = []
    for name, formula in formulas.items():
        try:
            r = ols_predictive(formula, train, test, name)
            ols_results.append(r)
        except Exception as e:
            print(f"  {name}: Error — {e}")

    ols_df = pd.DataFrame(ols_results)

    # Gradient Boosting
    ml_df = gradient_boosting_comparison(train, test)

    # Full comparison table
    print(f"\n{'='*65}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  Statistical Models (OLS):")
    print(ols_df[["model", "RMSE", "MAE", "R2_test"]].to_string(index=False))
    print(f"\n  ML Models (GBM — supplementary only):")
    print(ml_df.to_string(index=False))

    if save_path:
        full = pd.concat([ols_df.rename(columns={"R2_test": "R2"}),
                          ml_df], ignore_index=True)
        full.to_csv(save_path, index=False)
        print(f"\n[SAVE] Model comparison saved → {save_path}")

    return {"ols": ols_df, "ml": ml_df}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean = os.path.join(base, "data", "traffic_clean.csv")
    save  = os.path.join(base, "results", "model_comparison.csv")
    df = pd.read_csv(clean, parse_dates=["date_time"])
    run_model_comparison(df, save_path=save)
