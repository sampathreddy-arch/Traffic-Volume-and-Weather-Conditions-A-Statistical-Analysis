
"""
main.py
=======
Master Execution Pipeline
Traffic Volume and Weather Conditions: A Statistical Analysis

Course: B.Tech Statistics and Probability (Modules I - X)
Dataset: UCI Metro Interstate Traffic Volume (n = 40,564 clean observations)

PURPOSE:
    Execute the complete end-to-end analytical pipeline reproducibly:
    1. Data Preprocessing & Integrity Verification (Module I)
    2. Descriptive Statistics & Distribution Analysis (Module I)
    3. Probability Analysis & Bayes' Theorem (Module II)
    4. Hypothesis Testing: H1 to H5 (Modules VII & VIII)
    5. Correlation & Covariance Analysis (Module IX)
    6. Multiple Regression & Confounding Analysis (Module X)
    7. Residual Diagnostics & HAC Inference (Module X)
    8. Out-of-Sample Predictive Model Comparison (Module X)
    9. Publication-Quality Visualizations (Modules I, IX, X)
"""

import os
import sys
import time
import warnings
import pandas as pd

# Suppress warnings for clean reporting
warnings.filterwarnings("ignore")

# Add src to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

sys.path.insert(0, SRC_DIR)

from data_preprocessing import run_preprocessing
from descriptive_statistics import run_descriptive_statistics
from probability_analysis import run_probability_analysis
from hypothesis_testing import run_hypothesis_testing
from correlation_analysis import run_correlation_analysis
from regression_analysis import run_regression_analysis
from diagnostics import run_diagnostics
from model_comparison import run_model_comparison
from visualization import run_all_visualizations


def main():
    start_time = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n" + "=" * 75)
    print("  B.TECH STATISTICS & PROBABILITY PROJECT: COMPLETE PIPELINE")
    print("  Title: Traffic Volume and Weather Conditions: A Statistical Analysis")
    print("  Dataset: Metro Interstate 94 (Minneapolis-St Paul) | UCI ML Repository")
    print("=" * 75)

    raw_path = os.path.join(DATA_DIR, "Metro_Interstate_Traffic_Volume.csv")
    clean_path = os.path.join(DATA_DIR, "traffic_clean.csv")

    # STEP 1: Preprocessing
    print("\n>>> [1/9] PREPROCESSING DATA")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    df = run_preprocessing(raw_path, clean_path)

    # STEP 2: Descriptive Statistics
    print("\n>>> [2/9] RUNNING DESCRIPTIVE STATISTICS (Module I)")
    desc_csv = os.path.join(RESULTS_DIR, "descriptive_statistics.csv")
    desc_results = run_descriptive_statistics(df, save_path=desc_csv)

    # STEP 3: Probability Analysis
    print("\n>>> [3/9] RUNNING PROBABILITY & BAYES ANALYSIS (Module II)")
    prob_csv = os.path.join(RESULTS_DIR, "probability_results.csv")
    prob_results = run_probability_analysis(df, save_path=prob_csv)

    # STEP 4: Hypothesis Testing
    print("\n>>> [4/9] RUNNING HYPOTHESIS TESTING H1-H5 (Modules VII & VIII)")
    hyp_csv = os.path.join(RESULTS_DIR, "statistical_results.csv")
    hyp_results = run_hypothesis_testing(df, save_path=hyp_csv)

    # STEP 5: Correlation & Covariance
    print("\n>>> [5/9] RUNNING CORRELATION & COVARIANCE (Module IX)")
    corr_csv = os.path.join(RESULTS_DIR, "correlation_results.csv")
    corr_results = run_correlation_analysis(df, save_path=corr_csv)

    # STEP 6: Regression Models
    print("\n>>> [6/9] RUNNING REGRESSION & CONFOUNDING ANALYSIS (Module X)")
    reg_csv = os.path.join(RESULTS_DIR, "regression_results.csv")
    reg_results = run_regression_analysis(df, save_path=reg_csv)

    # STEP 7: Diagnostics
    print("\n>>> [7/9] RUNNING RESIDUAL DIAGNOSTICS & ASSUMPTION CHECKS (Module X)")
    diag_results = run_diagnostics(reg_results, save_dir=FIGURES_DIR)

    # STEP 8: Model Comparison
    print("\n>>> [8/9] RUNNING CHRONOLOGICAL MODEL COMPARISON (Module X)")
    comp_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comp_results = run_model_comparison(df, save_path=comp_csv)

    # STEP 9: Visualizations
    print("\n>>> [9/9] GENERATING PUBLICATION-GRADE VISUALIZATIONS")
    saved_figs = run_all_visualizations(df, regression_dict=reg_results, save_dir=FIGURES_DIR)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"  ALL 9 PIPELINE MODULES COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print(f"  Clean Observations Analyzed: {len(df):,}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR} ({len(saved_figs)} plots)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
