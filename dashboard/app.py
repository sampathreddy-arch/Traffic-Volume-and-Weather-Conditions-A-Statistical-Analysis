"""
dashboard/app.py
================
Streamlit Interactive Academic Dashboard
Traffic Volume and Weather Conditions: A Statistical Analysis

To run:
    streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Traffic & Weather Statistical Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "traffic_clean.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

st.title("🚗 Traffic Volume and Weather Conditions: A Statistical Analysis")
st.markdown("**B.Tech Project | Statistics and Probability (Modules I–X) | UCI Metro Interstate 94**")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, parse_dates=["date_time"])
    return None

df = load_data()

# Sidebar
st.sidebar.header("Academic Navigation")
page = st.sidebar.radio(
    "Select View",
    ["Overview & Metrics", "Hypothesis Testing (H1-H5)", "Interactive Bayes & OLS Tools", "Publication Figures"]
)

if df is not None:
    st.sidebar.info(f"Loaded {len(df):,} clean observations ({df['date_time'].min().year}–{df['date_time'].max().year})")

if page == "Overview & Metrics":
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Size (n)", "40,564", "Deduplicated")
    col2.metric("Mean Traffic Volume", "3,291 veh/h", "Std: 1,985")
    col3.metric("Diurnal & Calendar R²", "83.79%", "Dominant Factor")

    col4, col5, col6 = st.columns(3)
    col4.metric("Isolated Weather R²", "2.90%", "Confounded")
    col5.metric("Incremental Weather ΔR²", "+0.09%", "p < 10⁻¹⁰ (Modest)")
    col6.metric("Durbin-Watson Stat", "0.412", "HAC SEs required")

    st.subheader("Predictive Model Comparison (Chronological 80/20 Out-of-Sample Split)")
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        cdf = pd.read_csv(comp_path)
        st.dataframe(cdf.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.4f}"}))
    else:
        st.info("Run main.py to populate model comparisons.")

elif page == "Hypothesis Testing (H1-H5)":
    st.subheader("Syllabus Pre-Specified Hypothesis Testing Matrix (Modules VII & VIII)")
    hyp_path = os.path.join(RESULTS_DIR, "statistical_results.csv")
    if os.path.exists(hyp_path):
        hdf = pd.read_csv(hyp_path)
        st.dataframe(hdf)
    else:
        st.info("Run main.py to populate statistical test results.")

    st.markdown("""
    ### Statistical & Methodological Defense:
    - **H1 (Rain vs Dry):** In unadjusted tests, rain shows little effect ($p=0.36$). Once calendar confounders (hour, weekday, month) are adjusted, rain remains statistically non-significant ($p=0.997$).
    - **H2 (Precipitation Intensity):** Kruskal-Wallis detects omnibus variation ($p=0.011$), but effect size is negligible ($\eta^2 = 0.0002$).
    - **H3 (Interaction):** Rain association does not statistically differ between peak and off-peak periods ($p=1.00$).
    - **H4 (Nonlinear Temperature):** Temperature exhibits statistically significant quadratic curvature ($p = 7.5 \times 10^{-5}$).
    - **H5 (Explanatory Power):** Time variables account for $83.8\%$ of variance; weather adds a modest increment ($\Delta R^2 = 0.0009$).
    """)

elif page == "Interactive Bayes & OLS Tools":
    st.subheader("Interactive Bayes' Theorem Risk Calculator (Module II)")
    col1, col2 = st.columns(2)
    with col1:
        p_rain = st.slider("Prior P(Rain)", 0.01, 0.20, 0.0506, step=0.005)
        p_high_rain = st.slider("Likelihood P(High Congestion | Rain)", 0.10, 0.60, 0.2689, step=0.01)
        p_high_dry = st.slider("Likelihood P(High Congestion | Dry)", 0.10, 0.60, 0.2492, step=0.01)
        
        # Total probability
        p_high = (p_high_rain * p_rain) + (p_high_dry * (1 - p_rain))
        p_rain_high = (p_high_rain * p_rain) / p_high

    with col2:
        st.metric("Posterior P(Rain | High Congestion)", f"{p_rain_high * 100:.2f}%")
        st.metric("Relative Risk Ratio", f"{(p_high_rain / p_high_dry):.2f}x")
        st.caption("P(Rain | High) = [P(High | Rain) · P(Rain)] / P(High)")

elif page == "Publication Figures":
    st.subheader("Academic Figures Gallery")
    if os.path.exists(FIGURES_DIR):
        figs = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")])
        selected_fig = st.selectbox("Choose a Figure to View", figs)
        if selected_fig:
            fig_path = os.path.join(FIGURES_DIR, selected_fig)
            st.image(fig_path, use_column_width=True)
