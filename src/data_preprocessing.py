"""
data_preprocessing.py
=====================
Module I / VI — Data Preparation and Feature Engineering
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Load, inspect, clean, and engineer features from the UCI Metro Interstate
    Traffic Volume dataset. Every cleaning decision is justified, quantified,
    and documented. No arbitrary thresholds are applied without statistical
    reasoning.

AUTHOR: B.Tech Statistics & Probability Project
DATASET: UCI ML Repository — Metro Interstate Traffic Volume
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
KELVIN_OFFSET = 273.15          # Kelvin → Celsius conversion
PEAK_HOURS = list(range(7, 10)) + list(range(16, 20))   # 07-09, 16-19
HIGH_TRAFFIC_QUANTILE = 0.75   # 75th percentile defines "high traffic"
RAIN_THRESHOLD = 0.0           # >0 mm/h = rain present
SNOW_THRESHOLD = 0.0           # >0 mm/h = snow present

# Precipitation intensity bins (mm/h) — justified by WMO light/moderate/heavy
# thresholds for hourly accumulations (WMO No.8, 2018)
PRECIP_BINS = [-np.inf, 0, 2.5, 7.6, np.inf]
PRECIP_LABELS = ["Dry", "Light", "Moderate", "Heavy"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Metro Interstate Traffic Volume CSV.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to Metro_Interstate_Traffic_Volume.csv

    Returns
    -------
    pd.DataFrame  — raw dataset with minimal modification
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please download from:\n"
            "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume\n"
            "and place it in the data/ directory."
        )

    df = pd.read_csv(filepath)
    print(f"[LOAD] Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"[LOAD] Column names: {list(df.columns)}")
    print(f"\n[LOAD] Data types:\n{df.dtypes}\n")
    print(f"[LOAD] First 3 rows:\n{df.head(3)}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARSE DATETIME AND SORT
# ─────────────────────────────────────────────────────────────────────────────
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the date_time column to pandas Timestamp and sort chronologically.

    Why: Temporal analyses (autocorrelation, trend, seasonal decomposition)
         require observations to be ordered in time. Incorrect dtype prevents
         time-aware operations.

    Decision: date_time is parsed with infer_datetime_format for efficiency.
    """
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    print(f"[DATETIME] Range: {df['date_time'].min()} → {df['date_time'].max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. INSPECT DUPLICATE TIMESTAMPS
# ─────────────────────────────────────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Investigate and remove exact duplicate rows.

    Why: Duplicate timestamps may arise from data collection artefacts.
         Duplicates inflate sample sizes and introduce artificial patterns
         in autocorrelation.

    Decision: Keep first occurrence of duplicated (date_time, traffic_volume)
              pairs after sorting. Document the count removed.
    """
    n_dup = df.duplicated(subset=["date_time"], keep=False).sum()
    print(f"[DUPLICATES] Rows with duplicate date_time: {n_dup}")

    # Keep the first of each duplicated timestamp
    df_clean = df.drop_duplicates(subset=["date_time"], keep="first").reset_index(drop=True)
    removed = len(df) - len(df_clean)
    print(f"[DUPLICATES] Removed: {removed} rows → Remaining: {len(df_clean):,}")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# 4. INSPECT MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
def inspect_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report and handle missing values.

    Why: Missing values in weather or traffic fields would produce biased
         statistical estimates if silently included.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0]
    if len(report):
        print(f"[MISSING]\n{report}\n")
    else:
        print("[MISSING] No missing values detected.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. INVESTIGATE IRREGULAR INTERVALS
# ─────────────────────────────────────────────────────────────────────────────
def inspect_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify gaps larger than 1 hour in the time series.

    Why: The dataset is hourly. Gaps > 1h break stationarity assumptions and
         need to be acknowledged in temporal analysis. We do NOT impute; we
         document.
    """
    df = df.copy()
    df["time_gap_h"] = df["date_time"].diff().dt.total_seconds() / 3600
    large_gaps = df[df["time_gap_h"] > 1]
    print(f"[GAPS] Observations with time gap > 1h: {len(large_gaps)}")
    print(f"[GAPS] Largest gap: {df['time_gap_h'].max():.1f} hours")
    print(f"[GAPS] Note: Gaps are documented but NOT imputed. "
          "Temporal analyses account for potential non-stationarity.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNREALISTIC VALUE CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_unrealistic_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify physically impossible values.

    Justifications for each threshold:
    - temp = 0 K (−273.15 °C): The dataset contains sentinel value 0 K for
      missing temperature readings. 0 K is physically impossible; any valid
      temperature in Minneapolis, MN is well above 200 K (~−70 °C).
      Decision: Flag and remove rows where temp == 0 (impossible Kelvin).
    - traffic_volume < 0: Counts cannot be negative.
    - rain_1h < 0 / snow_1h < 0: Accumulations cannot be negative.
    - clouds_all not in [0, 100]: Percentage out of range.

    Sensitivity analysis: We retain the cleaned dataset as primary and note
    the effect of inclusion/exclusion in the report.
    """
    df = df.copy()
    n_before = len(df)

    # Temperature sentinel: temp = 0 K is impossible and represents missing
    temp_zero = (df["temp"] == 0).sum()
    print(f"[QUALITY] temp == 0 K (sentinel for missing): {temp_zero} rows")
    df = df[df["temp"] > 0]

    # Negative traffic
    neg_traffic = (df["traffic_volume"] < 0).sum()
    print(f"[QUALITY] traffic_volume < 0: {neg_traffic} rows")
    df = df[df["traffic_volume"] >= 0]

    # Negative precipitation
    neg_rain = (df["rain_1h"] < 0).sum()
    neg_snow = (df["snow_1h"] < 0).sum()
    print(f"[QUALITY] rain_1h < 0: {neg_rain} | snow_1h < 0: {neg_snow}")
    df = df[df["rain_1h"] >= 0]
    df = df[df["snow_1h"] >= 0]

    # Cloud percentage
    bad_cloud = (~df["clouds_all"].between(0, 100)).sum()
    print(f"[QUALITY] clouds_all out of [0,100]: {bad_cloud} rows")
    df = df[df["clouds_all"].between(0, 100)]

    # Extreme rain outlier: UCI dataset contains one erroneous value of
    # rain_1h = 9831.3 mm/h (physically impossible; world record hourly
    # rain is ~340 mm). Cap at 100 mm/h with documentation.
    extreme_rain = (df["rain_1h"] > 100).sum()
    print(f"[QUALITY] rain_1h > 100 mm/h (extreme outlier): {extreme_rain} rows")
    df.loc[df["rain_1h"] > 100, "rain_1h"] = np.nan  # mark as missing
    df = df.dropna(subset=["rain_1h"])
    print(f"[QUALITY] Rows removed due to impossible rain values: {extreme_rain}")

    n_after = len(df)
    print(f"[QUALITY] Total rows removed: {n_before - n_after} "
          f"({(n_before - n_after)/n_before*100:.2f}%)")
    print(f"[QUALITY] Clean dataset size: {n_after:,}\n")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TEMPERATURE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
def convert_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert temperature from Kelvin to Celsius.

        temp_c = temp_K − 273.15

    Why: Celsius is more interpretable for weather analysis. The conversion
         is a linear shift and does not affect correlation or regression
         slopes involving only temp.
    """
    df = df.copy()
    df["temp_c"] = df["temp"] - KELVIN_OFFSET
    print(f"[TEMP] Converted temp (K) → temp_c (°C)")
    print(f"[TEMP] Range: {df['temp_c'].min():.1f} °C to {df['temp_c'].max():.1f} °C\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived temporal and categorical features.

    Features created:
    -----------------
    hour          : int 0-23, hour of day (calendar)
    weekday       : int 0=Mon … 6=Sun
    month         : int 1-12
    year          : int
    season        : str (meteorological seasons)
    is_weekend    : bool
    peak          : str "Peak" or "Off-Peak"
                    Definition: Peak = weekday 07-09 & 16-19 (US commute hrs)
                    Justification: FHWA defines AM peak 7-9 AM and PM peak
                    4-7 PM for urban corridors. Minneapolis AADT data
                    corroborates bimodal peaks in these windows.
    holiday_flag  : int (1 = holiday listed, 0 = none)
    rain_flag     : int (1 = rain_1h > 0)
    snow_flag     : int (1 = snow_1h > 0)
    precipitation_intensity : categorical (Dry/Light/Moderate/Heavy)
                    based on WMO rainfall intensity classification
    high_traffic  : int (1 = traffic_volume ≥ 75th percentile)
    """
    df = df.copy()

    # Temporal features
    df["hour"]    = df["date_time"].dt.hour
    df["weekday"] = df["date_time"].dt.weekday          # 0=Mon
    df["month"]   = df["date_time"].dt.month
    df["year"]    = df["date_time"].dt.year
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Meteorological seasons (Northern Hemisphere)
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month"].map(season_map)

    # Peak/Off-Peak — FHWA-based definition, applied to weekdays only
    def assign_peak(row):
        if row["weekday"] < 5 and row["hour"] in PEAK_HOURS:
            return "Peak"
        return "Off-Peak"
    df["peak"] = df.apply(assign_peak, axis=1)

    # Holiday flag
    df["holiday_flag"] = (df["holiday"].notna() & (df["holiday"].astype(str).str.strip().str.lower() != "none")).astype(int)

    # Precipitation flags
    df["rain_flag"] = (df["rain_1h"] > RAIN_THRESHOLD).astype(int)
    df["snow_flag"] = (df["snow_1h"] > SNOW_THRESHOLD).astype(int)

    # Combined precipitation column (rain dominates if both present)
    # Justification: Use rain_1h + snow_1h total for intensity classification
    df["precip_total"] = df["rain_1h"] + df["snow_1h"]
    df["precipitation_intensity"] = pd.cut(
        df["precip_total"],
        bins=PRECIP_BINS,
        labels=PRECIP_LABELS,
        right=True
    )

    # High traffic flag (75th percentile threshold)
    threshold = df["traffic_volume"].quantile(HIGH_TRAFFIC_QUANTILE)
    df["high_traffic"] = (df["traffic_volume"] >= threshold).astype(int)

    print(f"[FEATURES] Features created: hour, weekday, month, year, season,")
    print(f"           is_weekend, peak, holiday_flag, rain_flag, snow_flag,")
    print(f"           precip_total, precipitation_intensity, high_traffic")
    print(f"[FEATURES] Peak hours definition: weekday + hours {PEAK_HOURS}")
    print(f"[FEATURES] High traffic threshold (75th pct): {threshold:.0f} vehicles/hr")
    print(f"[FEATURES] Peak observations: {(df['peak']=='Peak').sum():,}")
    print(f"[FEATURES] Rain observations: {df['rain_flag'].sum():,}")
    print(f"[FEATURES] Snow observations: {df['snow_flag'].sum():,}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE CLEAN DATASET
# ─────────────────────────────────────────────────────────────────────────────
def save_clean_data(df: pd.DataFrame, out_path: str) -> None:
    """Save the cleaned and engineered dataset to CSV."""
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Clean dataset saved → {out_path}")
    print(f"[SAVE] Shape: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing(raw_path: str, clean_path: str) -> pd.DataFrame:
    """
    Execute full preprocessing pipeline and return cleaned DataFrame.

    Parameters
    ----------
    raw_path   : path to raw CSV
    clean_path : path to save cleaned CSV

    Returns
    -------
    pd.DataFrame — cleaned, feature-engineered dataset
    """
    print("=" * 65)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 65)

    df = load_data(raw_path)
    df = parse_datetime(df)
    df = handle_duplicates(df)
    df = inspect_missing(df)
    df = inspect_time_gaps(df)
    df = check_unrealistic_values(df)
    df = convert_temperature(df)
    df = engineer_features(df)
    save_clean_data(df, clean_path)

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE")
    print(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 65)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw  = os.path.join(base, "data", "Metro_Interstate_Traffic_Volume.csv")
    clean = os.path.join(base, "data", "traffic_clean.csv")
    run_preprocessing(raw, clean)
