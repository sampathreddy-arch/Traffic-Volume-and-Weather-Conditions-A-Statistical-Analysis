"""
visualization.py
================
Modules I, IX, X — Visualization
Traffic Volume and Weather Conditions: A Statistical Analysis

PURPOSE:
    Generate all publication-quality visualizations for EDA, correlation,
    hypothesis testing, regression, and distribution analysis.
    Every chart includes: title, axis labels, units, legend, and caption.

DESIGN STANDARDS:
    - Dark theme with academic color palette
    - Readable fonts (DejaVu Sans)
    - All axes labeled with units
    - Statistical annotations on relevant plots
    - Saved as high-res PNG (150 DPI)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Color palette ─────────────────────────────────────────────────────────
BG_DARK  = "#0f1117"
BG_PANEL = "#1a1d2e"
BG_ALT   = "#14172a"
BORDER   = "#3a3f5c"
TEXT     = "#c8d0e7"
MUTED    = "#8890a4"
ACCENT1  = "#7b68ee"   # purple
ACCENT2  = "#ff6b9d"   # pink
ACCENT3  = "#00d4aa"   # teal
ACCENT4  = "#ffd166"   # amber
ACCENT5  = "#06d6a0"   # green

PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
           "#a8dadc", "#457b9d", "#e63946", "#06d6a0", "#ffd166"]


def apply_theme(ax, xlabel="", ylabel="", title="", grid=True):
    """Apply dark theme to an axes."""
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
    if grid:
        ax.grid(True, color=BORDER, alpha=0.5, lw=0.5)


def new_fig(figsize=(14, 7), title=""):
    fig = plt.figure(figsize=figsize, facecolor=BG_DARK)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=14, fontweight="bold", y=1.01)
    return fig


def save_fig(fig, path, name):
    fpath = os.path.join(path, name)
    fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  [SAVE] {name}")
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAFFIC VOLUME DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_distribution(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume Distribution Analysis",
                 color=TEXT, fontsize=14, fontweight="bold")

    tv = df["traffic_volume"].dropna()

    # Histogram
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    ax.hist(tv, bins=80, color=ACCENT1, alpha=0.85, edgecolor="none")
    ax.axvline(tv.mean(),   color=ACCENT2, lw=2, ls="--", label=f"Mean={tv.mean():.0f}")
    ax.axvline(tv.median(), color=ACCENT3, lw=2, ls=":",  label=f"Median={tv.median():.0f}")
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Traffic Volume (vehicles/hr)", "Frequency",
                "Histogram of Traffic Volume")

    # Boxplot
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    bp = ax.boxplot(tv, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=ACCENT1, alpha=0.7),
                    medianprops=dict(color=ACCENT2, lw=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker=".", color=ACCENT4, alpha=0.3, ms=2))
    ax.set_xlabel("", color=TEXT)
    apply_theme(ax, "", "Traffic Volume (vehicles/hr)", "Boxplot of Traffic Volume")
    stats_text = (f"Q1={tv.quantile(0.25):.0f}\n"
                  f"Median={tv.median():.0f}\n"
                  f"Q3={tv.quantile(0.75):.0f}\n"
                  f"Skew={tv.skew():.3f}")
    ax.text(1.3, tv.max() * 0.9, stats_text, color=TEXT, fontsize=9,
            fontfamily="monospace", transform=ax.get_yaxis_transform(),
            ha="left", va="top")

    # Q-Q plot
    ax = axes[2]
    ax.set_facecolor(BG_PANEL)
    sample = tv.sample(min(5000, len(tv)), random_state=42)
    (osm, osr), (slope, intercept, r) = stats.probplot(sample)
    ax.scatter(osm, osr, s=6, alpha=0.5, color=ACCENT1, rasterized=True)
    x_l = np.array([osm[0], osm[-1]])
    ax.plot(x_l, slope * x_l + intercept, color=ACCENT2, lw=2.5,
            label=f"Normal line (r={r:.3f})")
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Theoretical Quantiles", "Sample Quantiles",
                "Q-Q Plot (Traffic Volume vs Normal)")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "01_traffic_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAFFIC BY HOUR
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_by_hour(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume by Hour of Day",
                 color=TEXT, fontsize=14, fontweight="bold")

    # Mean + CI by hour
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    hourly = df.groupby("hour")["traffic_volume"].agg(
        ["mean", "std", "count"]).reset_index()
    hourly["se"] = hourly["std"] / np.sqrt(hourly["count"])
    hourly["ci"] = 1.96 * hourly["se"]

    ax.bar(hourly["hour"], hourly["mean"], color=ACCENT1, alpha=0.8, width=0.7)
    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["ci"],
                    hourly["mean"] + hourly["ci"],
                    alpha=0.3, color=ACCENT2, label="95% CI")
    # Mark peak hours
    for ph in [7, 8, 9, 16, 17, 18, 19]:
        ax.axvline(ph - 0.5, color=ACCENT4, alpha=0.4, lw=0.5)
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Hour of Day (0-23)", "Mean Traffic Volume (vehicles/hr)",
                "Mean Traffic by Hour ± 95% CI")
    ax.set_xticks(range(0, 24, 2))

    # Weekday vs Weekend by hour
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    for is_we, label, color in [(0, "Weekday", ACCENT1), (1, "Weekend", ACCENT2)]:
        sub = df[df["is_weekend"] == is_we].groupby("hour")["traffic_volume"].mean()
        ax.plot(sub.index, sub.values, color=color, lw=2.5, label=label, marker="o", ms=4)
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Hour of Day (0-23)", "Mean Traffic Volume (vehicles/hr)",
                "Weekday vs Weekend Hourly Pattern")
    ax.set_xticks(range(0, 24, 2))

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "02_traffic_by_hour.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAFFIC BY WEEKDAY AND MONTH
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_by_calendar(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume by Calendar Variables",
                 color=TEXT, fontsize=14, fontweight="bold")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    # By weekday
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    wd_stats = df.groupby("weekday")["traffic_volume"].agg(["mean","std","count"])
    wd_stats["ci"] = 1.96 * wd_stats["std"] / np.sqrt(wd_stats["count"])
    bars = ax.bar(range(7), wd_stats["mean"], color=PALETTE[:7], alpha=0.85)
    ax.errorbar(range(7), wd_stats["mean"], yerr=wd_stats["ci"],
                fmt="none", color=MUTED, capsize=4, lw=1.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names, color=MUTED)
    apply_theme(ax, "Day of Week", "Mean Traffic Volume (vehicles/hr)",
                "Mean Traffic by Day of Week ± 95% CI")

    # By month
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    mo_stats = df.groupby("month")["traffic_volume"].agg(["mean","std","count"])
    mo_stats["ci"] = 1.96 * mo_stats["std"] / np.sqrt(mo_stats["count"])
    ax.bar(range(1, 13), mo_stats["mean"], color=PALETTE[:12], alpha=0.85)
    ax.errorbar(range(1, 13), mo_stats["mean"], yerr=mo_stats["ci"],
                fmt="none", color=MUTED, capsize=3, lw=1.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, color=MUTED, rotation=45)
    apply_theme(ax, "Month", "Mean Traffic Volume (vehicles/hr)",
                "Mean Traffic by Month ± 95% CI")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4, axis="y")
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "03_traffic_by_calendar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAFFIC BY WEATHER TYPE + RAIN/SNOW/PEAK
# ─────────────────────────────────────────────────────────────────────────────
def plot_weather_comparisons(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume Under Different Weather and Period Conditions",
                 color=TEXT, fontsize=14, fontweight="bold")
    axes = axes.flatten()

    # 1) By weather_main
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    wm_stats = df.groupby("weather_main")["traffic_volume"].median().sort_values()
    ax.barh(wm_stats.index, wm_stats.values, color=ACCENT1, alpha=0.85)
    ax.set_xlabel("Median Traffic Volume (vehicles/hr)", color=TEXT)
    apply_theme(ax, "Median Traffic Volume (vehicles/hr)", "",
                "Median Traffic by Weather Type")

    # 2) Rain vs Dry
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    rain_g = df.groupby("rain_flag")["traffic_volume"]
    medians = rain_g.median()
    counts  = rain_g.count()
    ax.bar(["Dry (rain=0)", "Rain (rain>0)"],
           [medians[0], medians[1]],
           color=[ACCENT3, ACCENT2], alpha=0.85, width=0.5)
    for i, (m, c) in enumerate(zip(medians, counts)):
        ax.text(i, m + 50, f"n={c:,}\nMed={m:.0f}", ha="center",
                color=TEXT, fontsize=9, fontweight="bold")
    apply_theme(ax, "Condition", "Median Traffic Volume (vehicles/hr)",
                "Rain vs Dry Traffic Volume")

    # 3) Peak vs Off-Peak
    ax = axes[2]
    ax.set_facecolor(BG_PANEL)
    pk_g = df.groupby("peak")["traffic_volume"]
    pk_med = pk_g.median()
    pk_cnt = pk_g.count()
    labels = pk_med.index.tolist()
    vals   = pk_med.values
    ax.bar(labels, vals, color=[ACCENT4, ACCENT1], alpha=0.85, width=0.5)
    for i, (m, c) in enumerate(zip(vals, pk_cnt)):
        ax.text(i, m + 50, f"n={c:,}\nMed={m:.0f}", ha="center",
                color=TEXT, fontsize=9, fontweight="bold")
    apply_theme(ax, "Period", "Median Traffic Volume (vehicles/hr)",
                "Peak vs Off-Peak Traffic Volume")

    # 4) Holiday vs Non-Holiday
    ax = axes[3]
    ax.set_facecolor(BG_PANEL)
    hol_g = df.groupby("holiday_flag")["traffic_volume"]
    hol_med = hol_g.median()
    hol_cnt = hol_g.count()
    m_non = hol_med.get(0, 0)
    m_hol = hol_med.get(1, 0)
    c_non = hol_cnt.get(0, 0)
    c_hol = hol_cnt.get(1, 0)
    ax.bar(["Non-Holiday", "Holiday"],
           [m_non, m_hol],
           color=[ACCENT1, ACCENT5], alpha=0.85, width=0.5)
    for i, (m, c) in enumerate(zip([m_non, m_hol], [c_non, c_hol])):
        ax.text(i, m + 50, f"n={c:,}\nMed={m:.0f}", ha="center",
                color=TEXT, fontsize=9, fontweight="bold")
    apply_theme(ax, "Holiday Status", "Median Traffic Volume (vehicles/hr)",
                "Holiday vs Non-Holiday Traffic")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "04_weather_comparisons.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCATTER PLOTS: TRAFFIC vs WEATHER VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_vs_weather(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume vs. Weather Variables (Scatter Plots)",
                 color=TEXT, fontsize=14, fontweight="bold")
    axes = axes.flatten()

    pairs = [
        ("temp_c",   "Temperature (°C)",     ACCENT1),
        ("rain_1h",  "Rainfall (mm/hr)",      ACCENT2),
        ("snow_1h",  "Snowfall (mm/hr)",       ACCENT3),
        ("clouds_all","Cloud Coverage (%)",   ACCENT4),
    ]

    for ax, (var, xlabel, color) in zip(axes, pairs):
        ax.set_facecolor(BG_PANEL)
        sub = df[[var, "traffic_volume"]].dropna()
        # Subsample for readability
        sample = sub.sample(min(5000, len(sub)), random_state=42)

        ax.scatter(sample[var], sample["traffic_volume"],
                   alpha=0.25, s=5, color=color, rasterized=True)

        # Add LOWESS-style smoothed line (using numpy polyfit)
        try:
            xv = sub[var].values
            yv = sub["traffic_volume"].values
            # Bin means for smooth overlay
            bins = pd.cut(sub[var], bins=40, labels=False)
            bm = sub.groupby(bins)["traffic_volume"].mean()
            bc = sub.groupby(bins)[var].mean()
            ax.plot(bc.values, bm.values, color="white", lw=2, alpha=0.8,
                    label="Bin mean")
        except Exception:
            pass

        # Pearson r annotation
        try:
            r, p = stats.pearsonr(
                sub[var].values, sub["traffic_volume"].values
            )
            ax.text(0.05, 0.95, f"Pearson r = {r:.4f}\np = {p:.2e}",
                    transform=ax.transAxes, color=TEXT, fontsize=9,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor=BG_DARK,
                              alpha=0.7, edgecolor=BORDER))
        except Exception:
            pass

        apply_theme(ax, xlabel, "Traffic Volume (vehicles/hr)",
                    f"Traffic vs {xlabel}")
        ax.legend(fontsize=8, facecolor=BG_PANEL, labelcolor=TEXT,
                  edgecolor=BORDER)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "05_traffic_vs_weather.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str) -> str:
    cols = ["traffic_volume", "temp_c", "rain_1h", "snow_1h",
            "clouds_all", "hour", "weekday", "month"]
    sub  = df[cols].dropna()
    cmat = sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)

    # Custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    mask = np.zeros_like(cmat, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # Show full matrix

    sns.heatmap(cmat, ax=ax, annot=True, fmt=".3f", cmap=cmap,
                center=0, vmin=-1, vmax=1,
                annot_kws={"fontsize": 9, "color": TEXT},
                linewidths=0.5, linecolor=BORDER,
                cbar_kws={"shrink": 0.8})

    ax.set_title("Pearson Correlation Matrix — Traffic & Weather Variables",
                 color=TEXT, fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors=MUTED, labelsize=9)
    plt.xticks(rotation=35, ha="right", color=MUTED)
    plt.yticks(rotation=0, color=MUTED)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=9)

    plt.tight_layout()
    return save_fig(fig, save_dir, "06_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAFFIC TREND OVER TIME
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_trend(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume Trend Over Time",
                 color=TEXT, fontsize=14, fontweight="bold")

    # Daily mean
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    daily = df.set_index("date_time")["traffic_volume"].resample("D").mean()
    ax.fill_between(daily.index, daily.values, alpha=0.4, color=ACCENT1)
    ax.plot(daily.index, daily.values, color=ACCENT1, lw=0.8, alpha=0.9)
    # 30-day rolling mean
    rolling = daily.rolling(30, center=True).mean()
    ax.plot(rolling.index, rolling.values, color=ACCENT2, lw=2.5,
            label="30-day rolling mean")
    ax.legend(fontsize=10, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Date", "Mean Traffic Volume (vehicles/hr)",
                "Daily Mean Traffic Volume with 30-Day Rolling Average")

    # By year boxplot
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    years = sorted(df["year"].unique())
    data_by_year = [df.loc[df["year"] == y, "traffic_volume"].dropna().values
                    for y in years]
    try:
        bp = ax.boxplot(data_by_year, tick_labels=years, patch_artist=True,
                        medianprops=dict(color=ACCENT2, lw=2))
    except TypeError:
        bp = ax.boxplot(data_by_year, labels=years, patch_artist=True,
                        medianprops=dict(color=ACCENT2, lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(MUTED)
    apply_theme(ax, "Year", "Traffic Volume (vehicles/hr)",
                "Traffic Volume Distribution by Year")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "07_traffic_trend.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEMPERATURE — DISTRIBUTION AND NONLINEAR FIT
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_analysis(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
    fig.suptitle("Temperature Analysis: Distribution and Traffic Relationship",
                 color=TEXT, fontsize=14, fontweight="bold")

    # Temperature distribution
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    temp = df["temp_c"].dropna()
    ax.hist(temp, bins=70, color=ACCENT1, alpha=0.8, edgecolor="none",
            density=True)
    xr = np.linspace(temp.min(), temp.max(), 300)
    mu, sigma = temp.mean(), temp.std()
    normal_curve = stats.norm.pdf(xr, mu, sigma)
    ax.plot(xr, normal_curve, color=ACCENT2, lw=2.5,
            label=f"Normal(μ={mu:.1f}, σ={sigma:.1f})")
    ax.axvline(mu, color=ACCENT3, lw=1.5, ls="--", label=f"Mean = {mu:.1f}°C")
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Temperature (°C)", "Density",
                "Temperature Distribution with Normal Overlay")

    # Traffic vs Temperature with polynomial fit
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    sub = df[["temp_c", "traffic_volume"]].dropna()
    sample = sub.sample(min(6000, len(sub)), random_state=42)
    ax.scatter(sample["temp_c"], sample["traffic_volume"],
               alpha=0.2, s=5, color=ACCENT1, rasterized=True)

    # Bin means
    tc_bins = pd.cut(sub["temp_c"], bins=30, labels=False)
    bm = sub.groupby(tc_bins)["traffic_volume"].mean()
    bc = sub.groupby(tc_bins)["temp_c"].mean()
    ax.scatter(bc.values, bm.values, color=ACCENT3, s=40, zorder=5,
               label="Bin means")

    # Polynomial fit (degree 2)
    valid = bc.notna() & bm.notna()
    if valid.sum() > 5:
        z2 = np.polyfit(bc[valid], bm[valid], 2)
        p2 = np.poly1d(z2)
        x_line = np.linspace(bc[valid].min(), bc[valid].max(), 200)
        ax.plot(x_line, p2(x_line), color=ACCENT2, lw=3, label="Polynomial fit (deg=2)")

    r, _ = stats.pearsonr(sub["temp_c"], sub["traffic_volume"])
    ax.text(0.05, 0.95, f"Pearson r = {r:.4f}",
            transform=ax.transAxes, color=TEXT, fontsize=10,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor=BG_DARK,
                      alpha=0.7, edgecolor=BORDER))
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "Temperature (°C)", "Traffic Volume (vehicles/hr)",
                "Traffic vs Temperature with Polynomial Fit")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "08_temperature_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRECIPITATION INTENSITY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def plot_precipitation_intensity(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_DARK)
    fig.suptitle("Traffic Volume by Precipitation Intensity",
                 color=TEXT, fontsize=14, fontweight="bold")

    groups = ["Dry", "Light", "Moderate", "Heavy"]
    colors_g = [ACCENT3, ACCENT1, ACCENT4, ACCENT2]

    # Boxplot by intensity
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    data_g = [df.loc[df["precipitation_intensity"] == g, "traffic_volume"].dropna().values
              for g in groups]
    valid_groups = [(g, d, c) for g, d, c in zip(groups, data_g, colors_g) if len(d) > 0]
    if valid_groups:
        names, vals, cols = zip(*valid_groups)
        try:
            bp = ax.boxplot(vals, tick_labels=names, patch_artist=True,
                            medianprops=dict(color="white", lw=2))
        except TypeError:
            bp = ax.boxplot(vals, labels=names, patch_artist=True,
                            medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color(MUTED)
        apply_theme(ax, "Precipitation Intensity", "Traffic Volume (vehicles/hr)",
                    "Traffic by Precipitation Intensity")

    # Median + n
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    med_vals = []
    ns = []
    valid_names = []
    for g, d, c in zip(groups, data_g, colors_g):
        if len(d) > 0:
            med_vals.append(np.median(d))
            ns.append(len(d))
            valid_names.append(g)
    x = range(len(valid_names))
    bars = ax.bar(x, med_vals, color=[c for g, d, c in valid_groups],
                  alpha=0.85, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, color=MUTED)
    for i, (m, n) in enumerate(zip(med_vals, ns)):
        ax.text(i, m + 50, f"n={n:,}\nMed={m:.0f}",
                ha="center", color=TEXT, fontsize=8, fontweight="bold")
    apply_theme(ax, "Precipitation Intensity", "Median Traffic Volume (vehicles/hr)",
                "Median Traffic by Precipitation Intensity")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4, axis="y")
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "09_precipitation_intensity.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. REGRESSION RESULTS VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_regression_results(regression_dict: dict, save_dir: str) -> str:
    """
    Visualize coefficients with 95% CI for Model 3 (Combined).
    """
    m3 = regression_dict.get("m3")
    if m3 is None:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG_DARK)
    fig.suptitle("Regression Results — Model 3: Combined (Time + Weather)",
                 color=TEXT, fontsize=14, fontweight="bold")

    # Weather coefficients
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    weather_vars = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
    params = []
    ci_lo, ci_hi = [], []
    labels = []
    pvals = []

    for v in weather_vars:
        if v in m3.params:
            params.append(m3.params[v])
            ci = m3.conf_int().loc[v]
            ci_lo.append(ci[0])
            ci_hi.append(ci[1])
            labels.append(v)
            pvals.append(m3.pvalues[v])

    y_pos = range(len(params))
    colors_coef = [ACCENT3 if p < 0.05 else ACCENT4 for p in pvals]
    ax.barh(y_pos, params, color=colors_coef, alpha=0.8)
    ax.errorbar(params, y_pos,
                xerr=[[p - l for p, l in zip(params, ci_lo)],
                       [h - p for p, h in zip(params, ci_hi)]],
                fmt="none", color=MUTED, capsize=5, lw=2)
    ax.axvline(0, color=TEXT, lw=1, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=MUTED)
    apply_theme(ax, "Coefficient (vehicles/hr per unit change)", "",
                "Weather Coefficients with 95% CI")
    sig_patch = plt.Line2D([0], [0], color=ACCENT3, lw=5, label="p<0.05")
    ns_patch  = plt.Line2D([0], [0], color=ACCENT4, lw=5, label="p≥0.05")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9,
              facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)

    # R² comparison across models
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    model_names = ["M1\nTime-only", "M2\nWeather-only",
                   "M3\nCombined", "M4\nNonlinear"]
    r2_vals = []
    for key in ["m1", "m2", "m3", "m4"]:
        m = regression_dict.get(key)
        if m is not None:
            r2_vals.append(m.rsquared)
        else:
            r2_vals.append(0)

    bars = ax.bar(model_names[:len(r2_vals)], r2_vals[:len(r2_vals)],
                  color=PALETTE[:len(r2_vals)], alpha=0.85)
    for bar, r2 in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, r2 + 0.005,
                f"{r2:.4f}", ha="center", color=TEXT, fontsize=10,
                fontweight="bold")
    apply_theme(ax, "Model", "R² (Proportion of Variance Explained)",
                "Model R² Comparison")
    ax.set_ylim(0, max(r2_vals) * 1.15 if r2_vals else 1)

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "10_regression_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. PROBABILITY ANALYSIS VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_probability_analysis(df: pd.DataFrame, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_DARK)
    fig.suptitle("Conditional Probability Analysis",
                 color=TEXT, fontsize=14, fontweight="bold")

    # P(High Traffic | Weather Condition)
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    n = len(df)
    conditions = {
        "Rain": df["rain_flag"] == 1,
        "No Rain": df["rain_flag"] == 0,
        "Snow": df["snow_flag"] == 1,
        "No Snow": df["snow_flag"] == 0,
        "Peak": df["peak"] == "Peak",
        "Off-Peak": df["peak"] == "Off-Peak",
    }
    cond_probs = {}
    for cname, cmask in conditions.items():
        n_c = cmask.sum()
        if n_c == 0:
            continue
        n_hc = ((cmask) & (df["high_traffic"] == 1)).sum()
        cond_probs[cname] = n_hc / n_c

    colors_c = [ACCENT2, ACCENT3, ACCENT1, ACCENT5, ACCENT4, MUTED]
    bars = ax.bar(list(cond_probs.keys()),
                  list(cond_probs.values()),
                  color=colors_c[:len(cond_probs)], alpha=0.85)
    # Global P(High Traffic) reference line
    p_high = (df["high_traffic"] == 1).sum() / n
    ax.axhline(p_high, color="white", lw=1.5, ls="--",
               label=f"P(High Traffic) = {p_high:.4f}")
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    for bar, p in zip(bars, cond_probs.values()):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.005,
                f"{p:.3f}", ha="center", color=TEXT, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.65)
    plt.xticks(rotation=25)
    apply_theme(ax, "Condition", "P(High Traffic | Condition)",
                "Conditional Probability of High Traffic")

    # P(High Traffic | Weather Type)
    ax = axes[1]
    ax.set_facecolor(BG_PANEL)
    wt_probs = {}
    for wtype in df["weather_main"].unique():
        mask = df["weather_main"] == wtype
        n_w = mask.sum()
        if n_w < 50:
            continue
        wt_probs[wtype] = ((mask) & (df["high_traffic"] == 1)).sum() / n_w
    sorted_wt = dict(sorted(wt_probs.items(), key=lambda x: x[1], reverse=True))
    ax.barh(list(sorted_wt.keys()), list(sorted_wt.values()),
            color=PALETTE[:len(sorted_wt)], alpha=0.85)
    ax.axvline(p_high, color="white", lw=1.5, ls="--",
               label=f"Overall P(High) = {p_high:.3f}")
    ax.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=TEXT, edgecolor=BORDER)
    apply_theme(ax, "P(High Traffic | Weather Type)", "",
                "Conditional Probability by Weather Type")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return save_fig(fig, save_dir, "11_probability_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_all_visualizations(df: pd.DataFrame,
                           regression_dict: dict = None,
                           save_dir: str = "results/figures") -> list:
    """
    Generate all visualizations and return list of saved paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*65)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*65)

    saved = []
    saved.append(plot_traffic_distribution(df, save_dir))
    saved.append(plot_traffic_by_hour(df, save_dir))
    saved.append(plot_traffic_by_calendar(df, save_dir))
    saved.append(plot_weather_comparisons(df, save_dir))
    saved.append(plot_traffic_vs_weather(df, save_dir))
    saved.append(plot_correlation_heatmap(df, save_dir))
    saved.append(plot_traffic_trend(df, save_dir))
    saved.append(plot_temperature_analysis(df, save_dir))
    saved.append(plot_precipitation_intensity(df, save_dir))
    if regression_dict:
        saved.append(plot_regression_results(regression_dict, save_dir))
    saved.append(plot_probability_analysis(df, save_dir))

    print(f"\n[DONE] {len(saved)} visualizations saved to: {save_dir}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean    = os.path.join(base, "data", "traffic_clean.csv")
    save_dir = os.path.join(base, "results", "figures")

    sys.path.insert(0, os.path.join(base, "src"))
    from regression_analysis import run_regression_analysis

    df = pd.read_csv(clean, parse_dates=["date_time"])
    reg = run_regression_analysis(df)
    run_all_visualizations(df, regression_dict=reg, save_dir=save_dir)
