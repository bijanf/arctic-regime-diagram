#!/usr/bin/env python3
"""Step 6: Plot wave diagnostics figure.

Two-panel figure:
  (a) Raincloud: K_s² % change relative to historical, with significance
      asterisks from Wilcoxon signed-rank test.
  (b) Scatter: historical R vs ΔKs² (%), emergent-constraint style
      with Theil-Sen robust regression line, Pearson and Spearman correlations.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.plotting.style import apply_style, SCENARIO_COLORS
from src.plotting.raincloud import raincloud

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Quality filters
EXCLUDE_MODELS = {"MCM-UA-1-0", "MPI-ESM1-2-HR"}

# Scenario/period groups
CHANGE_GROUPS = [
    ("ssp245", "mid_century",  "SSP2-4.5\nMid"),
    ("ssp245", "end_century",  "SSP2-4.5\nEnd"),
    ("ssp585", "mid_century",  "SSP5-8.5\nMid"),
    ("ssp585", "end_century",  "SSP5-8.5\nEnd"),
]

# X-positions with gap between SSP scenarios
XPOS = [0, 1.0, 2.5, 3.5]


def _compute_pct_change(df, col, hist_col_vals):
    """Compute per-model % change from historical for a column.

    Returns (labels, data_arrays, colors, model_names_per_group).
    """
    labels, data_arrays, colors, names_per_group = [], [], [], []
    for scenario, period, label in CHANGE_GROUPS:
        sub = df[(df["scenario"] == scenario) &
                 (df["period"] == period)]
        changes, model_names = [], []
        for _, row in sub.iterrows():
            if row["model"] in hist_col_vals.index:
                h = hist_col_vals.loc[row["model"]]
                if abs(h) > 1e-20:
                    changes.append((row[col] - h) / abs(h) * 100.0)
                    model_names.append(row["model"])
        if changes:
            labels.append(label)
            data_arrays.append(np.array(changes))
            colors.append(SCENARIO_COLORS.get(scenario, "grey"))
            names_per_group.append(model_names)
    return labels, data_arrays, colors, names_per_group


def plot_wave_figure(df: pd.DataFrame, output_path: Path):
    """Create the two-panel wave diagnostics figure with rainclouds."""
    apply_style()

    df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))

    # ══════════════════════════════════════════════════════════
    # Panel (a): Ks² % change — standard raincloud
    # ══════════════════════════════════════════════════════════
    ax = axes[0]

    df_ks = df.dropna(subset=["Ks_250"]).copy()
    hist_ks = df_ks[(df_ks["scenario"] == "historical") &
                    (df_ks["period"] == "historical")]
    hist_Ks = hist_ks.set_index("model")["Ks_250"]

    labels_ks, data_ks, colors_ks, names_ks = _compute_pct_change(
        df_ks, "Ks_250", hist_Ks)

    if len(data_ks) > 0:
        positions_ks = XPOS[:len(labels_ks)]

        medians = raincloud(
            ax, data_ks, positions_ks, colors_ks, labels_ks,
            violin_width=0.35, violin_alpha=0.25,
            box_width=0.18, box_alpha=0.55,
            dot_offset=-0.2, dot_size=12, dot_alpha=0.45,
            whis=(5, 95), show_counts=False, refline=0.0,
        )

        # Model counts (inside panel, just above axis bottom)
        ylim_bot = ax.get_ylim()[0]
        for i, d in enumerate(data_ks):
            n = np.sum(np.isfinite(d))
            ax.text(positions_ks[i], ylim_bot + 0.5, f"n={n}",
                    ha="center", va="bottom", fontsize=5, color="grey")

        # Significance asterisks (Wilcoxon signed-rank test, H0: median=0)
        ylim_top = ax.get_ylim()[1]
        for i, d in enumerate(data_ks):
            d_clean = d[np.isfinite(d)]
            if len(d_clean) >= 5:
                try:
                    _, p_val = sp_stats.wilcoxon(d_clean)
                except ValueError:
                    p_val = 1.0
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = ""
                if stars:
                    ax.text(positions_ks[i], ylim_top * 0.92, stars,
                            ha="center", va="top", fontsize=7,
                            fontweight="bold", color=colors_ks[i])

    ax.set_ylabel("$K_s^2$ change from historical (%)", fontsize=7.5)
    ax.set_title("(a) Stationary wave refractive index change",
                 fontsize=9, loc="left", fontweight="bold")

    # ══════════════════════════════════════════════════════════
    # Panel (b): R_historical vs ΔKs² scatter (emergent constraint)
    # ══════════════════════════════════════════════════════════
    ax = axes[1]

    # Historical R per model
    df_hist = df[(df["scenario"] == "historical") & (df["period"] == "historical")]
    hist_R = df_hist.dropna(subset=["R"]).set_index("model")["R"]

    # Historical Ks² per model (needed for % change denominator)
    hist_Ks_a = df_hist.dropna(subset=["Ks_250"]).set_index("model")["Ks_250"]

    # Collect all points for pooled regression
    all_r_vals, all_dks_vals = [], []
    scatter_labels = {
        ("ssp245", "mid_century"):  "SSP2-4.5 Mid",
        ("ssp245", "end_century"):  "SSP2-4.5 End",
        ("ssp585", "mid_century"):  "SSP5-8.5 Mid",
        ("ssp585", "end_century"):  "SSP5-8.5 End",
    }

    for scenario, period, _label in CHANGE_GROUPS:
        sub = df[(df["scenario"] == scenario) & (df["period"] == period)]
        r_pts, dks_pts = [], []
        for _, row in sub.iterrows():
            m = row["model"]
            if (m in hist_R.index and m in hist_Ks_a.index
                    and abs(hist_Ks_a.loc[m]) > 1e-20
                    and np.isfinite(row["Ks_250"]) and np.isfinite(hist_R.loc[m])):
                r_pts.append(hist_R.loc[m])
                dks_pts.append((row["Ks_250"] - hist_Ks_a.loc[m])
                               / abs(hist_Ks_a.loc[m]) * 100.0)
        if r_pts:
            marker = "o" if period == "mid_century" else "D"
            color = SCENARIO_COLORS.get(scenario, "grey")
            ax.scatter(r_pts, dks_pts, c=color, marker=marker, s=25,
                       alpha=0.6, edgecolors="none",
                       label=scatter_labels[(scenario, period)], zorder=2)
            all_r_vals.extend(r_pts)
            all_dks_vals.extend(dks_pts)

    # Pooled robust regression (Theil-Sen) + Pearson & Spearman stats
    if len(all_r_vals) >= 5:
        all_r_arr = np.array(all_r_vals)
        all_dks_arr = np.array(all_dks_vals)

        # Pearson (parametric)
        slope_ols, intercept_ols, r_val, p_val_pearson, _ = sp_stats.linregress(
            all_r_arr, all_dks_arr)

        # Spearman (rank-based, robust to outliers)
        rho_sp, p_val_spearman = sp_stats.spearmanr(all_r_arr, all_dks_arr)

        # Theil-Sen robust regression line
        ts_slope, ts_intercept, _, _ = sp_stats.theilslopes(
            all_dks_arr, all_r_arr)
        x_fit = np.linspace(all_r_arr.min(), all_r_arr.max(), 100)
        ax.plot(x_fit, ts_slope * x_fit + ts_intercept,
                color="black", lw=1.0, ls="--", zorder=3)

        # Annotation with both statistics
        p_pear = f"p = {p_val_pearson:.2f}" if p_val_pearson >= 0.001 else "p < 0.001"
        p_spear = f"p = {p_val_spearman:.1e}" if p_val_spearman >= 0.001 else "p < 0.001"
        ann = (f"Pearson r = {r_val:.2f}, {p_pear}\n"
               f"Spearman \u03c1 = {rho_sp:.2f}, {p_spear}")
        ax.text(0.05, 0.05, ann,
                transform=ax.transAxes, fontsize=6, va="bottom",
                linespacing=1.4)
        logger.info("Panel (b) Pearson r = %.3f, p = %.2e | "
                    "Spearman rho = %.3f, p = %.2e | n = %d",
                    r_val, p_val_pearson, rho_sp, p_val_spearman,
                    len(all_r_arr))

    ax.axhline(0, color="grey", lw=0.5, ls="-", zorder=1)
    ax.set_xlabel(r"$\mathcal{R}$ (historical)", fontsize=7.5)
    ax.set_ylabel("$K_s^2$ change from historical (%)", fontsize=7.5)
    ax.set_title("(b) Refractive index change vs nonlinearity",
                 fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7,
              handletextpad=0.3, borderpad=0.4)

    fig.tight_layout(w_pad=3.0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    logger.info("Saved wave diagnostics figure to %s", output_path)
    return fig


def main():
    cfg = load_config()

    csv_path = PROJECT_ROOT / "data" / "diagnostics" / "regime_diagnostics.csv"
    if not csv_path.exists():
        logger.error("Diagnostics CSV not found: %s", csv_path)
        logger.error("Run 02_compute_diagnostics.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    output_path = PROJECT_ROOT / "figures" / "wave_diagnostics.pdf"
    plot_wave_figure(df, output_path)


if __name__ == "__main__":
    main()
