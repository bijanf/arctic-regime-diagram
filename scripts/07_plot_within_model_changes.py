#!/usr/bin/env python3
"""Step 7: Within-model ΔR vs ΔKs² scatter and R-regime binning.

Two-panel figure:
  (a) Scatter: per-model ΔR vs ΔKs² (%), isolating the warming signal
      from structural model differences. Theil-Sen + Spearman statistics.
  (b) Raincloud: ΔKs² grouped by historical R regime bins (Low / Medium / High),
      with Kruskal-Wallis and pairwise Mann-Whitney U tests.
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
from src.plotting.raincloud import raincloud
from src.plotting.style import SCENARIO_COLORS, apply_style

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Quality filters
EXCLUDE_MODELS = {"MCM-UA-1-0", "MPI-ESM1-2-HR"}

# Scenario/period groups for scatter
CHANGE_GROUPS = [
    ("ssp245", "mid_century",  "SSP2-4.5 Mid"),
    ("ssp245", "end_century",  "SSP2-4.5 End"),
    ("ssp585", "mid_century",  "SSP5-8.5 Mid"),
    ("ssp585", "end_century",  "SSP5-8.5 End"),
]

# R-regime bin thresholds
R_BINS = [
    ("Low\n(R < 0.30)", 0.0, 0.30),
    ("Medium\n(0.30 ≤ R < 0.50)", 0.30, 0.50),
    ("High\n(R ≥ 0.50)", 0.50, np.inf),
]


def _build_change_table(df):
    """Build a table of per-model within-model changes from historical.

    Returns DataFrame with columns:
        model, scenario, period, R_hist, Ks_hist, delta_R, delta_Ks_pct
    """
    df_hist = df[(df["scenario"] == "historical") & (df["period"] == "historical")]
    hist_R = df_hist.dropna(subset=["R"]).set_index("model")["R"]
    hist_Ks = df_hist.dropna(subset=["Ks_250"]).set_index("model")["Ks_250"]

    rows = []
    for scenario, period, label in CHANGE_GROUPS:
        sub = df[(df["scenario"] == scenario) & (df["period"] == period)]
        for _, row in sub.iterrows():
            m = row["model"]
            if (m in hist_R.index and m in hist_Ks.index
                    and np.isfinite(row["R"]) and np.isfinite(row["Ks_250"])
                    and np.isfinite(hist_R.loc[m]) and np.isfinite(hist_Ks.loc[m])
                    and abs(hist_Ks.loc[m]) > 1e-20):
                rows.append({
                    "model": m,
                    "scenario": scenario,
                    "period": period,
                    "label": label,
                    "R_hist": hist_R.loc[m],
                    "Ks_hist": hist_Ks.loc[m],
                    "R_future": row["R"],
                    "Ks_future": row["Ks_250"],
                    "delta_R": row["R"] - hist_R.loc[m],
                    "delta_Ks_pct": (row["Ks_250"] - hist_Ks.loc[m])
                                    / abs(hist_Ks.loc[m]) * 100.0,
                })
    return pd.DataFrame(rows)


def plot_within_model_figure(df: pd.DataFrame, output_path: Path):
    """Create the two-panel within-model change figure."""
    apply_style()

    df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()
    changes = _build_change_table(df)

    if len(changes) == 0:
        logger.warning("No within-model change data — skipping figure")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))

    # ══════════════════════════════════════════════════════════
    # Panel (a): ΔR vs ΔKs² scatter — within-model changes
    # ══════════════════════════════════════════════════════════
    ax = axes[0]

    all_dr, all_dks = [], []
    for scenario, period, label in CHANGE_GROUPS:
        sub = changes[(changes["scenario"] == scenario) &
                      (changes["period"] == period)]
        if len(sub) == 0:
            continue
        marker = "o" if period == "mid_century" else "D"
        color = SCENARIO_COLORS.get(scenario, "grey")
        ax.scatter(sub["delta_R"], sub["delta_Ks_pct"],
                   c=color, marker=marker, s=25, alpha=0.6,
                   edgecolors="none", label=label, zorder=2)
        all_dr.extend(sub["delta_R"].values)
        all_dks.extend(sub["delta_Ks_pct"].values)

    # Robust regression + statistics
    if len(all_dr) >= 5:
        dr_arr = np.array(all_dr)
        dks_arr = np.array(all_dks)

        # Spearman
        rho_sp, p_sp = sp_stats.spearmanr(dr_arr, dks_arr)

        # Theil-Sen
        ts_slope, ts_intercept, _, _ = sp_stats.theilslopes(dks_arr, dr_arr)
        x_fit = np.linspace(dr_arr.min(), dr_arr.max(), 100)
        ax.plot(x_fit, ts_slope * x_fit + ts_intercept,
                color="black", lw=1.0, ls="--", zorder=3)

        # Pearson for reference
        r_pear, p_pear = sp_stats.pearsonr(dr_arr, dks_arr)

        p_sp_str = f"p = {p_sp:.1e}" if p_sp >= 0.001 else "p < 0.001"
        p_pear_str = f"p = {p_pear:.2f}" if p_pear >= 0.001 else "p < 0.001"
        ann = (f"Spearman \u03c1 = {rho_sp:.2f}, {p_sp_str}\n"
               f"Pearson r = {r_pear:.2f}, {p_pear_str}")
        ax.text(0.05, 0.95, ann, transform=ax.transAxes, fontsize=6,
                va="top", linespacing=1.4)
        logger.info("Panel (a) within-model: Spearman rho = %.3f, p = %.2e | "
                    "Pearson r = %.3f, p = %.2e | n = %d",
                    rho_sp, p_sp, r_pear, p_pear, len(dr_arr))

    ax.axhline(0, color="grey", lw=0.5, ls="-", zorder=1)
    ax.axvline(0, color="grey", lw=0.5, ls="-", zorder=1)
    ax.set_xlabel(r"$\Delta\mathcal{R}$ (future $-$ historical)", fontsize=7.5)
    ax.set_ylabel(r"$\Delta K_s^2$ change from historical (%)", fontsize=7.5)
    ax.set_title(r"(a) Within-model: $\Delta\mathcal{R}$ vs $\Delta K_s^2$",
                 fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.7,
              handletextpad=0.3, borderpad=0.4)

    # ══════════════════════════════════════════════════════════
    # Panel (b): R-regime binning — raincloud of ΔKs² per bin
    # ══════════════════════════════════════════════════════════
    ax = axes[1]

    # Use SSP5-8.5 end-century for the clearest signal
    sub_end = changes[(changes["scenario"] == "ssp585") &
                      (changes["period"] == "end_century")]

    bin_labels, bin_data, bin_colors = [], [], []
    bin_palette = ["#2166ac", "#f4a582", "#b2182b"]  # blue → orange → red

    for i, (label, lo, hi) in enumerate(R_BINS):
        mask = (sub_end["R_hist"] >= lo) & (sub_end["R_hist"] < hi)
        vals = sub_end.loc[mask, "delta_Ks_pct"].values
        if len(vals) > 0:
            bin_labels.append(label)
            bin_data.append(vals)
            bin_colors.append(bin_palette[i])

    if len(bin_data) >= 2:
        positions = list(range(len(bin_data)))
        raincloud(
            ax, bin_data, positions, bin_colors, bin_labels,
            violin_width=0.35, violin_alpha=0.25,
            box_width=0.18, box_alpha=0.55,
            dot_offset=-0.2, dot_size=14, dot_alpha=0.45,
            whis=(5, 95), show_counts=False, refline=0.0,
        )

        # Model counts
        ylim_bot = ax.get_ylim()[0]
        for i, d in enumerate(bin_data):
            n = np.sum(np.isfinite(d))
            ax.text(positions[i], ylim_bot + 0.5, f"n={n}",
                    ha="center", va="bottom", fontsize=5, color="grey")

        # Kruskal-Wallis test across bins
        if len(bin_data) >= 2:
            clean_bins = [d[np.isfinite(d)] for d in bin_data]
            if all(len(b) >= 2 for b in clean_bins):
                kw_stat, kw_p = sp_stats.kruskal(*clean_bins)
                kw_str = f"p = {kw_p:.3f}" if kw_p >= 0.001 else "p < 0.001"
                ax.text(0.95, 0.95,
                        f"Kruskal-Wallis\nH = {kw_stat:.1f}, {kw_str}",
                        transform=ax.transAxes, fontsize=6,
                        ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="#cccccc",
                                  alpha=0.8))
                logger.info("Kruskal-Wallis: H = %.2f, p = %.4f", kw_stat, kw_p)

                # Pairwise Mann-Whitney U tests
                for i in range(len(clean_bins)):
                    for j in range(i + 1, len(clean_bins)):
                        if len(clean_bins[i]) >= 3 and len(clean_bins[j]) >= 3:
                            u_stat, u_p = sp_stats.mannwhitneyu(
                                clean_bins[i], clean_bins[j],
                                alternative="two-sided")
                            logger.info("Mann-Whitney U (%s vs %s): "
                                        "U = %.1f, p = %.4f",
                                        bin_labels[i].split('\n')[0],
                                        bin_labels[j].split('\n')[0],
                                        u_stat, u_p)

    ax.set_ylabel(r"$\Delta K_s^2$ change from historical (%)", fontsize=7.5)
    ax.set_title("(b) SSP5-8.5 end-century by R regime",
                 fontsize=9, loc="left", fontweight="bold")

    fig.tight_layout(w_pad=3.0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    logger.info("Saved within-model changes figure to %s", output_path)
    return fig


def main():
    load_config()

    csv_path = PROJECT_ROOT / "data" / "diagnostics" / "regime_diagnostics.csv"
    if not csv_path.exists():
        logger.error("Diagnostics CSV not found: %s", csv_path)
        logger.error("Run 02_compute_diagnostics.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    output_path = PROJECT_ROOT / "figures" / "within_model_changes.pdf"
    plot_within_model_figure(df, output_path)


if __name__ == "__main__":
    main()
