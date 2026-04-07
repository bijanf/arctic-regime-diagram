#!/usr/bin/env python3
"""Step 5: Plot Eady growth rate diagnostics with raincloud plots.

Two-panel figure:
  (a) Raincloud: Absolute Eady growth rate (dry & moist, split-violin)
      by scenario/period including historical baseline.
  (b) Horizontal strip: Moist enhancement factor F grouped by scenario/period
      — reveals the systematic Clausius-Clapeyron shift with warming.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.plotting.raincloud import (
    _box,
    _half_violin,
    _strip,
    horizontal_strip,
)
from src.plotting.style import SCENARIO_COLORS, apply_style

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Quality filters
EXCLUDE_MODELS = {"MCM-UA-1-0", "MPI-ESM1-2-HR"}
R_MAX = 1.5

# All scenario/period groups for panel (a) — includes historical
ALL_GROUPS = [
    ("historical", "historical", "Hist."),
    ("ssp245", "mid_century",   "SSP2-4.5\nMid"),
    ("ssp245", "end_century",   "SSP2-4.5\nEnd"),
    ("ssp585", "mid_century",   "SSP5-8.5\nMid"),
    ("ssp585", "end_century",   "SSP5-8.5\nEnd"),
]

# Groups for panel (b) F strip (includes historical)
F_GROUPS = [
    ("historical", "historical", "Historical"),
    ("ssp245", "mid_century",   "SSP2-4.5 Mid"),
    ("ssp245", "end_century",   "SSP2-4.5 End"),
    ("ssp585", "mid_century",   "SSP5-8.5 Mid"),
    ("ssp585", "end_century",   "SSP5-8.5 End"),
]

# X-positions: Historical alone, then SSP groups with gap between scenarios
XPOS_ALL = [0, 1.5, 2.5, 4.0, 5.0]


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters."""
    df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()
    df = df[df["R"] < R_MAX]
    df = df[df["eady_dry"] > 0.01]
    return df


def plot_eady_figure(df: pd.DataFrame, output_path: Path):
    """Create the two-panel Eady figure with raincloud + strip chart."""
    apply_style()

    df_eady = df.dropna(subset=["eady_dry", "eady_moist"]).copy()
    df_eady = _filter_df(df_eady)
    if len(df_eady) == 0:
        logger.warning("No Eady data available — skipping figure")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))

    # ── Panel (a): Split-violin raincloud for dry & moist absolute values ──
    ax = axes[0]

    # Collect absolute Eady growth rate values for each group
    labels, data_dry, data_moist, colors, counts = [], [], [], [], []
    for scenario, period, label in ALL_GROUPS:
        sub = df_eady[(df_eady["scenario"] == scenario) &
                      (df_eady["period"] == period)]
        if len(sub) == 0:
            continue
        labels.append(label)
        data_dry.append(sub["eady_dry"].values)
        data_moist.append(sub["eady_moist"].values)
        colors.append(SCENARIO_COLORS.get(scenario, "grey"))
        counts.append(len(sub))

    if len(data_dry) > 0:
        positions = XPOS_ALL[:len(labels)]

        for i in range(len(data_dry)):
            pos = positions[i]
            col = colors[i]
            d_dry = data_dry[i]
            d_moist = data_moist[i]

            # Dry: half-violin LEFT, strip LEFT, box slightly left
            _half_violin(ax, d_dry, pos - 0.05, col, side="left",
                         width=0.35, alpha=0.2, kde_bw="scott")
            _box(ax, d_dry, pos - 0.12, col, width=0.1, alpha=0.45,
                 whis=(5, 95))
            _strip(ax, d_dry, pos - 0.12, col, offset=-0.12,
                   size=10, alpha=0.4, seed=42, jitter_width=0.04)

            # Moist: half-violin RIGHT, strip RIGHT, box slightly right
            _half_violin(ax, d_moist, pos + 0.05, col, side="right",
                         width=0.35, alpha=0.35, kde_bw="scott")
            _box(ax, d_moist, pos + 0.12, col, width=0.1, alpha=0.7,
                 whis=(5, 95))
            _strip(ax, d_moist, pos + 0.12, col, offset=0.12,
                   size=10, alpha=0.55, seed=43, jitter_width=0.04)

        # Axis setup
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6.5)

        # Model counts (inside panel, near bottom)
        ylim = ax.get_ylim()
        for i, n in enumerate(counts):
            ax.text(positions[i], ylim[0] + (ylim[1] - ylim[0]) * 0.01,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=5, color="grey")

        # Legend for dry/moist distinction
        from matplotlib.patches import Patch
        legend_dry = Patch(facecolor="#999999", alpha=0.35, label="Dry")
        legend_moist = Patch(facecolor="#999999", alpha=0.7, label="Moist")
        ax.legend(handles=[legend_dry, legend_moist], fontsize=6,
                  loc="upper right", handlelength=1.2, handleheight=0.8)
    else:
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)

    ax.set_ylabel("Eady growth rate (day$^{-1}$)", fontsize=7.5)
    ax.set_title("(a) Baroclinic instability", fontsize=9,
                 loc="left", fontweight="bold")

    # ── Panel (b): Horizontal strip chart of F by scenario/period ──
    ax = axes[1]

    f_data, f_positions, f_colors, f_labels = [], [], [], []
    for idx, (scenario, period, label) in enumerate(F_GROUPS):
        sub = df_eady[(df_eady["scenario"] == scenario) &
                      (df_eady["period"] == period)]
        if len(sub) == 0:
            continue
        f_data.append(sub["eady_ratio"].values)
        f_positions.append(len(F_GROUPS) - 1 - idx)  # reverse so historical on top
        f_colors.append(SCENARIO_COLORS.get(scenario, "grey"))
        f_labels.append(label)

    if len(f_data) > 0:
        horizontal_strip(ax, f_data, f_positions, f_colors, f_labels,
                         box_height=0.3, dot_size=14, dot_alpha=0.45,
                         mean_size=45, seed=42)

        # Reference line at F=1 (no enhancement)
        ax.axvline(1.0, color="#aaaaaa", ls=":", lw=0.5, zorder=0)

        # Tight x-limits to zoom into the actual range
        all_f = np.concatenate(f_data)
        f_min = np.nanmin(all_f)
        f_max = np.nanmax(all_f)
        pad = (f_max - f_min) * 0.15
        ax.set_xlim(f_min - pad, f_max + pad)

        ax.set_xlabel("Moist enhancement factor $F$", fontsize=7.5)

        # Add n= counts on the right
        for data, pos in zip(f_data, f_positions, strict=False):
            n = np.sum(np.isfinite(data))
            ax.text(ax.get_xlim()[1], pos, f" n={n}",
                    ha="left", va="center", fontsize=5, color="grey",
                    clip_on=False)

    ax.set_title("(b) Moist enhancement by scenario", fontsize=9,
                 loc="left", fontweight="bold")

    fig.tight_layout(w_pad=3.0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    logger.info("Saved Eady figure to %s", output_path)
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

    output_path = PROJECT_ROOT / "figures" / "eady_growth_rate.pdf"
    plot_eady_figure(df, output_path)


if __name__ == "__main__":
    main()
