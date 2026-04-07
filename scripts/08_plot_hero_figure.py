#!/usr/bin/env python3
"""Step 8: Hero figure — 3-panel summary of nonlinear PV dynamics driving Arctic amplification.

Layout (GridSpec 2×2):
  (a) Regime diagram (R vs D) — simplified, no mid-century, ERA5 only
  (b) ΔKs² distribution — raincloud showing near-universal Ks² decline by scenario
  (c) Causal chain schematic — full width bottom strip
"""

import logging
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.raincloud import raincloud
from src.plotting.style import (
    REANALYSIS_STYLES,
    WINDOW_MARKERS,
    apply_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Quality filters
EXCLUDE_MODELS = {"MCM-UA-1-0", "MPI-ESM1-2-HR"}

# Brighter, more distinct colors for clarity
COL_HIST = "#1f77b4"  # Strong blue
COL_SSP245 = "#ff7f0e"  # Bright orange
COL_SSP585 = "#d62728"  # Bright red


# ═══════════════════════════════════════════════════════════════════
# Panel (a): Regime Diagram
# ═══════════════════════════════════════════════════════════════════


def _plot_panel_a(ax, df, df_reanalysis):
    """Simplified regime diagram: no mid-century, ERA5 only."""
    R_range = (0.15, 0.60)
    D_range = (0.015, 0.055)
    R_threshold = 0.30
    D_threshold = 0.03

    # --- Quadrant shading (fill_between for precise control) ---
    ax.fill_between(
        [R_range[0], R_threshold], D_range[0], D_threshold, color="#2166ac", alpha=0.07, zorder=0
    )
    ax.fill_between(
        [R_threshold, R_range[1]], D_range[0], D_threshold, color="#f4a582", alpha=0.07, zorder=0
    )
    ax.fill_between(
        [R_range[0], R_threshold], D_threshold, D_range[1], color="#92c5de", alpha=0.07, zorder=0
    )
    ax.fill_between(
        [R_threshold, R_range[1]], D_threshold, D_range[1], color="#b2182b", alpha=0.08, zorder=0
    )

    # --- Threshold lines ---
    ax.axvline(R_threshold, color="#999999", ls="--", lw=0.7, zorder=1)
    ax.axhline(D_threshold, color="#999999", ls="--", lw=0.7, zorder=1)

    # --- Quadrant labels ---
    ax.text(
        R_range[0] + 0.008,
        D_range[0] + 0.001,
        "Dry\u2013Linear",
        ha="left",
        va="bottom",
        fontsize=6.5,
        style="italic",
        color="#888888",
        zorder=10,
    )
    ax.text(
        R_range[1] - 0.008,
        D_range[0] + 0.001,
        "Dry\u2013Nonlinear",
        ha="right",
        va="bottom",
        fontsize=6.5,
        style="italic",
        color="#888888",
        zorder=10,
    )
    ax.text(
        R_range[0] + 0.008,
        D_range[1] - 0.001,
        "Moist\u2013Linear",
        ha="left",
        va="top",
        fontsize=6.5,
        style="italic",
        color="#888888",
        zorder=10,
    )
    ax.text(
        R_range[1] - 0.008,
        D_range[1] - 0.001,
        "Moist\u2013Nonlinear",
        ha="right",
        va="top",
        fontsize=7,
        style="italic",
        color="#b2182b",
        fontweight="bold",
        zorder=10,
    )

    # --- Scatter individual models (no mid-century) ---
    hist = df[(df["scenario"] == "historical") & (df["period"] == "historical")]
    ssp245_end = df[(df["scenario"] == "ssp245") & (df["period"] == "end_century")]
    ssp585_end = df[(df["scenario"] == "ssp585") & (df["period"] == "end_century")]

    # Historical — circles, strong blue
    ax.scatter(
        hist["R"],
        hist["D"],
        c=COL_HIST,
        marker="o",
        s=28,
        alpha=0.45,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
        label="_nolegend_",
    )

    # SSP2-4.5 end — squares, bright orange (distinct from SSP5-8.5)
    ax.scatter(
        ssp245_end["R"],
        ssp245_end["D"],
        c=COL_SSP245,
        marker="s",
        s=28,
        alpha=0.50,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
        label="_nolegend_",
    )

    # SSP5-8.5 end — diamonds, bright red
    ax.scatter(
        ssp585_end["R"],
        ssp585_end["D"],
        c=COL_SSP585,
        marker="D",
        s=28,
        alpha=0.50,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
        label="_nolegend_",
    )

    # --- Multi-model means with SEM error bars ---
    mean_data = []
    for label, sub, color, marker in [
        ("Historical", hist, COL_HIST, "o"),
        ("SSP2-4.5 End", ssp245_end, COL_SSP245, "s"),
        ("SSP5-8.5 End", ssp585_end, COL_SSP585, "D"),
    ]:
        if len(sub) == 0:
            continue
        R_m, D_m = sub["R"].mean(), sub["D"].mean()
        R_sem = sub["R"].std() / np.sqrt(len(sub))
        D_sem = sub["D"].std() / np.sqrt(len(sub))
        ax.errorbar(
            R_m,
            D_m,
            xerr=R_sem,
            yerr=D_sem,
            fmt=marker,
            color=color,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1.0,
            capsize=3,
            zorder=6,
        )
        mean_data.append((label, R_m, D_m, color))

    # --- Bold arrows connecting means: Hist → SSP2-4.5 → SSP5-8.5 ---
    if len(mean_data) >= 2:
        # Historical → SSP2-4.5
        ax.annotate(
            "",
            xy=(mean_data[1][1], mean_data[1][2]),
            xytext=(mean_data[0][1], mean_data[0][2]),
            arrowprops=dict(
                arrowstyle="-|>", color=COL_SSP245, lw=1.5, mutation_scale=11, alpha=0.7
            ),
            zorder=5,
        )
    if len(mean_data) >= 3:
        # Historical → SSP5-8.5 (thicker, on top)
        ax.annotate(
            "",
            xy=(mean_data[2][1], mean_data[2][2]),
            xytext=(mean_data[0][1], mean_data[0][2]),
            arrowprops=dict(arrowstyle="-|>", color=COL_SSP585, lw=2.2, mutation_scale=13),
            zorder=5,
        )

    # --- ERA5 reanalysis ---
    if df_reanalysis is not None:
        era5 = df_reanalysis[df_reanalysis["dataset"] == "ERA5"].sort_values("center_year")
        if len(era5) > 0:
            REANALYSIS_STYLES["ERA5"]
            R_vals = era5["R"].values
            D_vals = era5["D"].values
            windows = era5["window"].values if "window" in era5.columns else None
            ax.plot(R_vals, D_vals, color="black", ls="-", lw=1.8, zorder=7)
            for i, (r, d) in enumerate(zip(R_vals, D_vals, strict=False)):
                win = windows[i] if windows is not None else None
                marker_w = WINDOW_MARKERS.get(win, "o")
                ax.scatter(
                    r,
                    d,
                    c="black",
                    marker=marker_w,
                    s=40,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=8,
                )
            if len(R_vals) >= 2:
                ax.annotate(
                    "",
                    xy=(R_vals[-1], D_vals[-1]),
                    xytext=(R_vals[-2], D_vals[-2]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=9),
                    zorder=7,
                )
            ax.text(
                R_vals[-1] + 0.008,
                D_vals[-1] + 0.0005,
                "ERA5",
                fontsize=7,
                fontweight="bold",
                va="bottom",
                zorder=9,
            )

    # --- "X% of models above R = 0.30" annotation ---
    if len(ssp585_end) > 0:
        frac_above = (ssp585_end["R"] > R_threshold).mean()
        pct = int(round(frac_above * 100))
        ax.text(
            0.97,
            0.50,
            f"{pct}% of SSP5-8.5\nmodels exceed\nR = {R_threshold:.2f}",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            zorder=10,
        )

    # --- Legend ---
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COL_HIST,
            markeredgecolor="white",
            markersize=6,
            label="Historical (1980\u20132000)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COL_SSP245,
            markeredgecolor="white",
            markersize=6,
            label="SSP2-4.5 (2080\u20132100)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=COL_SSP585,
            markeredgecolor="white",
            markersize=6,
            label="SSP5-8.5 (2080\u20132100)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            ls="-",
            markerfacecolor="black",
            markersize=5,
            lw=1.5,
            label="ERA5",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=6,
        handletextpad=0.4,
        borderpad=0.5,
        labelspacing=0.4,
    )

    # --- Axes ---
    ax.set_xlim(R_range)
    ax.set_ylim(D_range)
    ax.set_xlabel(
        r"Nonlinearity ratio  $R = \langle|\sigma^\prime|\rangle / \bar{\sigma}$", fontsize=8
    )
    ax.set_ylabel(r"Diabatic number  $D$", fontsize=8)
    ax.set_title("(a) Regime diagram", fontsize=9, loc="left", fontweight="bold", pad=8)


# ═══════════════════════════════════════════════════════════════════
# Panel (b): ΔKs² Distribution (raincloud)
# ═══════════════════════════════════════════════════════════════════

CHANGE_GROUPS = [
    ("ssp245", "mid_century", "SSP2-4.5\nMid"),
    ("ssp245", "end_century", "SSP2-4.5\nEnd"),
    ("ssp585", "mid_century", "SSP5-8.5\nMid"),
    ("ssp585", "end_century", "SSP5-8.5\nEnd"),
]


def _compute_dks_per_group(df):
    """Compute per-model ΔKs² (%) for each scenario/period group."""
    df_hist = df[(df["scenario"] == "historical") & (df["period"] == "historical")]
    hist_Ks = df_hist.dropna(subset=["Ks_250"]).set_index("model")["Ks_250"]

    groups = {}
    for scenario, period, label in CHANGE_GROUPS:
        sub = df[(df["scenario"] == scenario) & (df["period"] == period)]
        vals = []
        for _, row in sub.iterrows():
            m = row["model"]
            if (
                m in hist_Ks.index
                and np.isfinite(row["Ks_250"])
                and np.isfinite(hist_Ks.loc[m])
                and abs(hist_Ks.loc[m]) > 1e-20
            ):
                vals.append((row["Ks_250"] - hist_Ks.loc[m]) / abs(hist_Ks.loc[m]) * 100.0)
        groups[label] = np.array(vals)
    return groups


def _plot_panel_b(ax, df):
    """Raincloud showing ΔKs² distribution by scenario/period."""
    groups = _compute_dks_per_group(df)

    labels = list(groups.keys())
    data_list = [groups[k] for k in labels]
    positions = list(range(len(labels)))
    colors = [COL_SSP245, COL_SSP245, COL_SSP585, COL_SSP585]

    raincloud(
        ax,
        data_list,
        positions,
        colors,
        labels,
        violin_width=0.40,
        violin_alpha=0.25,
        box_width=0.20,
        box_alpha=0.55,
        dot_offset=-0.22,
        dot_size=16,
        dot_alpha=0.50,
        whis=(5, 95),
        show_counts=True,
        refline=0.0,
        refline_kw={"color": "#999999", "ls": "-", "lw": 0.7},
    )

    # Annotate % below zero for each group
    for _i, (label, data) in enumerate(zip(labels, data_list, strict=False)):
        n = len(data)
        n_below = (data < 0).sum()
        pct = n_below / n * 100 if n > 0 else 0
        median = np.median(data)
        logger.info(
            "Panel (b) %s: median=%.1f%%, %d/%d below 0 (%.0f%%)",
            label.replace("\n", " "),
            median,
            n_below,
            n,
            pct,
        )

    # Annotate the SSP5-8.5 end-century result prominently
    ssp585_end = groups["SSP5-8.5\nEnd"]
    n = len(ssp585_end)
    n_below = (ssp585_end < 0).sum()
    pct = n_below / n * 100
    med = np.median(ssp585_end)
    ann = (
        f"SSP5-8.5 end-century:\n"
        f"median = {med:.0f}%\n"
        f"{n_below}/{n} models ({pct:.0f}%) show decline"
    )
    ax.text(
        0.97,
        0.97,
        ann,
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        ha="right",
        fontweight="bold",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_ylabel(r"$\Delta K_s^2$ change from historical (%)", fontsize=8)
    ax.set_title(
        r"(b) Stationary wavenumber response ($\Delta K_s^2$)",
        fontsize=9,
        loc="left",
        fontweight="bold",
        pad=8,
    )


# ═══════════════════════════════════════════════════════════════════
# Panel (c): Causal Chain Schematic
# ═══════════════════════════════════════════════════════════════════


def _plot_panel_c(ax, pct_ks_decline=None):
    """Causal chain: 4 nodes connected by arrows + feedback loop."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Format Ks² decline percentage from data (fallback to "most" if unavailable)
    ks_label = (
        f"{pct_ks_decline:.0f}% of models\n(panel b)"
        if pct_ks_decline is not None
        else "most models\n(panel b)"
    )

    # "4× global" Arctic warming: Rantanen et al. (2022), Nat. Commun. 13, 708
    nodes = [
        (0.095, "Arctic\nwarming", r"$\Delta T_s$: 4$\times$ global", "#e0e0e0", "#999999"),
        (
            0.335,
            "R, D increase",
            r"$\bar{\sigma}\downarrow$, moisture$\uparrow$"
            "\n(panel a)",
            "#c6dbef",
            "#6baed6",
        ),
        (0.575, "$K_s^2$ decreases", ks_label, "#fdd0a2", "#e6550d"),
        (
            0.835,
            "Wave trapping\n& amplification",
            "SSW, blocking,\njet shifts",
            "#fcbba1",
            "#cb181d",
        ),
    ]

    node_w = 0.19
    node_h = 0.48
    node_y = 0.52  # vertical center

    for x_c, title, subtitle, fc, ec in nodes:
        x0 = x_c - node_w / 2
        y0 = node_y - node_h / 2
        rect = FancyBboxPatch(
            (x0, y0),
            node_w,
            node_h,
            boxstyle="round,pad=0.025",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
            zorder=3,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

        # Title (bold, larger)
        ax.text(
            x_c,
            node_y + 0.08,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            zorder=4,
            linespacing=1.2,
        )
        # Subtitle (clear, not too small)
        ax.text(
            x_c,
            node_y - 0.12,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            color="#333333",
            linespacing=1.2,
            zorder=4,
        )

    # --- Forward arrows between nodes ---
    arrow_colors = ["#777777", "#2171b5", "#cb181d"]
    for i in range(len(nodes) - 1):
        x_start = nodes[i][0] + node_w / 2 + 0.008
        x_end = nodes[i + 1][0] - node_w / 2 - 0.008
        ax.annotate(
            "",
            xy=(x_end, node_y),
            xytext=(x_start, node_y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=arrow_colors[i], lw=2.5, mutation_scale=16),
            zorder=2,
        )

    # --- Feedback arrow (curved, from node 4 back to node 1, below) ---
    y_bot = node_y - node_h / 2
    ax.annotate(
        "",
        xy=(nodes[0][0], y_bot - 0.02),
        xytext=(nodes[3][0], y_bot - 0.02),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#cb181d",
            lw=1.8,
            ls="--",
            mutation_scale=14,
            connectionstyle="arc3,rad=0.4",
        ),
        zorder=2,
    )

    # Feedback label
    ax.text(
        0.465,
        y_bot - 0.28,
        "positive feedback",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
        color="#cb181d",
        fontweight="bold",
        zorder=4,
    )


# ═══════════════════════════════════════════════════════════════════
# Panel connectors
# ═══════════════════════════════════════════════════════════════════


def _add_panel_connectors(fig, ax_a, ax_b, ax_c):
    """Add thin dotted lines from schematic nodes up to the relevant panels."""
    con_a = ConnectionPatch(
        xyA=(0.335, 1.0),
        coordsA=ax_c.transAxes,
        xyB=(0.5, 0.0),
        coordsB=ax_a.transAxes,
        color="#aaaaaa",
        lw=0.8,
        ls=":",
        zorder=0,
    )
    fig.add_artist(con_a)

    con_b = ConnectionPatch(
        xyA=(0.575, 1.0),
        coordsA=ax_c.transAxes,
        xyB=(0.5, 0.0),
        coordsB=ax_b.transAxes,
        color="#aaaaaa",
        lw=0.8,
        ls=":",
        zorder=0,
    )
    fig.add_artist(con_b)


# ═══════════════════════════════════════════════════════════════════
# Main figure assembly
# ═══════════════════════════════════════════════════════════════════


def plot_hero_figure(df, df_reanalysis, output_path):
    """Create the 3-panel hero figure."""
    apply_style()

    fig = plt.figure(figsize=(7.09, 6.0))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[3.0, 1.1],
        width_ratios=[1.1, 1.0],
        hspace=0.35,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _plot_panel_a(ax_a, df, df_reanalysis)
    _plot_panel_b(ax_b, df)

    # Compute Ks² decline percentage from data for panel (c) schematic
    dks_groups = _compute_dks_per_group(df)
    ssp585_end_dks = dks_groups.get("SSP5-8.5\nEnd", np.array([]))
    pct_ks_decline = None
    if len(ssp585_end_dks) > 0:
        pct_ks_decline = (ssp585_end_dks < 0).sum() / len(ssp585_end_dks) * 100

    _plot_panel_c(ax_c, pct_ks_decline=pct_ks_decline)
    _add_panel_connectors(fig, ax_a, ax_b, ax_c)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    logger.info("Saved hero figure to %s", output_path)
    plt.close(fig)

    return fig


def main():
    csv_path = PROJECT_ROOT / "data" / "diagnostics" / "regime_diagnostics.csv"
    rean_path = PROJECT_ROOT / "data" / "diagnostics" / "reanalysis_diagnostics.csv"

    if not csv_path.exists():
        logger.error("Diagnostics CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()
    logger.info("Loaded %d rows from %s (after exclusions)", len(df), csv_path)

    df_reanalysis = None
    if rean_path.exists():
        df_reanalysis = pd.read_csv(rean_path)
        logger.info("Loaded %d reanalysis rows from %s", len(df_reanalysis), rean_path)

    output_path = PROJECT_ROOT / "figures" / "hero_figure.pdf"
    plot_hero_figure(df, df_reanalysis, output_path)


if __name__ == "__main__":
    main()
