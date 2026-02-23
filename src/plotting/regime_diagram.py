"""Generate the 4-quadrant regime diagram (R vs D) for publication.

The diagram shows the Arctic atmosphere migrating from a dry-linear regime
(low R, low D) to a moist-nonlinear regime (high R, high D) under warming.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .style import (
    apply_style, SCENARIO_COLORS, PERIOD_MARKERS, PERIOD_LABELS,
)

logger = logging.getLogger(__name__)


def plot_regime_diagram(
    df: pd.DataFrame,
    output_path: str | Path = "figures/regime_diagram.pdf",
    figsize: tuple = (5.5, 5.0),
    R_range: tuple = (0.0, 0.5),
    D_range: tuple = (0.0, 4.0),
    R_threshold: float = 0.1,
    D_threshold: float = 1.0,
    show_trajectories: bool = True,
    show_multi_model_mean: bool = True,
) -> plt.Figure:
    """Create the 4-quadrant regime diagram.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'model', 'period', 'scenario', 'R', 'D'.
        Optionally: 'R_std', 'D_std' for error bars.
    output_path : str or Path
        Where to save the figure.
    figsize : tuple
        Figure size in inches.
    R_range, D_range : tuple
        Axis limits for R and D.
    R_threshold : float
        Vertical dashed line separating linear / nonlinear.
    D_threshold : float
        Horizontal dashed line separating dry / moist.
    show_trajectories : bool
        If True, draw arrows connecting multi-model means across periods.
    show_multi_model_mean : bool
        If True, plot large markers for the multi-model mean.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # --- Quadrant shading ---
    ax.axvspan(R_range[0], R_threshold, ymin=0,
               ymax=(D_threshold - D_range[0]) / (D_range[1] - D_range[0]),
               alpha=0.06, color="#2166ac", zorder=0)
    ax.axvspan(R_threshold, R_range[1], ymin=0,
               ymax=(D_threshold - D_range[0]) / (D_range[1] - D_range[0]),
               alpha=0.06, color="#f4a582", zorder=0)
    ax.axvspan(R_range[0], R_threshold,
               ymin=(D_threshold - D_range[0]) / (D_range[1] - D_range[0]),
               ymax=1, alpha=0.06, color="#92c5de", zorder=0)
    ax.axvspan(R_threshold, R_range[1],
               ymin=(D_threshold - D_range[0]) / (D_range[1] - D_range[0]),
               ymax=1, alpha=0.06, color="#b2182b", zorder=0)

    # --- Threshold lines ---
    ax.axvline(R_threshold, color="grey", ls="--", lw=0.8, zorder=1)
    ax.axhline(D_threshold, color="grey", ls="--", lw=0.8, zorder=1)

    # --- Quadrant labels ---
    pad = 0.02
    ax.text(R_threshold / 2, D_range[0] + pad * (D_range[1] - D_range[0]),
            "Dry–Linear", ha="center", va="bottom", fontsize=6.5,
            style="italic", color="grey")
    ax.text((R_threshold + R_range[1]) / 2,
            D_range[0] + pad * (D_range[1] - D_range[0]),
            "Dry–Nonlinear", ha="center", va="bottom", fontsize=6.5,
            style="italic", color="grey")
    ax.text(R_threshold / 2, D_range[1] - pad * (D_range[1] - D_range[0]),
            "Moist–Linear", ha="center", va="top", fontsize=6.5,
            style="italic", color="grey")
    ax.text((R_threshold + R_range[1]) / 2,
            D_range[1] - pad * (D_range[1] - D_range[0]),
            "Moist–Nonlinear", ha="center", va="top", fontsize=6.5,
            style="italic", color="#b2182b", fontweight="bold")

    # --- Scatter individual models ---
    _plot_group(ax, df, "historical", "historical", alpha=0.25, size=20)
    for scenario in ["ssp245", "ssp585"]:
        for period_key, period_name in [("mid_century", "mid"),
                                         ("end_century", "end")]:
            label_key = f"{scenario}_{period_name}"
            sub = df[(df["scenario"] == scenario) & (df["period"] == period_key)]
            if len(sub) == 0:
                continue
            color = SCENARIO_COLORS[scenario]
            marker = PERIOD_MARKERS[period_key]
            ax.scatter(sub["R"], sub["D"], c=color, marker=marker,
                       s=20, alpha=0.25, edgecolors="none", zorder=2)

    # --- Multi-model means ---
    if show_multi_model_mean:
        means = _compute_means(df)
        _plot_means(ax, means, zorder=5)

    # --- Trajectories ---
    if show_trajectories and show_multi_model_mean:
        means = _compute_means(df)
        _plot_trajectories(ax, means, zorder=4)

    # --- Legend ---
    _add_legend(ax)

    # --- Axes ---
    ax.set_xlim(R_range)
    ax.set_ylim(D_range)
    ax.set_xlabel(r"Nonlinearity ratio  $R = \langle|\sigma^\prime|\rangle / \bar{\sigma}$")
    ax.set_ylabel(r"Diabatic number  $D = L_v \cdot \overline{P} \,/\, "
                  r"(c_p \cdot |\partial T/\partial y| \cdot f_0 \cdot L_d \cdot H)$")

    fig.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    logger.info("Saved regime diagram to %s", output_path)

    return fig


def _plot_group(ax, df, scenario, period, alpha=0.3, size=20):
    """Plot scatter for one scenario-period group."""
    sub = df[(df["scenario"] == scenario) & (df["period"] == period)]
    if len(sub) == 0:
        return
    color = SCENARIO_COLORS.get(scenario, "grey")
    marker = PERIOD_MARKERS.get(period, "o")
    ax.scatter(sub["R"], sub["D"], c=color, marker=marker, s=size,
               alpha=alpha, edgecolors="none", zorder=2)


def _compute_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-model mean R, D (and std) per scenario-period."""
    groups = df.groupby(["scenario", "period"]).agg(
        R_mean=("R", "mean"),
        D_mean=("D", "mean"),
        R_sem=("R", "std"),
        D_sem=("D", "std"),
        n=("R", "count"),
    ).reset_index()
    # Standard error of the mean
    groups["R_sem"] = groups["R_sem"] / np.sqrt(groups["n"])
    groups["D_sem"] = groups["D_sem"] / np.sqrt(groups["n"])
    return groups


def _plot_means(ax, means: pd.DataFrame, zorder=5):
    """Plot multi-model mean markers with error bars."""
    for _, row in means.iterrows():
        color = SCENARIO_COLORS.get(row["scenario"], "grey")
        marker = PERIOD_MARKERS.get(row["period"], "o")
        ax.errorbar(
            row["R_mean"], row["D_mean"],
            xerr=row["R_sem"], yerr=row["D_sem"],
            fmt=marker, color=color, markersize=8,
            markeredgecolor="white", markeredgewidth=0.5,
            elinewidth=0.8, capsize=2, zorder=zorder,
        )


def _plot_trajectories(ax, means: pd.DataFrame, zorder=4):
    """Draw arrows connecting multi-model means across time periods."""
    for scenario in ["ssp245", "ssp585"]:
        color = SCENARIO_COLORS[scenario]
        points = {}

        # Historical
        hist = means[(means["scenario"] == "historical") &
                     (means["period"] == "historical")]
        if len(hist) > 0:
            points["historical"] = (
                hist.iloc[0]["R_mean"], hist.iloc[0]["D_mean"]
            )

        for period in ["mid_century", "end_century"]:
            sub = means[(means["scenario"] == scenario) &
                        (means["period"] == period)]
            if len(sub) > 0:
                points[period] = (
                    sub.iloc[0]["R_mean"], sub.iloc[0]["D_mean"]
                )

        # Draw arrows
        trajectory = ["historical", "mid_century", "end_century"]
        for i in range(len(trajectory) - 1):
            if trajectory[i] in points and trajectory[i + 1] in points:
                x0, y0 = points[trajectory[i]]
                x1, y1 = points[trajectory[i + 1]]
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=1.2,
                        mutation_scale=10,
                    ),
                    zorder=zorder,
                )


def _add_legend(ax):
    """Add a compact legend for scenarios and periods."""
    handles = []

    # Historical
    handles.append(plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor=SCENARIO_COLORS["historical"],
        markersize=6, label=PERIOD_LABELS["historical"],
    ))

    # SSP2-4.5
    for period, label_key in [("mid_century", "ssp245_mid"),
                               ("end_century", "ssp245_end")]:
        handles.append(plt.Line2D(
            [0], [0], marker=PERIOD_MARKERS[period], color="w",
            markerfacecolor=SCENARIO_COLORS["ssp245"],
            markersize=6, label=PERIOD_LABELS[label_key],
        ))

    # SSP5-8.5
    for period, label_key in [("mid_century", "ssp585_mid"),
                               ("end_century", "ssp585_end")]:
        handles.append(plt.Line2D(
            [0], [0], marker=PERIOD_MARKERS[period], color="w",
            markerfacecolor=SCENARIO_COLORS["ssp585"],
            markersize=6, label=PERIOD_LABELS[label_key],
        ))

    ax.legend(handles=handles, loc="upper left", fontsize=6.5)
