"""Generate the 4-quadrant regime diagram (R vs D) for publication.

The diagram shows the Arctic atmosphere migrating from a dry-linear regime
(low R, low D) to a moist-nonlinear regime (high R, high D) under warming.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .style import (
    PERIOD_LABELS,
    PERIOD_MARKERS,
    REANALYSIS_STYLES,
    SCENARIO_COLORS,
    WINDOW_MARKERS,
    apply_style,
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
    show_model_trajectories: bool = False,
    df_reanalysis: pd.DataFrame | None = None,
    exclude_models: set | None = None,
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
    show_model_trajectories : bool
        If True, draw thin arrows for each model's hist→mid→end trajectory.
    df_reanalysis : pd.DataFrame or None
        Reanalysis trajectory data.
    exclude_models : set or None
        Model names to exclude from the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_style()

    # Apply model exclusion filter
    if exclude_models:
        df = df[~df["model"].isin(exclude_models)].copy()
        logger.info("Excluded models: %s (%d rows remaining)",
                    exclude_models, len(df))

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

    # --- Quadrant labels (placed in corners to avoid overlap) ---
    pad_x = 0.01 * (R_range[1] - R_range[0])
    pad_y = 0.015 * (D_range[1] - D_range[0])
    # Bottom-left corner of Dry–Linear quadrant
    ax.text(R_range[0] + pad_x, D_range[0] + pad_y,
            "Dry–Linear", ha="left", va="bottom", fontsize=6.5,
            style="italic", color="grey", zorder=10)
    # Bottom-right corner of Dry–Nonlinear quadrant
    ax.text(R_range[1] - pad_x, D_range[0] + pad_y,
            "Dry–Nonlinear", ha="right", va="bottom", fontsize=6.5,
            style="italic", color="grey", zorder=10)
    # Just right of threshold line, near the top — avoids legend in upper-left
    ax.text(R_threshold - pad_x, D_threshold + pad_y,
            "Moist–Linear", ha="right", va="bottom", fontsize=6.5,
            style="italic", color="grey", zorder=10)
    # Top-right corner of Moist–Nonlinear quadrant
    ax.text(R_range[1] - pad_x, D_range[1] - pad_y,
            "Moist–Nonlinear", ha="right", va="top", fontsize=6.5,
            style="italic", color="#b2182b", fontweight="bold", zorder=10)

    # --- Scatter individual models ---
    _plot_group(ax, df, "historical", "historical", alpha=0.25, size=20)
    for scenario in ["ssp245", "ssp585"]:
        for period_key, _period_name in [("mid_century", "mid"),
                                         ("end_century", "end")]:
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
        if show_trajectories:
            _plot_trajectories(ax, means, zorder=4)

    # --- Individual model trajectories (SSP5-8.5 only) ---
    if show_model_trajectories:
        _plot_model_trajectories(ax, df, zorder=1)

    # --- Reanalysis trajectories ---
    if df_reanalysis is not None and len(df_reanalysis) > 0:
        _plot_reanalysis(ax, df_reanalysis, zorder=6)

    # --- R-D correlation annotation ---
    _annotate_RD_correlation(ax, df)

    # --- Legend ---
    _add_legend(ax, df_reanalysis=df_reanalysis)

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


def _plot_reanalysis(ax, df_rean, zorder=6):
    """Plot reanalysis trajectories with distinct markers per window."""
    for dataset in df_rean["dataset"].unique():
        sub = df_rean[df_rean["dataset"] == dataset].sort_values("center_year")
        if len(sub) == 0:
            continue

        style = REANALYSIS_STYLES.get(dataset, {
            "color": "black", "ls": "-", "lw": 1.5, "ms": 5,
            "label": dataset,
        })

        R_vals = sub["R"].values
        D_vals = sub["D"].values
        windows = sub["window"].values if "window" in sub.columns else None

        # Draw connected trajectory line
        ax.plot(R_vals, D_vals,
                color=style["color"], ls=style["ls"], lw=style["lw"],
                zorder=zorder)

        # Draw a distinct marker at each window
        for i, (r, d) in enumerate(zip(R_vals, D_vals, strict=False)):
            win = windows[i] if windows is not None else None
            marker = WINDOW_MARKERS.get(win, "o")
            ax.scatter(r, d, c=style["color"], marker=marker,
                       s=style["ms"] ** 2, edgecolors="white",
                       linewidths=0.5, zorder=zorder + 1)

        # Add arrow on last segment to show direction
        if len(R_vals) >= 2:
            ax.annotate(
                "", xy=(R_vals[-1], D_vals[-1]),
                xytext=(R_vals[-2], D_vals[-2]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=style["color"],
                    lw=style["lw"],
                    mutation_scale=8,
                ),
                zorder=zorder,
            )


def _plot_model_trajectories(ax, df, zorder=1):
    """Draw thin semi-transparent arrows for each model's hist→mid→end trajectory.

    Only SSP5-8.5 is shown to avoid clutter.
    """
    scenario = "ssp585"
    color = SCENARIO_COLORS[scenario]

    models = df["model"].unique()
    for model in models:
        points = {}
        # Historical point
        hist = df[(df["model"] == model) & (df["scenario"] == "historical")
                  & (df["period"] == "historical")]
        if len(hist) > 0:
            points["historical"] = (hist.iloc[0]["R"], hist.iloc[0]["D"])

        for period in ["mid_century", "end_century"]:
            sub = df[(df["model"] == model) & (df["scenario"] == scenario)
                     & (df["period"] == period)]
            if len(sub) > 0:
                points[period] = (sub.iloc[0]["R"], sub.iloc[0]["D"])

        # Draw arrows along trajectory
        trajectory = ["historical", "mid_century", "end_century"]
        for i in range(len(trajectory) - 1):
            if trajectory[i] in points and trajectory[i + 1] in points:
                x0, y0 = points[trajectory[i]]
                x1, y1 = points[trajectory[i + 1]]
                if np.isfinite([x0, y0, x1, y1]).all():
                    ax.annotate(
                        "", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=0.3,
                            alpha=0.12,
                            mutation_scale=6,
                        ),
                        zorder=zorder,
                    )


def _annotate_RD_correlation(ax, df):
    """Add R-D Pearson correlation annotation to the regime diagram."""
    valid = df.dropna(subset=["R", "D"])
    if len(valid) < 5:
        return
    r_corr, p_corr = pearsonr(valid["R"], valid["D"])
    p_str = f"p = {p_corr:.1e}" if p_corr >= 0.001 else "p < 0.001"
    ax.text(0.98, 0.02, f"r(R, D) = {r_corr:.2f}, {p_str}",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))
    logger.info("R-D correlation: r = %.3f, p = %.2e, n = %d",
                r_corr, p_corr, len(valid))


def _add_legend(ax, df_reanalysis=None):
    """Add a compact legend for scenarios, periods, and reanalysis."""
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

    # Reanalysis entries: dataset line + shared window markers
    if df_reanalysis is not None:
        # One entry per dataset (line style)
        for dataset in df_reanalysis["dataset"].unique():
            style = REANALYSIS_STYLES.get(dataset, {
                "color": "black", "ls": "-", "ms": 5, "label": dataset,
            })
            handles.append(plt.Line2D(
                [0], [0], marker="o", color=style["color"],
                ls=style["ls"], markerfacecolor=style["color"],
                markersize=5, lw=style.get("lw", 1.5),
                label=style["label"],
            ))
        # Window marker legend (shared across datasets)
        for win, marker in WINDOW_MARKERS.items():
            handles.append(plt.Line2D(
                [0], [0], marker=marker, color="w", ls="none",
                markerfacecolor="#666666", markeredgecolor="#333333",
                markeredgewidth=0.5, markersize=5,
                label=win,
            ))

    ax.legend(handles=handles, loc="upper left", fontsize=6.5)
