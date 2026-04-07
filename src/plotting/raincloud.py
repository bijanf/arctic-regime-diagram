"""Raincloud plot helper using pure matplotlib + scipy.

A raincloud plot combines:
  - Half-violin (KDE density, one side only) → shows distribution shape
  - Box plot (IQR + whiskers) → shows quartiles
  - Jittered strip (raw dots) → shows individual data points

Reference: Allen et al. (2019), Wellcome Open Research.
"""

import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import stats


def _half_violin(
    ax,
    data,
    pos,
    color,
    side="right",
    width=0.3,
    alpha=0.3,
    kde_bw="scott",
    exclude=None,
    clip=None,
):
    """Draw a half-violin (KDE density) on one side of a position.

    Parameters
    ----------
    ax : Axes
    data : array-like
        Values to compute KDE on.
    pos : float
        X-position center.
    color : str
        Fill color.
    side : str
        'right' or 'left'.
    width : float
        Maximum width of the violin in data coordinates.
    alpha : float
        Fill transparency.
    kde_bw : str or float
        Bandwidth for gaussian_kde.
    exclude : array-like or None
        Values to exclude from KDE (but still in the data range).
    clip : tuple or None
        (min, max) to clip the KDE evaluation range.
    """
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 4:
        return

    # Optionally exclude outliers from KDE computation
    d_kde = d
    if exclude is not None:
        mask = np.ones(len(d), dtype=bool)
        for val in exclude:
            mask &= ~np.isclose(d, val, atol=0.5)
        d_kde = d[mask]
        if len(d_kde) < 4:
            d_kde = d

    try:
        if isinstance(kde_bw, str):
            kde = stats.gaussian_kde(d_kde, bw_method=kde_bw)
        else:
            kde = stats.gaussian_kde(d_kde, bw_method=kde_bw)
    except (np.linalg.LinAlgError, ValueError):
        return

    y_min = d.min() if clip is None else clip[0]
    y_max = d.max() if clip is None else clip[1]
    pad = (y_max - y_min) * 0.05
    y_grid = np.linspace(y_min - pad, y_max + pad, 200)
    density = kde(y_grid)

    # Normalize density so max = width
    if density.max() > 0:
        density = density / density.max() * width

    if side == "right":
        ax.fill_betweenx(y_grid, pos, pos + density, alpha=alpha, color=color, lw=0, zorder=1)
        ax.plot(pos + density, y_grid, color=color, lw=0.5, alpha=0.6, zorder=1)
    else:
        ax.fill_betweenx(y_grid, pos - density, pos, alpha=alpha, color=color, lw=0, zorder=1)
        ax.plot(pos - density, y_grid, color=color, lw=0.5, alpha=0.6, zorder=1)


def _box(
    ax, data, pos, color, width=0.15, whis=(5, 95), alpha=0.6, median_color="white", median_lw=1.5
):
    """Draw a single box plot element at a given position.

    Parameters
    ----------
    ax : Axes
    data : array-like
    pos : float
        X-position center.
    color : str
        Box fill color.
    width : float
        Box width.
    whis : tuple
        (lower, upper) percentiles for whiskers.
    alpha : float
        Box fill alpha.
    median_color : str
    median_lw : float
    """
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return

    q25, median, q75 = np.percentile(d, [25, 50, 75])
    wlo, whi = np.percentile(d, list(whis))

    # Box
    box_x = pos - width / 2
    box_h = q75 - q25
    rect = FancyBboxPatch(
        (box_x, q25),
        width,
        box_h,
        boxstyle="round,pad=0.01",
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=0.6,
        zorder=3,
    )
    ax.add_patch(rect)

    # Median line
    ax.plot(
        [pos - width / 2.2, pos + width / 2.2],
        [median, median],
        color=median_color,
        lw=median_lw,
        solid_capstyle="round",
        zorder=4,
    )

    # Whiskers
    ax.plot([pos, pos], [wlo, q25], color=color, lw=0.8, zorder=2)
    ax.plot([pos, pos], [q75, whi], color=color, lw=0.8, zorder=2)

    # Whisker caps
    cap_w = width * 0.4
    ax.plot([pos - cap_w / 2, pos + cap_w / 2], [wlo, wlo], color=color, lw=0.8, zorder=2)
    ax.plot([pos - cap_w / 2, pos + cap_w / 2], [whi, whi], color=color, lw=0.8, zorder=2)

    return median


def _strip(
    ax,
    data,
    pos,
    color,
    offset=-0.15,
    size=12,
    alpha=0.5,
    seed=42,
    jitter_width=0.06,
    highlight_idx=None,
    highlight_size=22,
    zorder=2,
):
    """Draw jittered strip of individual data points.

    Parameters
    ----------
    ax : Axes
    data : array-like
    pos : float
        X-position center.
    color : str
    offset : float
        Horizontal offset from pos.
    size : float
        Marker area (pt²).
    alpha : float
    seed : int
        RNG seed for reproducible jitter.
    jitter_width : float
        Max jitter range.
    highlight_idx : list or None
        Indices of data points to highlight (e.g., outlier models).
    highlight_size : float
        Size of highlighted markers.
    zorder : int
    """
    d = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-jitter_width, jitter_width, len(d))

    # Regular dots
    regular_mask = np.ones(len(d), dtype=bool)
    if highlight_idx is not None:
        regular_mask[highlight_idx] = False

    ax.scatter(
        pos + offset + jitter[regular_mask],
        d[regular_mask],
        s=size,
        c=color,
        alpha=alpha,
        edgecolors="none",
        zorder=zorder,
    )

    # Highlighted dots (outliers)
    if highlight_idx is not None and len(highlight_idx) > 0:
        hi = np.array(highlight_idx)
        ax.scatter(
            pos + offset + jitter[hi],
            d[hi],
            s=highlight_size,
            c=color,
            alpha=min(alpha + 0.2, 0.9),
            edgecolors="black",
            linewidths=0.5,
            zorder=zorder + 1,
        )


def raincloud(
    ax,
    data_list,
    positions,
    colors,
    labels,
    violin_width=0.3,
    violin_side="right",
    violin_alpha=0.3,
    box_width=0.15,
    box_alpha=0.6,
    dot_offset=-0.18,
    dot_size=12,
    dot_alpha=0.5,
    whis=(5, 95),
    kde_bw="scott",
    exclude_from_kde=None,
    show_counts=True,
    count_y=None,
    count_fontsize=5.5,
    count_color="grey",
    refline=None,
    refline_kw=None,
):
    """Draw a full raincloud plot: half-violin + box + jittered strip.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data_list : list of array-like
        One array of values per group.
    positions : list of float
        X-positions for each group.
    colors : list of str
        Color per group.
    labels : list of str
        X-tick labels.
    violin_width : float
    violin_side : str
        'right' or 'left'.
    violin_alpha : float
    box_width : float
    box_alpha : float
    dot_offset : float
        Offset for strip dots (negative = left of center).
    dot_size : float
    dot_alpha : float
    whis : tuple
        Whisker percentiles.
    kde_bw : str or float
    exclude_from_kde : list of array-like or None
        Per-group values to exclude from KDE.
    show_counts : bool
        If True, annotate model count below each group.
    count_y : float or None
        Y-position for count text. If None, use axis bottom.
    count_fontsize : float
    count_color : str
    refline : float or None
        If given, draw a horizontal reference line.
    refline_kw : dict or None
        kwargs for the reference line.

    Returns
    -------
    list of float
        Median values per group.
    """
    medians = []

    if refline is not None:
        kw = {"color": "#888888", "ls": "-", "lw": 0.5, "zorder": 0}
        if refline_kw:
            kw.update(refline_kw)
        ax.axhline(refline, **kw)

    for i, (data, pos, color) in enumerate(zip(data_list, positions, colors, strict=False)):
        d = np.asarray(data, dtype=float)

        # KDE exclusions
        excl = None
        if exclude_from_kde is not None and i < len(exclude_from_kde):
            excl = exclude_from_kde[i]

        _half_violin(
            ax,
            d,
            pos,
            color,
            side=violin_side,
            width=violin_width,
            alpha=violin_alpha,
            kde_bw=kde_bw,
            exclude=excl,
        )

        med = _box(ax, d, pos, color, width=box_width, whis=whis, alpha=box_alpha)
        medians.append(med)

        _strip(ax, d, pos, color, offset=dot_offset, size=dot_size, alpha=dot_alpha)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    if show_counts:
        if count_y is None:
            count_y = ax.get_ylim()[0]
        for data, pos in zip(data_list, positions, strict=False):
            n = np.sum(np.isfinite(data))
            ax.text(
                pos,
                count_y,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=count_fontsize,
                color=count_color,
            )

    return medians


def horizontal_strip(
    ax,
    data_list,
    positions,
    colors,
    labels,
    box_height=0.25,
    dot_size=15,
    dot_alpha=0.5,
    mean_size=50,
    seed=42,
):
    """Horizontal strip chart: dots + box + mean diamond, stacked vertically.

    Parameters
    ----------
    ax : Axes
    data_list : list of array-like
        Values per group.
    positions : list of float
        Y-positions (vertical) for each group.
    colors : list of str
    labels : list of str
        Y-tick labels.
    box_height : float
    dot_size : float
    dot_alpha : float
    mean_size : float
        Size of the mean diamond marker.
    seed : int
    """
    rng = np.random.default_rng(seed)

    for data, pos, color in zip(data_list, positions, colors, strict=False):
        d = np.asarray(data, dtype=float)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue

        q25, med, q75 = np.percentile(d, [25, 50, 75])
        mean_val = d.mean()

        # Jittered dots
        jitter = rng.uniform(-box_height * 0.4, box_height * 0.4, len(d))
        ax.scatter(
            d, pos + jitter, s=dot_size, c=color, alpha=dot_alpha, edgecolors="none", zorder=2
        )

        # Horizontal box
        rect = FancyBboxPatch(
            (q25, pos - box_height / 2),
            q75 - q25,
            box_height,
            boxstyle="round,pad=0.01",
            facecolor=color,
            alpha=0.4,
            edgecolor=color,
            linewidth=0.6,
            zorder=3,
        )
        ax.add_patch(rect)

        # Median line
        ax.plot(
            [med, med],
            [pos - box_height / 2.2, pos + box_height / 2.2],
            color="white",
            lw=1.5,
            solid_capstyle="round",
            zorder=4,
        )

        # Mean diamond
        ax.scatter(
            [mean_val],
            [pos],
            s=mean_size,
            c=color,
            marker="D",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
