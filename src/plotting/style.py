"""Nature Communications figure style configuration."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature Comms: single column = 88 mm ≈ 3.46 in; double = 180 mm ≈ 7.09 in
# We use a slightly larger single-column figure for the regime diagram.
NATURE_COMMS_RCPARAMS = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 5,

    # Axes
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Ticks
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend
    "legend.frameon": False,
    "legend.handlelength": 1.5,

    # Figure
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # PDF
    "pdf.fonttype": 42,  # TrueType fonts in PDF
    "ps.fonttype": 42,
}

# Colour palette for scenarios
SCENARIO_COLORS = {
    "historical": "#2166ac",   # Blue
    "ssp245": "#f4a582",       # Light orange
    "ssp585": "#b2182b",       # Dark red
}

# Marker styles for time periods
PERIOD_MARKERS = {
    "historical": "o",
    "mid_century": "s",
    "end_century": "D",
}

# Labels for legend
PERIOD_LABELS = {
    "historical": "Historical (1980-2000)",
    "ssp245_mid": "SSP2-4.5 (2040-2060)",
    "ssp245_end": "SSP2-4.5 (2080-2100)",
    "ssp585_mid": "SSP5-8.5 (2040-2060)",
    "ssp585_end": "SSP5-8.5 (2080-2100)",
}

# Reanalysis styling (line style per dataset)
REANALYSIS_STYLES = {
    "ERA5": {"color": "#000000", "ls": "-", "lw": 2.0, "ms": 6,
             "label": "ERA5"},
    "NCEP-R1": {"color": "#555555", "ls": "--", "lw": 1.5, "ms": 5,
                "label": "NCEP/NCAR R1"},
    "NCEP-R2": {"color": "#888888", "ls": ":", "lw": 1.5, "ms": 5,
                "label": "NCEP-DOE R2"},
    "JRA-55": {"color": "#666666", "ls": "-.", "lw": 1.5, "ms": 4,
               "label": "JRA-55"},
    "MERRA-2": {"color": "#999999", "ls": "-.", "lw": 1.5, "ms": 4,
                "label": "MERRA-2"},
}

# Distinct markers for each 30-year climatological window
WINDOW_MARKERS = {
    "1961-1990": "o",    # circle
    "1981-2010": "s",    # square
    "1991-2020": "D",    # diamond
}


def apply_style():
    """Apply Nature Communications style to all subsequent plots."""
    mpl.rcParams.update(NATURE_COMMS_RCPARAMS)


def get_figure(figsize=None):
    """Create a styled figure and axes.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches. Default (5.5, 5.0).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    apply_style()
    if figsize is None:
        figsize = (5.5, 5.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return fig, ax
