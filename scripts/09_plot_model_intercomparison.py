#!/usr/bin/env python3
"""Step 9: Model intercomparison figure for SI.

Box plots of R and D stratified by model resolution and model-top height.
Addresses Reviewer 3 Comment 2.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.style import apply_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# CMIP6 model metadata: approximate atmospheric resolution (degrees) and model-top (hPa)
# Sources: CMIP6 ES-DOC, model documentation
MODEL_META = {
    "ACCESS-CM2": {
        "resolution": 1.875,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "AWI-CM-1-1-MR": {
        "resolution": 0.94,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "BCC-CSM2-MR": {
        "resolution": 1.125,
        "model_top": 1.46,
        "category_res": "low",
        "category_top": "low",
    },
    "CAMS-CSM1-0": {
        "resolution": 1.125,
        "model_top": 1.0,
        "category_res": "low",
        "category_top": "low",
    },
    "CAS-ESM2-0": {
        "resolution": 1.41,
        "model_top": 1.0,
        "category_res": "low",
        "category_top": "low",
    },
    "CESM2": {
        "resolution": 1.25,
        "model_top": 0.0006,
        "category_res": "low",
        "category_top": "high",
    },
    "CESM2-WACCM": {
        "resolution": 1.25,
        "model_top": 0.0006,
        "category_res": "low",
        "category_top": "high",
    },
    "CMCC-CM2-SR5": {
        "resolution": 1.25,
        "model_top": 1.0,
        "category_res": "low",
        "category_top": "low",
    },
    "CMCC-ESM2": {
        "resolution": 1.25,
        "model_top": 1.0,
        "category_res": "low",
        "category_top": "low",
    },
    "CNRM-CM6-1": {
        "resolution": 1.41,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "CNRM-CM6-1-HR": {
        "resolution": 0.5,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "CNRM-ESM2-1": {
        "resolution": 1.41,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "CanESM5": {"resolution": 2.81, "model_top": 1.0, "category_res": "low", "category_top": "low"},
    "E3SM-1-1": {
        "resolution": 1.0,
        "model_top": 0.1,
        "category_res": "high",
        "category_top": "high",
    },
    "EC-Earth3": {
        "resolution": 0.7,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "EC-Earth3-CC": {
        "resolution": 0.7,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "EC-Earth3-Veg": {
        "resolution": 0.7,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "EC-Earth3-Veg-LR": {
        "resolution": 1.125,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "FGOALS-f3-L": {
        "resolution": 1.25,
        "model_top": 2.2,
        "category_res": "low",
        "category_top": "low",
    },
    "FGOALS-g3": {
        "resolution": 2.0,
        "model_top": 2.2,
        "category_res": "low",
        "category_top": "low",
    },
    "FIO-ESM-2-0": {
        "resolution": 1.25,
        "model_top": 2.2,
        "category_res": "low",
        "category_top": "low",
    },
    "GFDL-CM4": {
        "resolution": 1.0,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "GFDL-ESM4": {
        "resolution": 1.0,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "GISS-E2-1-G": {
        "resolution": 2.5,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "GISS-E2-1-H": {
        "resolution": 2.5,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "IITM-ESM": {
        "resolution": 1.875,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "INM-CM4-8": {
        "resolution": 2.0,
        "model_top": 0.2,
        "category_res": "low",
        "category_top": "high",
    },
    "INM-CM5-0": {
        "resolution": 2.0,
        "model_top": 0.2,
        "category_res": "low",
        "category_top": "high",
    },
    "IPSL-CM6A-LR": {
        "resolution": 2.5,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "KIOST-ESM": {
        "resolution": 1.875,
        "model_top": 1.0,
        "category_res": "low",
        "category_top": "low",
    },
    "MIROC-ES2L": {
        "resolution": 2.81,
        "model_top": 3.0,
        "category_res": "low",
        "category_top": "low",
    },
    "MIROC6": {
        "resolution": 1.41,
        "model_top": 0.004,
        "category_res": "low",
        "category_top": "high",
    },
    "MPI-ESM1-2-HR": {
        "resolution": 0.94,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "MPI-ESM1-2-LR": {
        "resolution": 1.875,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "MRI-ESM2-0": {
        "resolution": 1.125,
        "model_top": 0.01,
        "category_res": "low",
        "category_top": "high",
    },
    "NESM3": {"resolution": 1.875, "model_top": 1.0, "category_res": "low", "category_top": "low"},
    "NorESM2-MM": {
        "resolution": 1.0,
        "model_top": 0.01,
        "category_res": "high",
        "category_top": "high",
    },
    "TaiESM1": {
        "resolution": 1.25,
        "model_top": 0.0006,
        "category_res": "low",
        "category_top": "high",
    },
}

EXCLUDE_MODELS = {"MCM-UA-1-0", "MPI-ESM1-2-HR"}


def main():
    apply_style()

    csv_path = PROJECT_ROOT / "data" / "diagnostics" / "regime_diagnostics.csv"
    df = pd.read_csv(csv_path)
    df = df[~df["model"].isin(EXCLUDE_MODELS)]

    # Add metadata
    df["resolution"] = df["model"].map(
        lambda m: MODEL_META.get(m, {}).get("category_res", "unknown")
    )
    df["model_top"] = df["model"].map(
        lambda m: MODEL_META.get(m, {}).get("category_top", "unknown")
    )

    # Historical only
    hist = df[(df["scenario"] == "historical") & (df["period"] == "historical")]
    hist = hist[hist["resolution"] != "unknown"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

    colors = {"high": "#2166ac", "low": "#b2182b"}

    # Panel (a): R by resolution
    ax = axes[0, 0]
    for i, (cat, _label) in enumerate(
        [
            ("high", f"High-res\n(<1°, n={len(hist[hist['resolution'] == 'high'])})"),
            ("low", f"Low-res\n(≥1°, n={len(hist[hist['resolution'] == 'low'])})"),
        ]
    ):
        data = hist[hist["resolution"] == cat]["R"].dropna()
        ax.boxplot(
            [data],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=colors[cat], alpha=0.3),
            medianprops=dict(color=colors[cat], lw=2),
            whiskerprops=dict(color=colors[cat]),
            capprops=dict(color=colors[cat]),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=colors[cat], alpha=0.5),
        )
        ax.scatter(
            np.random.normal(i, 0.08, len(data)), data, c=colors[cat], s=15, alpha=0.5, zorder=3
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-res\n(<1°)", "Low-res\n(≥1°)"])
    ax.set_ylabel(r"$\mathcal{R}$")
    ax.set_title("(a) R by resolution", fontsize=9, loc="left", fontweight="bold")
    ax.axhline(0.3, color="grey", ls="--", lw=0.5)

    # Panel (b): D by resolution
    ax = axes[0, 1]
    for i, cat in enumerate(["high", "low"]):
        data = hist[hist["resolution"] == cat]["D"].dropna()
        ax.boxplot(
            [data],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=colors[cat], alpha=0.3),
            medianprops=dict(color=colors[cat], lw=2),
            whiskerprops=dict(color=colors[cat]),
            capprops=dict(color=colors[cat]),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=colors[cat], alpha=0.5),
        )
        ax.scatter(
            np.random.normal(i, 0.08, len(data)), data, c=colors[cat], s=15, alpha=0.5, zorder=3
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-res\n(<1°)", "Low-res\n(≥1°)"])
    ax.set_ylabel(r"$\mathcal{D}$")
    ax.set_title("(b) D by resolution", fontsize=9, loc="left", fontweight="bold")
    ax.axhline(0.03, color="grey", ls="--", lw=0.5)

    # Panel (c): R by model-top
    ax = axes[1, 0]
    colors_top = {"high": "#1b7837", "low": "#762a83"}
    for i, (cat, _label) in enumerate(
        [
            ("high", f"High-top\n(<1 hPa, n={len(hist[hist['model_top'] == 'high'])})"),
            ("low", f"Low-top\n(≥1 hPa, n={len(hist[hist['model_top'] == 'low'])})"),
        ]
    ):
        data = hist[hist["model_top"] == cat]["R"].dropna()
        ax.boxplot(
            [data],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=colors_top[cat], alpha=0.3),
            medianprops=dict(color=colors_top[cat], lw=2),
            whiskerprops=dict(color=colors_top[cat]),
            capprops=dict(color=colors_top[cat]),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=colors_top[cat], alpha=0.5),
        )
        ax.scatter(
            np.random.normal(i, 0.08, len(data)), data, c=colors_top[cat], s=15, alpha=0.5, zorder=3
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-top\n(<1 hPa)", "Low-top\n(≥1 hPa)"])
    ax.set_ylabel(r"$\mathcal{R}$")
    ax.set_title("(c) R by model-top", fontsize=9, loc="left", fontweight="bold")
    ax.axhline(0.3, color="grey", ls="--", lw=0.5)

    # Panel (d): D by model-top
    ax = axes[1, 1]
    for i, cat in enumerate(["high", "low"]):
        data = hist[hist["model_top"] == cat]["D"].dropna()
        ax.boxplot(
            [data],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=colors_top[cat], alpha=0.3),
            medianprops=dict(color=colors_top[cat], lw=2),
            whiskerprops=dict(color=colors_top[cat]),
            capprops=dict(color=colors_top[cat]),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=colors_top[cat], alpha=0.5),
        )
        ax.scatter(
            np.random.normal(i, 0.08, len(data)), data, c=colors_top[cat], s=15, alpha=0.5, zorder=3
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-top\n(<1 hPa)", "Low-top\n(≥1 hPa)"])
    ax.set_ylabel(r"$\mathcal{D}$")
    ax.set_title("(d) D by model-top", fontsize=9, loc="left", fontweight="bold")
    ax.axhline(0.03, color="grey", ls="--", lw=0.5)

    fig.tight_layout()

    output_path = PROJECT_ROOT / "figures" / "model_intercomparison.pdf"
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    logger.info("Saved model intercomparison figure to %s", output_path)


if __name__ == "__main__":
    main()
