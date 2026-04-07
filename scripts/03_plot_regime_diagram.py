#!/usr/bin/env python3
"""Step 3: Generate the publication-quality regime diagram from diagnostics CSV."""

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.plotting.regime_diagram import plot_regime_diagram

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()

    # Load CMIP6 diagnostics
    csv_path = PROJECT_ROOT / "data" / "diagnostics" / "regime_diagnostics.csv"
    if not csv_path.exists():
        logger.error("Diagnostics CSV not found: %s", csv_path)
        logger.error("Run 02_compute_diagnostics.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    logger.info("Models: %s", sorted(df["model"].unique()))
    logger.info("Scenarios: %s", sorted(df["scenario"].unique()))

    # Load reanalysis diagnostics (optional)
    rean_path = PROJECT_ROOT / "data" / "diagnostics" / "reanalysis_diagnostics.csv"
    df_rean = None
    if rean_path.exists():
        df_rean = pd.read_csv(rean_path)
        logger.info("Loaded %d reanalysis rows from %s", len(df_rean), rean_path)
        logger.info("Datasets: %s", sorted(df_rean["dataset"].unique()))
    else:
        logger.info("No reanalysis diagnostics found (run 04_reanalysis_diagnostics.py)")

    # Plot
    plot_cfg = cfg["plot"]
    output_path = PROJECT_ROOT / "figures" / f"regime_diagram.{plot_cfg['format']}"

    # Models to exclude from analysis (quality filters)
    exclude = {"MCM-UA-1-0", "MPI-ESM1-2-HR"}

    fig = plot_regime_diagram(
        df,
        output_path=output_path,
        figsize=tuple(plot_cfg["figsize"]),
        R_range=tuple(plot_cfg["R_range"]),
        D_range=tuple(plot_cfg["D_range"]),
        R_threshold=plot_cfg["R_threshold"],
        D_threshold=plot_cfg["D_threshold"],
        show_model_trajectories=True,
        df_reanalysis=df_rean,
        exclude_models=exclude,
    )

    # Also save PNG for quick preview
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    logger.info("Also saved PNG preview: %s", png_path)

    logger.info("Done. Figure saved to %s", output_path)


if __name__ == "__main__":
    main()
