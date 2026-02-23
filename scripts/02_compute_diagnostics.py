#!/usr/bin/env python3
"""Step 2: Compute R (nonlinearity ratio) and D (diabatic number) per model/period.

Reads preprocessed NetCDF files from data/processed/ and outputs a CSV
summary table to data/diagnostics/regime_diagnostics.csv.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.physics.static_stability import static_stability, layer_mean_stability
from src.physics.nonlinearity import nonlinearity_ratio
from src.physics.diabatic import (
    meridional_T_gradient, arctic_mean_precipitation, diabatic_number,
)
from src.data.preprocess import subset_arctic, _get_plev_name

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_processed(model: str, exp: str, var: str,
                   period: str, data_dir: Path) -> xr.Dataset | None:
    """Load a preprocessed NetCDF file, returning None if missing."""
    fname = f"{model}_{exp}_{var}_{period}.nc"
    path = data_dir / fname
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None
    ds = xr.open_dataset(path)
    # Squeeze singleton dimensions from Pangeo (member_id, dcpp_init_year)
    for dim in list(ds.dims):
        if dim not in ("time", "plev", "lev", "lat", "lon", "latitude", "longitude"):
            if ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)
    return ds


def compute_R(ds_ta: xr.Dataset, cfg: dict) -> float:
    """Compute the nonlinearity ratio R from temperature data."""
    # Extract temperature
    ta = ds_ta["ta"] if "ta" in ds_ta else ds_ta[list(ds_ta.data_vars)[0]]

    # Compute static stability on all levels
    sigma = static_stability(ta)

    # Layer-mean stability (500-850 hPa)
    sigma_layer = layer_mean_stability(
        sigma,
        p_top=cfg["pressure"]["layer_top"],
        p_bot=cfg["pressure"]["layer_bot"],
    )

    # Nonlinearity ratio
    R = nonlinearity_ratio(sigma_layer)
    return R


def compute_D(ds_ta: xr.Dataset, ds_pr: xr.Dataset, cfg: dict,
              sigma_bar_value: float | None = None) -> float:
    """Compute the diabatic number D from temperature and precipitation."""
    # Temperature
    ta = ds_ta["ta"] if "ta" in ds_ta else ds_ta[list(ds_ta.data_vars)[0]]

    # Meridional temperature gradient at Arctic edge
    # Data may only cover 60-90°N with coarse resolution,
    # so use 60-75°N to ensure enough points for differentiation
    lat_min_avail = float(ta["lat"].min())
    edge_lat_min = max(cfg["domain"]["edge_lat_min"], lat_min_avail)
    edge_lat_max = 75.0  # Broad enough for coarse-res models

    dT_dy = meridional_T_gradient(
        ta,
        lat_min=edge_lat_min,
        lat_max=edge_lat_max,
        plev_level=cfg["pressure"]["gradient_level"],
    )

    # If gradient is near zero (limited latitude range), use a default
    if dT_dy < 1e-7:
        logger.warning("dT/dy too small (%.2e); using default 5e-6 K/m", dT_dy)
        dT_dy = 5e-6  # ~5 K over 1000 km

    # Precipitation
    pr = ds_pr["pr"] if "pr" in ds_pr else ds_pr[list(ds_pr.data_vars)[0]]
    pr_arctic = arctic_mean_precipitation(
        pr,
        lat_min=cfg["domain"]["arctic_lat_min"],
        lat_max=cfg["domain"]["arctic_lat_max"],
    )

    # Static stability for Ld computation
    if sigma_bar_value is None:
        sigma = static_stability(ta)
        sigma_layer = layer_mean_stability(
            sigma,
            p_top=cfg["pressure"]["layer_top"],
            p_bot=cfg["pressure"]["layer_bot"],
        )
        sigma_bar_value = float(sigma_layer.mean())

    # Diabatic number
    D = diabatic_number(
        pr_arctic, dT_dy, sigma_bar_value,
        Lv=cfg["constants"]["Lv"],
        cp=cfg["constants"]["cp"],
        f0=cfg["constants"]["f0"],
        H=cfg["constants"]["H"],
    )
    return D


def main():
    cfg = load_config()
    data_dir = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "data" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model in cfg["models"]:
        for period_name, period_cfg in cfg["periods"].items():
            mapping = cfg["period_experiment"][period_name]
            experiments = mapping if isinstance(mapping, list) else [mapping]

            for exp in experiments:
                logger.info("Computing: %s / %s / %s", model, exp, period_name)

                ds_ta = load_processed(model, exp, "ta", period_name, data_dir)
                ds_pr = load_processed(model, exp, "pr", period_name, data_dir)

                if ds_ta is None or ds_pr is None:
                    logger.warning("  Skipping (missing data)")
                    continue

                try:
                    R = compute_R(ds_ta, cfg)
                    D = compute_D(ds_ta, ds_pr, cfg)

                    results.append({
                        "model": model,
                        "period": period_name,
                        "scenario": exp,
                        "R": R,
                        "D": D,
                        "label": period_cfg["label"],
                    })
                    logger.info("  R = %.4f, D = %.4f", R, D)

                except Exception as e:
                    logger.error("  Failed: %s", e)
                    continue

                finally:
                    ds_ta.close()
                    ds_pr.close()

    # Save results
    df = pd.DataFrame(results)
    out_path = out_dir / "regime_diagnostics.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved diagnostics to %s (%d rows)", out_path, len(df))
    if len(df) > 0:
        summary = df.groupby(["scenario", "period"])[["R", "D"]].agg(["mean", "std", "count"])
        logger.info("\nSummary:\n%s", summary.to_string())
    else:
        logger.warning("No diagnostics computed!")


if __name__ == "__main__":
    main()
