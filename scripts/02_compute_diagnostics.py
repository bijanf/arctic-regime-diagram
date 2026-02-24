#!/usr/bin/env python3
"""Step 2: Compute R, D, Eady growth rate, and wave diagnostics per model/period.

Reads preprocessed NetCDF files from data/processed/ and outputs a CSV
summary table to data/diagnostics/regime_diagnostics.csv.

Columns: model, period, scenario, R, R_annual, R_djf, D,
         eady_dry, eady_moist, eady_ratio, Ld_mean_km, Ks_250, label
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
from src.physics.nonlinearity import nonlinearity_ratio, nonlinearity_ratio_seasonal
from src.physics.diabatic import (
    meridional_T_gradient, arctic_mean_precipitation, diabatic_number,
)
from src.physics.baroclinic import compute_eady_diagnostics
from src.physics.wave_diagnostics import compute_wave_diagnostics
from src.data.preprocess import subset_arctic, _get_plev_name

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_processed(model: str, exp: str, var: str,
                   period: str, data_dir: Path) -> xr.Dataset | None:
    """Load a preprocessed NetCDF file, returning None if missing or corrupt."""
    fname = f"{model}_{exp}_{var}_{period}.nc"
    path = data_dir / fname
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None

    # Reject files smaller than 1KB (empty shells)
    if path.stat().st_size < 1024:
        logger.warning("Suspiciously small file (%d bytes): %s",
                        path.stat().st_size, path)
        return None

    try:
        ds = xr.open_dataset(path)
    except ValueError:
        # Handle non-standard calendars (365_day, 360_day, etc.)
        try:
            ds = xr.open_dataset(path, use_cftime=True)
        except Exception as e:
            logger.warning("Cannot open %s: %s", path, e)
            return None
    except (OSError, RuntimeError) as e:
        # Handle corrupt/truncated NetCDF files
        logger.warning("Cannot open %s: %s", path, e)
        return None

    # Squeeze singleton dimensions from Pangeo (member_id, dcpp_init_year)
    for dim in list(ds.dims):
        if dim not in ("time", "plev", "lev", "lat", "lon", "latitude", "longitude"):
            if ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)

    # Reject files with zero time steps
    if ds.sizes.get("time", 0) == 0:
        logger.warning("Zero timesteps: %s", path)
        ds.close()
        return None

    # Reject files where primary variable is all-NaN
    var_name = [v for v in ds.data_vars][0]
    if ds[var_name].isnull().all():
        logger.warning("All-NaN data: %s", path)
        ds.close()
        return None

    return ds


def compute_R(ds_ta: xr.Dataset, cfg: dict) -> dict:
    """Compute the nonlinearity ratio R from temperature data.

    Returns dict with R_annual, R_djf, and sigma_bar for use in D computation.
    """
    # Extract temperature
    ta = ds_ta["ta"] if "ta" in ds_ta else ds_ta[list(ds_ta.data_vars)[0]]

    # Compute static stability on all levels
    sigma = static_stability(ta)

    # Lower-troposphere layer (700-1000 hPa) where Arctic amplification is strongest
    sigma_lower = layer_mean_stability(sigma, p_top=700.0, p_bot=1000.0)
    # Mid-troposphere layer (500-850 hPa)
    sigma_mid = layer_mean_stability(
        sigma,
        p_top=cfg["pressure"]["layer_top"],
        p_bot=cfg["pressure"]["layer_bot"],
    )

    # Annual R (spatio-temporal, lower troposphere for max signal)
    R_annual = nonlinearity_ratio(sigma_lower)

    # Winter (DJF) R — Arctic amplification is strongest in winter
    R_djf = nonlinearity_ratio_seasonal(sigma_lower, season="DJF")

    # σ̄ for Ld computation in D
    sigma_bar = float(sigma_mid.mean())

    return {"R_annual": R_annual, "R_djf": R_djf, "sigma_bar": sigma_bar}


def compute_D(ds_ta: xr.Dataset, ds_pr: xr.Dataset, cfg: dict,
              sigma_bar_value: float | None = None) -> float:
    """Compute the diabatic number D from temperature and precipitation."""
    # Temperature
    ta = ds_ta["ta"] if "ta" in ds_ta else ds_ta[list(ds_ta.data_vars)[0]]

    # Meridional temperature gradient at Arctic edge
    lat_min_avail = float(ta["lat"].min())
    edge_lat_min = max(cfg["domain"]["edge_lat_min"], lat_min_avail)
    edge_lat_max = min(cfg["domain"]["edge_lat_max"],
                       float(ta["lat"].max()))

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


def compute_eady(ds_ta: xr.Dataset, ds_ua: xr.Dataset, cfg: dict) -> dict:
    """Compute dry and moist Eady growth rate diagnostics.

    Parameters
    ----------
    ds_ta : xr.Dataset
        Temperature data on pressure levels.
    ds_ua : xr.Dataset
        Eastward wind data on pressure levels.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Keys: eady_dry (day⁻¹), eady_moist (day⁻¹), eady_ratio (F).
    """
    ta = ds_ta["ta"] if "ta" in ds_ta else ds_ta[list(ds_ta.data_vars)[0]]
    ua = ds_ua["ua"] if "ua" in ds_ua else ds_ua[list(ds_ua.data_vars)[0]]

    # Use Eady-specific domain from config (near jet, 50-70°N)
    eady_cfg = cfg.get("eady", {})
    return compute_eady_diagnostics(
        ua, ta,
        lat_min=eady_cfg.get("lat_min", 50.0),
        lat_max=eady_cfg.get("lat_max", 70.0),
        p_top=eady_cfg.get("p_top", cfg["pressure"]["layer_top"]),
        p_bot=eady_cfg.get("p_bot", cfg["pressure"]["layer_bot"]),
    )


def compute_waves(ds_ta: xr.Dataset, ds_ua: xr.Dataset,
                  cfg: dict, sigma_bar: float) -> dict:
    """Compute Ld map stats and refractive index diagnostics.

    Parameters
    ----------
    ds_ta : xr.Dataset
        Temperature data on pressure levels.
    ds_ua : xr.Dataset
        Eastward wind data on pressure levels.
    cfg : dict
        Configuration dictionary.
    sigma_bar : float
        Domain-mean layer-mean static stability.

    Returns
    -------
    dict
        Keys: Ld_mean_km, Ks_250.
    """
    ta = ds_ta["ta"] if "ta" in ds_ta else ds_ta[list(ds_ta.data_vars)[0]]
    ua = ds_ua["ua"] if "ua" in ds_ua else ds_ua[list(ds_ua.data_vars)[0]]

    # Use wave-specific domain from config
    wave_cfg = cfg.get("wave", {})
    return compute_wave_diagnostics(
        ua, ta, sigma_bar,
        lat_min=wave_cfg.get("Ld_lat_min", 55.0),
        lat_max=wave_cfg.get("Ld_lat_max", 75.0),
        p_top=cfg["pressure"]["layer_top"],
        p_bot=cfg["pressure"]["layer_bot"],
        f0=cfg["constants"]["f0"],
        Ks_plev=wave_cfg.get("Ks_plev", 250.0),
        Ks_lat_min=wave_cfg.get("Ks_lat_min", 50.0),
        Ks_lat_max=wave_cfg.get("Ks_lat_max", 70.0),
    )


# Models to exclude due to known data quality issues
EXCLUDE_MODELS = {"MCM-UA-1-0"}  # empty data shells


def main():
    cfg = load_config()

    # Prefer regridded data; fall back to processed
    data_dir = PROJECT_ROOT / "data" / "processed_regridded"
    if not data_dir.exists() or len(list(data_dir.glob("*.nc"))) == 0:
        logger.warning("Regridded data not found; using data/processed/")
        data_dir = PROJECT_ROOT / "data" / "processed"

    out_dir = PROJECT_ROOT / "data" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model in cfg["models"]:
        if model in EXCLUDE_MODELS:
            logger.info("Skipping excluded model: %s", model)
            continue
        for period_name, period_cfg in cfg["periods"].items():
            mapping = cfg["period_experiment"][period_name]
            experiments = mapping if isinstance(mapping, list) else [mapping]

            for exp in experiments:
                logger.info("Computing: %s / %s / %s", model, exp, period_name)

                ds_ta = load_processed(model, exp, "ta", period_name, data_dir)
                ds_pr = load_processed(model, exp, "pr", period_name, data_dir)

                if ds_ta is None or ds_pr is None:
                    logger.warning("  Skipping (missing ta or pr)")
                    continue

                # Optionally load ua (not required for R/D)
                ds_ua = load_processed(model, exp, "ua", period_name, data_dir)

                try:
                    R_dict = compute_R(ds_ta, cfg)
                    D = compute_D(ds_ta, ds_pr, cfg,
                                  sigma_bar_value=R_dict["sigma_bar"])

                    row = {
                        "model": model,
                        "period": period_name,
                        "scenario": exp,
                        "R": R_dict["R_djf"],
                        "R_annual": R_dict["R_annual"],
                        "R_djf": R_dict["R_djf"],
                        "D": D,
                        "eady_dry": np.nan,
                        "eady_moist": np.nan,
                        "eady_ratio": np.nan,
                        "Ld_mean_km": np.nan,
                        "Ks_250": np.nan,
                        "label": period_cfg["label"],
                    }

                    # Compute Eady and wave diagnostics if ua is available
                    if ds_ua is not None:
                        try:
                            eady = compute_eady(ds_ta, ds_ua, cfg)
                            row["eady_dry"] = eady["eady_dry"]
                            row["eady_moist"] = eady["eady_moist"]
                            row["eady_ratio"] = eady["eady_ratio"]
                            logger.info("  Eady: dry=%.3f, moist=%.3f, F=%.3f",
                                        eady["eady_dry"], eady["eady_moist"],
                                        eady["eady_ratio"])
                        except Exception as e:
                            logger.warning("  Eady computation failed: %s", e)

                        try:
                            waves = compute_waves(ds_ta, ds_ua, cfg,
                                                  R_dict["sigma_bar"])
                            row["Ld_mean_km"] = waves["Ld_mean_km"]
                            row["Ks_250"] = waves["Ks_250"]
                            logger.info("  Waves: Ld=%.0f km, K_s²=%.2e",
                                        waves["Ld_mean_km"], waves["Ks_250"])
                        except Exception as e:
                            logger.warning("  Wave diagnostics failed: %s", e)
                    else:
                        logger.info("  No ua data — skipping Eady and wave diagnostics")

                    results.append(row)
                    logger.info("  R_annual=%.4f, R_djf=%.4f, D=%.4f",
                                R_dict["R_annual"], R_dict["R_djf"], D)

                except Exception as e:
                    logger.error("  Failed: %s", e)
                    continue

                finally:
                    ds_ta.close()
                    ds_pr.close()
                    if ds_ua is not None:
                        ds_ua.close()

    # Save results
    df = pd.DataFrame(results)
    out_path = out_dir / "regime_diagnostics.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved diagnostics to %s (%d rows)", out_path, len(df))
    if len(df) > 0:
        summary = df.groupby(["scenario", "period"])[["R", "D"]].agg(["mean", "std", "count"])
        logger.info("\nSummary:\n%s", summary.to_string())
        # Also report Eady and wave diagnostics where available
        eady_cols = ["eady_dry", "eady_moist", "eady_ratio", "Ld_mean_km", "Ks_250"]
        df_eady = df.dropna(subset=["eady_dry"])
        if len(df_eady) > 0:
            eady_summary = df_eady.groupby(["scenario", "period"])[eady_cols].agg(
                ["mean", "std", "count"])
            logger.info("\nEady/Wave Summary:\n%s", eady_summary.to_string())
        else:
            logger.info("No models had ua data for Eady/wave diagnostics")
    else:
        logger.warning("No diagnostics computed!")


if __name__ == "__main__":
    main()
