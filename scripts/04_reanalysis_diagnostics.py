#!/usr/bin/env python3
"""Step 4: Download reanalysis data and compute R, D, Eady, and wave diagnostics.

Supports ERA5 (via CDS API) and NCEP/NCAR Reanalysis 1 (via NOAA PSL OPeNDAP).
Outputs reanalysis_diagnostics.csv for overlay on the CMIP6 regime diagram.
"""

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.physics.baroclinic import compute_eady_diagnostics
from src.physics.diabatic import (
    arctic_mean_precipitation,
    diabatic_number,
    meridional_T_gradient,
)
from src.physics.nonlinearity import nonlinearity_ratio, nonlinearity_ratio_seasonal
from src.physics.static_stability import layer_mean_stability, static_stability
from src.physics.wave_diagnostics import compute_wave_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Sliding 30-year climatological windows
WINDOWS = [
    (1961, 1990),
    (1981, 2010),
    (1991, 2020),
]


# ── Data access functions ─────────────────────────────────────────────────


def download_ncep_r1(data_dir):
    """Download NCEP/NCAR R1 monthly means via direct HTTP.

    Returns (ds_ta, ds_pr) with standard names: 'ta' in K on 'plev', 'pr' in kg/m²/s.
    """
    import urllib.request

    ta_cache = data_dir / "ncep_r1_ta_arctic.nc"
    pr_cache = data_dir / "ncep_r1_pr_arctic.nc"

    # Check for processed local cache
    if ta_cache.exists() and pr_cache.exists() and ta_cache.stat().st_size > 10000:
        logger.info("  Loading NCEP R1 from local cache")
        ds_ta = xr.open_dataset(ta_cache)
        ds_pr = xr.open_dataset(pr_cache)
        return ds_ta, ds_pr

    # Download full files from NOAA PSL
    ta_raw = data_dir / "ncep_r1_air.mon.mean.nc"
    pr_raw = data_dir / "ncep_r1_prate.sfc.gauss.nc"

    if not ta_raw.exists():
        ta_url = (
            "https://downloads.psl.noaa.gov/Datasets/"
            "ncep.reanalysis.derived/pressure/air.mon.mean.nc"
        )
        logger.info("  Downloading NCEP R1 temperature (~340 MB)...")
        urllib.request.urlretrieve(ta_url, ta_raw)
        logger.info("  Downloaded: %s", ta_raw)

    if not pr_raw.exists():
        # Try multiple URLs for precipitation
        pr_urls = [
            (
                "https://downloads.psl.noaa.gov/Datasets/"
                "ncep.reanalysis/surface_gauss/prate.sfc.gauss.mon.mean.nc"
            ),
            (
                "https://downloads.psl.noaa.gov/Datasets/"
                "ncep.reanalysis.derived/surface_gauss/prate.sfc.mon.mean.nc"
            ),
        ]
        for url in pr_urls:
            try:
                logger.info("  Downloading NCEP R1 precipitation from %s...", url.split("/")[-1])
                urllib.request.urlretrieve(url, pr_raw)
                logger.info("  Downloaded: %s", pr_raw)
                break
            except Exception as e:
                logger.warning("  Download failed: %s", e)
                continue

    if not pr_raw.exists():
        raise RuntimeError("Could not download NCEP R1 precipitation")

    # Load, subset, rename, and cache
    logger.info("  Processing NCEP R1 raw files...")
    ds_ta = xr.open_dataset(ta_raw)
    ds_ta = ds_ta.sel(lat=slice(90, 50), level=slice(1000, 200))
    ds_ta = ds_ta.rename({"air": "ta", "level": "plev"})
    # NCEP R1 derived temperature is in degC — convert to K
    ta_units = ds_ta["ta"].attrs.get("units", "")
    if "C" in ta_units or float(ds_ta["ta"].min()) < 100:
        ds_ta["ta"] = ds_ta["ta"] + 273.15
        logger.info("  Converted NCEP R1 ta from °C to K")
    ds_ta = ds_ta.sortby("lat").sortby("plev").load()
    ds_ta["ta"].encoding.clear()
    logger.info(
        "  NCEP R1 ta: %s, T=[%.1f, %.1f] K, plev=%s",
        dict(ds_ta.sizes),
        float(ds_ta["ta"].min()),
        float(ds_ta["ta"].max()),
        ds_ta.plev.values.tolist(),
    )

    ds_pr = xr.open_dataset(pr_raw)
    # Gaussian grid: check lat ordering for correct subsetting
    if float(ds_pr.lat[0]) > float(ds_pr.lat[-1]):
        ds_pr = ds_pr.sel(lat=slice(90, 50))  # descending
    else:
        ds_pr = ds_pr.sel(lat=slice(50, 90))  # ascending
    ds_pr = ds_pr.rename({"prate": "pr"})
    ds_pr = ds_pr.sortby("lat").load()  # force into memory
    # Clear inherited encoding that corrupts data on save (least_significant_digit=0)
    ds_pr["pr"].encoding.clear()
    logger.info(
        "  NCEP R1 pr: %s, pr=[%.2e, %.2e] kg/m²/s",
        dict(ds_pr.sizes),
        float(ds_pr["pr"].min()),
        float(ds_pr["pr"].max()),
    )

    # Save Arctic subset for fast re-runs
    ds_ta.to_netcdf(ta_cache)
    ds_pr.to_netcdf(pr_cache)
    logger.info("  Cached NCEP R1 Arctic subset to %s", data_dir)

    return ds_ta, ds_pr


def download_ncep_r1_ua(data_dir):
    """Download NCEP/NCAR R1 monthly mean u-wind via direct HTTP.

    Returns ds_ua with standard name 'ua' in m/s on 'plev' (hPa).
    """
    import urllib.request

    ua_cache = data_dir / "ncep_r1_ua_arctic.nc"

    if ua_cache.exists() and ua_cache.stat().st_size > 10000:
        logger.info("  Loading NCEP R1 ua from local cache")
        return xr.open_dataset(ua_cache)

    ua_raw = data_dir / "ncep_r1_uwnd.mon.mean.nc"
    if not ua_raw.exists():
        ua_url = (
            "https://downloads.psl.noaa.gov/Datasets/"
            "ncep.reanalysis.derived/pressure/uwnd.mon.mean.nc"
        )
        logger.info("  Downloading NCEP R1 u-wind (~340 MB)...")
        urllib.request.urlretrieve(ua_url, ua_raw)
        logger.info("  Downloaded: %s", ua_raw)

    ds_ua = xr.open_dataset(ua_raw)
    ds_ua = ds_ua.sel(lat=slice(90, 50), level=slice(1000, 200))
    ds_ua = ds_ua.rename({"uwnd": "ua", "level": "plev"})
    ds_ua = ds_ua.sortby("lat").sortby("plev").load()
    ds_ua["ua"].encoding.clear()
    logger.info(
        "  NCEP R1 ua: %s, u=[%.1f, %.1f] m/s",
        dict(ds_ua.sizes),
        float(ds_ua["ua"].min()),
        float(ds_ua["ua"].max()),
    )

    ds_ua.to_netcdf(ua_cache)
    logger.info("  Cached NCEP R1 ua to %s", data_dir)
    return ds_ua


def download_era5_cds(data_dir):
    """Download ERA5 monthly means via CDS API. Returns (ta_path, pr_path, ua_path)."""
    import cdsapi

    ta_path = data_dir / "era5_ta_monthly_arctic.nc"
    pr_path = data_dir / "era5_pr_monthly_arctic.nc"
    ua_path = data_dir / "era5_ua_monthly_arctic.nc"

    if ta_path.exists() and pr_path.exists() and ua_path.exists():
        logger.info("ERA5 CDS data already cached locally")
        return ta_path, pr_path, ua_path

    c = cdsapi.Client()

    # Temperature on pressure levels (monthly means)
    if not ta_path.exists():
        logger.info("Requesting ERA5 monthly ta from CDS (this may take a while)...")
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": ["temperature"],
                "pressure_level": ["300", "500", "600", "700", "850", "925", "1000"],
                "year": [str(y) for y in range(1959, 2024)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": ["00:00"],
                "area": [90, -180, 50, 180],  # N, W, S, E
                "data_format": "netcdf",
            },
            str(ta_path),
        )
        logger.info("  Saved ERA5 ta to %s", ta_path)

    # Total precipitation (monthly means, single levels)
    if not pr_path.exists():
        logger.info("Requesting ERA5 monthly pr from CDS...")
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": ["mean_total_precipitation_rate"],
                "year": [str(y) for y in range(1959, 2024)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": ["00:00"],
                "area": [90, -180, 50, 180],
                "data_format": "netcdf",
            },
            str(pr_path),
        )
        logger.info("  Saved ERA5 pr to %s", pr_path)

    # Eastward wind on pressure levels (monthly means)
    if not ua_path.exists():
        logger.info("Requesting ERA5 monthly ua from CDS...")
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": ["u_component_of_wind"],
                "pressure_level": ["300", "500", "600", "700", "850", "925", "1000"],
                "year": [str(y) for y in range(1959, 2024)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": ["00:00"],
                "area": [90, -180, 50, 180],
                "data_format": "netcdf",
            },
            str(ua_path),
        )
        logger.info("  Saved ERA5 ua to %s", ua_path)

    return ta_path, pr_path, ua_path


def load_era5_from_files(ta_path, pr_path, ua_path=None):
    """Load ERA5 from local NetCDF files, normalizing to standard names."""
    ds_ta = xr.open_dataset(ta_path)
    ds_pr = xr.open_dataset(pr_path)

    # ── Normalize temperature dataset ──
    rename_ta = {}
    # Coordinate names
    if "latitude" in ds_ta.dims:
        rename_ta["latitude"] = "lat"
    if "longitude" in ds_ta.dims:
        rename_ta["longitude"] = "lon"
    if "pressure_level" in ds_ta.dims:
        rename_ta["pressure_level"] = "plev"
    if "valid_time" in ds_ta.dims:
        rename_ta["valid_time"] = "time"
    # Variable name
    ta_varname = None
    for v in ds_ta.data_vars:
        if v in ("t", "temperature", "ta"):
            ta_varname = v
            break
    if ta_varname is None:
        ta_varname = list(ds_ta.data_vars)[0]
    if ta_varname != "ta":
        rename_ta[ta_varname] = "ta"
    if rename_ta:
        ds_ta = ds_ta.rename(rename_ta)

    # Drop non-essential coordinates that cause issues
    for coord in ["number", "expver"]:
        if coord in ds_ta.coords:
            ds_ta = ds_ta.drop_vars(coord)

    # ── Normalize precipitation dataset ──
    rename_pr = {}
    if "latitude" in ds_pr.dims:
        rename_pr["latitude"] = "lat"
    if "longitude" in ds_pr.dims:
        rename_pr["longitude"] = "lon"
    if "valid_time" in ds_pr.dims:
        rename_pr["valid_time"] = "time"
    # Variable name (CDS uses avg_tprate for mean total precipitation rate)
    pr_varname = None
    for v in ds_pr.data_vars:
        if v in (
            "tp",
            "mtpr",
            "avg_tprate",
            "total_precipitation",
            "mean_total_precipitation_rate",
            "pr",
            "prate",
        ):
            pr_varname = v
            break
    if pr_varname is None:
        pr_varname = list(ds_pr.data_vars)[0]
    if pr_varname != "pr":
        rename_pr[pr_varname] = "pr"
    if rename_pr:
        ds_pr = ds_pr.rename(rename_pr)
    for coord in ["number", "expver"]:
        if coord in ds_pr.coords:
            ds_pr = ds_pr.drop_vars(coord)

    # ERA5 precipitation: check units attribute
    pr_units = ds_pr["pr"].attrs.get("units", "")
    pr_max = float(ds_pr["pr"].max())
    if "m s" in pr_units and "kg" not in pr_units:
        # Units are m/s (meters of water per second) → convert to kg/m²/s
        ds_pr["pr"] = ds_pr["pr"] * 1000.0
        logger.info(
            "  Converted ERA5 pr from m/s to kg/m²/s (max=%.2e → %.2e)", pr_max, pr_max * 1000
        )
    else:
        logger.info("  ERA5 pr already in %s (max=%.2e), no conversion needed", pr_units, pr_max)

    # ERA5 pressure levels: CDS provides in hPa (300, 500, 600, 700, 850, 925, 1000)
    if "plev" in ds_ta.dims and float(ds_ta.plev.max()) > 2000:
        ds_ta["plev"] = ds_ta["plev"] / 100.0
        logger.info("  Converted ERA5 plev from Pa to hPa")

    # Sort ascending
    ds_ta = ds_ta.sortby("lat")
    ds_pr = ds_pr.sortby("lat")
    if "plev" in ds_ta.dims:
        ds_ta = ds_ta.sortby("plev")

    logger.info(
        "  ERA5 ta: %s, T=[%.1f, %.1f] K, plev=%s",
        dict(ds_ta.sizes),
        float(ds_ta["ta"].min()),
        float(ds_ta["ta"].max()),
        ds_ta.plev.values.tolist(),
    )
    logger.info(
        "  ERA5 pr: %s, pr=[%.2e, %.2e] kg/m²/s",
        dict(ds_pr.sizes),
        float(ds_pr["pr"].min()),
        float(ds_pr["pr"].max()),
    )

    # ── Normalize ua dataset (optional) ──
    ds_ua = None
    if ua_path is not None and Path(ua_path).exists():
        ds_ua = xr.open_dataset(ua_path)
        rename_ua = {}
        if "latitude" in ds_ua.dims:
            rename_ua["latitude"] = "lat"
        if "longitude" in ds_ua.dims:
            rename_ua["longitude"] = "lon"
        if "pressure_level" in ds_ua.dims:
            rename_ua["pressure_level"] = "plev"
        if "valid_time" in ds_ua.dims:
            rename_ua["valid_time"] = "time"
        ua_varname = None
        for v in ds_ua.data_vars:
            if v in ("u", "u_component_of_wind", "ua"):
                ua_varname = v
                break
        if ua_varname is None:
            ua_varname = list(ds_ua.data_vars)[0]
        if ua_varname != "ua":
            rename_ua[ua_varname] = "ua"
        if rename_ua:
            ds_ua = ds_ua.rename(rename_ua)
        for coord in ["number", "expver"]:
            if coord in ds_ua.coords:
                ds_ua = ds_ua.drop_vars(coord)
        if "plev" in ds_ua.dims and float(ds_ua.plev.max()) > 2000:
            ds_ua["plev"] = ds_ua["plev"] / 100.0
            logger.info("  Converted ERA5 ua plev from Pa to hPa")
        ds_ua = ds_ua.sortby("lat")
        if "plev" in ds_ua.dims:
            ds_ua = ds_ua.sortby("plev")
        logger.info(
            "  ERA5 ua: %s, u=[%.1f, %.1f] m/s",
            dict(ds_ua.sizes),
            float(ds_ua["ua"].min()),
            float(ds_ua["ua"].max()),
        )

    return ds_ta, ds_pr, ds_ua


# ── Diagnostics computation ──────────────────────────────────────────────


def compute_reanalysis_diagnostics(ds_ta, ds_pr, cfg, dataset_name, ds_ua=None, windows=None):
    """Compute R, D, Eady, and wave diagnostics for sliding time windows.

    Parameters
    ----------
    ds_ta : xr.Dataset with 'ta' (K) on 'plev' (hPa), 'lat', 'lon', 'time'
    ds_pr : xr.Dataset with 'pr' (kg/m²/s) on 'lat', 'lon', 'time'
    cfg : dict from load_config()
    dataset_name : str (e.g., 'ERA5', 'NCEP-R1')
    ds_ua : xr.Dataset, optional
        Eastward wind with 'ua' on 'plev', 'lat', 'lon', 'time'.
    windows : list of (start_year, end_year) tuples

    Returns list of dicts with diagnostics per window.
    """
    if windows is None:
        windows = WINDOWS

    results = []

    for start_yr, end_yr in windows:
        center = (start_yr + end_yr) // 2
        label = f"{start_yr}-{end_yr}"
        logger.info("  %s window %s (center %d)...", dataset_name, label, center)

        # Select time window
        try:
            ta_window = ds_ta["ta"].sel(time=slice(f"{start_yr}-01", f"{end_yr}-12"))
            pr_window = ds_pr["pr"].sel(time=slice(f"{start_yr}-01", f"{end_yr}-12"))
        except Exception:
            # cftime or non-standard time
            ta_times = ds_ta.time.values
            pr_times = ds_pr.time.values
            ta_window = ds_ta["ta"].isel(
                time=[
                    i
                    for i, t in enumerate(ta_times)
                    if hasattr(t, "year") and start_yr <= t.year <= end_yr
                ]
            )
            pr_window = ds_pr["pr"].isel(
                time=[
                    i
                    for i, t in enumerate(pr_times)
                    if hasattr(t, "year") and start_yr <= t.year <= end_yr
                ]
            )

        if len(ta_window.time) < 12:
            logger.warning(
                "    Not enough data for %s (only %d months)", label, len(ta_window.time)
            )
            continue

        logger.info("    ta: %d months, pr: %d months", len(ta_window.time), len(pr_window.time))

        try:
            # ── Compute R ──
            sigma = static_stability(ta_window)
            sigma_lower = layer_mean_stability(sigma, p_top=700.0, p_bot=1000.0)

            R_annual = nonlinearity_ratio(sigma_lower)
            try:
                R_djf = nonlinearity_ratio_seasonal(sigma_lower, season="DJF")
            except Exception:
                R_djf = R_annual  # fallback

            sigma_mid = layer_mean_stability(
                sigma,
                p_top=cfg["pressure"]["layer_top"],
                p_bot=cfg["pressure"]["layer_bot"],
            )
            sigma_bar = float(sigma_mid.mean())

            # ── Compute D ──
            # Meridional T gradient
            lat_name = "lat"
            lat_min_avail = float(ta_window[lat_name].min())
            edge_lat_min = max(cfg["domain"]["edge_lat_min"], lat_min_avail)
            edge_lat_max = min(cfg["domain"]["edge_lat_max"], float(ta_window[lat_name].max()))

            dT_dy = meridional_T_gradient(
                ta_window,
                lat_min=edge_lat_min,
                lat_max=edge_lat_max,
                plev_level=cfg["pressure"]["gradient_level"],
            )
            if dT_dy < 1e-7:
                logger.warning("    dT/dy too small; using default 5e-6 K/m")
                dT_dy = 5e-6

            # Arctic-mean precipitation
            pr_arctic = arctic_mean_precipitation(
                pr_window,
                lat_min=cfg["domain"]["arctic_lat_min"],
                lat_max=cfg["domain"]["arctic_lat_max"],
            )

            D = diabatic_number(
                pr_arctic,
                dT_dy,
                sigma_bar,
                Lv=cfg["constants"]["Lv"],
                cp=cfg["constants"]["cp"],
                f0=cfg["constants"]["f0"],
                H=cfg["constants"]["H"],
            )

            row = {
                "dataset": dataset_name,
                "window": label,
                "center_year": center,
                "R": R_djf,
                "R_annual": R_annual,
                "R_djf": R_djf,
                "D": D,
                "sigma_bar": sigma_bar,
                "eady_dry": np.nan,
                "eady_moist": np.nan,
                "eady_ratio": np.nan,
                "Ld_mean_km": np.nan,
                "Ks_250": np.nan,
            }

            # Use climatological (time-mean) fields for Eady/Wave to limit memory
            ta_clim = ta_window.mean(dim="time")
            del sigma, ta_window, pr_window
            gc.collect()

            # Compute Eady and wave diagnostics if ua is available
            if ds_ua is not None:
                try:
                    ua_window = ds_ua["ua"].sel(time=slice(f"{start_yr}-01", f"{end_yr}-12"))
                except Exception:
                    ua_times = ds_ua.time.values
                    ua_window = ds_ua["ua"].isel(
                        time=[
                            i
                            for i, t in enumerate(ua_times)
                            if hasattr(t, "year") and start_yr <= t.year <= end_yr
                        ]
                    )

                if len(ua_window.time) >= 12:
                    ua_clim = ua_window.mean(dim="time")
                    del ua_window

                    try:
                        eady = compute_eady_diagnostics(
                            ua_clim,
                            ta_clim,
                            lat_min=cfg["domain"]["arctic_lat_min"],
                            lat_max=cfg["domain"]["arctic_lat_max"],
                            p_top=cfg["pressure"]["layer_top"],
                            p_bot=cfg["pressure"]["layer_bot"],
                        )
                        row["eady_dry"] = eady["eady_dry"]
                        row["eady_moist"] = eady["eady_moist"]
                        row["eady_ratio"] = eady["eady_ratio"]
                    except Exception as e:
                        logger.warning("    Eady failed: %s", e)

                    try:
                        waves = compute_wave_diagnostics(
                            ua_clim,
                            ta_clim,
                            sigma_bar,
                            lat_min=cfg["domain"]["arctic_lat_min"],
                            lat_max=cfg["domain"]["arctic_lat_max"],
                            p_top=cfg["pressure"]["layer_top"],
                            p_bot=cfg["pressure"]["layer_bot"],
                            f0=cfg["constants"]["f0"],
                        )
                        row["Ld_mean_km"] = waves["Ld_mean_km"]
                        row["Ks_250"] = waves["Ks_250"]
                    except Exception as e:
                        logger.warning("    Wave diagnostics failed: %s", e)

            results.append(row)
            logger.info("    R_annual=%.4f, R_djf=%.4f, D=%.4f", R_annual, R_djf, D)

        except Exception as e:
            logger.error("    Failed for %s %s: %s", dataset_name, label, e)
            import traceback

            traceback.print_exc()
            continue

    return results


# ── Main pipeline ────────────────────────────────────────────────────────


def main():
    cfg = load_config()
    data_dir = PROJECT_ROOT / "data" / "reanalysis"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PROJECT_ROOT / "data" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ── 1. NCEP/NCAR Reanalysis 1 ──
    logger.info("=" * 60)
    logger.info("Processing NCEP/NCAR Reanalysis 1...")
    logger.info("=" * 60)
    try:
        ds_ta_ncep, ds_pr_ncep = download_ncep_r1(data_dir=data_dir)
        # Try to download ua for NCEP R1
        ds_ua_ncep = None
        try:
            ds_ua_ncep = download_ncep_r1_ua(data_dir=data_dir)
        except Exception as e:
            logger.warning("NCEP R1 ua download failed (non-fatal): %s", e)

        results_ncep = compute_reanalysis_diagnostics(
            ds_ta_ncep,
            ds_pr_ncep,
            cfg,
            "NCEP-R1",
            ds_ua=ds_ua_ncep,
            windows=WINDOWS,
        )
        all_results.extend(results_ncep)
        logger.info("NCEP R1: %d windows computed", len(results_ncep))
    except Exception as e:
        logger.error("NCEP R1 failed: %s", e)
        import traceback

        traceback.print_exc()

    # ── 2. ERA5 via CDS API ──
    logger.info("=" * 60)
    logger.info("Processing ERA5 (CDS API)...")
    logger.info("=" * 60)
    try:
        ta_path, pr_path, ua_path = download_era5_cds(data_dir)
        ds_ta_era5, ds_pr_era5, ds_ua_era5 = load_era5_from_files(ta_path, pr_path, ua_path)
        results_era5 = compute_reanalysis_diagnostics(
            ds_ta_era5,
            ds_pr_era5,
            cfg,
            "ERA5",
            ds_ua=ds_ua_era5,
            windows=WINDOWS,
        )
        all_results.extend(results_era5)
        logger.info("ERA5: %d windows computed", len(results_era5))
    except Exception as e:
        logger.error("ERA5 CDS failed: %s", e)
        import traceback

        traceback.print_exc()

    # ── Save results ──
    if all_results:
        df = pd.DataFrame(all_results)
        out_path = out_dir / "reanalysis_diagnostics.csv"
        df.to_csv(out_path, index=False)
        logger.info("\nSaved reanalysis diagnostics to %s (%d rows)", out_path, len(df))
        logger.info("\n%s", df.to_string(index=False))
    else:
        logger.error("No reanalysis diagnostics computed!")


if __name__ == "__main__":
    main()
