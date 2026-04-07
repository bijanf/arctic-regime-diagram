#!/usr/bin/env python3
"""Step 1c: Regrid all processed data to a common 2° grid.

Reads NetCDF files from data/processed/ and interpolates to a uniform
2°×2° lat-lon grid covering 50-90°N, saving results to data/processed_regridded/.

Uses xarray.interp() (bilinear interpolation) since xESMF is not available.
This ensures all models are directly comparable despite native resolution
differences (0.5° to 2.8°).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.preprocess import _get_lat_name, _get_lon_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Target grid: 2° resolution, 50-90°N (wide enough for Eady at 50-70°N
# and wave diagnostics at 55-75°N)
TARGET_RES = 2.0
TARGET_LAT = np.arange(50.0, 91.0, TARGET_RES)  # 50, 52, ..., 90
TARGET_LON = np.arange(0.0, 360.0, TARGET_RES)  # 0, 2, ..., 358


def regrid_file(src_path: Path, dst_path: Path) -> bool:
    """Regrid a single NetCDF file to the common 2° grid.

    Parameters
    ----------
    src_path : Path
        Input file path.
    dst_path : Path
        Output file path.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        ds = xr.open_dataset(src_path)
    except ValueError:
        try:
            ds = xr.open_dataset(src_path, use_cftime=True)
        except Exception as e:
            logger.warning("Cannot open %s: %s", src_path, e)
            return False
    except (OSError, RuntimeError) as e:
        logger.warning("Cannot open %s: %s", src_path, e)
        return False

    # Squeeze singleton dimensions
    for dim in list(ds.dims):
        if (
            dim not in ("time", "plev", "lev", "lat", "lon", "latitude", "longitude")
            and ds.sizes[dim] == 1
        ):
            ds = ds.squeeze(dim, drop=True)

    # Validate
    if ds.sizes.get("time", 0) == 0:
        logger.warning("Zero timesteps: %s", src_path)
        ds.close()
        return False

    var_name = list(ds.data_vars)[0]
    if ds[var_name].isnull().all():
        logger.warning("All-NaN data: %s", src_path)
        ds.close()
        return False

    # Detect coordinate names
    try:
        lat_name = _get_lat_name(ds)
        lon_name = _get_lon_name(ds)
    except ValueError as e:
        logger.warning("Cannot find coords in %s: %s", src_path, e)
        ds.close()
        return False

    src_lat = ds[lat_name].values

    # Only include target lats within the source data range (with 1° margin)
    lat_min_src = src_lat.min()
    lat_max_src = src_lat.max()
    target_lat = TARGET_LAT[(lat_min_src + 1.0 <= TARGET_LAT) & (lat_max_src - 1.0 >= TARGET_LAT)]
    if len(target_lat) == 0:
        # Fall back: use all target points within source range
        target_lat = TARGET_LAT[(lat_min_src <= TARGET_LAT) & (lat_max_src >= TARGET_LAT)]

    if len(target_lat) == 0:
        logger.warning(
            "No target lat points within source range [%.1f, %.1f]: %s",
            lat_min_src,
            lat_max_src,
            src_path,
        )
        ds.close()
        return False

    # Interpolate to common grid
    try:
        ds_regridded = ds.interp(
            {lat_name: target_lat, lon_name: TARGET_LON},
            method="linear",
            kwargs={"fill_value": None},  # extrapolate at edges
        )
    except Exception as e:
        logger.warning("Interpolation failed for %s: %s", src_path, e)
        ds.close()
        return False

    # Rename coordinates to standard names
    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        ds_regridded = ds_regridded.rename(rename_map)

    # Save
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ds_regridded.to_netcdf(dst_path)
    ds.close()
    ds_regridded.close()
    return True


def main():
    load_config()
    src_dir = PROJECT_ROOT / "data" / "processed"
    dst_dir = PROJECT_ROOT / "data" / "processed_regridded"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find all NetCDF files
    nc_files = sorted(src_dir.glob("*.nc"))
    logger.info("Found %d files in %s", len(nc_files), src_dir)

    success = 0
    failed = 0
    skipped = 0

    for i, src_path in enumerate(nc_files):
        dst_path = dst_dir / src_path.name

        # Skip if already regridded
        if dst_path.exists():
            skipped += 1
            continue

        # Skip known corrupt files
        if src_path.stat().st_size < 1024:
            logger.warning(
                "Skipping small file (%d bytes): %s", src_path.stat().st_size, src_path.name
            )
            failed += 1
            continue

        logger.info("[%d/%d] Regridding %s", i + 1, len(nc_files), src_path.name)

        if regrid_file(src_path, dst_path):
            success += 1
        else:
            failed += 1

    logger.info("Regridding complete: %d success, %d failed, %d skipped", success, failed, skipped)

    # Verify: check that all regridded files have the same grid
    regridded_files = sorted(dst_dir.glob("*.nc"))
    if len(regridded_files) > 0:
        ds_check = xr.open_dataset(regridded_files[0])
        lat_check = ds_check["lat"].values
        lon_check = ds_check["lon"].values
        ds_check.close()
        logger.info(
            "Target grid: %d lat × %d lon (lat=[%.0f, %.0f], lon=[%.0f, %.0f])",
            len(lat_check),
            len(lon_check),
            lat_check.min(),
            lat_check.max(),
            lon_check.min(),
            lon_check.max(),
        )


if __name__ == "__main__":
    main()
