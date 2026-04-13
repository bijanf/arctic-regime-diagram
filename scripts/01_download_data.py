#!/usr/bin/env python3
"""Step 1: Download and preprocess CMIP6 data from Pangeo Google Cloud.

For each model × experiment × variable, this script:
  1. Opens the Pangeo CMIP6 Zarr catalog
  2. Loads ta, pr, and ua lazily
  3. Subsets to 55-90°N (wider for meridional gradient) and pressure levels
  4. Slices to the required time periods
  5. Saves processed NetCDF files to data/processed/

Note: ua (eastward wind) is downloaded separately from ta/pr because not all
models provide it. Models lacking ua will simply have no ua files, and downstream
diagnostics (Eady growth rate, refractive index) will gracefully skip them.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.catalog import find_common_models, open_catalog
from src.data.download import load_dataset
from src.data.preprocess import (
    select_period,
    select_pressure_layer,
    subset_arctic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _clean_encoding(ds):
    """Remove conflicting _FillValue/missing_value from variable encodings."""
    for var in ds.variables:
        enc = ds[var].encoding
        if "missing_value" in enc and "_FillValue" in enc:
            del enc["missing_value"]
    return ds


# Members to try in order for each model
MEMBER_ATTEMPTS = ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3", "r2i1p1f1"]


def try_load(catalog, var, exp, model):
    """Try multiple member IDs until one works."""
    for member in MEMBER_ATTEMPTS:
        try:
            ds = load_dataset(catalog, var, exp, model, member_id=member)
            logger.info("  Loaded with member %s", member)
            return ds
        except (ValueError, KeyError):
            continue
    raise ValueError(f"No member found for {model}/{exp}/{var}")


def main():
    cfg = load_config()
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = open_catalog()

    models = find_common_models(
        catalog,
        variables=["ta", "pr"],
        experiments=cfg["experiments"],
        model_list=cfg["models"],
    )
    logger.info("Processing %d models: %s", len(models), models)

    # Use 55°N as southern boundary to capture edge gradient region
    lat_min_download = 55.0
    lat_max_download = 90.0

    for model in models:
        for exp in cfg["experiments"]:
            # Determine time periods for this experiment
            periods_to_process = []
            for period_name, period_cfg in cfg["periods"].items():
                mapping = cfg["period_experiment"][period_name]
                if isinstance(mapping, list):
                    if exp in mapping:
                        periods_to_process.append((period_name, period_cfg))
                elif mapping == exp:
                    periods_to_process.append((period_name, period_cfg))

            if not periods_to_process:
                continue

            for var in ["ta", "pr"]:
                # Check if all periods already downloaded for this model/exp/var
                all_exist = all(
                    (out_dir / f"{model}_{exp}_{var}_{pn}.nc").exists()
                    for pn, _ in periods_to_process
                )
                if all_exist:
                    logger.info("Skipping %s/%s/%s (all periods exist)", model, exp, var)
                    continue

                logger.info("=== %s / %s / %s ===", model, exp, var)

                try:
                    ds = try_load(catalog, var, exp, model)
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping %s/%s/%s: %s", model, exp, var, e)
                    continue

                for period_name, period_cfg in periods_to_process:
                    out_path = out_dir / f"{model}_{exp}_{var}_{period_name}.nc"
                    if out_path.exists():
                        logger.info("  %s already exists, skipping", out_path.name)
                        continue

                    logger.info("  Period: %s", period_name)

                    try:
                        ds_period = select_period(ds, period_cfg["start"], period_cfg["end"])

                        if var == "ta":
                            ds_sub = subset_arctic(
                                ds_period,
                                lat_min_download,
                                lat_max_download,
                            )
                            ds_sub = select_pressure_layer(
                                ds_sub,
                                cfg["pressure"]["full_range_top"],
                                cfg["pressure"]["full_range_bot"],
                            )
                        else:
                            ds_sub = subset_arctic(
                                ds_period,
                                lat_min_download,
                                lat_max_download,
                            )

                        _clean_encoding(ds_sub)
                        ds_sub.to_netcdf(out_path)
                        logger.info("  Saved: %s", out_path)

                    except Exception as e:
                        logger.error("  Failed %s/%s/%s/%s: %s", model, exp, var, period_name, e)
                        continue

    # ── Download ua (eastward wind) separately ──
    # ua is not required for R/D, so we download it for all models that
    # have it without constraining the ta/pr model intersection.
    logger.info("=" * 60)
    logger.info("Downloading ua (eastward wind on pressure levels)...")
    logger.info("=" * 60)

    for model in cfg["models"]:
        for exp in cfg["experiments"]:
            periods_to_process = []
            for period_name, period_cfg in cfg["periods"].items():
                mapping = cfg["period_experiment"][period_name]
                if isinstance(mapping, list):
                    if exp in mapping:
                        periods_to_process.append((period_name, period_cfg))
                elif mapping == exp:
                    periods_to_process.append((period_name, period_cfg))

            if not periods_to_process:
                continue

            # Check if all ua periods already exist
            all_exist = all(
                (out_dir / f"{model}_{exp}_ua_{pn}.nc").exists() for pn, _ in periods_to_process
            )
            if all_exist:
                logger.info("Skipping %s/%s/ua (all periods exist)", model, exp)
                continue

            logger.info("=== %s / %s / ua ===", model, exp)

            try:
                ds = try_load(catalog, "ua", exp, model)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping %s/%s/ua: %s", model, exp, e)
                continue

            for period_name, period_cfg in periods_to_process:
                out_path = out_dir / f"{model}_{exp}_ua_{period_name}.nc"
                if out_path.exists():
                    logger.info("  %s already exists, skipping", out_path.name)
                    continue

                logger.info("  Period: %s", period_name)

                try:
                    ds_period = select_period(ds, period_cfg["start"], period_cfg["end"])
                    # ua is on pressure levels like ta
                    ds_sub = subset_arctic(
                        ds_period,
                        lat_min_download,
                        lat_max_download,
                    )
                    ds_sub = select_pressure_layer(
                        ds_sub,
                        cfg["pressure"]["full_range_top"],
                        cfg["pressure"]["full_range_bot"],
                    )
                    _clean_encoding(ds_sub)
                    ds_sub.to_netcdf(out_path)
                    logger.info("  Saved: %s", out_path)

                except Exception as e:
                    logger.error("  Failed %s/%s/ua/%s: %s", model, exp, period_name, e)
                    continue

    logger.info("Download and preprocessing complete.")


if __name__ == "__main__":
    main()
