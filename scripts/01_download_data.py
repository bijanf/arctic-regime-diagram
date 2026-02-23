#!/usr/bin/env python3
"""Step 1: Download and preprocess CMIP6 data from Pangeo Google Cloud.

For each model × experiment × variable, this script:
  1. Opens the Pangeo CMIP6 Zarr catalog
  2. Loads ta and pr lazily
  3. Subsets to Arctic domain (60-90°N) and relevant pressure levels
  4. Slices to the required time periods
  5. Regrids to a common 2° grid
  6. Saves processed NetCDF files to data/processed/
"""

import logging
import sys
from pathlib import Path

import xarray as xr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.catalog import open_catalog, find_common_models, search_catalog
from src.data.download import load_dataset
from src.data.preprocess import (
    subset_arctic, select_period, select_pressure_layer, regrid_to_common,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open catalog
    catalog = open_catalog()

    # Find common models
    models = find_common_models(
        catalog,
        variables=["ta", "pr"],
        experiments=cfg["experiments"],
        model_list=cfg["models"],
    )
    logger.info("Processing %d models: %s", len(models), models)

    fallback = cfg.get("fallback_members", {})
    preferred = cfg.get("preferred_member", "r1i1p1f1")

    for model in models:
        member = fallback.get(model, preferred)

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
                logger.info("=== %s / %s / %s / %s ===", model, exp, var, member)

                try:
                    ds = load_dataset(catalog, var, exp, model,
                                      member_id=member)
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping %s/%s/%s: %s", model, exp, var, e)
                    continue

                for period_name, period_cfg in periods_to_process:
                    logger.info("  Period: %s", period_name)

                    try:
                        # Time slice
                        ds_period = select_period(
                            ds, period_cfg["start"], period_cfg["end"]
                        )

                        if var == "ta":
                            # Subset Arctic and pressure levels
                            ds_sub = subset_arctic(
                                ds_period,
                                cfg["domain"]["arctic_lat_min"],
                                cfg["domain"]["arctic_lat_max"],
                            )
                            ds_sub = select_pressure_layer(
                                ds_sub,
                                cfg["pressure"]["full_range_top"],
                                cfg["pressure"]["full_range_bot"],
                            )
                        else:
                            # pr: just Arctic subset (no pressure levels)
                            ds_sub = subset_arctic(
                                ds_period,
                                cfg["domain"]["arctic_lat_min"],
                                cfg["domain"]["arctic_lat_max"],
                            )

                        # Regrid
                        method = (cfg["regrid"]["method_ta"] if var == "ta"
                                  else cfg["regrid"]["method_pr"])
                        try:
                            ds_regridded = regrid_to_common(
                                ds_sub,
                                target_res=cfg["regrid"]["target_resolution"],
                                method=method,
                            )
                        except Exception as e:
                            logger.warning(
                                "Regridding failed for %s/%s/%s/%s: %s. "
                                "Saving without regridding.",
                                model, exp, var, period_name, e
                            )
                            ds_regridded = ds_sub

                        # Save
                        fname = f"{model}_{exp}_{var}_{period_name}.nc"
                        out_path = out_dir / fname
                        ds_regridded.to_netcdf(out_path)
                        logger.info("  Saved: %s", out_path)

                    except Exception as e:
                        logger.error(
                            "  Failed %s/%s/%s/%s: %s",
                            model, exp, var, period_name, e
                        )
                        continue

    logger.info("Download and preprocessing complete.")


if __name__ == "__main__":
    main()
