#!/usr/bin/env python3
"""Fast batch download: one model at a time with timeout protection."""

import logging
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.catalog import open_catalog, search_catalog
from src.data.preprocess import select_period, select_pressure_layer, subset_arctic

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEMBER_ATTEMPTS = ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3", "r2i1p1f1"]
GRID_ATTEMPTS = ["gn", "gr", "gr1"]
TIMEOUT_PER_DATASET = 120  # seconds


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Dataset load timed out")


def try_load_dataset(catalog, var, exp, model):
    """Try all member/grid combos to load a dataset."""
    for member in MEMBER_ATTEMPTS:
        for grid in GRID_ATTEMPTS:
            sub = search_catalog(catalog, variable_id=var,
                                 experiment_id=exp, source_id=model,
                                 table_id="Amon", member_id=member,
                                 grid_label=grid)
            if len(sub.df) == 0:
                continue
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TIMEOUT_PER_DATASET)
                dsets = sub.to_dataset_dict(
                    zarr_kwargs={"consolidated": True, "use_cftime": True},
                    progressbar=False,
                )
                signal.alarm(0)
                key = list(dsets.keys())[0]
                logger.info("  Loaded %s (%s/%s)", key, member, grid)
                return dsets[key]
            except TimeoutError:
                logger.warning("  Timeout for %s/%s/%s/%s/%s",
                               model, exp, var, member, grid)
                signal.alarm(0)
                continue
            except Exception:
                signal.alarm(0)
                continue
    return None


def main():
    cfg = load_config()
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = open_catalog()
    models = cfg["models"]
    lat_min, lat_max = 55.0, 90.0

    succeeded = []
    failed = []

    for i, model in enumerate(models):
        logger.info("=== [%d/%d] %s ===", i + 1, len(models), model)

        # Check if already fully downloaded
        needed_files = []
        for exp in cfg["experiments"]:
            for period_name, _ in cfg["periods"].items():
                mapping = cfg["period_experiment"][period_name]
                exps = mapping if isinstance(mapping, list) else [mapping]
                if exp not in exps:
                    continue
                for var in ["ta", "pr"]:
                    f = out_dir / f"{model}_{exp}_{var}_{period_name}.nc"
                    if not f.exists():
                        needed_files.append((exp, var, period_name))

        if not needed_files:
            logger.info("  Already complete, skipping")
            succeeded.append(model)
            continue

        # Group needed files by (exp, var) to load each dataset once
        exp_var_periods = {}
        for exp, var, period_name in needed_files:
            key = (exp, var)
            if key not in exp_var_periods:
                exp_var_periods[key] = []
            exp_var_periods[key].append(period_name)

        model_ok = True
        for (exp, var), periods in exp_var_periods.items():
            logger.info("  Loading %s/%s/%s...", model, exp, var)
            ds = try_load_dataset(catalog, var, exp, model)
            if ds is None:
                logger.warning("  FAILED to load %s/%s/%s", model, exp, var)
                model_ok = False
                continue

            for period_name in periods:
                out_path = out_dir / f"{model}_{exp}_{var}_{period_name}.nc"
                period_cfg = cfg["periods"][period_name]
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TIMEOUT_PER_DATASET)

                    ds_period = select_period(ds, period_cfg["start"],
                                              period_cfg["end"])
                    ds_sub = subset_arctic(ds_period, lat_min, lat_max)
                    if var == "ta":
                        ds_sub = select_pressure_layer(
                            ds_sub,
                            cfg["pressure"]["full_range_top"],
                            cfg["pressure"]["full_range_bot"],
                        )
                    ds_sub.to_netcdf(out_path)
                    signal.alarm(0)
                    logger.info("    Saved %s", out_path.name)
                except TimeoutError:
                    signal.alarm(0)
                    logger.warning("    Timeout saving %s/%s/%s",
                                   model, exp, period_name)
                except Exception as e:
                    signal.alarm(0)
                    logger.error("    Error %s/%s/%s: %s",
                                 model, exp, period_name, e)

        if model_ok:
            succeeded.append(model)
        else:
            failed.append(model)

    logger.info("\n=== DOWNLOAD SUMMARY ===")
    logger.info("Succeeded: %d models", len(succeeded))
    logger.info("Failed: %d models: %s", len(failed), failed)

    total_files = len(list(out_dir.glob("*.nc")))
    total_models = len(set(f.name.split("_")[0]
                           for f in out_dir.glob("*.nc")))
    logger.info("Total: %d files, %d distinct models", total_files, total_models)


if __name__ == "__main__":
    main()
