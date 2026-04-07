"""Lazy-load CMIP6 datasets from Google Cloud Zarr stores."""

import logging

import xarray as xr

from .catalog import open_catalog, search_catalog

logger = logging.getLogger(__name__)


def load_dataset(
    catalog,
    variable_id: str,
    experiment_id: str,
    source_id: str,
    member_id: str = "r1i1p1f1",
    table_id: str = "Amon",
    grid_label: str = "gn",
) -> xr.Dataset:
    """Load a single CMIP6 dataset lazily from Zarr on Google Cloud.

    Parameters
    ----------
    catalog : intake_esm.core.esm_datastore
        Pangeo CMIP6 catalog (already opened).
    variable_id : str
        Variable name (e.g., 'ta').
    experiment_id : str
        Experiment (e.g., 'historical').
    source_id : str
        Model name (e.g., 'CESM2').
    member_id : str
        Ensemble member (default 'r1i1p1f1').
    table_id : str
        MIP table (default 'Amon').
    grid_label : str
        Grid label (default 'gn'; falls back to 'gr' if not found).

    Returns
    -------
    xr.Dataset
        Lazily loaded xarray Dataset backed by Zarr/Dask.
    """
    sub = search_catalog(
        catalog,
        variable_id=variable_id,
        experiment_id=experiment_id,
        source_id=source_id,
        table_id=table_id,
        member_id=member_id,
        grid_label=grid_label,
    )

    # Fall back to other grids if primary grid not available
    if len(sub.df) == 0:
        for alt_grid in ["gr", "gr1", "gn"]:
            if alt_grid == grid_label:
                continue
            logger.info(
                "No '%s' grid for %s/%s/%s, trying '%s'",
                grid_label,
                source_id,
                experiment_id,
                variable_id,
                alt_grid,
            )
            sub = search_catalog(
                catalog,
                variable_id=variable_id,
                experiment_id=experiment_id,
                source_id=source_id,
                table_id=table_id,
                member_id=member_id,
                grid_label=alt_grid,
            )
            if len(sub.df) > 0:
                break

    if len(sub.df) == 0:
        raise ValueError(
            f"No dataset found for {source_id}/{experiment_id}/{variable_id}/{member_id}"
        )

    dsets = sub.to_dataset_dict(
        zarr_kwargs={"consolidated": True, "use_cftime": True},
        progressbar=False,
    )

    # to_dataset_dict returns a dict with keys; take the first one
    key = list(dsets.keys())[0]
    ds = dsets[key]
    logger.info(
        "Loaded %s (%s, %s, %s): %s", key, source_id, experiment_id, variable_id, list(ds.dims)
    )
    return ds


def load_variable(
    variable_id: str,
    experiment_id: str,
    source_id: str,
    member_id: str = "r1i1p1f1",
    catalog=None,
) -> xr.Dataset:
    """Convenience wrapper: open catalog if needed, then load dataset.

    Parameters
    ----------
    variable_id, experiment_id, source_id, member_id : str
        Standard CMIP6 identifiers.
    catalog : intake_esm catalog, optional
        If None, opens the Pangeo catalog automatically.

    Returns
    -------
    xr.Dataset
    """
    if catalog is None:
        catalog = open_catalog()
    return load_dataset(catalog, variable_id, experiment_id, source_id, member_id=member_id)
