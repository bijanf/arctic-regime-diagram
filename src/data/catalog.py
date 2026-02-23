"""Search and filter CMIP6 datasets from the Pangeo Google Cloud catalog."""

from __future__ import annotations

import logging
from typing import Optional

import intake

logger = logging.getLogger(__name__)

PANGEO_CATALOG_URL = (
    "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
)


def open_catalog():
    """Open the Pangeo CMIP6 intake-esm catalog.

    Returns
    -------
    intake_esm.core.esm_datastore
        The full Pangeo CMIP6 catalog.
    """
    logger.info("Opening Pangeo CMIP6 catalog from %s", PANGEO_CATALOG_URL)
    return intake.open_esm_datastore(PANGEO_CATALOG_URL)


def search_catalog(
    catalog,
    variable_id: str,
    experiment_id: str | list[str],
    source_id: Optional[str | list[str]] = None,
    table_id: str = "Amon",
    member_id: Optional[str | list[str]] = None,
    grid_label: str = "gn",
) -> intake.open_esm_datastore:
    """Search the catalog for matching datasets.

    Parameters
    ----------
    catalog : intake_esm.core.esm_datastore
        Pangeo CMIP6 catalog.
    variable_id : str
        CMIP6 variable name (e.g., 'ta', 'pr').
    experiment_id : str or list of str
        Experiment name(s) (e.g., 'historical', 'ssp585').
    source_id : str or list of str, optional
        Model name(s). If None, searches all models.
    table_id : str
        MIP table (default 'Amon').
    member_id : str or list of str, optional
        Ensemble member(s). If None, searches all.
    grid_label : str
        Grid label (default 'gn').

    Returns
    -------
    intake_esm.core.esm_datastore
        Filtered catalog subset.
    """
    search_kwargs = dict(
        variable_id=variable_id,
        experiment_id=experiment_id,
        table_id=table_id,
        grid_label=grid_label,
    )
    if source_id is not None:
        search_kwargs["source_id"] = source_id
    if member_id is not None:
        search_kwargs["member_id"] = member_id

    result = catalog.search(**search_kwargs)
    logger.info(
        "Found %d dataset(s) for %s / %s",
        len(result.df),
        variable_id,
        experiment_id,
    )
    return result


def find_common_models(
    catalog,
    variables: list[str],
    experiments: list[str],
    model_list: list[str],
    table_id: str = "Amon",
) -> list[str]:
    """Find models that have all required variables across all experiments.

    Parameters
    ----------
    catalog : intake_esm.core.esm_datastore
        Pangeo CMIP6 catalog.
    variables : list of str
        Required variable IDs (e.g., ['ta', 'pr']).
    experiments : list of str
        Required experiment IDs.
    model_list : list of str
        Candidate model names from models.yaml.
    table_id : str
        MIP table.

    Returns
    -------
    list of str
        Sorted list of models available for all variable-experiment combos.
    """
    available = None
    for var in variables:
        for exp in experiments:
            # Search both gn and gr grids
            sub_gn = search_catalog(
                catalog,
                variable_id=var,
                experiment_id=exp,
                source_id=model_list,
                table_id=table_id,
                grid_label="gn",
            )
            sub_gr = search_catalog(
                catalog,
                variable_id=var,
                experiment_id=exp,
                source_id=model_list,
                table_id=table_id,
                grid_label="gr",
            )
            models_here = (
                set(sub_gn.df["source_id"].unique())
                | set(sub_gr.df["source_id"].unique())
            )
            available = models_here if available is None else available & models_here

    common = sorted(available & set(model_list)) if available else []
    logger.info("Common models across all var/exp combos: %s", common)
    return common
