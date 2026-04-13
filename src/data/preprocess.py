"""Preprocess CMIP6 data: Arctic subsetting, time slicing, regridding."""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def subset_arctic(ds: xr.Dataset, lat_min: float = 60.0, lat_max: float = 90.0) -> xr.Dataset:
    """Select latitudes within the Arctic band.

    Handles both ascending and descending latitude coordinates, and
    common coordinate names (lat, latitude).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a latitude dimension.
    lat_min, lat_max : float
        Latitude bounds (°N).

    Returns
    -------
    xr.Dataset
        Arctic subset.
    """
    lat_name = _get_lat_name(ds)
    lat_vals = ds[lat_name].values

    if lat_vals[0] > lat_vals[-1]:
        # Descending latitudes
        return ds.sel({lat_name: slice(lat_max, lat_min)})
    return ds.sel({lat_name: slice(lat_min, lat_max)})


def subset_edge(ds: xr.Dataset, lat_min: float = 55.0, lat_max: float = 65.0) -> xr.Dataset:
    """Select the Arctic-edge latitude band for meridional gradients."""
    lat_name = _get_lat_name(ds)
    lat_vals = ds[lat_name].values

    if lat_vals[0] > lat_vals[-1]:
        return ds.sel({lat_name: slice(lat_max, lat_min)})
    return ds.sel({lat_name: slice(lat_min, lat_max)})


def select_period(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Slice dataset to a time period.

    Parameters
    ----------
    ds : xr.Dataset
    start, end : str
        ISO date strings (e.g., '1980-01-01').

    Returns
    -------
    xr.Dataset
    """
    time_name = _get_time_name(ds)
    # Truncate end date to YYYY-MM to avoid 360-day calendar issues
    parts = end.split("-")
    if len(parts) == 3:
        end = f"{parts[0]}-{parts[1]}"
    return ds.sel({time_name: slice(start, end)})


def select_pressure_layer(ds: xr.Dataset, p_top: float, p_bot: float) -> xr.Dataset:
    """Select pressure levels between p_top and p_bot (hPa).

    Handles both ascending and descending pressure coordinates.
    Automatically converts Pa to hPa if needed.
    """
    plev_name = _get_plev_name(ds)
    plev_vals = ds[plev_name].values.copy()

    # Convert Pa to hPa if values are > 10000
    scale = 1.0
    if np.nanmax(plev_vals) > 10000:
        scale = 0.01
        plev_vals = plev_vals * scale

    p_lo = min(p_top, p_bot)
    p_hi = max(p_top, p_bot)

    if ds[plev_name].values[0] > ds[plev_name].values[-1]:
        # Descending pressure
        return ds.sel({plev_name: slice(p_hi / scale, p_lo / scale)})
    return ds.sel({plev_name: slice(p_lo / scale, p_hi / scale)})


def area_weights(ds: xr.Dataset) -> xr.DataArray:
    """Compute cosine-of-latitude area weights.

    Parameters
    ----------
    ds : xr.Dataset
        Must have a latitude coordinate.

    Returns
    -------
    xr.DataArray
        Weights proportional to cos(lat), normalized to sum to 1.
    """
    lat_name = _get_lat_name(ds)
    weights = np.cos(np.deg2rad(ds[lat_name]))
    weights = weights / weights.sum()
    return weights


def regrid_to_common(
    ds: xr.Dataset, target_res: float = 2.0, method: str = "bilinear"
) -> xr.Dataset:
    """Regrid a dataset to a regular lat-lon grid using xESMF.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    target_res : float
        Target resolution in degrees.
    method : str
        Regridding method ('bilinear' or 'conservative').

    Returns
    -------
    xr.Dataset
        Regridded dataset.
    """
    import xesmf as xe

    lat_name = _get_lat_name(ds)
    lon_name = _get_lon_name(ds)

    # Build target grid
    lat_out = np.arange(-90 + target_res / 2, 90, target_res)
    lon_out = np.arange(0 + target_res / 2, 360, target_res)
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], lat_out),
            "lon": (["lon"], lon_out),
        }
    )

    # Rename to standard names for xESMF
    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    regridder = xe.Regridder(ds, ds_out, method, periodic=True)
    ds_regridded = regridder(ds, keep_attrs=True)
    return ds_regridded


# --- Coordinate name helpers ---


def _get_lat_name(ds: xr.Dataset) -> str:
    """Detect the latitude coordinate name."""
    for name in ["lat", "latitude", "LAT", "Latitude"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Cannot find latitude coordinate in {list(ds.coords)}")


def _get_lon_name(ds: xr.Dataset) -> str:
    """Detect the longitude coordinate name."""
    for name in ["lon", "longitude", "LON", "Longitude"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Cannot find longitude coordinate in {list(ds.coords)}")


def _get_time_name(ds: xr.Dataset) -> str:
    """Detect the time coordinate name."""
    for name in ["time", "Time", "TIME"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Cannot find time coordinate in {list(ds.coords)}")


def _get_plev_name(ds: xr.Dataset) -> str:
    """Detect the pressure level coordinate name."""
    for name in ["plev", "lev", "level", "pressure", "air_pressure"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Cannot find pressure coordinate in {list(ds.coords)}")
