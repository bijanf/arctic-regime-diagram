"""Compute the diabatic number D.

D quantifies the ratio of diabatic (moist/latent) heating to dry baroclinic
energy scales:

    D = (Lv · pr_arctic) / (cp · |∂T/∂y|_edge · f₀ · Ld · H)

where:
  - pr_arctic : Arctic-mean precipitation rate (kg m⁻² s⁻¹)
  - |∂T/∂y|_edge : meridional temperature gradient at Arctic edge (55-65°N)
  - Ld : Rossby deformation radius = N·H/f₀
  - N : Brunt-Väisälä frequency computed from σ̄
"""

import logging

import numpy as np
import xarray as xr

from ..data.preprocess import (
    area_weights, subset_arctic, subset_edge,
    _get_lat_name, _get_plev_name,
)

logger = logging.getLogger(__name__)

# Physical constants (defaults)
LV = 2.501e6    # J/kg, latent heat of vaporization
CP = 1004.0     # J/(kg·K)
F0 = 1.2e-4     # s⁻¹, Coriolis at ~55°N
H_SCALE = 8500.0  # m, scale height
R_EARTH = 6.371e6  # m, Earth radius


def meridional_T_gradient(
    T: xr.DataArray,
    lat_min: float = 55.0,
    lat_max: float = 65.0,
    plev_level: float = 700.0,
    time_dim: str = "time",
) -> float:
    """Compute |∂T/∂y| averaged over the Arctic edge band.

    Approximates the meridional gradient at a single pressure level,
    averaged over the edge latitudes and all longitudes.

    Parameters
    ----------
    T : xr.DataArray
        Temperature (K) on pressure levels, covering at least 55-65°N.
    lat_min, lat_max : float
        Latitude bounds for the edge region.
    plev_level : float
        Pressure level (hPa) for gradient computation.
    time_dim : str
        Time dimension name.

    Returns
    -------
    float
        |∂T/∂y| in K/m (positive definite).
    """
    ds_temp = T.to_dataset(name="ta")
    plev_name = _get_plev_name(ds_temp)
    lat_name = _get_lat_name(ds_temp)

    # Select pressure level
    plev_vals = T[plev_name].values
    if np.nanmax(plev_vals) > 10000:
        plev_level_u = plev_level * 100.0
    else:
        plev_level_u = plev_level
    T_level = T.sel({plev_name: plev_level_u}, method="nearest")

    # Subset to edge latitudes
    ds_edge = subset_edge(T_level.to_dataset(name="ta"), lat_min, lat_max)
    T_edge = ds_edge["ta"]

    # Time mean
    T_clim = T_edge.mean(dim=time_dim)

    # Zonal mean
    lon_name = None
    for name in ["lon", "longitude", "LON"]:
        if name in T_clim.dims:
            lon_name = name
            break
    if lon_name:
        T_clim = T_clim.mean(dim=lon_name)

    # ∂T/∂lat in K/degree
    dT_dlat = T_clim.differentiate(lat_name)  # K per degree

    # Convert to K/m: dy = R_earth · dφ, dφ in radians
    # So ∂T/∂y = (∂T/∂lat_deg) / (R_earth · π/180)
    dT_dy = dT_dlat / (R_EARTH * np.pi / 180.0)

    # Mean absolute value
    grad_mag = float(np.abs(dT_dy).mean())
    logger.info("|∂T/∂y| at %.0f hPa, %.0f-%.0f°N = %.2e K/m",
                plev_level, lat_min, lat_max, grad_mag)
    return grad_mag


def rossby_deformation_radius(sigma_bar: float, H: float = H_SCALE,
                              f0: float = F0) -> float:
    """Compute the Rossby deformation radius Ld = N·H/f₀.

    N is the Brunt-Väisälä frequency, related to σ̄ by N² ≈ g·σ̄ / T_ref
    or more directly N = sqrt(σ̄) · Δp / H for the layer-averaged form.

    Here we use: N² = σ̄ · (g/T_ref) where T_ref ~ 250 K for mid-troposphere,
    giving N = sqrt(σ̄ · g / T_ref).

    Then Ld = N · H / f₀.

    Parameters
    ----------
    sigma_bar : float
        Climatological mean static stability (dimensionless, ~1e-6 to 1e-5).
    H : float
        Scale height (m).
    f0 : float
        Coriolis parameter (s⁻¹).

    Returns
    -------
    float
        Rossby deformation radius Ld (m).
    """
    g = 9.81
    T_ref = 250.0  # K, representative mid-tropospheric temperature
    N_squared = abs(sigma_bar) * g / T_ref
    N = np.sqrt(max(N_squared, 1e-10))  # Ensure non-negative
    Ld = N * H / f0
    logger.info("N = %.3e s⁻¹, Ld = %.0f km", N, Ld / 1000)
    return Ld


def arctic_mean_precipitation(
    pr: xr.DataArray,
    lat_min: float = 60.0,
    lat_max: float = 90.0,
    time_dim: str = "time",
) -> float:
    """Compute area-weighted Arctic-mean precipitation.

    Parameters
    ----------
    pr : xr.DataArray
        Precipitation rate (kg m⁻² s⁻¹).
    lat_min, lat_max : float
        Arctic latitude bounds.
    time_dim : str
        Time dimension name.

    Returns
    -------
    float
        Arctic-mean precipitation (kg m⁻² s⁻¹).
    """
    ds_pr = pr.to_dataset(name="pr")
    ds_arctic = subset_arctic(ds_pr, lat_min, lat_max)
    pr_arctic = ds_arctic["pr"]

    # Time mean
    pr_clim = pr_arctic.mean(dim=time_dim)

    # Area-weighted mean
    weights = area_weights(ds_arctic)
    lat_name = _get_lat_name(ds_arctic)

    pr_mean = float((pr_clim * weights).sum(dim=lat_name).mean())
    logger.info("Arctic-mean precipitation = %.2e kg/(m²·s)", pr_mean)
    return pr_mean


def diabatic_number(
    pr_arctic: float,
    dT_dy: float,
    sigma_bar: float,
    Lv: float = LV,
    cp: float = CP,
    f0: float = F0,
    H: float = H_SCALE,
) -> float:
    """Compute the diabatic number D.

    D = (Lv · pr_arctic) / (cp · |∂T/∂y| · f₀ · Ld · H)

    Parameters
    ----------
    pr_arctic : float
        Arctic-mean precipitation rate (kg m⁻² s⁻¹).
    dT_dy : float
        |∂T/∂y| meridional temperature gradient (K/m).
    sigma_bar : float
        Climatological mean static stability.
    Lv : float
        Latent heat of vaporization (J/kg).
    cp : float
        Specific heat at constant pressure (J/(kg·K)).
    f0 : float
        Coriolis parameter (s⁻¹).
    H : float
        Scale height (m).

    Returns
    -------
    float
        Diabatic number D (dimensionless).
    """
    Ld = rossby_deformation_radius(sigma_bar, H=H, f0=f0)

    numerator = Lv * pr_arctic
    denominator = cp * dT_dy * f0 * Ld * H

    if denominator == 0:
        logger.warning("Denominator is zero in diabatic number; returning NaN")
        return float("nan")

    D = numerator / denominator
    logger.info("D = %.3f  (numerator=%.2e, denominator=%.2e)",
                D, numerator, denominator)
    return D
