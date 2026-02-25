"""Rossby deformation radius maps and stationary wave refractive index.

Deformation radius:
    Ld(lat) = √σ̄ · Δp / f(lat)

Stationary wave refractive index (Karoly & Hoskins, 1982):
    K_s² = β* / ū_M
    β* = β - ∂²ū/∂y²
    β = 2Ω·cos(φ)/a
"""

import logging

import numpy as np
import xarray as xr

from ..data.preprocess import (
    _get_plev_name, _get_lat_name, _get_lon_name,
    area_weights, subset_arctic,
)
from .static_stability import static_stability, layer_mean_stability

logger = logging.getLogger(__name__)

# Physical constants
OMEGA = 7.2921e-5    # rad/s, Earth rotation rate
A_EARTH = 6.371e6    # m, Earth radius
G = 9.81             # m/s²
RD = 287.05          # J/(kg·K)
H_SCALE = 8500.0     # m, scale height


def deformation_radius_map(sigma_field: xr.DataArray,
                            lat: xr.DataArray,
                            dp: float = 35000.0,
                            f0_fixed: float | None = None) -> xr.DataArray:
    """Compute Rossby deformation radius Ld(lat, lon) from local σ̄.

    Ld = √σ̄ · Δp / f(lat)

    Parameters
    ----------
    sigma_field : xr.DataArray
        Layer-mean static stability (time-mean or instantaneous).
        Dims: (lat, lon) or (time, lat, lon).
    lat : xr.DataArray
        Latitude coordinate (degrees).
    dp : float
        Pressure depth of averaging layer (Pa). Default 35000 Pa (500-850 hPa).
    f0_fixed : float, optional
        If given, use a fixed Coriolis parameter instead of f(lat).

    Returns
    -------
    xr.DataArray
        Deformation radius Ld in meters.
    """
    sigma_pos = sigma_field.clip(min=1e-20)

    if f0_fixed is not None:
        f = f0_fixed
    else:
        f = 2.0 * OMEGA * np.abs(np.sin(np.deg2rad(lat)))
        # Avoid division by zero near equator
        f = f.where(f > 1e-10, 1e-10)

    Ld = np.sqrt(sigma_pos) * dp / f
    Ld.name = "Ld"
    Ld.attrs["units"] = "m"
    Ld.attrs["long_name"] = "Rossby deformation radius"
    return Ld


def zonal_mean_wind(ua: xr.DataArray,
                     time_dim: str = "time") -> xr.DataArray:
    """Compute zonal-mean, time-mean zonal wind ū(lat, plev).

    Parameters
    ----------
    ua : xr.DataArray
        Eastward wind (m/s) with dims (time, plev, lat, lon).
    time_dim : str
        Time dimension name.

    Returns
    -------
    xr.DataArray
        ū(lat, plev) in m/s.
    """
    ds_temp = ua.to_dataset(name="ua") if isinstance(ua, xr.DataArray) else ua
    lon_name = _get_lon_name(ds_temp)

    u_bar = ua.mean(dim=lon_name)
    if time_dim in u_bar.dims:
        u_bar = u_bar.mean(dim=time_dim)

    u_bar.name = "u_bar"
    u_bar.attrs["units"] = "m s-1"
    u_bar.attrs["long_name"] = "Zonal-mean zonal wind"
    return u_bar


def beta_star(u_bar: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """Compute β* = β - ∂²ū/∂y² on spherical coordinates.

    Uses the correct spherical form:
        ∂²ū/∂y² = (1/(a²·cos(φ))) · d/dφ[cos(φ) · dū/dφ]

    β = 2Ω·cos(φ)/a

    Parameters
    ----------
    u_bar : xr.DataArray
        Zonal-mean zonal wind ū(lat, plev).
    lat : xr.DataArray
        Latitude in degrees.

    Returns
    -------
    xr.DataArray
        β* (m⁻¹ s⁻¹).
    """
    ds_temp = u_bar.to_dataset(name="u_bar") if isinstance(u_bar, xr.DataArray) else u_bar
    lat_name = _get_lat_name(ds_temp)

    phi = np.deg2rad(lat)
    cos_lat = np.cos(phi)
    # Avoid division by zero at poles
    cos_lat_safe = cos_lat.where(np.abs(cos_lat) > 1e-6, 1e-6)

    # β = 2Ω·cos(φ)/a
    beta = 2.0 * OMEGA * cos_lat / A_EARTH

    # Correct spherical metric for d²ū/dy²:
    # d²ū/dy² = (1/(a²·cos(φ))) · d/dφ[cos(φ) · dū/dφ]
    # where differentiate gives d/d(lat_deg), so we need * (π/180) for d/dφ
    deg2rad = np.pi / 180.0

    # Step 1: dū/dφ = dū/d(lat_deg) * (1/deg2rad)... but differentiate
    # gives dū/d(lat_deg), so cos(φ)·dū/dφ = cos(lat) * du_dlat / deg2rad
    du_dlat = u_bar.differentiate(lat_name)  # (m/s) / degree
    cos_du = cos_lat * du_dlat  # cos(φ) · dū/d(lat_deg)

    # Step 2: d/dφ[cos(φ) · dū/dφ]
    # = d/d(lat_deg)[cos(φ) · dū/d(lat_deg)] / deg2rad²
    # (one factor for the outer d/dφ, one already implicit in du_dlat)
    d_cos_du = cos_du.differentiate(lat_name)  # d/d(lat_deg) of cos_du

    # Full expression: (1/(a²·cos(φ))) · d/dφ[cos(φ)·dū/dφ]
    # d/dφ = d/d(lat_deg) / deg2rad, applied twice gives / deg2rad²
    d2u_dy2 = d_cos_du / (A_EARTH**2 * cos_lat_safe * deg2rad**2)

    beta_s = beta - d2u_dy2
    beta_s.name = "beta_star"
    beta_s.attrs["units"] = "m-1 s-1"
    beta_s.attrs["long_name"] = "Meridional gradient of absolute vorticity"
    return beta_s


def refractive_index_squared(u_bar: xr.DataArray, lat: xr.DataArray,
                              plev_name: str | None = None,
                              N: xr.DataArray | None = None,
                              k: int | None = None) -> xr.DataArray:
    """Compute stationary wave refractive index K_s²(lat, plev).

    K_s² = β* / ū_M

    For a specific zonal wavenumber k with vertical propagation:
        n_k² = β*/ū - k² - f₀²/(4N²H²)

    Parameters
    ----------
    u_bar : xr.DataArray
        Zonal-mean zonal wind ū(lat, plev).
    lat : xr.DataArray
        Latitude in degrees.
    plev_name : str, optional
        Pressure coordinate name.
    N : xr.DataArray, optional
        Brunt-Väisälä frequency N(lat, plev). Required if k is given.
    k : int, optional
        Zonal wavenumber. If None, return total K_s² = β*/ū.

    Returns
    -------
    xr.DataArray
        K_s² or n_k² (m⁻²).
    """
    beta_s = beta_star(u_bar, lat)

    # Avoid division by zero in regions of weak wind
    u_safe = u_bar.where(np.abs(u_bar) > 0.5, np.nan)

    Ks2 = beta_s / u_safe

    if k is not None:
        # Convert zonal wavenumber to angular wavenumber at each latitude
        phi = np.deg2rad(lat)
        cos_phi = np.cos(phi).clip(min=0.01)
        k_angular = k / (A_EARTH * cos_phi)  # rad/m

        Ks2 = Ks2 - k_angular ** 2

        if N is not None:
            f0 = 2.0 * OMEGA * np.sin(phi)
            N_safe = N.where(N > 1e-6, 1e-6)
            vertical_term = f0 ** 2 / (4.0 * N_safe ** 2 * H_SCALE ** 2)
            Ks2 = Ks2 - vertical_term

    Ks2.name = "Ks2"
    Ks2.attrs["units"] = "m-2"
    if k is not None:
        Ks2.attrs["long_name"] = f"Refractive index squared (k={k})"
    else:
        Ks2.attrs["long_name"] = "Stationary wave refractive index squared"
    return Ks2


def compute_wave_diagnostics(ua: xr.DataArray, T: xr.DataArray,
                              sigma_bar: float,
                              lat_min: float = 55.0, lat_max: float = 75.0,
                              p_top: float = 500.0, p_bot: float = 850.0,
                              f0: float = 1.2e-4,
                              dp: float = 35000.0,
                              Ks_plev: float = 250.0,
                              Ks_lat_min: float = 50.0,
                              Ks_lat_max: float = 70.0) -> dict:
    """Compute Ld and K_s² diagnostics for a model/period.

    Parameters
    ----------
    ua : xr.DataArray
        Eastward wind on pressure levels.
    T : xr.DataArray
        Temperature on pressure levels.
    sigma_bar : float
        Domain-mean layer-mean static stability (from R computation).
    lat_min, lat_max : float
        Latitude bounds for Ld averaging.
    p_top, p_bot : float
        Pressure layer bounds (hPa) for Ld computation.
    f0 : float
        Reference Coriolis parameter (s⁻¹).
    dp : float
        Pressure layer depth (Pa) for Ld.
    Ks_plev : float
        Pressure level (hPa) for K_s² diagnostic.
    Ks_lat_min, Ks_lat_max : float
        Latitude bounds for K_s² averaging.

    Returns
    -------
    dict
        Keys: Ld_mean_km, Ks_250.
    """
    ds_temp = T.to_dataset(name="ta") if isinstance(T, xr.DataArray) else T
    plev_name = _get_plev_name(ds_temp)
    lat_name = _get_lat_name(ds_temp)
    lat = T[lat_name]

    # ── Ld from spatial σ̄ field ──
    sigma = static_stability(T, plev_name=plev_name)
    sigma_layer = layer_mean_stability(sigma, p_top=p_top, p_bot=p_bot)
    # Time-mean
    if "time" in sigma_layer.dims:
        sigma_clim = sigma_layer.mean(dim="time")
    else:
        sigma_clim = sigma_layer

    Ld = deformation_radius_map(sigma_clim, lat, dp=dp)

    # Domain-mean Ld
    ds_Ld = Ld.to_dataset(name="Ld")
    ds_Ld_arctic = subset_arctic(ds_Ld, lat_min, lat_max)
    weights = area_weights(ds_Ld_arctic)
    lat_name_sub = _get_lat_name(ds_Ld_arctic)
    Ld_mean = float(
        (ds_Ld_arctic["Ld"] * weights).sum(dim=lat_name_sub).mean()
    )
    Ld_mean_km = Ld_mean / 1000.0

    # ── K_s² from zonal-mean wind ──
    u_bar = zonal_mean_wind(ua)
    beta_s = beta_star(u_bar, lat)

    # K_s² at specified pressure level
    plev_vals = u_bar[plev_name].values if plev_name in u_bar.dims else None
    if plev_vals is not None:
        if float(np.nanmax(plev_vals)) > 10000:
            Ks_plev_u = Ks_plev * 100.0
        else:
            Ks_plev_u = Ks_plev

        u_bar_level = u_bar.sel({plev_name: Ks_plev_u}, method="nearest")
        beta_s_level = beta_s.sel({plev_name: Ks_plev_u}, method="nearest")

        u_safe = u_bar_level.where(np.abs(u_bar_level) > 0.5, np.nan)
        Ks2_level = beta_s_level / u_safe

        # Average over specified latitude band
        lat_vals = Ks2_level[lat_name].values
        lat_mask = (lat_vals >= Ks_lat_min) & (lat_vals <= Ks_lat_max)
        Ks2_band = Ks2_level.isel({lat_name: lat_mask})
        Ks_250 = float(Ks2_band.mean())
    else:
        Ks_250 = float("nan")

    logger.info("Ld_mean=%.0f km, K_s²(%.0f hPa, %.0f-%.0f°N)=%.2e m⁻²",
                Ld_mean_km, Ks_plev, Ks_lat_min, Ks_lat_max, Ks_250)

    return {
        "Ld_mean_km": Ld_mean_km,
        "Ks_250": Ks_250,
    }
