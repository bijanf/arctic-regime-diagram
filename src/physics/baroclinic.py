"""Eady growth rate (dry and moist) for baroclinic instability diagnostics.

Dry Eady growth rate (Lindzen & Farrell, 1980):
    σ_E = 0.31 · f(lat) · |∂u/∂z| / N

where:
    f(lat) = 2Ω·sin(lat)    (latitude-dependent Coriolis)
    |∂u/∂z|                  (vertical wind shear, from ua on pressure levels)
    N = Brunt-Väisälä frequency

Moist Eady growth rate (Emanuel et al., 1987):
    σ_moist = σ_E × F(RH, T)
    F = 1 + (Lv·qs)/(cp·T) · (RH/100)^γ
"""

import logging

import numpy as np
import xarray as xr

from ..data.preprocess import _get_plev_name, _get_lat_name, area_weights, subset_arctic
from .static_stability import static_stability, layer_mean_stability

logger = logging.getLogger(__name__)

# Physical constants
OMEGA = 7.2921e-5   # rad/s, Earth rotation rate
G = 9.81             # m/s²
RD = 287.05          # J/(kg·K), dry air gas constant
CP = 1004.0          # J/(kg·K), specific heat at constant pressure
LV = 2.501e6         # J/kg, latent heat of vaporization


def coriolis_parameter(lat: xr.DataArray) -> xr.DataArray:
    """Compute latitude-dependent Coriolis parameter f = 2Ω·sin(lat).

    Parameters
    ----------
    lat : xr.DataArray
        Latitude in degrees.

    Returns
    -------
    xr.DataArray
        Coriolis parameter (s⁻¹).
    """
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat))


def brunt_vaisala_from_sigma(sigma: xr.DataArray, T: xr.DataArray,
                              plev_name: str | None = None) -> xr.DataArray:
    """Compute Brunt-Väisälä frequency N from static stability σ and T.

    From the relation:
        N² = σ · p² · g² / (Rd · T)²

    which gives:
        N = p·g/(Rd·T) · √σ

    Parameters
    ----------
    sigma : xr.DataArray
        Static stability parameter σ (m² s⁻² Pa⁻²).
    T : xr.DataArray
        Temperature (K) on pressure levels.
    plev_name : str, optional
        Pressure coordinate name. Auto-detected if None.

    Returns
    -------
    xr.DataArray
        Brunt-Väisälä frequency N (s⁻¹). Only valid where σ > 0.
    """
    if plev_name is None:
        ds_temp = T.to_dataset(name="ta") if isinstance(T, xr.DataArray) else T
        plev_name = _get_plev_name(ds_temp)

    plev = T[plev_name]
    # Ensure pressure in Pa
    plev_Pa = plev.copy()
    if float(plev.max()) < 10000:
        plev_Pa = plev * 100.0

    # Clip sigma to positive values (only stable stratification gives real N)
    sigma_pos = sigma.clip(min=1e-20)

    N = plev_Pa * G / (RD * T) * np.sqrt(sigma_pos)
    N.name = "N"
    N.attrs["units"] = "s-1"
    N.attrs["long_name"] = "Brunt-Vaisala frequency"
    return N


def vertical_wind_shear(ua: xr.DataArray, T: xr.DataArray,
                         plev_name: str | None = None) -> xr.DataArray:
    """Compute vertical wind shear ∂u/∂z from ua on pressure levels.

    Uses the hydrostatic relation:
        ∂u/∂z = (∂u/∂p) · (dp/dz) = (∂u/∂p) · (-ρg) = (∂u/∂p) · (-p·g/(Rd·T))

    Parameters
    ----------
    ua : xr.DataArray
        Eastward wind (m/s) on pressure levels.
    T : xr.DataArray
        Temperature (K) on pressure levels.
    plev_name : str, optional
        Pressure coordinate name. Auto-detected if None.

    Returns
    -------
    xr.DataArray
        Vertical wind shear |∂u/∂z| (s⁻¹).
    """
    if plev_name is None:
        ds_temp = ua.to_dataset(name="ua") if isinstance(ua, xr.DataArray) else ua
        plev_name = _get_plev_name(ds_temp)

    plev = ua[plev_name]
    # Ensure pressure in Pa
    plev_Pa = plev.copy()
    if float(plev.max()) < 10000:
        plev_Pa = plev * 100.0

    # ∂u/∂p via centred finite differences
    du_dp = ua.differentiate(plev_name)
    # If plev was in hPa, differentiate gives (m/s)/hPa; convert to (m/s)/Pa
    if float(plev.max()) < 10000:
        du_dp = du_dp / 100.0

    # ∂u/∂z = ∂u/∂p · (-p·g/(Rd·T))
    du_dz = du_dp * (-plev_Pa * G / (RD * T))

    du_dz.name = "du_dz"
    du_dz.attrs["units"] = "s-1"
    du_dz.attrs["long_name"] = "Vertical wind shear"
    return du_dz


def eady_growth_rate(ua: xr.DataArray, T: xr.DataArray,
                      sigma: xr.DataArray, lat: xr.DataArray,
                      plev_name: str | None = None) -> xr.DataArray:
    """Compute dry Eady growth rate σ_E = 0.31 · f · |∂u/∂z| / N.

    Parameters
    ----------
    ua : xr.DataArray
        Eastward wind (m/s) on pressure levels.
    T : xr.DataArray
        Temperature (K) on pressure levels.
    sigma : xr.DataArray
        Static stability (m² s⁻² Pa⁻²) on pressure levels.
    lat : xr.DataArray
        Latitude coordinate (degrees).
    plev_name : str, optional
        Pressure coordinate name. Auto-detected if None.

    Returns
    -------
    xr.DataArray
        Eady growth rate (s⁻¹).
    """
    f = coriolis_parameter(lat)
    N = brunt_vaisala_from_sigma(sigma, T, plev_name=plev_name)
    du_dz = vertical_wind_shear(ua, T, plev_name=plev_name)

    # Soft floor: clip N to minimum 5e-4 s⁻¹ instead of NaN-masking at 2e-3.
    # The old threshold (2e-3) clipped away weakly-stable grid points where
    # the climate-change signal lives. The soft clip preserves data while
    # still preventing division-by-zero.
    N_safe = N.clip(min=5e-4)

    sigma_E = 0.31 * np.abs(f) * np.abs(du_dz) / N_safe
    sigma_E.name = "eady_growth_rate"
    sigma_E.attrs["units"] = "s-1"
    sigma_E.attrs["long_name"] = "Dry Eady growth rate"
    return sigma_E


def moist_eady_enhancement(T: xr.DataArray, plev_name: str | None = None,
                            RH: float = 80.0, gamma: float = 0.6) -> xr.DataArray:
    """Compute the moist enhancement factor F(RH, T) for Eady growth rate.

    F = 1 + (Lv · qs) / (cp · T) · (RH/100)^γ

    where qs = 0.622 · es / (p - es) and es follows Teten's formula.

    Parameters
    ----------
    T : xr.DataArray
        Temperature (K) on pressure levels.
    RH : float
        Relative humidity (%). Default 80% for Arctic.
    gamma : float
        Exponent for RH dependence. Default 0.6 (midpoint of 0.5-0.8 range).

    Returns
    -------
    xr.DataArray
        Moist enhancement factor F (dimensionless, ≥ 1).
    """
    if plev_name is None:
        ds_temp = T.to_dataset(name="ta") if isinstance(T, xr.DataArray) else T
        plev_name = _get_plev_name(ds_temp)

    plev = T[plev_name]
    # Ensure pressure in Pa
    plev_Pa = plev.copy()
    if float(plev.max()) < 10000:
        plev_Pa = plev * 100.0

    # Saturation vapour pressure (Teten's formula)
    T_celsius = T - 273.15
    es = 611.2 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))

    # Saturation specific humidity
    qs = 0.622 * es / (plev_Pa - es).clip(min=1.0)

    # Moist enhancement — clip to physically reasonable range [1, 3]
    # (F > 3 indicates unphysical saturation at low pressures)
    F = 1.0 + (LV * qs) / (CP * T) * (RH / 100.0) ** gamma
    F = F.clip(min=1.0, max=3.0)

    F.name = "moist_enhancement"
    F.attrs["units"] = "1"
    F.attrs["long_name"] = "Moist Eady enhancement factor"
    return F


def moist_eady_growth_rate(ua: xr.DataArray, T: xr.DataArray,
                            sigma: xr.DataArray, lat: xr.DataArray,
                            plev_name: str | None = None,
                            RH: float = 80.0, gamma: float = 0.6) -> xr.DataArray:
    """Compute moist Eady growth rate: σ_moist = σ_E × F(RH, T).

    Parameters
    ----------
    ua : xr.DataArray
        Eastward wind (m/s) on pressure levels.
    T : xr.DataArray
        Temperature (K) on pressure levels.
    sigma : xr.DataArray
        Static stability (m² s⁻² Pa⁻²) on pressure levels.
    lat : xr.DataArray
        Latitude coordinate (degrees).
    plev_name : str, optional
        Pressure coordinate name. Auto-detected if None.
    RH : float
        Relative humidity (%). Default 80%.
    gamma : float
        RH exponent. Default 0.6.

    Returns
    -------
    xr.DataArray
        Moist Eady growth rate (s⁻¹).
    """
    sigma_E = eady_growth_rate(ua, T, sigma, lat, plev_name=plev_name)
    F = moist_eady_enhancement(T, plev_name=plev_name, RH=RH, gamma=gamma)

    sigma_moist = sigma_E * F
    sigma_moist.name = "moist_eady_growth_rate"
    sigma_moist.attrs["units"] = "s-1"
    sigma_moist.attrs["long_name"] = "Moist Eady growth rate"
    return sigma_moist


def compute_eady_diagnostics(ua: xr.DataArray, T: xr.DataArray,
                              lat_min: float = 50.0, lat_max: float = 70.0,
                              p_top: float = 500.0, p_bot: float = 850.0,
                              RH: float = 80.0, gamma: float = 0.6) -> dict:
    """Compute domain-mean, layer-averaged Eady growth rate diagnostics.

    Parameters
    ----------
    ua : xr.DataArray
        Eastward wind (m/s) on pressure levels.
    T : xr.DataArray
        Temperature (K) on pressure levels.
    lat_min, lat_max : float
        Latitude bounds for domain averaging (°N).
    p_top, p_bot : float
        Pressure layer bounds (hPa).
    RH : float
        Assumed relative humidity (%).
    gamma : float
        RH exponent for moist enhancement.

    Returns
    -------
    dict
        Keys: eady_dry (day⁻¹), eady_moist (day⁻¹), eady_ratio (F, dimensionless).
    """
    ds_temp = T.to_dataset(name="ta") if isinstance(T, xr.DataArray) else T
    plev_name = _get_plev_name(ds_temp)
    lat_name = _get_lat_name(ds_temp)
    lat = T[lat_name]

    # Compute static stability
    sigma = static_stability(T, plev_name=plev_name)

    # Compute Eady growth rates on full grid
    sigma_E = eady_growth_rate(ua, T, sigma, lat, plev_name=plev_name)
    F = moist_eady_enhancement(T, plev_name=plev_name, RH=RH, gamma=gamma)
    sigma_moist = sigma_E * F

    # Layer-average over pressure
    sigma_E_layer = layer_mean_stability(sigma_E, p_top=p_top, p_bot=p_bot)
    sigma_moist_layer = layer_mean_stability(sigma_moist, p_top=p_top, p_bot=p_bot)

    # Subset to domain
    ds_dry = sigma_E_layer.to_dataset(name="eady_dry")
    ds_dry = subset_arctic(ds_dry, lat_min, lat_max)
    ds_moist = sigma_moist_layer.to_dataset(name="eady_moist")
    ds_moist = subset_arctic(ds_moist, lat_min, lat_max)

    # Area-weighted time-space mean
    weights = area_weights(ds_dry)
    lat_name_sub = _get_lat_name(ds_dry)

    eady_dry_mean = float(
        (ds_dry["eady_dry"].mean(dim="time") * weights)
        .sum(dim=lat_name_sub).mean()
    )
    eady_moist_mean = float(
        (ds_moist["eady_moist"].mean(dim="time") * weights)
        .sum(dim=lat_name_sub).mean()
    )

    # Convert from s⁻¹ to day⁻¹
    SEC_PER_DAY = 86400.0
    eady_dry_day = eady_dry_mean * SEC_PER_DAY
    eady_moist_day = eady_moist_mean * SEC_PER_DAY

    # Compute ratio directly from domain-mean values (not from averaging F)
    # This avoids the inconsistency where averaged F ≠ eady_moist/eady_dry
    eady_ratio = eady_moist_day / eady_dry_day if eady_dry_day > 0 else float("nan")

    logger.info("Eady dry=%.3f day⁻¹, moist=%.3f day⁻¹, F=%.3f",
                eady_dry_day, eady_moist_day, eady_ratio)

    return {
        "eady_dry": eady_dry_day,
        "eady_moist": eady_moist_day,
        "eady_ratio": eady_ratio,
    }
