"""Compute static stability from temperature on pressure levels.

Static stability parameter (Holton 2004):
    σ(p) = -(Rd·T) / (p·θ) · (∂θ/∂p)

which, using θ = T·(p₀/p)^κ, simplifies to:
    σ(p) = -(T/θ)·(Rd/p)·(∂θ/∂p)
"""

import numpy as np
import xarray as xr

from ..data.preprocess import _get_plev_name

# Default physical constants
RD = 287.05  # J/(kg·K)
CP = 1004.0  # J/(kg·K)
KAPPA = RD / CP  # ~0.2854
P0 = 1000.0  # hPa reference pressure


def potential_temperature(
    T: xr.DataArray, plev_hPa: xr.DataArray, p0: float = P0, kappa: float = KAPPA
) -> xr.DataArray:
    """Compute potential temperature θ = T·(p₀/p)^κ.

    Parameters
    ----------
    T : xr.DataArray
        Air temperature (K) on pressure levels.
    plev_hPa : xr.DataArray
        Pressure levels in hPa.
    p0 : float
        Reference pressure in hPa.
    kappa : float
        Rd / cp.

    Returns
    -------
    xr.DataArray
        Potential temperature (K).
    """
    theta = T * (p0 / plev_hPa) ** kappa
    theta.name = "theta"
    theta.attrs["units"] = "K"
    theta.attrs["long_name"] = "Potential temperature"
    return theta


def static_stability(
    T: xr.DataArray,
    plev_name: str | None = None,
    Rd: float = RD,
    kappa: float = KAPPA,
    p0: float = P0,
) -> xr.DataArray:
    """Compute static stability σ(p).

    σ = -(Rd·T) / (p·θ) · (∂θ/∂p)

    where pressure is in Pa for the derivative and ∂θ/∂p is computed
    via centred finite differences.

    Parameters
    ----------
    T : xr.DataArray
        Temperature (K) with a pressure-level dimension.
    plev_name : str, optional
        Name of the pressure coordinate. Auto-detected if None.
    Rd : float
        Gas constant for dry air (J/(kg·K)).
    kappa : float
        Rd / cp.
    p0 : float
        Reference pressure (hPa).

    Returns
    -------
    xr.DataArray
        Static stability parameter σ (s² m⁻² or equivalently K m⁻¹ scaled).
        Positive values indicate stable stratification.
    """
    ds_temp = T.to_dataset(name="ta") if isinstance(T, xr.DataArray) else T
    if plev_name is None:
        plev_name = _get_plev_name(ds_temp)

    plev = T[plev_name]

    # Ensure pressure is in Pa for differentiation
    plev_pa = plev.copy()
    if float(plev.max()) < 10000:
        # Pressure in hPa, convert to Pa
        plev_pa = plev * 100.0

    # Compute θ
    plev_hPa = plev_pa / 100.0
    theta = potential_temperature(T, plev_hPa, p0=p0, kappa=kappa)

    # ∂θ/∂p in K/Pa
    dtheta_dp = theta.differentiate(plev_name)
    # If plev was in hPa, differentiate gives K/hPa; convert to K/Pa
    if float(plev.max()) < 10000:
        dtheta_dp = dtheta_dp / 100.0

    # σ = -(Rd·T) / (p·θ) · (∂θ/∂p)
    sigma = -(Rd * T) / (plev_pa * theta) * dtheta_dp
    sigma.name = "sigma"
    sigma.attrs["units"] = "m^2 s^-2 Pa^-2"
    sigma.attrs["long_name"] = "Static stability parameter"
    return sigma


def layer_mean_stability(
    sigma: xr.DataArray, p_top: float = 500.0, p_bot: float = 850.0
) -> xr.DataArray:
    """Average static stability over a pressure layer using dp-weighting.

    Uses pressure-thickness (dp) weights instead of a simple mean, which
    properly accounts for the varying spacing of pressure levels.

    Parameters
    ----------
    sigma : xr.DataArray
        Static stability on pressure levels.
    p_top, p_bot : float
        Top and bottom of averaging layer (hPa).

    Returns
    -------
    xr.DataArray
        Pressure-layer averaged σ.
    """
    ds_sigma = sigma.to_dataset(name="sigma")
    plev_name = _get_plev_name(ds_sigma)
    plev_vals = sigma[plev_name].values

    # Determine units
    if np.nanmax(plev_vals) > 10000:
        # Pa
        p_top_u = p_top * 100.0
        p_bot_u = p_bot * 100.0
    else:
        p_top_u = p_top
        p_bot_u = p_bot

    p_lo = min(p_top_u, p_bot_u)
    p_hi = max(p_top_u, p_bot_u)

    if plev_vals[0] > plev_vals[-1]:
        layer = sigma.sel({plev_name: slice(p_hi, p_lo)})
    else:
        layer = sigma.sel({plev_name: slice(p_lo, p_hi)})

    # dp-weighted average: use pressure intervals as weights
    layer_plev = layer[plev_name].values
    if len(layer_plev) < 2:
        return layer.mean(dim=plev_name)

    dp = np.abs(np.gradient(layer_plev))
    dp_weights = xr.DataArray(dp, dims=[plev_name], coords={plev_name: layer_plev})
    dp_weights = dp_weights / dp_weights.sum()

    return (layer * dp_weights).sum(dim=plev_name)
