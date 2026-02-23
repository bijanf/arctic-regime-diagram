"""Compute the nonlinearity ratio R = <|σ'|> / σ̄.

R quantifies the amplitude of static stability fluctuations relative to
the climatological mean. Key thresholds from the paper:
  - R ≈ 0.1 : onset of nonlinear regime
  - R > 0.3 : subcritical instability regime
"""

import numpy as np
import xarray as xr

from ..data.preprocess import area_weights, _get_lat_name


def climatological_mean(sigma: xr.DataArray,
                        time_dim: str = "time") -> xr.DataArray:
    """Compute the time-mean static stability σ̄.

    Parameters
    ----------
    sigma : xr.DataArray
        Layer-averaged static stability (time, lat, lon).
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    xr.DataArray
        Climatological mean σ̄.
    """
    return sigma.mean(dim=time_dim)


def monthly_anomalies(sigma: xr.DataArray,
                      time_dim: str = "time") -> xr.DataArray:
    """Compute monthly anomalies σ' = σ - σ̄.

    Parameters
    ----------
    sigma : xr.DataArray
        Layer-averaged static stability.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    xr.DataArray
        Anomalies σ'.
    """
    sigma_bar = climatological_mean(sigma, time_dim=time_dim)
    return sigma - sigma_bar


def nonlinearity_ratio(sigma: xr.DataArray,
                       time_dim: str = "time") -> float:
    """Compute the nonlinearity ratio R = <|σ'|>_time / σ̄.

    Both the numerator and denominator are area-weighted spatial means
    over the Arctic domain (assumed to be the full lat extent of the input).

    Parameters
    ----------
    sigma : xr.DataArray
        Layer-averaged static stability on Arctic domain (time, lat, lon).
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    float
        Nonlinearity ratio R (dimensionless, positive).
    """
    sigma_bar = climatological_mean(sigma, time_dim=time_dim)
    sigma_prime = sigma - sigma_bar

    # Time-mean of absolute anomalies
    abs_sigma_prime_mean = np.abs(sigma_prime).mean(dim=time_dim)

    # Area-weighted spatial mean
    ds_temp = sigma.to_dataset(name="sigma")
    weights = area_weights(ds_temp)
    lat_name = _get_lat_name(ds_temp)

    numerator = (abs_sigma_prime_mean * weights).sum(dim=lat_name).mean()
    denominator = (sigma_bar * weights).sum(dim=lat_name).mean()

    R = float(numerator / denominator)
    return R


def nonlinearity_ratio_with_uncertainty(
    sigma: xr.DataArray, time_dim: str = "time", n_bootstrap: int = 1000
) -> tuple[float, float]:
    """Compute R with bootstrap uncertainty estimate.

    Parameters
    ----------
    sigma : xr.DataArray
        Layer-averaged static stability on Arctic domain.
    time_dim : str
        Time dimension name.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    R_mean : float
        Mean nonlinearity ratio.
    R_std : float
        Standard deviation from bootstrap.
    """
    R_mean = nonlinearity_ratio(sigma, time_dim=time_dim)

    n_time = sigma.sizes[time_dim]
    R_boot = np.zeros(n_bootstrap)

    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        idx = rng.choice(n_time, size=n_time, replace=True)
        sigma_boot = sigma.isel({time_dim: idx})
        R_boot[i] = nonlinearity_ratio(sigma_boot, time_dim=time_dim)

    return R_mean, float(np.std(R_boot))
