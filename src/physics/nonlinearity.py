"""Compute the nonlinearity ratio R = <|σ'|> / σ̄.

R quantifies the amplitude of static stability fluctuations relative to
the climatological mean. It captures both temporal variability and spatial
heterogeneity of σ across the Arctic domain.

Key thresholds from the paper:
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
    """Compute the nonlinearity ratio R including spatio-temporal variability.

    R = sqrt(<σ'²>_space,time) / <σ̄>_space

    where σ' = σ(t,x) - <σ̄>_space (departure from the domain-mean climatology)
    and averages are area-weighted over the Arctic domain.

    This captures both temporal variability AND spatial heterogeneity,
    consistent with the PV-stability framework where local departures
    from the background state drive nonlinear dynamics.

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
    ds_temp = sigma.to_dataset(name="sigma")
    weights = area_weights(ds_temp)
    lat_name = _get_lat_name(ds_temp)

    # Domain-mean climatology: single scalar
    sigma_clim = climatological_mean(sigma, time_dim=time_dim)
    sigma_bar_domain = float((sigma_clim * weights).sum(dim=lat_name).mean())

    # Departures from domain-mean climatology (captures space + time variability)
    sigma_prime = sigma - sigma_bar_domain

    # RMS of departures (area-weighted, then time-averaged)
    sigma_prime_sq = sigma_prime ** 2
    # Area-weighted spatial mean of σ'² at each timestep
    variance_t = (sigma_prime_sq * weights).sum(dim=lat_name).mean(dim="lon")
    # Time-mean variance
    mean_variance = float(variance_t.mean(dim=time_dim))
    rms_sigma_prime = np.sqrt(mean_variance)

    R = rms_sigma_prime / abs(sigma_bar_domain)
    return R


def nonlinearity_ratio_seasonal(
    sigma: xr.DataArray, season: str = "DJF", time_dim: str = "time"
) -> float:
    """Compute R for a specific season (e.g., DJF for winter).

    Parameters
    ----------
    sigma : xr.DataArray
        Layer-averaged static stability.
    season : str
        Season code: 'DJF', 'MAM', 'JJA', 'SON'.
    time_dim : str
        Time dimension name.

    Returns
    -------
    float
        Seasonal nonlinearity ratio R.
    """
    # Select season
    sigma_season = sigma.sel(
        {time_dim: sigma[time_dim].dt.season == season}
    )
    if sigma_season.sizes[time_dim] == 0:
        return float("nan")
    return nonlinearity_ratio(sigma_season, time_dim=time_dim)


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
