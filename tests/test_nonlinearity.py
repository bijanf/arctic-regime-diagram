"""Tests for the nonlinearity ratio computation."""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics.nonlinearity import (
    climatological_mean,
    monthly_anomalies,
    nonlinearity_ratio,
)


def _make_sigma_dataset(n_time=120, n_lat=5, n_lon=4, mean_val=1e-5, anomaly_frac=0.1):
    """Create synthetic static stability data with known R.

    Parameters
    ----------
    mean_val : float
        Climatological mean σ̄.
    anomaly_frac : float
        σ' amplitudes will be drawn so that R ≈ anomaly_frac.
    """
    lat = np.linspace(65, 85, n_lat)
    lon = np.linspace(0, 270, n_lon)
    time = np.arange(n_time)

    rng = np.random.default_rng(42)

    # Create σ with controlled anomalies
    # σ = σ̄ + σ', where |σ'| ~ anomaly_frac * σ̄
    sigma_bar = mean_val
    sigma_prime_std = anomaly_frac * sigma_bar  # std of anomalies

    shape = (n_time, n_lat, n_lon)
    sigma_vals = sigma_bar + rng.normal(0, sigma_prime_std, size=shape)

    ds = xr.Dataset(
        {"sigma": (["time", "lat", "lon"], sigma_vals)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds["sigma"]


class TestClimatologicalMean:
    def test_mean_shape(self):
        """Time mean should drop the time dimension."""
        sigma = _make_sigma_dataset()
        sigma_bar = climatological_mean(sigma)
        assert "time" not in sigma_bar.dims

    def test_mean_value(self):
        """Mean should be close to the prescribed value."""
        sigma = _make_sigma_dataset(n_time=10000, mean_val=1e-5)
        sigma_bar = climatological_mean(sigma)
        np.testing.assert_allclose(float(sigma_bar.mean()), 1e-5, rtol=0.05)


class TestMonthlyAnomalies:
    def test_anomalies_zero_mean(self):
        """Anomalies should have near-zero time mean."""
        sigma = _make_sigma_dataset(n_time=10000)
        prime = monthly_anomalies(sigma)
        mean_prime = float(prime.mean())
        assert abs(mean_prime) < 1e-8


class TestNonlinearityRatio:
    def test_known_R(self):
        """R should approximately match the prescribed anomaly fraction."""
        target_R = 0.15
        sigma = _make_sigma_dataset(n_time=5000, anomaly_frac=target_R)
        R = nonlinearity_ratio(sigma)

        # For Gaussian anomalies, <|σ'|> ≈ √(2/π) · std(σ')
        # So R ≈ √(2/π) · anomaly_frac ≈ 0.798 · anomaly_frac
        expected_R = np.sqrt(2 / np.pi) * target_R
        np.testing.assert_allclose(R, expected_R, rtol=0.1)

    def test_zero_anomalies(self):
        """Constant σ should give R = 0."""
        lat = np.linspace(65, 85, 5)
        lon = np.linspace(0, 270, 4)
        time = np.arange(100)

        sigma_vals = np.full((100, 5, 4), 1e-5)
        sigma = xr.DataArray(
            sigma_vals,
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": lat, "lon": lon},
        )
        R = nonlinearity_ratio(sigma)
        assert R < 1e-10

    def test_R_positive(self):
        """R should always be non-negative."""
        sigma = _make_sigma_dataset()
        R = nonlinearity_ratio(sigma)
        assert R >= 0

    def test_R_increases_with_anomaly(self):
        """Larger anomalies should produce larger R."""
        R_small = nonlinearity_ratio(_make_sigma_dataset(anomaly_frac=0.05, n_time=2000))
        R_large = nonlinearity_ratio(_make_sigma_dataset(anomaly_frac=0.30, n_time=2000))
        assert R_large > R_small
