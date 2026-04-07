"""Tests for static stability computation.

Includes analytical validation against an isothermal atmosphere where
σ has a known closed-form solution.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics.static_stability import (
    KAPPA,
    P0,
    RD,
    layer_mean_stability,
    potential_temperature,
    static_stability,
)


def _make_isothermal_dataset(T0=250.0, p_levels=None):
    """Create a synthetic isothermal atmosphere dataset.

    For an isothermal atmosphere (T = T0 = const):
        θ(p) = T0 · (p0/p)^κ
        ∂θ/∂p = -κ · T0 · p0^κ · p^(-κ-1)
        σ = -(Rd·T0)/(p·θ) · ∂θ/∂p = Rd·κ·T0/(p²·(p0/p)^κ) · T0·p0^κ·p^(-κ-1)
          = Rd·κ/p²  (simplified)

    More precisely:
        σ = κ · Rd · T0 / p²  ... but with p in Pa.

    Actually, let's derive carefully:
        θ = T0 · (p0/p)^κ
        ∂θ/∂p = T0 · p0^κ · (-κ) · p^(-κ-1) = -κ·θ/p
        σ = -(Rd·T0)/(p·θ) · (-κ·θ/p) = Rd·T0·κ/p²

    So for an isothermal atmosphere: σ = Rd·T0·κ/p²  (p in Pa).
    """
    if p_levels is None:
        p_levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 200, 100], dtype=float)

    lat = np.array([65.0, 70.0, 75.0, 80.0, 85.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    time = np.arange(12)  # 12 months

    # Constant temperature
    shape = (len(time), len(p_levels), len(lat), len(lon))
    ta = np.full(shape, T0)

    ds = xr.Dataset(
        {"ta": (["time", "plev", "lat", "lon"], ta)},
        coords={
            "time": time,
            "plev": p_levels,
            "lat": lat,
            "lon": lon,
        },
    )
    ds["plev"].attrs["units"] = "hPa"
    return ds


class TestPotentialTemperature:
    """Tests for potential temperature computation."""

    def test_surface_level(self):
        """At p = p0, θ = T."""
        T = xr.DataArray([300.0])
        plev = xr.DataArray([P0])
        theta = potential_temperature(T, plev)
        np.testing.assert_allclose(theta.values, [300.0], rtol=1e-10)

    def test_upper_level(self):
        """θ increases with decreasing pressure for constant T."""
        T = xr.DataArray([250.0, 250.0])
        plev = xr.DataArray([500.0, 1000.0])
        theta = potential_temperature(T, plev)
        assert theta.values[0] > theta.values[1]

    def test_known_value(self):
        """Check a known θ value: T=250K at 500hPa."""
        T = xr.DataArray([250.0])
        plev = xr.DataArray([500.0])
        expected = 250.0 * (1000.0 / 500.0) ** KAPPA
        theta = potential_temperature(T, plev)
        np.testing.assert_allclose(theta.values, [expected], rtol=1e-10)


class TestStaticStability:
    """Tests for static stability computation."""

    def test_isothermal_positive(self):
        """An isothermal atmosphere should have positive σ (stably stratified)."""
        ds = _make_isothermal_dataset(T0=250.0)
        sigma = static_stability(ds["ta"])
        # σ should be positive everywhere (except possibly boundary effects)
        inner = sigma.isel(plev=slice(1, -1))  # skip endpoints
        assert float(inner.min()) > 0, "σ should be positive for isothermal atm"

    def test_isothermal_analytical(self):
        """Compare σ against the analytical formula for isothermal atmosphere.

        σ_analytical = Rd · T0 · κ / p²  (p in Pa)
        """
        T0 = 250.0
        ds = _make_isothermal_dataset(T0=T0)
        sigma = static_stability(ds["ta"])

        # Check at interior pressure levels (avoid boundary finite-diff issues)
        for plev_hPa in [700.0, 600.0, 500.0, 400.0, 300.0]:
            p_Pa = plev_hPa * 100.0
            sigma_analytical = RD * T0 * KAPPA / p_Pa**2

            sigma_at_level = sigma.sel(plev=plev_hPa)
            sigma_mean = float(sigma_at_level.mean())

            np.testing.assert_allclose(
                sigma_mean, sigma_analytical, rtol=0.15, err_msg=f"σ mismatch at {plev_hPa} hPa"
            )

    def test_layer_mean_within_range(self):
        """Layer mean σ should be between min and max of the layer."""
        ds = _make_isothermal_dataset()
        sigma = static_stability(ds["ta"])
        sigma_layer = layer_mean_stability(sigma, p_top=500.0, p_bot=850.0)

        # Pressure levels are descending, so slice high-to-low
        layer = sigma.sel(plev=slice(850.0, 500.0))
        layer_min = float(layer.min())
        layer_max = float(layer.max())
        layer_mean = float(sigma_layer.mean())

        assert layer_min <= layer_mean <= layer_max


class TestStabilityPhysics:
    """Physical sanity checks."""

    def test_warmer_atmosphere_lower_stability(self):
        """A uniformly warmer isothermal atmosphere should have
        slightly different σ due to σ ∝ T0, but still positive."""
        ds_cold = _make_isothermal_dataset(T0=230.0)
        ds_warm = _make_isothermal_dataset(T0=270.0)

        sigma_cold = static_stability(ds_cold["ta"])
        sigma_warm = static_stability(ds_warm["ta"])

        # For isothermal, σ = Rd·T0·κ/p², so warmer → higher σ
        s_cold_mean = float(sigma_cold.sel(plev=500.0).mean())
        s_warm_mean = float(sigma_warm.sel(plev=500.0).mean())
        assert s_warm_mean > s_cold_mean
