#!/usr/bin/env python3
"""Plot ERA5 10-hPa anomaly figures (z and T) for revision R1.

Two-panel layout:
  (a) Spatial anomaly (recent - historical), Welch t-test hatching where p >= 0.05.
  (b) Polar-cap (60-90N, area-weighted) annual time series with linear trend.

Periods: recent = 1995-2024, historical = 1941-1970.

Inputs:
  paper/data/reanalysis/era5_z10hpa_monthly.nc  (z in m^2/s^2)
  paper/data/reanalysis/era5_t10hpa_monthly.nc  (t in K)

Outputs:
  figures/era5_z10hpa_anomaly.{png,pdf}
  figures/era5_t10hpa_anomaly.{png,pdf}
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "paper" / "data" / "reanalysis"
FIG_DIR = PROJECT_ROOT / "figures"
G0 = 9.80665

HIST = (1941, 1970)
RECENT = (1995, 2024)


def _open_era5(path, varname):
    ds = xr.open_dataset(path)
    rn = {}
    if "valid_time" in ds.dims:
        rn["valid_time"] = "time"
    if "latitude" in ds.dims:
        rn["latitude"] = "lat"
    if "longitude" in ds.dims:
        rn["longitude"] = "lon"
    if rn:
        ds = ds.rename(rn)
    for c in ("number", "expver"):
        if c in ds.coords:
            ds = ds.drop_vars(c)
    var = None
    for v in ds.data_vars:
        if v == varname or v in {"z", "t", "temperature", "geopotential"}:
            var = v
            break
    if var is None:
        var = list(ds.data_vars)[0]
    da = ds[var]
    if "pressure_level" in da.dims:
        da = da.squeeze("pressure_level", drop=True)
    return da.sortby("lat")


def _annual_mean(da):
    return da.groupby("time.year").mean("time")


def _area_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def _polar_cap_series(da_annual, lat_min=60.0, lat_max=90.0):
    sub = da_annual.sel(lat=slice(lat_min, lat_max))
    w = _area_weights(sub["lat"])
    return sub.weighted(w).mean(("lat", "lon"))


def _welch_ttest_map(hist_annual, recent_annual):
    h = hist_annual.values
    r = recent_annual.values
    _, p = stats.ttest_ind(r, h, axis=0, equal_var=False, nan_policy="omit")
    return p


def _linear_trend(years, values):
    res = stats.linregress(years, values)
    r, pval = stats.pearsonr(years, values)
    return res.slope, res.intercept, r, pval


def _plot_panel_a(ax, diff, pval, *, title, cbar_label, cmap, levels, extend):
    lon = diff["lon"].values
    lat = diff["lat"].values
    lon_c = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_c)
    lon_p = lon_c[order]
    data = diff.values[:, order]
    pv = pval[:, order]

    cf = ax.contourf(
        lon_p, lat, data, levels=levels, cmap=cmap, extend=extend,
        transform=ccrs.PlateCarree(),
    )
    ax.contourf(
        lon_p, lat, pv, levels=[0.05, 1.0],
        colors="none", hatches=["////"],
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines(linewidth=0.5, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="gray")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.75, pad=0.05)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _plot_panel_b(ax, years, series, *, title, ylabel, unit_label):
    slope, intercept, r, p = _linear_trend(years, series)
    trend = intercept + slope * years
    ax.plot(years, series, color="steelblue", alpha=0.5, linewidth=0.8, label=None)
    run = np.convolve(series, np.ones(11) / 11, mode="same")
    # mask edges
    run[:5] = np.nan
    run[-5:] = np.nan
    ax.plot(years, run, color="steelblue", linewidth=2.0, label="11-yr running mean")
    ax.plot(years, trend, color="crimson", linestyle="--", linewidth=1.5,
            label=f"Trend: {slope * 10:+.2f}{unit_label}/decade")
    total = slope * (years[-1] - years[0])
    ax.text(
        0.97, 0.97,
        f"Total: {total:+.1f}{unit_label} over {int(years[-1] - years[0])} yrs\n"
        f"r = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    return slope, r, p


def make_figure(var_kind):
    if var_kind == "z":
        path = DATA_DIR / "era5_z10hpa_monthly.nc"
        da = _open_era5(path, "z") / G0  # geopotential -> geopotential height (m)
        panel_a_title = (
            "(a) 10-hPa height change\n"
            f"({RECENT[0]}-{RECENT[1]}) minus ({HIST[0]}-{HIST[1]})\n"
            "Hatching: not significant (p $\\geq$ 0.05, annual means)"
        )
        cbar_label = "Geopotential height anomaly (m)"
        cmap = "Blues_r"
        levels = np.linspace(-140, 20, 17)
        extend = "both"
        ylabel = "Polar-cap mean (60-90$^\\circ$N)\n10-hPa geopotential height (m)"
        unit_label = " m"
        suptitle = "ERA5: Stratospheric geopotential height anomaly at 10 hPa"
        panel_b_title = "ERA5: Stratospheric geopotential height anomaly at 10 hPa"
        stem = "era5_z10hpa_anomaly"
    elif var_kind == "t":
        path = DATA_DIR / "era5_t10hpa_monthly.nc"
        da = _open_era5(path, "t")
        panel_a_title = (
            "(a) 10-hPa temperature change\n"
            f"({RECENT[0]}-{RECENT[1]}) minus ({HIST[0]}-{HIST[1]})\n"
            "Hatching: not significant (p $\\geq$ 0.05, annual means)"
        )
        cbar_label = "Temperature anomaly (K)"
        cmap = "RdBu_r"
        levels = np.linspace(-5, 5, 21)
        extend = "both"
        ylabel = "Polar-cap mean (60-90$^\\circ$N)\n10-hPa temperature (K)"
        unit_label = " K"
        suptitle = "ERA5: Stratospheric temperature anomaly at 10 hPa"
        panel_b_title = "ERA5: Stratospheric temperature anomaly at 10 hPa"
        stem = "era5_t10hpa_anomaly"
    else:
        raise ValueError(var_kind)

    ann = _annual_mean(da).load()
    hist = ann.sel(year=slice(HIST[0], HIST[1]))
    rec = ann.sel(year=slice(RECENT[0], RECENT[1]))
    diff = rec.mean("year") - hist.mean("year")
    pval = _welch_ttest_map(hist, rec)

    pc = _polar_cap_series(ann).values
    years = ann["year"].values.astype(float)

    fig = plt.figure(figsize=(14, 5.5))
    ax_a = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
    ax_b = fig.add_subplot(1, 2, 2)
    _plot_panel_a(
        ax_a, diff, pval,
        title=panel_a_title, cbar_label=cbar_label,
        cmap=cmap, levels=levels, extend=extend,
    )
    slope, r_val, p_val = _plot_panel_b(
        ax_b, years, pc, title=panel_b_title, ylabel=ylabel, unit_label=unit_label,
    )
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[{var_kind}] slope = {slope * 10:+.3f}{unit_label}/decade, r = {r_val:.3f}, p = {p_val:.3e}")
    return slope, r_val, p_val


def main():
    kinds = sys.argv[1:] or ["z", "t"]
    for k in kinds:
        make_figure(k)


if __name__ == "__main__":
    main()
