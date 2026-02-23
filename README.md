# CMIP6 Arctic Regime Diagram

Multi-model CMIP6 evidence for the Arctic atmosphere's transition from a dry-linear to a moist-nonlinear regime under climate change. Accompanies the manuscript *"Towards a theoretical understanding of Arctic amplification dynamics"* (Fallah et al.).

## Overview

This repository computes two dimensionless parameters from CMIP6 model output and generates a publication-quality regime diagram (Fig. 7 of the paper):

| Parameter | Formula | Physical meaning |
|-----------|---------|-----------------|
| **R** (Nonlinearity ratio) | R = ⟨\|σ'⟩\| / σ̄ | Amplitude of static stability fluctuations relative to the climatological mean |
| **D** (Diabatic number) | D = Lv·P / (cp·\|∂T/∂y\|·f₀·Ld·H) | Ratio of latent heating to dry baroclinic energy |

Key thresholds: R ≈ 0.1 (nonlinear onset), R > 0.3 (subcritical instability), D ≈ 1 (moist-dominated).

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate arctic_regime

# 2. Run the full pipeline
bash scripts/run_all.sh

# Or run steps individually:
python scripts/01_download_data.py      # Download CMIP6 data from Pangeo
python scripts/02_compute_diagnostics.py # Compute R and D per model
python scripts/03_plot_regime_diagram.py # Generate figure
```

## Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `01_download_data.py` | Pangeo Google Cloud catalog | `data/processed/*.nc` |
| 2 | `02_compute_diagnostics.py` | Processed NetCDF files | `data/diagnostics/regime_diagnostics.csv` |
| 3 | `03_plot_regime_diagram.py` | Diagnostics CSV | `figures/regime_diagram.pdf` |

## CMIP6 Models

13 models with `ta` and `pr` across historical, SSP2-4.5, and SSP5-8.5:

ACCESS-CM2, ACCESS-ESM1-5, CanESM5, CESM2, CNRM-CM6-1, EC-Earth3, GFDL-ESM4, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0, NorESM2-LM, UKESM1-0-LL

## Key Formulas

**Static stability** (Holton 2004):
```
σ(p) = -(Rd·T)/(p·θ) · ∂θ/∂p
```

**Nonlinearity ratio**:
```
R = ⟨|σ'|⟩_time / σ̄     (area-weighted, 60-90°N, 500-850 hPa)
```

**Diabatic number**:
```
D = (Lv · pr_arctic) / (cp · |∂T/∂y|_edge · f₀ · Ld · H)
```
where Ld = N·H/f₀ is computed self-consistently from σ̄.

## Configuration

All tunable parameters are in `config/config.yaml` (physical constants, domain bounds, pressure levels, plotting options) and `config/models.yaml` (CMIP6 model list).

## Tests

```bash
pytest tests/ -v
```

## Repository Structure

```
arctic_regime_diagram/
├── config/           # YAML configuration
├── src/
│   ├── data/         # Catalog search, download, preprocessing
│   ├── physics/      # Static stability, R, D computations
│   └── plotting/     # Nature Comms style, regime diagram
├── scripts/          # Pipeline steps
├── tests/            # Analytical validation tests
├── data/             # Raw/processed/diagnostics (.gitignored)
└── figures/          # Output figures
```

## License

MIT

## Citation

If you use this code, please cite:

> Fallah, B. et al. "Towards a theoretical understanding of Arctic amplification dynamics." *npj Climate and Atmospheric Science* (2026).
