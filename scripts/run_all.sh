#!/usr/bin/env bash
# Run the full Arctic regime diagram pipeline.
#
# Usage:
#   bash scripts/run_all.sh
#
# Prerequisites:
#   conda activate arctic_regime
#   (or equivalent environment with dependencies installed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Arctic Regime Diagram Pipeline"
echo "============================================"
echo ""

echo "[1/7] Downloading and preprocessing CMIP6 data (ta, pr, ua)..."
python "$SCRIPT_DIR/01_download_data.py"
echo ""

echo "[2/7] Regridding all data to common 2-degree grid..."
python "$SCRIPT_DIR/01c_regrid.py"
echo ""

echo "[3/7] Computing CMIP6 diagnostics (R, D, Eady, Ld, K_s²)..."
python "$SCRIPT_DIR/02_compute_diagnostics.py"
echo ""

echo "[4/7] Computing reanalysis diagnostics (ERA5 + NCEP/NCAR R1)..."
python "$SCRIPT_DIR/04_reanalysis_diagnostics.py"
echo ""

echo "[5/7] Generating regime diagram..."
python "$SCRIPT_DIR/03_plot_regime_diagram.py"
echo ""

echo "[6/7] Generating Eady growth rate figure..."
python "$SCRIPT_DIR/05_plot_eady.py"
echo ""

echo "[7/7] Generating wave diagnostics figure..."
python "$SCRIPT_DIR/06_plot_wave_diagnostics.py"
echo ""

echo "============================================"
echo " Pipeline complete!"
echo " CMIP6 diagnostics:      $PROJECT_DIR/data/diagnostics/regime_diagnostics.csv"
echo " Reanalysis diagnostics: $PROJECT_DIR/data/diagnostics/reanalysis_diagnostics.csv"
echo " Figures:"
echo "   Regime diagram:       $PROJECT_DIR/figures/regime_diagram.pdf"
echo "   Eady growth rate:     $PROJECT_DIR/figures/eady_growth_rate.pdf"
echo "   Wave diagnostics:     $PROJECT_DIR/figures/wave_diagnostics.pdf"
echo "============================================"
