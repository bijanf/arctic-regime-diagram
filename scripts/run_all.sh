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

echo "[1/4] Downloading and preprocessing CMIP6 data..."
python "$SCRIPT_DIR/01_download_data.py"
echo ""

echo "[2/4] Computing CMIP6 R and D diagnostics..."
python "$SCRIPT_DIR/02_compute_diagnostics.py"
echo ""

echo "[3/4] Computing reanalysis diagnostics (ERA5 + NCEP/NCAR R1)..."
python "$SCRIPT_DIR/04_reanalysis_diagnostics.py"
echo ""

echo "[4/4] Generating regime diagram..."
python "$SCRIPT_DIR/03_plot_regime_diagram.py"
echo ""

echo "============================================"
echo " Pipeline complete!"
echo " CMIP6 diagnostics:      $PROJECT_DIR/data/diagnostics/regime_diagnostics.csv"
echo " Reanalysis diagnostics: $PROJECT_DIR/data/diagnostics/reanalysis_diagnostics.csv"
echo " Figure:                 $PROJECT_DIR/figures/regime_diagram.pdf"
echo "============================================"
