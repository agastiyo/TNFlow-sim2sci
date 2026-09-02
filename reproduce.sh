#!/usr/bin/env bash
set -euo pipefail

# Reproduce TNFlow paper results from a pretrained checkpoint.
# Requires: tnflow_checkpoint.pt in the project root,
#           data/{spectra,manifest}.parquet, data/JWST_DiSCO-TNOs/

echo "=== Evaluating on synthetic test + OOD splits (Table 1) ==="
python3.10 -m src.pipeline.test

echo ""
echo "=== JWST transfer experiment (Appendix C) ==="
python3.10 -m src.pipeline.apply_jwst
