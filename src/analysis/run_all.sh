#!/usr/bin/env bash
# Regenerate every table and figure from data/ into outputs/, then check the
# surrogate / Sobol results against results/.
set -euo pipefail
cd "$(dirname "$0")"
python build_master.py
python gpr_sobol.py --phenotype LAP
python gpr_sobol.py --phenotype CP
python make_tables.py
python fig5_outputs_validation.py
python fig6_consistency.py
python fig7_sobol.py
python figS_input_distributions.py
python verify.py
