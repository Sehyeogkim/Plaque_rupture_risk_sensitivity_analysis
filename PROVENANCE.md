# Figure and data map

| Manuscript item | Input data | Analysis script |
|---|---|---|
| Fig. 5A–C: stress and peak locations | `data/simulation/stress_LAP.csv` | `src/analysis/build_master.py`, `fig5_outputs_validation.py` |
| Fig. 5D: FFR versus stenosis | `data/simulation/FFR.csv`, `data/inputs/lhs_design_1000.csv`, `data/validation/ffr_literature/` | `src/analysis/fig5_outputs_validation.py` |
| Fig. 5E: fibrous-cap thickness | `data/validation/fc_thickness/` | `src/analysis/fig5_outputs_validation.py` |
| Fig. 6: directional consistency | `data/simulation/stress_{LAP,CP}.csv`, `FFR.csv`, `plaque_burden.csv`, `data/inputs/lhs_design_1000.csv` | `src/analysis/build_master.py`, `fig6_consistency.py` |
| Fig. 7: Sobol indices | `results/sensitivity/{LAP,CP}/sobol_{ind,grp}.csv` | `src/analysis/gpr_sobol.py`, `fig7_sobol.py` |
| Supplementary input distributions | `data/inputs/lhs_design_1000.csv` | `src/analysis/figS_input_distributions.py` |

`src/simulation/` contains the forward-model code. The per-case stress, FFR, and plaque-burden CSV files in `data/simulation/` are the starting point for reproducing the analysis figures.
