# Material-dominated sensitivity of coronary plaque vulnerability to rupture

Code and data accompanying

> S. H. Kim\*, W. Choi\*, H. J. Kim. *Material-dominated sensitivity of coronary plaque
> vulnerability to rupture.* Submitted to *Computers in Biology and Medicine* (2026).
> \*Equal contribution. Department of Mechanical Engineering, KAIST.

The study generates input–output datasets for two idealized coronary plaque phenotypes —
low-attenuation plaque (**LAP**, 12 inputs) and calcified plaque (**CP**, 18 inputs) — with a
snapshot-based structural stress estimator (1D reduced-order hemodynamics → steady 3D
fluid → quasi-static 3D structure). It evaluates two vulnerability indices — peak systolic
plaque structural stress (**PSS**) and its systolic–diastolic change (**ΔPSS**) — against
five high-risk plaque features, and quantifies their drivers with Gaussian-process
surrogates and Sobol global sensitivity analysis.

```
data/            simulation inputs and per-case outputs (everything the analysis needs)
src/analysis/    surrogate + Sobol analysis, tables and figures 5–7   ← reproduces the paper
src/simulation/  forward-model code (CAD, meshing, 1D/3D CFD, ANSYS, post-processing)
results/         reference outputs as reported in the paper
PROVENANCE.md    figure-to-data-and-code map
```

## Reproduce the paper's results

Everything from the per-case simulation outputs onward runs on an ordinary workstation
(Python ≥ 3.10; versions used are pinned in `requirements.txt`).

```bash
pip install -r requirements.txt
cd src/analysis
python build_master.py                  # merge inputs and outputs            (seconds)
python gpr_sobol.py --phenotype LAP     # GPR, 5-fold CV, Sobol indices       (1–2 h)
python gpr_sobol.py --phenotype CP      #                                     (1.5–3 h)
python make_tables.py                   # every number in the Results section
python fig5_outputs_validation.py       # Fig. 5
python fig6_consistency.py              # Fig. 6
python fig7_sobol.py                    # Fig. 7
python figS_input_distributions.py      # input-distribution supplement
python verify.py                        # compare outputs/ with results/
```

`run_all.sh` runs the whole chain. To check the tables and Fig. 7 without re-fitting the
surrogates, use `python make_tables.py --reference` and `python fig7_sobol.py --reference`.
Everything is written to `outputs/` (git-ignored); `results/` is never overwritten.

## Data

All quantities are in CGS units (cm, g, s; stress and moduli in dyne/cm², 1 kPa =
10⁴ dyne/cm²; pressures in mmHg).

| File | Rows | Content |
|---|---|---|
| `data/inputs/lhs_design_1000.csv` | 1000 | Latin-hypercube design used in the analysis; row index = `case_id`. Shared by LAP and CP (LAP ignores the calcification columns). |
| `data/simulation/stress_LAP.csv` | 936 | PSS, ΔPSS (`del_PSS`), coordinates and location indices (`*_loc_index` = I_z, `*_circum_index` = I_θ) of each peak |
| `data/simulation/stress_CP.csv` | 879 | same for CP |
| `data/simulation/FFR.csv` | 1000 | hyperaemic FFR 3 cm distal to the lesion (1D model); ΔFFR = 1 − FFR |
| `data/simulation/plaque_burden.csv` | 1000 | plaque burden at the MLA cross-section [%] |
| `data/simulation/failed_cases_{LAP,CP}.csv` | 64 / 121 | cases without a usable result and the stage at which they failed |
| `data/validation/fc_thickness/` | 3 × 5 | auxiliary fibrous-cap-thickness set (Fig. 5E): inputs `parameterD_level{0,1,2}.csv` (E_FC = 1.2, 1.6, 2.0 MPa), outputs `PSS_result_expD_stiff*.csv` |
| `data/validation/ffr_literature/` | — | FFR–DoS curves digitised from Kristensen et al. 2010, Dai et al. 2021, Nijjer et al. 2013 and Lee et al. 2018 (Fig. 5D) |

Design-file columns and paper symbols:

| column | symbol | unit | | column | symbol | unit |
|---|---|---|---|---|---|---|
| `DOS` | DoS | – (fraction) | | `E_vessel` | E_vessel | dyne/cm² |
| `lesion_length` | L_lesion | cm | | `E_fc` | E_FC | dyne/cm² |
| `lumen_axial_skewness` | γ_z | – | | `E_lipid` | E_lipid | dyne/cm² |
| `alpha` | θ_lipid,max | deg | | `E_cal` | E_calc | dyne/cm² |
| `lipid_length_ratio` | l_lipid | – | | `SBP` | P_sys | mmHg |
| `PI` | PI | – | | `PP` | ΔP | mmHg |
| `fraction` | f_calc | – | | `tau` | τ | s |
| `ca_axial_skewness` | s_ax | – | | `DBP` | P_sys − ΔP | mmHg (derived, not an input) |
| `ca_shoulder_skewness` | s_circ | – | | `fc_av_th` | fibrous-cap thickness | cm (fixed, 0.01) |
| `ca_strength_ratio` | w | – | | `d_fc_ca` | d_buf | cm |

## Forward simulations

`src/simulation/` contains the forward-model code. The CP calcification mesher reads the
bundled `data/inputs/lhs_design_1000.csv` by case index. Other stages retain the
workstation-specific paths used during the campaign. Running the pipeline requires
licensed software (Autodesk Inventor 2026, Simmetrix MeshSim, ANSYS Mechanical APDL
2025 R1) and an in-house 1D solver. See [`src/simulation/README.md`](src/simulation/README.md)
for the script for each stage.

## Provenance

[`PROVENANCE.md`](PROVENANCE.md) maps each data figure to its input files and analysis script.

## License

Code: MIT (`LICENSE`). Data in `data/` and `results/`: CC BY 4.0. The digitised
literature curves in `data/validation/ffr_literature/` remain the work of their original
authors; cite the original publications if you reuse them.

## Citation

See `CITATION.cff`. Correspondence: Hyun Jin Kim (kim.hyunjin@kaist.ac.kr).
