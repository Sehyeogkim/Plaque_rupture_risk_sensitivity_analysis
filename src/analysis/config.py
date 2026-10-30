"""
Shared paths, input-parameter definitions and unit conversions.

All quantities in the data files are in CGS units (cm, g, s, dyne/cm^2):
  stress / Young's modulus [dyne/cm^2]   1 kPa = 1e4 dyne/cm^2
  pressure inputs SBP, PP  [mmHg]
  lengths                  [cm]
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"            # reference results reported in the paper
OUTPUTS = ROOT / "outputs"            # everything the analysis scripts regenerate

DESIGN_CSV = DATA / "inputs" / "lhs_design_1000.csv"
SIM = DATA / "simulation"
VALID = DATA / "validation"

DYN_TO_KPA = 1e-4
PHENOTYPES = ("LAP", "CP")
STRESS_OUTPUTS = ("PSS", "del_PSS")

# ---- Sobol input sets ----------------------------------------------------
# fc_av_th is constant (0.01 cm) and DBP = SBP - PP is collinear, so those are
# the only design columns that never enter the surrogate.
MORPH_LAP = ["DOS", "lesion_length", "lumen_axial_skewness",
             "alpha", "lipid_length_ratio", "PI"]
MAT_LAP = ["E_vessel", "E_fc", "E_lipid"]
HEMO = ["SBP", "PP", "tau"]

# CP keeps every calcification parameter (18 inputs).  The four auxiliary
# geometry-generation variables sit in the morphological group, E_cal in the
# material group.
MORPH_CP = MORPH_LAP + ["fraction", "d_fc_ca", "ca_axial_skewness",
                        "ca_shoulder_skewness", "ca_strength_ratio"]
MAT_CP = MAT_LAP + ["E_cal"]

INPUT_SETS = {
    "LAP": (MORPH_LAP, MAT_LAP, HEMO),
    "CP": (MORPH_CP, MAT_CP, HEMO),
}

# column name in the data -> symbol used in the paper
DISPLAY_NAMES = {
    "DOS": r"$DoS$",
    "lesion_length": r"$L_{\mathrm{lesion}}$",
    "lumen_axial_skewness": r"$\gamma_z$",
    "alpha": r"$\theta_{\mathrm{lipid,max}}$",
    "lipid_length_ratio": r"$l_{\mathrm{lipid}}$",
    "PI": r"$PI$",
    "E_vessel": r"$E_{\mathrm{vessel}}$",
    "E_fc": r"$E_{\mathrm{FC}}$",
    "E_lipid": r"$E_{\mathrm{lipid}}$",
    "SBP": r"$P_{\mathrm{sys}}$",
    "PP": r"$\Delta P$",
    "tau": r"$\tau$",
    "fraction": r"$f_{\mathrm{calc}}$",
    "E_cal": r"$E_{\mathrm{calc}}$",
    "d_fc_ca": r"$d_{\mathrm{buf}}$",
    "ca_axial_skewness": r"$s_{\mathrm{ax}}$",
    "ca_shoulder_skewness": r"$s_{\mathrm{circ}}$",
    "ca_strength_ratio": r"$w$",
}


def input_set(phenotype):
    """Return (input_names, group_of_each_input) for a phenotype."""
    morph, mat, hemo = INPUT_SETS[phenotype]
    names = morph + mat + hemo
    groups = (["Morphology"] * len(morph) + ["Material"] * len(mat)
              + ["Hemodynamics"] * len(hemo))
    return names, groups
