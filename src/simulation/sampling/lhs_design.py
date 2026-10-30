"""Latin-hypercube design for the 1,000-case study.

The simulations were run on ``data/inputs/lhs_design_1000.csv``. This script
draws a new table using the truncated log-normal w distribution of that design.
It does not reproduce the campaign rows or overwrite the bundled CSV.

Independent samples are concatenated by row:

  morphology     uniform Latin hypercube. Fibrous-cap thickness is fixed
                 at 0.01 cm.
  w              calcification strength ratio, its own one-dimensional
                 Latin hypercube mapped to a log-normal truncated to
                 [0.1, 10] (shape 1, scale 1)
  material       uniform Young's moduli, stored as integers [dyne/cm^2]
  pressure       uniform SBP and pulse pressure, rejecting draws with
                 DBP = SBP - PP <= 40 mmHg. DBP is stored but is not an input.
  tau            uniform on [0.5, 2.0] s

Units match the rest of the repository: lengths in cm, moduli in dyne/cm^2,
pressures in mmHg.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import lognorm, qmc

N_SAMPLES = 1000
FC_THICKNESS_CM = 0.01
DBP_MIN_MMHG = 40.0

# name -> (lower, upper, decimal places). None means leave the float as drawn.
MORPHOLOGY = {
    "DOS": (0.25, 0.75, 4),
    "lesion_length": (1.0, 3.0, 4),
    "lumen_axial_skewness": (-0.5, 0.5, 4),
    "lipid_length_ratio": (0.5, 0.8, 4),
    "alpha": (120.0, 270.0, 1),
    "PI": (0.9, 1.2, 3),
    "d_fc_ca": (0.005, 0.02, 4),
    "fraction": (0.1, 0.8, 4),
    "ca_axial_skewness": (0.0, 0.8, 4),
    "ca_shoulder_skewness": (0.0, 0.8, 4),
}
STRENGTH_RATIO = ("ca_strength_ratio", 0.1, 10.0, 4)  # truncated log-normal
LOGNORM_S = 1.0
LOGNORM_SCALE = 1.0

MATERIAL = {
    "E_vessel": (1.0e6, 1.4e7),    # 100-1400 kPa
    "E_fc": (4.0e6, 2.3e7),        # 400-2300 kPa
    "E_lipid": (1.0e4, 1.0e6),     # 1-100 kPa
    "E_cal": (0.7e10, 25e10),      # 0.7-25 GPa
}

PRESSURE = {
    "SBP": (100.0, 170.0, 2),
    "PP": (35.0, 80.0, 2),
}
TAU = ("tau", 0.5, 2.0)

COLUMNS = [
    *MORPHOLOGY,
    STRENGTH_RATIO[0],
    "fc_av_th",
    *MATERIAL,
    *PRESSURE,
    TAU[0],
    "DBP",
]


def _uniform_lhs(spec, n, seed):
    names = list(spec)
    lower = np.array([spec[k][0] for k in names])
    upper = np.array([spec[k][1] for k in names])
    sample = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    scaled = qmc.scale(sample, lower, upper)
    df = pd.DataFrame(scaled, columns=names)
    for name in names:
        digits = spec[name][2] if len(spec[name]) > 2 else None
        if digits is None:
            continue
        df[name] = df[name].round(digits)
        if digits == 0:
            df[name] = df[name].astype(np.int64)
    return df


def _strength_ratio(n, seed):
    name, lo, hi, digits = STRENGTH_RATIO
    sample = qmc.LatinHypercube(d=1, seed=seed).random(n)[:, 0]
    q_lo = lognorm.cdf(lo, s=LOGNORM_S, scale=LOGNORM_SCALE)
    q_hi = lognorm.cdf(hi, s=LOGNORM_S, scale=LOGNORM_SCALE)
    values = lognorm.ppf(q_lo + sample * (q_hi - q_lo), s=LOGNORM_S, scale=LOGNORM_SCALE)
    return pd.Series(np.round(values, digits), name=name)


def _pressures(n, seed):
    """SBP and PP, refilled until every row has DBP > 40 mmHg."""
    names = list(PRESSURE)
    collected = []
    remaining = n
    current_seed = seed
    while remaining > 0:
        block = _uniform_lhs(PRESSURE, remaining, current_seed)
        dbp = block["SBP"] - block["PP"]
        keep = block.loc[dbp > DBP_MIN_MMHG].copy()
        keep["DBP"] = dbp.loc[keep.index].round(PRESSURE["SBP"][2])
        collected.append(keep)
        remaining -= len(keep)
        current_seed += 1
    out = pd.concat(collected, ignore_index=True).head(n)
    return out[names + ["DBP"]]


def sample_design(n=N_SAMPLES, seed=42):
    """Return one design table. ``seed`` starts four independent draws."""
    if n <= 0:
        raise ValueError("n must be > 0")
    morph = _uniform_lhs(MORPHOLOGY, n, seed)
    strength = _strength_ratio(n, seed + 1)
    material = _uniform_lhs(
        {k: (*bounds, 0) for k, bounds in MATERIAL.items()}, n, seed + 2
    )
    pressure = _pressures(n, seed + 3)
    tau_name, tau_lo, tau_hi = TAU
    tau = _uniform_lhs({tau_name: (tau_lo, tau_hi)}, n, seed + 4)
    design = pd.concat(
        [morph, strength, material, pressure.drop(columns="DBP"), tau], axis=1
    )
    design.insert(design.columns.get_loc("E_vessel"), "fc_av_th", FC_THICKNESS_CM)
    design["DBP"] = pressure["DBP"].to_numpy()
    return design[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="lhs_design.csv",
        help="output path (the bundled data/inputs file is never the default)",
    )
    args = ap.parse_args()
    design = sample_design(args.n, args.seed)
    design.to_csv(args.out, index=False)
    print(f"Saved {args.out}  ({len(design)} rows x {design.shape[1]} columns)")


if __name__ == "__main__":
    main()
