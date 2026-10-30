"""
Step 1 - merge the per-case simulation outputs with the design inputs.

  data/simulation/stress_{LAP,CP}.csv   PSS, del_PSS and peak locations
  data/inputs/lhs_design_1000.csv        sampled inputs (row index = case_id)
  data/simulation/FFR.csv                hyperaemic distal FFR (1D model)
  data/simulation/plaque_burden.csv      PB at the MLA cross-section

-> outputs/master_{LAP,CP}.csv, one row per valid case (936 LAP, 879 CP).

Axial class from I_z: proximal (< -1/3), mid / MLA-centred (|I_z| <= 1/3),
distal (> 1/3).  Circumferential class from I_theta: centre (< 0.5),
shoulder (>= 0.5); it is already stored in the stress tables.
"""
import numpy as np
import pandas as pd

from config import DESIGN_CSV, SIM, OUTPUTS, PHENOTYPES

THIRD = 1.0 / 3.0


def axial_class(v):
    return np.where(v < -THIRD, "proximal",
                    np.where(v > THIRD, "distal", "mid"))


def main():
    OUTPUTS.mkdir(exist_ok=True)
    par = pd.read_csv(DESIGN_CSV)
    par["case_id"] = np.arange(len(par))
    ffr = pd.read_csv(SIM / "FFR.csv")
    pb = pd.read_csv(SIM / "plaque_burden.csv")

    for kind in PHENOTYPES:
        s = pd.read_csv(SIM / f"stress_{kind}.csv")
        # the stress table also carries the realised lesion length; keep the
        # design value from the LHS table
        s = s.drop(columns=["lesion_length"])
        s["PSS_axial_loc"] = axial_class(s.PSS_loc_index)
        s["delPSS_axial_loc"] = axial_class(s.delPSS_loc_index)

        m = (s.merge(par, on="case_id", how="left")
              .merge(ffr, on="case_id", how="left")
              .merge(pb, on="case_id", how="left"))
        m["delta_FFR"] = 1.0 - m["FFR"]
        m.to_csv(OUTPUTS / f"master_{kind}.csv", index=False)

        print(f"{kind}: n={len(m)}  FFR non-null {m.FFR.notna().sum()}  "
              f"PB non-null {m['PB(%)'].notna().sum()}")
        for pre in ("PSS", "delPSS"):
            a = m[f"{pre}_axial_loc"].value_counts(normalize=True) * 100
            c = m[f"{pre}_circum_class"].value_counts(normalize=True) * 100
            print(f"  {pre:<7} proximal {a.get('proximal', 0):4.1f}%  "
                  f"mid {a.get('mid', 0):4.1f}%  distal {a.get('distal', 0):4.1f}%  |  "
                  f"centre {c.get('center', 0):4.1f}%  shoulder {c.get('shoulder', 0):4.1f}%")


if __name__ == "__main__":
    main()
