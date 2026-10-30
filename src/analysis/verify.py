"""
Compare the regenerated surrogate / Sobol outputs (outputs/sensitivity/) with the
reference results shipped in results/sensitivity/, and print the largest deviation
for every quantity reported in the paper.

Exit status 1 if any CV R2 or Sobol index differs by more than --tol.
"""
import argparse
import sys

import pandas as pd

from config import OUTPUTS, PHENOTYPES, RESULTS


def compare(kind, fname, keys, cols, tol):
    ref = pd.read_csv(RESULTS / "sensitivity" / kind / fname)
    new = pd.read_csv(OUTPUTS / "sensitivity" / kind / fname)
    m = ref.merge(new, on=keys, suffixes=("_ref", "_new"), how="outer", indicator=True)
    if (m._merge != "both").any():
        print(f"  {kind} {fname}: row sets differ")
        return False
    ok = True
    for c in cols:
        d = (m[f"{c}_ref"] - m[f"{c}_new"]).abs().max()
        flag = "ok" if d <= tol else "DIFF"
        ok &= d <= tol
        print(f"  {kind:<4}{fname:<16}{c:<10} max |diff| = {d:.2e}  {flag}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    tol = ap.parse_args().tol
    ok = True
    for kind in PHENOTYPES:
        ok &= compare(kind, "cv_summary.csv", ["output"], ["R2_mean", "R2_sd"], tol)
        ok &= compare(kind, "sobol_ind.csv", ["output", "scale", "parameter"],
                      ["S1_raw", "ST_raw", "S1_conf", "ST_conf"], tol)
        ok &= compare(kind, "sobol_grp.csv", ["output", "scale", "group"],
                      ["S1_raw", "ST_raw"], tol)
    print("\nreproduced within tolerance" if ok else "\ndifferences above tolerance")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
