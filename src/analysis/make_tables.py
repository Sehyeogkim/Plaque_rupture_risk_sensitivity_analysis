"""
Step 3 - every number quoted in the Results section.

Reads outputs/master_*.csv and outputs/sensitivity/*  (or, with --reference,
the shipped results/sensitivity/* so the tables can be checked without
re-running the surrogate).

-> outputs/tables/
  T1_magnitude.csv        PSS / del_PSS median, IQR, range [kPa]
  T2_location.csv         circumferential / axial peak-location fractions
  T3_consistency.csv      Spearman rho vs the five high-risk features
  T4_surrogate.csv        5-fold CV (log scale) + pooled physical-scale accuracy
  T5_sobol_individual.csv S1 / ST, physical ("orig") scale
  T6_sobol_group.csv      group S1 / ST, physical scale
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from config import DYN_TO_KPA, OUTPUTS, PHENOTYPES, RESULTS, STRESS_OUTPUTS

SCALE = "orig"
# (column, label, cohort, expected sign)  - Table 2 of the paper
FEATURES = [("PB(%)", "PB", "LAP", +1),
            ("delta_FFR", "delta_FFR", "LAP", +1),
            ("E_lipid", "E_lipid", "LAP", -1),
            ("alpha", "theta_lipid_max", "LAP", +1),
            ("fraction", "f_calc", "CP", -1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true",
                    help="read surrogate/Sobol results from results/ instead of outputs/")
    args = ap.parse_args()
    sens_root = (RESULTS if args.reference else OUTPUTS) / "sensitivity"
    out = OUTPUTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    master = {k: pd.read_csv(OUTPUTS / f"master_{k}.csv") for k in PHENOTYPES}
    pd.set_option("display.width", 200)

    # ---- T1 magnitude ------------------------------------------------------
    rows = []
    for k, m in master.items():
        r = {"model": k, "n": len(m)}
        for col, lab in (("PSS", "PSS"), ("del_PSS", "dPSS")):
            v = m[col] * DYN_TO_KPA
            r.update({f"{lab}_min": v.min(), f"{lab}_median": v.median(),
                      f"{lab}_IQR_lo": v.quantile(.25), f"{lab}_IQR_hi": v.quantile(.75),
                      f"{lab}_max": v.max()})
        rows.append(r)
    T1 = pd.DataFrame(rows).round(2)

    # ---- T2 location -------------------------------------------------------
    rows = []
    for k, m in master.items():
        for pre, lab in (("PSS", "PSS"), ("delPSS", "dPSS")):
            c = m[f"{pre}_circum_class"].value_counts(normalize=True) * 100
            a = m[f"{pre}_axial_loc"].value_counts(normalize=True) * 100
            rows.append({"model": k, "metric": lab, "n": len(m),
                         "shoulder_pct": c.get("shoulder", 0), "center_pct": c.get("center", 0),
                         "proximal_pct": a.get("proximal", 0), "mid_pct": a.get("mid", 0),
                         "distal_pct": a.get("distal", 0)})
    T2 = pd.DataFrame(rows).round(2)

    # ---- T3 directional consistency -----------------------------------------
    rows = []
    for col, lab, cohort, sign in FEATURES:
        d = master[cohort]
        for vi in STRESS_OUTPUTS:
            s = d[[col, vi]].dropna()
            rho, p = stats.spearmanr(s[col], s[vi])
            rows.append({"feature": lab, "cohort": cohort, "index": vi, "n": len(s),
                         "expected_sign": sign, "rho": rho, "p": p,
                         "concordant": bool(np.sign(rho) == sign)})
    T3 = pd.DataFrame(rows)
    # DoS drives PB and delta_FFR (Discussion)
    lap = master["LAP"]
    extra = {f"rho(DoS,{c})": stats.spearmanr(lap.DOS, lap[c])[0]
             for c in ("PB(%)", "delta_FFR", "PSS", "del_PSS")}

    # ---- T4 surrogate ------------------------------------------------------
    rows = []
    for k in PHENOTYPES:
        cv = pd.read_csv(sens_root / k / "cv_summary.csv")
        oof = pd.read_csv(sens_root / k / "oof_predictions.csv")
        for _, c in cv.iterrows():
            o = oof[(oof.output == c.output) & oof.y_pred.notna()]
            yt, yp = o.y_true.to_numpy(), o.y_pred.to_numpy()
            ape = np.abs(yp - yt) / yt * 100
            rows.append({"model": k, "output": c.output, "n": len(o),
                         "R2_log_mean": c.R2_mean, "R2_log_sd": c.R2_sd,
                         "RMSE_log_mean": c.RMSE_mean, "RMSE_log_sd": c.RMSE_sd,
                         "R2_orig_pooled": r2_score(yt, yp),
                         "median_abs_rel_err_pct": np.median(ape)})
    T4 = pd.DataFrame(rows).round(4)

    # ---- T5 / T6 Sobol -----------------------------------------------------
    ind = pd.concat([pd.read_csv(sens_root / k / "sobol_ind.csv").assign(model=k)
                     for k in PHENOTYPES])
    grp = pd.concat([pd.read_csv(sens_root / k / "sobol_grp.csv").assign(model=k)
                     for k in PHENOTYPES])
    ind, grp = ind[ind.scale == SCALE], grp[grp.scale == SCALE]
    cols = ["model", "output", "parameter", "group", "S1_raw", "S1_conf", "ST_raw", "ST_conf",
            "S1_ci_includes_zero", "ST_ci_includes_zero"]
    T5 = ind[cols].sort_values(["model", "output", "S1_raw"], ascending=[True, True, False])
    T6 = grp[["model", "output", "group", "S1_raw", "S1_conf", "ST_raw", "ST_conf"]]

    for name, t in (("T1_magnitude", T1), ("T2_location", T2), ("T3_consistency", T3),
                    ("T4_surrogate", T4), ("T5_sobol_individual", T5),
                    ("T6_sobol_group", T6)):
        t.to_csv(out / f"{name}.csv", index=False)

    print("== T1 magnitude [kPa]\n", T1.to_string(index=False))
    print("\n== T2 peak location [%]\n", T2.to_string(index=False))
    print("\n== T3 directional consistency\n",
          T3[["feature", "cohort", "index", "expected_sign", "rho", "concordant"]]
          .round(3).to_string(index=False))
    print("  ", {k: round(v, 3) for k, v in extra.items()})
    print("\n== T4 surrogate\n", T4.to_string(index=False))
    for k in PHENOTYPES:
        for o in STRESS_OUTPUTS:
            t = T5[(T5.model == k) & (T5.output == o)]
            print(f"\n== Sobol {k} {o}  sum S1 = {t.S1_raw.sum():.3f}  sum ST = {t.ST_raw.sum():.3f}  "
                  f"min raw = {min(t.S1_raw.min(), t.ST_raw.min()):.4f}")
            print(t.head(6)[["parameter", "S1_raw", "ST_raw"]].round(3).to_string(index=False))
    print("\n== group indices\n", T6.round(3).to_string(index=False))
    print(f"\ntables -> {out}")


if __name__ == "__main__":
    main()
