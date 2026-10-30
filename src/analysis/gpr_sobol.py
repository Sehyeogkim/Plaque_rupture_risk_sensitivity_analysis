"""
Step 2 - GPR surrogate + variance-based Sobol sensitivity analysis.

For each phenotype (LAP: 12 inputs, CP: 18 inputs) and each stress index
(PSS, del_PSS):

  1. inputs min-max scaled to the range of the LHS design
  2. output natural-log transformed, then standardised (normalize_y=True,
     so within each CV fold the standardisation uses the training part only)
  3. kernel: constant * ARD squared-exponential + white-noise nugget,
     hyper-parameters by maximising the log marginal likelihood
  4. five-fold cross-validation -> R2 / RMSE on the log scale, out-of-fold
     predictions kept for physical-scale metrics (see make_tables.py)
  5. retrain on the full dataset
  6. Saltelli sampling (N = 2048, no second-order terms) on the surrogate,
     individual and grouped (Morphology / Material / Hemodynamics) Sobol
     indices with 1,000 bootstrap resamples (95% CI). Every Sobol input,
     including w (ca_strength_ratio), is uniform on its observed design range.
     This Sobol target differs from the truncated log-normal w used to generate
     the simulation design. Indices are computed on both the
     fitted log scale ("log") and, after inverse-transforming the predictions,
     on the physical scale ("orig").  The paper reports "orig".

Raw estimates are kept (S1_raw, ST_raw); a clipped-at-zero copy (S1, ST) is
stored alongside for plotting.

Usage:
    python gpr_sobol.py --phenotype LAP
    python gpr_sobol.py --phenotype CP
Outputs -> outputs/sensitivity/{LAP,CP}/
"""
import argparse
import warnings
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from config import DESIGN_CSV, DISPLAY_NAMES, OUTPUTS, SIM, STRESS_OUTPUTS, input_set

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
N_SOBOL = 2048
N_BOOT = 1000
# Additional optimiser restarts in GaussianProcessRegressor (random_state=42),
# i.e. 1 + N_RESTARTS initialisations; the best log marginal likelihood is kept.
# The reported results were produced with 5.
N_RESTARTS = 5


def build_kernel(n_dim):
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))


def fit_gpr(X, y, n_restarts):
    gpr = GaussianProcessRegressor(kernel=build_kernel(X.shape[1]),
                                   n_restarts_optimizer=n_restarts,
                                   normalize_y=True, random_state=SEED)
    gpr.fit(X, y)
    return gpr


def run_one(job):
    output, X, y, case_ids, names, groups, n_restarts = job
    n = len(y)

    # ---- five-fold cross-validation ---------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds, oof = [], np.full(n, np.nan)
    for k, (tr, te) in enumerate(kf.split(X), start=1):
        pred = fit_gpr(X[tr], y[tr], n_restarts).predict(X[te])
        oof[te] = pred
        folds.append({"output": output, "fold": k,
                      "n_train": len(tr), "n_test": len(te),
                      "R2": r2_score(y[te], pred),
                      "RMSE": float(np.sqrt(mean_squared_error(y[te], pred)))})

    # ---- retrain on the full dataset --------------------------------------
    gpr = fit_gpr(X, y, n_restarts)
    ls = np.atleast_1d(gpr.kernel_.k1.k2.length_scale)
    hyper = {"output": output, "n_samples": n,
             "constant": float(gpr.kernel_.k1.k1.constant_value),
             "noise": float(gpr.kernel_.k2.noise_level),
             "log_marginal_likelihood": float(
                 gpr.log_marginal_likelihood(gpr.kernel_.theta))}
    hyper.update({f"ls_{p}": float(v) for p, v in zip(names, ls)})

    # ---- Sobol on the surrogate (all inputs uniform on observed ranges) --
    prob_ind = {"num_vars": len(names), "names": names,
                "bounds": [[0.0, 1.0]] * len(names)}
    prob_grp = {**prob_ind, "groups": groups}
    Xs_ind = sobol_sample.sample(prob_ind, N_SOBOL, calc_second_order=False, seed=SEED)
    Xs_grp = sobol_sample.sample(prob_grp, N_SOBOL, calc_second_order=False, seed=SEED)
    yi_log, yg_log = gpr.predict(Xs_ind), gpr.predict(Xs_grp)
    scales = {"log": (yi_log, yg_log), "orig": (np.exp(yi_log), np.exp(yg_log))}

    clip = lambda a: np.clip(np.asarray(a, dtype=float), 0.0, None)
    group_names = list(dict.fromkeys(groups))
    ind_parts, grp_parts = [], []
    for scale, (yi, yg) in scales.items():
        Si = sobol_analyze.analyze(prob_ind, yi, calc_second_order=False,
                                   num_resamples=N_BOOT, conf_level=0.95, seed=SEED)
        Sg = sobol_analyze.analyze(prob_grp, yg, calc_second_order=False,
                                   num_resamples=N_BOOT, conf_level=0.95, seed=SEED)
        a = pd.DataFrame({
            "output": output, "scale": scale, "parameter": names,
            "display_name": [DISPLAY_NAMES[p] for p in names], "group": groups,
            "S1_raw": Si["S1"], "S1": clip(Si["S1"]), "S1_conf": Si["S1_conf"],
            "ST_raw": Si["ST"], "ST": clip(Si["ST"]), "ST_conf": Si["ST_conf"],
        })
        a["S1_ci_includes_zero"] = a.S1_raw - a.S1_conf <= 0
        a["ST_ci_includes_zero"] = a.ST_raw - a.ST_conf <= 0
        a["rank_S1"] = a["S1"].rank(ascending=False, method="min").astype(int)
        a["rank_ST"] = a["ST"].rank(ascending=False, method="min").astype(int)
        ind_parts.append(a)
        grp_parts.append(pd.DataFrame({
            "output": output, "scale": scale, "group": group_names,
            "S1_raw": Sg["S1"], "S1": clip(Sg["S1"]), "S1_conf": Sg["S1_conf"],
            "ST_raw": Sg["ST"], "ST": clip(Sg["ST"]), "ST_conf": Sg["ST_conf"],
        }))

    oof_df = pd.DataFrame({"output": output, "case_id": case_ids,
                           "y_true_log": y, "y_pred_log": oof,
                           "y_true": np.exp(y), "y_pred": np.exp(oof)})
    r2 = [f["R2"] for f in folds]
    print(f"[OK] {output:<8s} n={n}  R2={np.mean(r2):.4f} +/- {np.std(r2, ddof=1):.4f}",
          flush=True)
    return folds, hyper, pd.concat(ind_parts), pd.concat(grp_parts), oof_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype", choices=["LAP", "CP"], required=True)
    ap.add_argument("--restarts", type=int, default=N_RESTARTS,
                    help="GPR optimiser restarts (default: value used for the paper)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    names, groups = input_set(args.phenotype)
    out_dir = (OUTPUTS / "sensitivity" / args.phenotype if args.out_dir is None
               else __import__("pathlib").Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.phenotype}: {len(names)} inputs, restarts={args.restarts}")

    par = pd.read_csv(DESIGN_CSV)
    par["case_id"] = np.arange(len(par))
    lo, hi = par[names].min(), par[names].max()
    norm = par.copy()
    norm[names] = (par[names] - lo) / (hi - lo)
    pd.DataFrame({"parameter": names, "design_min": lo.values,
                  "design_max": hi.values, "group": groups}
                 ).to_csv(out_dir / "input_ranges.csv", index=False)

    res = pd.read_csv(SIM / f"stress_{args.phenotype}.csv")[["case_id", *STRESS_OUTPUTS]]
    m = res.merge(norm[["case_id"] + names], on="case_id", how="inner")
    jobs = []
    for out in STRESS_OUTPUTS:
        d = m.dropna(subset=names + [out])
        d = d[d[out] > 0]
        jobs.append((out, d[names].to_numpy(float), np.log(d[out].to_numpy(float)),
                     d["case_id"].to_numpy(), names, groups, args.restarts))

    with get_context("spawn").Pool(processes=len(jobs)) as pool:
        results = pool.map(run_one, jobs)

    folds = pd.DataFrame([f for r in results for f in r[0]])
    summary = (folds.groupby("output")
               .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                    n_folds=("fold", "count")).reset_index())
    folds.to_csv(out_dir / "cv_foldwise.csv", index=False)
    summary.to_csv(out_dir / "cv_summary.csv", index=False)
    pd.DataFrame([r[1] for r in results]).to_csv(out_dir / "kernel_hyperparams.csv", index=False)
    pd.concat([r[2] for r in results]).to_csv(out_dir / "sobol_ind.csv", index=False)
    pd.concat([r[3] for r in results]).to_csv(out_dir / "sobol_grp.csv", index=False)
    pd.concat([r[4] for r in results]).to_csv(out_dir / "oof_predictions.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
