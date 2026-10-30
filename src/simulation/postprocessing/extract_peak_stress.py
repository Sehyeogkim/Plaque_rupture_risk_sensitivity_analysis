"""
LAP (type1_expB_new) post-processing with a radius sweep.

Implements Journal_definition.md exactly:

  sigma_bar_n(t) = mean of maximum principal stress over the spherical
                   neighbourhood of radius r around node n, restricted to
                   the fibrous-cap luminal surface (Group 5).

  PSS      = max_n sigma_bar_n(t_sys)
  del_PSS  = max_n ( sigma_bar_n(t_sys) - sigma_bar_n(t_dia) )     (NO /2)

  d_z      = z_peak - z_MLA                       [cm]
  axial    : upstream (d_z < -0.15), MLA (|d_z| <= 0.15), downstream (d_z > 0.15)
  I_theta  = |theta_peak| / (theta_lipid/2)       (alpha_theta returns the half-arc)
  circum   : center (I_theta < 0.5), shoulder (I_theta >= 0.5)

Two corrections relative to post_processing_updated_bf_type2_expB_new.py:

 1. VALID-NODE MASK.  The vtu carries every node of the full solid model but
    only the fibrous-cap tets, so ~84% of the nodes are orphans with stress
    identically 0.  The old script KD-treed all 218k points and averaged them
    in, diluting each sphere mean ~5x.  Here the mask is the exact set of
    nodes referenced by a tet.
 2. PSS is the max of the sphere-averaged stress (journal definition), not the
    95th percentile of the raw stresses inside the winning sphere.

r = None means no smoothing: the raw nodal peak.
"""
from pathlib import Path
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.neighbors import radius_neighbors_graph
from multiprocessing import get_context

from utils.utils_geo_mesh.main_CAD import CAD_instance_from_idx
from utils.utils_geo_mesh.utils_CAD import y_center_lumen, alpha_theta

BASE = Path("/home/jeff/project/55c_final_defense_0512/Figure_final_0731")
SOLID_DATA_DIR = Path("/home/jeff/project/54_analysis/solid_data")
PARAMETER_CSV = BASE / "pre_data" / "parameter_0728.csv"
OUT_DIR = BASE / "data_0731_LAP"

RESULT_SUBDIR = os.environ.get("FC_RESULT_SUBDIR", "ansys_results_type1_expB_new")
GROUP_TAG = 5                                    # cap luminal surface
Z_EPSILON = 0.02                                 # cm, trim each end of the cap
MLA_BAND_CM = 0.15                               # +/- 1.5 mm
CIRCUM_THRESHOLD = 0.5

RADII = [None, 0.005, 0.010, 0.015, 0.020]


def radius_tag(r):
    # "raw", not "None": pandas parses the string "None" back as NaN.
    return "raw" if r is None else f"{r:.3f}"


def sphere_average(points, values, r):
    """Mean of `values` over the r-sphere around every point (self included)."""
    if r is None:
        return values
    A = radius_neighbors_graph(points, r, mode="connectivity",
                               include_self=True, n_jobs=1)
    counts = np.asarray(A.sum(axis=1)).ravel()
    return np.asarray(A @ values).ravel() / counts


def circum_index(cad, x, y, z):
    """|theta| / (theta_lipid/2); alpha_theta already returns the half-arc."""
    y_c = y_center_lumen(cad, z)
    v = np.array([x, y - y_c], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return float("nan")
    cos_val = np.dot(v, np.array([0.0, 1.0])) / n
    angle = np.arccos(np.clip(cos_val, -1.0, 1.0))
    half_arc = alpha_theta(cad, z)
    if not np.isfinite(half_arc) or half_arc <= 0:
        return float("nan")
    return float(angle / half_arc)


def loc_index(crit_z, z_mlp, lesion_length):
    """Piecewise-normalised axial index, asymmetric MLP handled.

    -1 = proximal lipid edge, 0 = MLP, +1 = distal lipid edge.
    Each side is normalised by its own MLP-to-edge distance, so a lesion whose
    MLP sits off-centre still maps onto [-1, +1].
    """
    denom = (lesion_length / 2.0 - z_mlp) if crit_z > z_mlp \
        else (z_mlp + lesion_length / 2.0)
    return (crit_z - z_mlp) / denom if denom != 0 else float("nan")


def classify_axial(d_z):
    if not np.isfinite(d_z):
        return "unknown"
    if d_z < -MLA_BAND_CM:
        return "upstream"
    if d_z > MLA_BAND_CM:
        return "downstream"
    return "MLA"


def classify_circum(i_theta):
    if not np.isfinite(i_theta):
        return "unknown"
    return "center" if i_theta < CIRCUM_THRESHOLD else "shoulder"


def process_case(case_index):
    case_index = int(case_index)
    d = SOLID_DATA_DIR / f"case_{case_index}" / RESULT_SUBDIR
    peak_path = d / f"FC_case_{case_index}_peak.vtu"
    low_path = d / f"FC_case_{case_index}_low.vtu"
    if not peak_path.exists() or not low_path.exists():
        return [], f"case_{case_index}: missing vtu"

    try:
        pk = pv.read(peak_path)
        lo = pv.read(low_path)
    except Exception as e:
        return [], f"case_{case_index}: read failed ({e})"

    if pk.n_points != lo.n_points:
        return [], f"case_{case_index}: peak/low node count mismatch"

    if "Group" not in pk.point_data:
        return [], f"case_{case_index}: no Group field"

    pts = pk.points
    group = pk.point_data["Group"]
    sp = pk.point_data["Principal_stress"]
    sl = lo.point_data["Principal_stress"]

    # --- valid nodes: exactly those referenced by a fibrous-cap tet ---------
    try:
        conn = pk.cells_dict[10]                      # VTK_TETRA
    except KeyError:
        return [], f"case_{case_index}: no linear tets"
    valid = np.zeros(pk.n_points, dtype=bool)
    valid[np.unique(conn)] = True

    mask = (group == GROUP_TAG) & valid
    if not mask.any():
        return [], f"case_{case_index}: no valid Group {GROUP_TAG} nodes"

    # --- suspect-data filter: diastole must not exceed systole -------------
    ep = pk.point_data["EQV_stress"][mask]
    el = lo.point_data["EQV_stress"][mask]
    if np.nanmean(el) > np.nanmean(ep):
        return [], f"case_{case_index}: suspect data (low > peak)"

    # --- trim the two cap ends --------------------------------------------
    idx = np.where(mask)[0]
    z = pts[idx, 2]
    keep = (z > z.min() + Z_EPSILON) & (z < z.max() - Z_EPSILON)
    idx = idx[keep]
    if len(idx) == 0:
        return [], f"case_{case_index}: nothing left after z-trim"

    sub = pts[idx]
    stress_peak = sp[idx].astype(float)
    stress_delta = (sp[idx] - sl[idx]).astype(float)   # NO /2

    # --- geometry ----------------------------------------------------------
    try:
        cad = CAD_instance_from_idx(case_index, str(PARAMETER_CSV))
        z_mla = float(cad.z_peak)
        lesion_length = float(cad.lesion_length)
    except Exception as e:
        return [], f"case_{case_index}: CAD failed ({e})"

    rows = []
    for r in RADII:
        avg_peak = sphere_average(sub, stress_peak, r)
        avg_delta = sphere_average(sub, stress_delta, r)

        i_pss = int(np.argmax(avg_peak))
        # del_PSS = max_n (sigma_sys - sigma_dia), with the sign retained.
        i_dps = int(np.argmax(avg_delta))

        row = {
            "case_id": case_index,
            "radius_cm": 0.0 if r is None else r,
            "radius_tag": radius_tag(r),
            "n_nodes": len(idx),
            "z_MLA": z_mla,
            "lesion_length": lesion_length,
            "PSS": float(avg_peak[i_pss]),
            "del_PSS": float(avg_delta[i_dps]),
        }

        for prefix, i, in (("PSS", i_pss), ("delPSS", i_dps)):
            x, y, zz = (float(v) for v in sub[i])
            d_z = zz - z_mla
            it = circum_index(cad, x, y, zz)
            row[f"{prefix}_x"] = x
            row[f"{prefix}_y"] = y
            row[f"{prefix}_z"] = zz
            row[f"{prefix}_dz"] = d_z
            row[f"{prefix}_dz_mm"] = d_z * 10.0
            row[f"{prefix}_loc_index"] = loc_index(zz, z_mla, lesion_length)
            row[f"{prefix}_axial_class"] = classify_axial(d_z)
            row[f"{prefix}_circum_index"] = it
            row[f"{prefix}_circum_class"] = classify_circum(it)

        rows.append(row)

    return rows, None


def _worker(case_index):
    try:
        return process_case(case_index)
    except Exception:
        return [], f"case_{case_index}: UNCAUGHT\n{traceback.format_exc()}"


COLUMNS = (
    ["case_id", "radius_cm", "radius_tag", "n_nodes", "z_MLA", "lesion_length",
     "PSS", "del_PSS"]
    + [f"{p}_{s}" for p in ("PSS", "delPSS")
       for s in ("x", "y", "z", "dz", "dz_mm", "loc_index", "axial_class",
                 "circum_index", "circum_class")]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--type", choices=("LAP", "CP"), default="LAP")
    ap.add_argument("--prefix", type=str, default=None)
    args = ap.parse_args()

    global RESULT_SUBDIR
    RESULT_SUBDIR = ("ansys_results_type1_expB_new" if args.type == "LAP"
                     else "ansys_results_type2_expB_new")
    os.environ["FC_RESULT_SUBDIR"] = RESULT_SUBDIR      # inherited by workers
    prefix = args.prefix or args.type
    print(f"type={args.type}  subdir={RESULT_SUBDIR}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_indices = list(range(args.cases))

    with get_context("spawn").Pool(processes=args.procs) as pool:
        results = pool.map(_worker, case_indices, chunksize=4)

    all_rows, skips = [], []
    for rows, err in results:
        all_rows.extend(rows)
        if err:
            skips.append(err)

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df = df.sort_values(["radius_cm", "case_id"]).reset_index(drop=True)

    combined = args.out_dir / f"{prefix}_all_radii.csv"
    df.to_csv(combined, index=False)

    for tag, sub in df.groupby("radius_tag"):
        sub.drop(columns=["radius_tag"]).to_csv(
            args.out_dir / f"{prefix}_r{tag}.csv", index=False)

    n_cases = df["case_id"].nunique()
    print(f"\n=== LAP extraction complete ===")
    print(f"cases with results : {n_cases}")
    print(f"radii              : {len(RADII)}")
    print(f"rows               : {len(df)}  (expect {n_cases * len(RADII)})")
    print(f"combined           : {combined}")
    print(f"per-radius CSVs    : {args.out_dir}/LAP_r<tag>.csv")

    if skips:
        skip_path = args.out_dir / f"{prefix}_skipped.txt"
        skip_path.write_text("\n".join(skips) + "\n")
        print(f"skipped            : {len(skips)}  -> {skip_path}")
        from collections import Counter
        reasons = Counter(s.split(": ", 1)[1].split(" (")[0].split("\n")[0]
                          for s in skips)
        for reason, n in reasons.most_common():
            print(f"    {n:5d}  {reason}")


if __name__ == "__main__":
    main()
