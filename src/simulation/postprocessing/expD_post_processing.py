"""Extract raw nodal PSS and signed systolic-diastolic ΔPSS for Experiment D.

For each case and surface group, use valid fibrous-cap nodes after trimming
0.02 cm from each axial end. PSS is the maximum peak maximum-principal stress;
ΔPSS is the maximum nodewise peak-minus-low stress, without dividing by two.
EQV stress columns use the same operators. Output files have a ``_raw`` suffix
so this script never overwrites the bundled Fig. 5E CSV files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
from multiprocessing import get_context

SOLID_DATA_DIR = Path("/home/jeff/project/61_experimentD/geo_expD")
PARAMETER_CSV_PATH = Path(__file__).parent / "pre_data/parameterD.csv"
STRESS_FIELDS = ["EQV_stress", "Principal_stress"]
Z_EPSILON = 0.02


from utils.utils_geo_mesh.main_CAD import CAD_instance_from_idx

def get_z_MLP(case_index: int, parameter_csv_path: str):
    """Get the z-coordinate of the MLP of a given case."""
    cad_instance = CAD_instance_from_idx(case_index, parameter_csv_path)
    z_peak = cad_instance.z_peak
    lesion_length = cad_instance.lesion_length
    return z_peak, lesion_length


def process_case(args):
    """Process a single case and return {case_id, PSS, delta_PSS}."""
    case_index, group_tag = args
    case_dir = SOLID_DATA_DIR / f"case_{case_index}" / "ansys_expD"
    peak_path = case_dir / f"FC_case_{case_index}_peak.vtu"
    low_path = case_dir / f"FC_case_{case_index}_low.vtu"

    if not peak_path.exists() or not low_path.exists():
        print(f"[SKIP] case_{case_index}: VTU files not found")
        return None

    # Read VTU files
    fc_peak = pv.read(peak_path)
    fc_low = pv.read(low_path)

    if fc_peak.n_points != fc_low.n_points:
        print(f"[SKIP] case_{case_index}: peak/low node count mismatch")
        return None
    if "Group" not in fc_peak.point_data:
        print(f"[SKIP] case_{case_index}: no Group field")
        return None
    try:
        conn = fc_peak.cells_dict[10]  # linear fibrous-cap tetrahedra
    except KeyError:
        print(f"[SKIP] case_{case_index}: no linear tets")
        return None
    valid = np.zeros(fc_peak.n_points, dtype=bool)
    valid[np.unique(conn)] = True
    mask_group5 = (fc_peak.point_data["Group"] == group_tag) & valid
    g5_points = fc_peak.points[mask_group5]
    if len(g5_points) == 0:
        print(f"[SKIP] case_{case_index}: no Group {group_tag} nodes")
        return None

    # Skip suspect cases where low mean stress > peak mean stress (Group 5, EQV)
    peak_eqv = fc_peak.point_data["EQV_stress"]
    low_eqv = fc_low.point_data["EQV_stress"]
    peak_g5_mean = np.mean(peak_eqv[mask_group5])
    low_g5_mean = np.mean(low_eqv[mask_group5])
    if low_g5_mean > peak_g5_mean:
        print(f"[SKIP] case_{case_index}: suspect data (low_mean={low_g5_mean:.2e} > peak_mean={peak_g5_mean:.2e})")
        return None

    # Z masking
    z_min, z_max = g5_points[:, 2].min(), g5_points[:, 2].max()
    mask_z = ((fc_peak.points[:, 2] > z_min + Z_EPSILON)
              & (fc_peak.points[:, 2] < z_max - Z_EPSILON))
    mask_group5_z = mask_group5 & mask_z
    z_masked_group5_indices = np.where(mask_group5_z)[0]

    if len(z_masked_group5_indices) == 0:
        print(f"[SKIP] case_{case_index}: no nodes after z-masking")
        return None

    # Compute raw nodal maxima for each stress field.
    points_overall = fc_peak.points
    row = {"case_id": case_index}
    pss_crit_idx = None
    delpss_crit_idx = None
    for sf in STRESS_FIELDS:
        label = "EQV" if sf == "EQV_stress" else "Principal"
        peak = np.asarray(fc_peak.point_data[sf])[z_masked_group5_indices]
        delta = peak - np.asarray(fc_low.point_data[sf])[z_masked_group5_indices]
        pss_idx = int(np.argmax(peak))
        delpss_idx = int(np.argmax(delta))
        row[f"PSS_{label}"] = float(peak[pss_idx])
        row[f"del_PSS_{label}"] = float(delta[delpss_idx])
        if label == "Principal":
            pss_crit_idx = int(z_masked_group5_indices[pss_idx])
            delpss_crit_idx = int(z_masked_group5_indices[delpss_idx])

    # --- Location labeling for Principal critical nodes (PSS and del_PSS) ---
    try:
        z_mlp, lesion_length = get_z_MLP(case_index, str(PARAMETER_CSV_PATH))
        z_mlp = float(z_mlp)
        lesion_length = float(lesion_length)
    except Exception as e:
        print(f"[SKIP] case_{case_index}: get_z_MLP failed ({e})")
        return None

    def compute_loc_index(crit_z):
        # Piecewise normalization handling asymmetric MLP position.
        # +1 at distal lipid edge, 0 at MLP, -1 at proximal lipid edge.
        if crit_z > z_mlp:
            denom = lesion_length / 2.0 - z_mlp      # MLP -> distal edge
        else:
            denom = z_mlp + lesion_length / 2.0      # MLP -> proximal edge
        return (crit_z - z_mlp) / denom if denom != 0 else float("nan")

    pss_xyz = points_overall[pss_crit_idx]
    row["PSS_crit_x"] = float(pss_xyz[0])
    row["PSS_crit_y"] = float(pss_xyz[1])
    row["PSS_crit_z"] = float(pss_xyz[2])
    row["PSS_loc_index"] = compute_loc_index(row["PSS_crit_z"])

    delpss_xyz = points_overall[delpss_crit_idx]
    row["delPSS_crit_x"] = float(delpss_xyz[0])
    row["delPSS_crit_y"] = float(delpss_xyz[1])
    row["delPSS_crit_z"] = float(delpss_xyz[2])
    row["delPSS_loc_index"] = compute_loc_index(row["delPSS_crit_z"])

    #row["z_MLP"] = z_mlp
    #row["lesion_length"] = lesion_length

    print(f"[OK] case_{case_index}: PSS_Principal={row['PSS_Principal']:.2e}, "
          f"PSS_loc={row['PSS_loc_index']:.3f}, delPSS_loc={row['delPSS_loc_index']:.3f}")
    return row



if __name__ == "__main__":
    case_indices = list(range(250))
    out_dir = Path(__file__).parent

    for group_tag, label in [(5, "tag5_lumenFC"), (7, "tag7_lipidFC")]:
        output_path = out_dir / f"PSS_result_expD_raw_{label}.csv"
        args_list = [(idx, group_tag) for idx in case_indices]

        try:
            with get_context("spawn").Pool(processes=20) as pool:
                results = pool.map(process_case, args_list)
        except PermissionError:
            print("Multiprocessing unavailable. Falling back to serial execution.")
            results = [process_case(a) for a in args_list]

        results = [r for r in results if r is not None]
        df = pd.DataFrame(results, columns=[
            "case_id",
            "PSS_EQV", "del_PSS_EQV",
            "PSS_Principal", "del_PSS_Principal",
            "PSS_crit_x", "PSS_crit_y", "PSS_crit_z", "PSS_loc_index",
            "delPSS_crit_x", "delPSS_crit_y", "delPSS_crit_z", "delPSS_loc_index",
        ])
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} rows to {output_path}")
