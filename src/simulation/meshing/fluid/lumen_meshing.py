"""
Lumen fluid meshing runner (geo_0723, Simmetrix) -- 65_final_0723.

For each case under geo_0723/case_<i>:
  1. lumen.stp -> lumen.vtk (gmsh + meshio) -> lumen.vtp (vtk)
       (LumenMeshing._stp_to_vtk / _vtk_to_vtp)
  2. Simmetrix volume meshing (grid_size + boundary layers)
  3. assemble the mesh-complete/ tree
  4. tree.dat + xyzts.dat from the CAD parameters (parameter.csv row == case_id)

Outputs per case:
  fluid_0723/case_<i>/meshing/mesh-complete/...
  fluid_0723/case_<i>/tree.dat
  fluid_0723/case_<i>/xyzts.dat

Run on cvbml01 with the `ansys_new` conda env (gmsh/meshio/vtk + master_python egg-link):
  /home/jeff/miniconda3/envs/ansys_new/bin/python lumen_meshing.py

Adapted from 54_analysis/lumen_Simmetrix_meshing.py (fluid part only).
"""

import json
import argparse
import shutil
from pathlib import Path

from lumen_codes.cad_meshing import VesselCADModel, LumenMeshing
import utils.utils_lumen.runner as runner


# ---- user config ----
CASE_IDS = [0]
GRID_SIZE = 0.015
BOUNDARY_LAYER_THICKNESS = 0.005
NUMBER_OF_BOUNDARY_LAYERS = 3
NUM_TREE_POINTS = 50

# ---- paths ----
HERE = Path(__file__).resolve().parent
GEO_DIR = HERE / "geo_0723"
FLUID_ROOT = HERE / "fluid_0723"
PRE_DATA = HERE / "pre_data"
PARAMETER_CSV = PRE_DATA / "parameter.csv"
CONFIG_PATH = PRE_DATA / "solver_path.json"


class LumenMeshingCase:
    def __init__(self, case_id, exec_dir, parameter_csv_path, config_path):
        self.case_id = case_id
        self.exec_dir = Path(exec_dir)
        self.parameter_csv_path = Path(parameter_csv_path)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.meshing_dir = self.exec_dir / "meshing"
        self.mesh_complete_dir = self.meshing_dir / "mesh-complete"
        self.tree_path = self.exec_dir / "tree.dat"
        self.xyzt_path = self.exec_dir / "xyzts.dat"

        self.meshing_dir.mkdir(parents=True, exist_ok=True)

    def run(self, stp_path, grid_size=GRID_SIZE,
            number_of_boundary_layers=NUMBER_OF_BOUNDARY_LAYERS):
        stp_src = Path(stp_path)
        if not stp_src.exists():
            raise FileNotFoundError(f"Input STP not found: {stp_src}")

        # keep every meshing artifact inside fluid_0723/case_xxx/meshing
        stp_dst = self.meshing_dir / stp_src.name
        shutil.copy2(stp_src, stp_dst)

        # 1) stp -> vtk -> vtp (triangulated surface with inlet/wall/outlet ids)
        vtk_path = LumenMeshing._stp_to_vtk(str(stp_dst))
        vtp_path = LumenMeshing._vtk_to_vtp(vtk_path)

        # 2) Simmetrix volume meshing
        runner.run_simmetrix_license(license_path=self.config["simmetrix"]["license"])
        runner.run_simmetrix_meshing(
            vtp_path=vtp_path,
            exe_dir=self.meshing_dir,
            simmetrix_path=self.config["simmetrix"]["meshing"],
            grid_size=grid_size,
            boundary_layer_thickness=BOUNDARY_LAYER_THICKNESS,
            number_of_boundary_layers=number_of_boundary_layers,
        )

        # 3) organize mesh-complete/ tree
        LumenMeshing._assemble_mesh_tree(self.meshing_dir)

        # 4) tree.dat + xyzts.dat from the CAD morphology parameters
        cad = VesselCADModel.CAD_instance_from_idx(self.case_id, self.parameter_csv_path)
        cad.tree_xyzts_generator(
            tree_path=self.tree_path,
            xyzt_path=self.xyzt_path,
            num_points=NUM_TREE_POINTS,
        )


def parse_case_ids(case_args):
    if not case_args:
        return CASE_IDS

    case_ids = []
    for raw in case_args:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                if end < start:
                    raise ValueError(f"Invalid case range: {part}")
                case_ids.extend(range(start, end + 1))
            else:
                case_ids.append(int(part))

    return sorted(set(case_ids))


def is_case_complete(case_dir):
    return all([
        (case_dir / "meshing" / "mesh-complete" / "mesh-complete.mesh.vtu").exists(),
        (case_dir / "tree.dat").exists(),
        (case_dir / "xyzts.dat").exists(),
    ])


def discover_case_ids(update_only):
    discovered = []
    for case_dir in sorted(GEO_DIR.glob("case_*")):
        try:
            cid = int(case_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        if update_only and is_case_complete(FLUID_ROOT / case_dir.name):
            continue
        discovered.append(cid)

    return discovered


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run lumen fluid meshing for selected geo_0723 cases."
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case id, comma list, or inclusive range (examples: 12, 1,4,9, 10-20).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every case under geo_0723, regardless of existing outputs.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Run only cases missing final outputs under fluid_0723.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=GRID_SIZE,
        help=f"Global mesh size passed to Simmetrix (default: {GRID_SIZE}).",
    )
    parser.add_argument(
        "--boundary-layers",
        type=int,
        default=NUMBER_OF_BOUNDARY_LAYERS,
        help=f"Number of boundary layers (default: {NUMBER_OF_BOUNDARY_LAYERS}).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    for req in (GEO_DIR, PARAMETER_CSV, CONFIG_PATH):
        if not req.exists():
            raise FileNotFoundError(f"Required input missing: {req}")

    FLUID_ROOT.mkdir(parents=True, exist_ok=True)

    if args.cases and (args.all or args.update):
        parser.error("--case cannot be combined with --all or --update")

    if args.all and args.update:
        parser.error("--all and --update are mutually exclusive")

    if args.cases:
        case_ids = parse_case_ids(args.cases)
    elif args.all:
        case_ids = discover_case_ids(update_only=False)
    elif args.update:
        case_ids = discover_case_ids(update_only=True)
    else:
        case_ids = CASE_IDS

    if args.update and not case_ids:
        print("[Done] --update found no incomplete cases")
        (HERE / "lumen_meshing_failed.txt").write_text("[]")
        return

    failed = []
    total = len(case_ids)
    for idx, cid in enumerate(case_ids, start=1):
        case_dir = GEO_DIR / f"case_{cid}"
        stp_path = case_dir / "lumen.stp"
        if not stp_path.exists():
            print(f"[Skip {idx}/{total}] case_{cid}: lumen.stp not found")
            failed.append(cid)
            continue

        out_dir = FLUID_ROOT / f"case_{cid}"
        try:
            case = LumenMeshingCase(cid, out_dir, PARAMETER_CSV, CONFIG_PATH)
            case.run(
                stp_path=stp_path,
                grid_size=args.grid_size,
                number_of_boundary_layers=args.boundary_layers,
            )
            print(f"[Done {idx}/{total}] case_{cid}: mesh-complete + tree.dat + xyzts.dat")
        except Exception as e:
            print(f"[Fail {idx}/{total}] case_{cid}: {e}")
            failed.append(cid)

    if failed:
        print(f"Failed cases: {failed}")
    (HERE / "lumen_meshing_failed.txt").write_text(str(failed))


if __name__ == "__main__":
    main()
