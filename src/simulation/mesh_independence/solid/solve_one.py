#!/usr/bin/env python
"""Single solid solve for the mesh-independence study: ver2 SFE traction on an
arbitrary LAP .msh, for either BC.

Mesh-independence numerics use vector traction on ver2, not the ||T|| /MAP
pressure in structural/pymapdl_solve_lap_map.py:
  - traction (Tx,Ty,Tz) applied as SFE via utils_bc.Apply_Traction  [NOT /MAP]
  - PCG toler 1e-5, STATIC, NLGEOM OFF  (production settings, untouched on purpose)
  - msh_to_cdb monkeypatched to blank cell_sets (LAP mesh meshio IndexError fix)

The mesh path and the wall csv are free parameters, so the same code path
drives peak and low on each of the three test meshes.

Usage: solve_one.py <case> <bc> <mesh_msh> <wall_csv> <nproc> <exec_dir> <port> <out_vtu>
"""
import sys, time, pathlib

# Worker lives next to the structural solvers. utils is still the campaign tree.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "structural"))
sys.path.insert(0, "/home/jeff/project/54_analysis")

import numpy as np, pyvista as pv, meshio
import ansys.mapdl.core as pymapdl
from ansys.mapdl.reader import save_as_archive
import pymapdl_worker as v2
import utils.utils_ansys.utils_prep as utils_prep

EXEC = "/opt/cvbml/softwares/ansys_inc/v251/ansys/bin/ansys251"
case     = int(sys.argv[1])
bc       = sys.argv[2]                       # "peak" | "low"
mesh     = pathlib.Path(sys.argv[3])
wall_csv = pathlib.Path(sys.argv[4])
nproc    = int(sys.argv[5])
exec_dir = pathlib.Path(sys.argv[6])
port     = int(sys.argv[7])
out_vtu  = sys.argv[8]

Q       = pathlib.Path("/home/jeff/project/54_analysis")
param   = Q / "pre_data" / "parameter_0728.csv"
matjson = Q / "pre_data" / "material.json"
exec_dir.mkdir(parents=True, exist_ok=True)


def _msh_to_cdb(self, gmsh_path, type=1):
    print("======= .msh to .cdb (mesh-indep, cell_sets blanked) ========", flush=True)
    gmsh_mesh = meshio.read(gmsh_path, file_format="gmsh")
    gmsh_mesh.cell_sets = {}
    m = pv.from_meshio(gmsh_mesh)
    tags = np.array(m.cell_data["gmsh:physical"])
    print("gmsh:physical tags:", set(tags), flush=True)
    for i in range(1, 8):   # LAP mesh = type1 tags 1..7
        if i not in tags:
            raise Exception(f"gmsh:physical tag {i} missing in {gmsh_path}")
    m.points *= 0.1
    self.nodes_by_tag, _ = utils_prep.surface_nid_dic(m, Linear=False, Terminal_display=False)
    cdb = self.ansys_exec_dir / f"total_solid{type}.cdb"
    if cdb.exists():
        return str(cdb)
    save_as_archive(cdb, m)
    return str(cdb)


v2.PYMAPDL_worker.pymapdl_msh_to_cdb = _msh_to_cdb


def _apply_traction_vector(mapdl, interpolated_df, element_array, txt_path="apply_traction.txt"):
    """ws2's Apply_Traction: the traction VECTOR, one SFE per local direction.

    ws1's copy of master_python is stale -- it writes a single
        sfe, e, 1, PRES, 0, ||T||
    i.e. the whole traction magnitude on face 1, which with keyopt(7,2,1) is the element
    LOCAL X (in-plane) direction. Loading a thin stiff cap in-plane instead of normal
    inflates cap stress ~100x: on this mesh it gave 12.5 MPa against production's 115 kPa.
    ws2 (which produced the production peaks) has the vector form below, with the
    magnitude form commented out. Overriding here rather than editing the shared repo,
    which other jobs on this host also import.
    """
    df = interpolated_df
    scale_factor = 1.0

    elems, conn = zip(*[(arr[-10], [arr[-8], arr[-7], arr[-6]]) for arr in element_array])
    elems = np.array(elems, dtype=np.int32)
    conn = np.array(conn, dtype=np.int32)

    node_ids = df["node_ID"].to_numpy(dtype=np.int32)
    tx = (df["Tx"].to_numpy() * scale_factor).astype(float)
    ty = (df["Ty"].to_numpy() * scale_factor).astype(float)
    tz = (df["Tz"].to_numpy() * scale_factor).astype(float)

    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    conn_idx = np.array([[node_id_to_idx[nid] for nid in elem] for elem in conn])
    Tx_e = np.round(tx[conn_idx].mean(axis=1), decimals=2).astype(float)
    Ty_e = np.round(ty[conn_idx].mean(axis=1), decimals=2).astype(float)
    Tz_e = np.round(tz[conn_idx].mean(axis=1), decimals=2).astype(float)

    with open(txt_path, "w") as f:
        for e, x, y, z in zip(elems, Tx_e, Ty_e, Tz_e):
            f.write(f"sfe, {e}, 1, PRES, 0, {x}\n")
            f.write(f"sfe, {e}, 2, PRES, 0, {y}\n")
            f.write(f"sfe, {e}, 3, PRES, 0, {z}\n")
    mapdl.input(txt_path)


import utils.utils_ansys.utils_bc as _ubc
_ubc.Apply_Traction = _apply_traction_vector
v2.utils_bc.Apply_Traction = _apply_traction_vector

# wall_folder_path = exec_dir so apply_traction_<bc>.txt is per-run (no cross-run race)
w = v2.PYMAPDL_worker(case, str(exec_dir), exec_dir, str(param), nproc=nproc)
t0 = time.time()
w.cdb_path = w.pymapdl_msh_to_cdb(mesh)

w.mapdl = None
for attempt in range(4):
    try:
        w.mapdl = pymapdl.launch_mapdl(jobname=f"mi_{case}_{bc}", exec_file=EXEC,
                    run_location=str(exec_dir), loglevel="ERROR", override=True,
                    port=port + attempt, nproc=nproc)
        break
    except Exception as e:
        print(f"[{bc} {case}] launch attempt {attempt} port {port+attempt} FAILED: {e}", flush=True)
        time.sleep(8)
if w.mapdl is None:
    raise SystemExit(f"[{bc} {case}] MAPDL failed to start")

w.mapdl.units("CGS")
w.bc_type = bc
w.casename = f"case_{case}_{bc}"
w.mapdl.jobname = w.casename
w.pymapdl_prep_bc(wall_csv)          # SFE traction (Tx,Ty,Tz)
w.pymapdl_prep_mat(str(matjson))
w.pymapdl_solve()
data = w._extract_fc_results()
w._build_fc_vtu(data, str(out_vtu))
try:
    w.mapdl.exit()
except Exception:
    pass
print(f"[{bc} {case}] DONE in {time.time()-t0:.1f}s -> {out_vtu}", flush=True)
