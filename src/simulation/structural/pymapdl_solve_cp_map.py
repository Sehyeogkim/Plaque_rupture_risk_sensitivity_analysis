# CP solver (peak or low) via /MAP -> PRESSURE ||T||, same method as the LAP low run.
# CP uses the type2 mesh, which is exactly what ver2 was written for (tag_max=8;
# vessel=4/fc=5/sides=6/lipid=7), so only prep_bc is swapped to the /MAP form.
# Usage: pymapdl_solve_cp_map.py <case> <bc: peak|low> <nproc> <exec_dir> <port> <out_vtu>
import sys, time, os, pathlib
import numpy as np, pandas as pd, pyvista as pv, meshio
import ansys.mapdl.core as pymapdl
from ansys.mapdl.reader import save_as_archive
import pymapdl_worker as v2
import utils.utils_ansys.utils_prep as utils_prep

EXEC = "/opt/cvbml/softwares/ansys_inc/v251/ansys/bin/ansys251"
case    = int(sys.argv[1])
bc      = sys.argv[2]                                  # "peak" or "low"
nproc   = int(sys.argv[3])
exec_dir= pathlib.Path(sys.argv[4]).resolve()          # absolute — /MAP READ needs abs path
port    = int(sys.argv[5])
out_vtu = sys.argv[6]
assert bc in ("peak", "low")

Q       = pathlib.Path("/home/jeff/project/54_analysis")
mesh    = Q / f"CP_mesh/solid_CP_{case}.msh"
wall_csv= Q / "post_data" / "wall_expB_new" / bc / f"wall_{bc}_case_{case}.csv"   # x,y,z,Tx,Ty,Tz
param   = Q / "pre_data" / "parameter_0728.csv"        # E_vessel/E_fc/E_lipid/E_cal
matjson = Q / "pre_data" / "material.json"             # cal = mat_id 8
exec_dir.mkdir(parents=True, exist_ok=True)

# --- ||T|| pressure CSV: x,y,z,Pmag ---
df = pd.read_csv(wall_csv)
pmag = np.sqrt(df["Tx"].to_numpy()**2 + df["Ty"].to_numpy()**2 + df["Tz"].to_numpy()**2)
mag_csv = exec_dir / f"wall_{bc}mag_{case}.csv"
pd.DataFrame({"x": df["x"], "y": df["y"], "z": df["z"], "Pmag": np.round(pmag, 4)}).to_csv(mag_csv, index=False)
print(f"[CP{bc} {case}] ||T|| csv: {mag_csv} (max Pmag={pmag.max():.4g})", flush=True)

# --- msh_to_cdb: CP = type2, tags 1..8; blank cell_sets (meshio IndexError guard) ---
def _msh_to_cdb(self, gmsh_path, type=2):
    print("======= .msh to .cdb (CP/type2, cell_sets blanked) ========", flush=True)
    gmsh_mesh = meshio.read(gmsh_path, file_format="gmsh")
    gmsh_mesh.cell_sets = {}
    m = pv.from_meshio(gmsh_mesh)
    tags = np.array(m.cell_data["gmsh:physical"])
    print("gmsh:physical tags:", set(tags), flush=True)
    for i in range(1, 9):   # CP type2 = tags 1..8 (8 = calcification)
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

# --- prep_bc via /MAP: ||T|| as NORMAL pressure (keyopt(7,2,0)) ---
def _prep_bc_map(self, wall_csv_path):
    self.mapdl.prep7()
    cdb_name = pathlib.Path(self.cdb_path).name
    self.mapdl.cdread("db", cdb_name)
    self.mapdl.nlgeom("OFF"); self.mapdl.shpp("OFF"); self.mapdl.allsel()
    wall_in_vessel_tag, wall_in_fc_tag, sides_tag, lipid_in_fc_tag = 4, 5, 6, 7
    utils_prep.fix_bc_from_nid_list(self.mapdl, self.nodes_by_tag[sides_tag])
    utils_prep.Create_cm_from_nid_list(self.mapdl, self.nodes_by_tag[wall_in_vessel_tag], "WALL_IN_VESSEL", False)
    utils_prep.Create_cm_from_nid_list(self.mapdl, self.nodes_by_tag[wall_in_fc_tag], "WALL_IN_FC", False)
    utils_prep.Create_cm_from_nid_list(self.mapdl, self.nodes_by_tag[lipid_in_fc_tag], "LIPID_IN_FC", False)
    self.mapdl.et(7, "SURF154")
    self.mapdl.keyopt(7, 2, 0)      # face 1 = NORMAL pressure (||T||)
    self.mapdl.keyopt(7, 4, 0)
    self.mapdl.keyopt(7, 7, 1)
    self.mapdl.keyopt(7, 11, 2)
    for wall in ["WALL_IN_FC", "WALL_IN_VESSEL"]:
        self.mapdl.cmsel(type_='S', name=wall, entity='NODE')
        self.mapdl.type(7)
        self.mapdl.esurf()
    self.mapdl.esel('S', 'TYPE', '', 7)
    print(f"[/MAP] SURF154_FACES: {len(self.mapdl.mesh.enum)} elements", flush=True)
    self.mapdl.nsle()
    self.mapdl.cm('TARGET_NODES', 'NODE')
    self.mapdl.allsel()
    self.mapdl.finish()
    print(f"[/MAP] applying {self.bc_type} ||T|| pressure from {wall_csv_path}", flush=True)
    self.mapdl.run('/MAP')
    self.mapdl.target('TARGET_NODES')
    self.mapdl.ftype('CSV', 0)
    self.mapdl.read(str(wall_csv_path), 1, '', 1, 2, 3, 4)   # NSKIP=1, x=1,y=2,z=3,p=4(||T||)
    self.mapdl.run('MAP,,2,,0,20')
    self.mapdl.run("WRITEMAP,'mapped.sfe'")
    self.mapdl.finish()
    return
v2.PYMAPDL_worker.pymapdl_prep_bc = _prep_bc_map

# --- solve: apply mapped.sfe, PCG (CP has cal E~1e11 next to lipid E~4e4 -> ill-conditioned) ---
def _solve_map(self):
    t = time.time()
    self.mapdl.allsel()
    self.mapdl.slashsolu()
    self.mapdl.input("mapped.sfe")
    print("[/MAP] applied mapped ||T|| pressure from mapped.sfe", flush=True)
    self.mapdl.antype("STATIC")
    self.mapdl.nlgeom("OFF")
    self.mapdl.eqslv(lab='PCG', toler=1e-6)
    _lev = os.environ.get('PCG_LEVDIFF')
    if _lev:
        self.mapdl.run(f'PCGOPT,{_lev}')
    self.mapdl.time(1); self.mapdl.autots("ON")
    self.mapdl.nsubst(nsbstp=1, nsdbmn=1, nsbmx=10, carry="OFF")
    self.mapdl.kbc(0); self.mapdl.allsel()
    self.mapdl.solve(verbose=True)
    self.mapdl.finish()
    print(f"[CP{bc} {case}] solve {time.time()-t:.1f}s", flush=True)
v2.PYMAPDL_worker.pymapdl_solve = _solve_map

# --- run ---
w = v2.PYMAPDL_worker(case, str(exec_dir), exec_dir, str(param), nproc=nproc)
t0 = time.time()
w.cdb_path = w.pymapdl_msh_to_cdb(mesh, type=2)
w.mapdl = None
for attempt in range(4):
    try:
        w.mapdl = pymapdl.launch_mapdl(jobname=f"CP{case}_{bc}", exec_file=EXEC,
                    run_location=str(exec_dir), loglevel="ERROR", override=True,
                    port=port+attempt, nproc=nproc)
        break
    except Exception as e:
        print(f"[CP{bc} {case}] launch attempt {attempt} port {port+attempt} FAILED: {e}", flush=True)
        time.sleep(8)
if w.mapdl is None:
    raise SystemExit(f"[CP{bc} {case}] MAPDL failed to start")
w.mapdl.units("CGS")
w.bc_type = bc
w.casename = f"case_{case}_{bc}"
w.mapdl.jobname = w.casename
w.pymapdl_prep_bc(mag_csv)
w.pymapdl_prep_mat(str(matjson))
w.pymapdl_solve()
data = w._extract_fc_results()
w._build_fc_vtu(data, str(out_vtu))
try: w.mapdl.exit()
except Exception: pass
print(f"[CP{bc} {case}] DONE in {time.time()-t0:.1f}s -> {out_vtu}", flush=True)
