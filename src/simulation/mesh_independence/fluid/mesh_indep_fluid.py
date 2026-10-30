"""
Fluid mesh-independence meshing for 54_analysis, case 0 only.
Usage: mesh_indep_fluid.py <grid_size>       e.g. 0.015

lumen_Simmetrix_meshing.LumenMeshingCase 와 동일한 경로를 쓰되 tree/xyzts 생성은 건너뛴다
(steady 3D 는 1D json 의 Q,R 만 필요하고 tree 는 안 씀).
boundary layer 설정(두께 0.005, 3층)은 production 과 동일하게 고정 -> core grid size 만 변화.
결과: mesh_independent_test_fluid/mesh/<grid>/meshing/mesh-complete/
"""
import json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, "/home/jeff/repo/master_python/src")
sys.path.insert(0, "/home/jeff/project/54_analysis")

from lumen_codes.cad_meshing import LumenMeshing
import utils.utils_lumen.runner as runner

ROOT = Path("/home/jeff/project/54_analysis")
BASE = ROOT / "mesh_independent_test_fluid"
STP  = ROOT / "solid_data" / "case_0" / "lumen.stp"
CFG  = json.load(open(ROOT / "pre_data" / "solver_path.json"))

BL_THICKNESS = 0.005      # production 과 동일 (고정)
BL_LAYERS    = 3          # production 과 동일 (고정)


def main():
    grid = sys.argv[1]
    outdir = BASE / "mesh" / grid
    meshdir = outdir / "meshing"
    meshdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[start] grid={grid} -> {meshdir}", flush=True)

    stp_dst = meshdir / STP.name
    shutil.copy2(STP, stp_dst)

    vtk_path = LumenMeshing._stp_to_vtk(str(stp_dst))
    vtp_path = LumenMeshing._vtk_to_vtp(vtk_path)

    runner.run_simmetrix_license(license_path=CFG["simmetrix"]["license"])
    runner.run_simmetrix_meshing(
        vtp_path=vtp_path,
        exe_dir=meshdir,
        simmetrix_path=CFG["simmetrix"]["meshing"],
        grid_size=float(grid),
        boundary_layer_thickness=BL_THICKNESS,
        number_of_boundary_layers=BL_LAYERS,
    )
    LumenMeshing._assemble_mesh_tree(meshdir)

    # volume_mesh.sms 는 출력의 대부분이고 이후 단계에서 안 쓰므로 제거 (ws1 디스크 93%)
    sms = meshdir / "volume_mesh.sms"
    if sms.exists():
        mb = sms.stat().st_size / 1e6
        sms.unlink()
        print(f"[clean] volume_mesh.sms 삭제 ({mb:.0f} MB)", flush=True)

    print(f"[done] grid={grid} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
