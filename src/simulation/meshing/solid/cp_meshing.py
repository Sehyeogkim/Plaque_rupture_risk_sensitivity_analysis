#!/usr/bin/env python3
"""Mesh ONE calcified-plaque (Type-2) case and keep only the final .msh.

Pipeline is HXT_meshing_II_main's, with the size/algorithm policy the user set:
  * never finer than 0.05  -- sizes walk 0.05 -> coarser, never 0.045/0.04
  * hard cap 5M tets
  * HXT first, Delaunay as the fallback (per-size, then per-algorithm)
Everything under type2_mesh/ is deleted afterwards, pass or fail; only
CP_mesh/solid_CP_<case>.msh survives. These meshes are ~0.5-1 GB each, so the
scratch has to go or the disk fills within a few dozen cases.

Usage: cp_meshing.py <case> [nproc]
Env:   SIZES (default "0.05,0.055,0.06,0.065,0.07"), ALGOS (default "10,1"),
       MAXTET (default 5000000)
Prints one final `STATUS {json}` line.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMULATION = HERE.parents[1]
sys.path.insert(0, str(SIMULATION / 'lib'))
sys.path.insert(0, str(SIMULATION / 'geometry' / 'calcification'))
os.chdir(HERE)

CP_OUT = HERE / 'CP_mesh'
CP_OUT.mkdir(exist_ok=True)


def main():
    case = int(sys.argv[1])
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    sizes = [float(s) for s in os.environ.get('SIZES', '0.05,0.055,0.06,0.065,0.07').split(',')]
    algos = [int(a) for a in os.environ.get('ALGOS', '10,1').split(',')]
    max_tet = int(os.environ.get('MAXTET', 5_000_000))

    final = CP_OUT / f'solid_CP_{case}.msh'
    if final.exists():
        print('STATUS ' + json.dumps({'case': case, 'status': 'skipped'}))
        return

    t0 = time.time()
    from HXT_type2_utils import HXT_mesh_II
    m = HXT_mesh_II(case_index=case, nproc=nproc)
    rst = m.rst_dir

    def cleanup():
        shutil.rmtree(rst, ignore_errors=True)

    def fail(stage, err):
        cleanup()
        print('STATUS ' + json.dumps({'case': case, 'status': 'failed', 'stage': stage,
                                      'error': str(err)[:200],
                                      'seconds': round(time.time() - t0, 1)}))

    buf = io.StringIO()
    # ---- geometry prep: lipid stl, calcification carve, smooth, back to STEP ----
    try:
        with contextlib.redirect_stdout(buf):
            HXT_mesh_II.get_lipid_stl(m.lipid_path, m.fc_path, m.lipid_stl_path)
            HXT_mesh_II.create_cal_mesh(m.lipid_path, m.fc_offset_path,
                                        m.raw_cal_msh_path, m.nproc)
            m.Voronoi_tesselation_KDTREE(m.raw_cal_msh_path)
            # 'laplacian' is what the June production run used
            # (batch_production.py passes it explicitly); it shrinks inward, so the
            # smoothed calcification stays inside the lipid. 'taubin' preserves
            # volume and pushed the surface out, tripping does_ca_inside_lipid on
            # ~half the cases. Try laplacian first, taubin only as a fallback.
            inside = False
            for mode in os.environ.get('SMOOTH', 'laplacian,taubin').split(','):
                m.smooth_vtu_to_stl(m.prog_cal_vtu_path, m.smooth_cal_stl_path,
                                    smooth_mode=mode)
                inside = HXT_mesh_II.does_ca_inside_lipid(m.lipid_stl_path,
                                                          m.smooth_cal_stl_path)
                if inside:
                    break
        if not inside:
            return fail('ca_outside_lipid', 'calcification not inside lipid core')
        with contextlib.redirect_stdout(buf):
            HXT_mesh_II.scale_stl(m.smooth_cal_stl_path, m.scaled_smo_cal_stl_path,
                                  scale_factor=10.0)
            HXT_mesh_II.stl_to_step(m.scaled_smo_cal_stl_path, m.smooth_cal_step_path,
                                    m.scripts_dir)
    except Exception as e:
        return fail('geometry', e)

    # ---- size/algorithm sweep: first combination inside budget wins ----
    # The 600 s budget covers ONLY the meshing sweep. Calcification generation
    # (voxelise -> smooth -> STL -> STEP via FreeCAD) runs before this and is not
    # charged against it -- that stage is deterministic and can't be sped up by
    # giving up early, whereas a gmsh attempt can hang indefinitely.
    mesh_budget = float(os.environ.get('MESH_BUDGET', 600))
    sweep_t0 = time.time()
    tried = []
    for algo in algos:
        for size in sizes:
            if time.time() - sweep_t0 > mesh_budget:
                cleanup()
                print('STATUS ' + json.dumps({'case': case, 'status': 'mesh_budget',
                       'error': 'mesh sweep exceeded %ds' % mesh_budget,
                       'tried': tried, 'seconds': round(time.time() - t0, 1)}))
                return
            try:
                with contextlib.redirect_stdout(buf):
                    st = m.solid_gmshing(mesh_size=size, nproc=nproc, order=2,
                                         second_order_linear=True, high_order_optimize=0,
                                         box_vout=0.2, optimize=True, write=True,
                                         algorithm_3d=algo)
            except BaseException as e:
                # gmsh does NOT raise when HXT/Delaunay gives up -- it logs and
                # writes an empty .msh, then get_mesh_info's meshio read fails and
                # calls sys.exit. SystemExit is not an Exception, so catching only
                # Exception killed the whole case at the first size and skipped the
                # rest of the sweep.
                if isinstance(e, KeyboardInterrupt):
                    raise
                tried.append({'algo': algo, 'size': size, 'err': type(e).__name__ + ':' + str(e)[:70]})
                continue
            ok = (0 < st['n_tet'] < max_tet and st['n_negative'] == 0 and st['min_quality'] > 0)
            tried.append({'algo': algo, 'size': size, 'n_tet': st['n_tet'],
                          'min_q': round(st['min_quality'], 5), 'neg': st['n_negative'],
                          'ok': ok})
            if ok:
                src = m.final_solid_msh_path
                if not src.exists():
                    return fail('write', 'gmsh reported success but no .msh on disk')
                shutil.move(str(src), str(final))
                cleanup()
                print('STATUS ' + json.dumps({
                    'case': case, 'status': 'done', 'algo': algo, 'mesh_size': size,
                    'n_tet': st['n_tet'], 'min_quality': round(st['min_quality'], 5),
                    'mb': round(final.stat().st_size / 1048576, 1),
                    'seconds': round(time.time() - t0, 1), 'tried': tried}))
                return

    cleanup()
    print('STATUS ' + json.dumps({'case': case, 'status': 'exhausted', 'tried': tried,
                                  'seconds': round(time.time() - t0, 1)}))


if __name__ == '__main__':
    main()
