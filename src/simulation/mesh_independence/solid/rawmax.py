#!/usr/bin/env python
"""Radius-free cap-stress metrics compared across meshes.

Candidates that need NO averaging radius (unlike PSS):
  raw max      : max nodal principal stress on the cap
  p99 / p99.9  : upper percentiles of the same nodal population (singularity-robust)

Also reports WHERE each metric lands, because a single-node max can be
location-noisy even when its value is stable.

Usage: rawmax.py <label> [<label> ...]
       reads  <RESULTS>/FC_case_<CASE>_a6_<label>_{peak,low}.vtu
"""
import sys
import numpy as np
import pyvista as pv

RESULTS = '/mnt/hdd/jeff/mesh_indep_solid/results'
CASE = 0
Z_EPS = 0.02


def cap(m):
    if 'WALL_IN_FC' in m.point_data:
        return np.asarray(m.point_data['WALL_IN_FC']).astype(bool)
    return np.asarray(m.point_data['Group']) == 5


rows = []
for lab in sys.argv[1:]:
    pk = pv.read(f'{RESULTS}/FC_case_{CASE}_a6_{lab}_peak.vtu')
    lo = pv.read(f'{RESULTS}/FC_case_{CASE}_a6_{lab}_low.vtu')
    mw = cap(pk)
    P = np.asarray(pk.points)
    wz = P[mw][:, 2]
    mz = (P[:, 2] > wz.min() + Z_EPS) & (P[:, 2] < wz.max() - Z_EPS)
    gz = np.where(mw & mz)[0]

    sp = np.asarray(pk.point_data['Principal_stress'], float)[gz]
    sl = np.asarray(lo.point_data['Principal_stress'], float)[gz]
    sd = sp - sl                      # full cyclic range, matching del_PSS
    pts = P[gz]
    v = ~np.isnan(sp)
    vi = np.where(v)[0]

    r = {'label': lab, 'n': int(v.sum())}
    for nm, f in (('peak', sp), ('delta', sd)):
        fv = f[vi]
        k = vi[int(np.nanargmax(fv))]
        r[nm + '_max'] = float(f[k])
        r[nm + '_xyz'] = pts[k]
        r[nm + '_p99'] = float(np.nanpercentile(fv, 99))
        r[nm + '_p999'] = float(np.nanpercentile(fv, 99.9))
    rows.append(r)

for nm in ('peak', 'delta'):
    print(f'===== {nm} =====')
    print(f"{'mesh':10s}{'n_cap':>9s}{'raw max':>13s}{'p99.9':>13s}{'p99':>13s}")
    for r in rows:
        print(f"{r['label']:10s}{r['n']:9d}{r[nm+'_max']:13.4e}"
              f"{r[nm+'_p999']:13.4e}{r[nm+'_p99']:13.4e}")
    b = rows[0]
    for r in rows[1:]:
        d = np.linalg.norm(r[nm + '_xyz'] - b[nm + '_xyz']) * 10
        print(f"   argmax location {r['label']} vs {b['label']}: {d:.4f} mm")
    for key, lbl in (('_max', 'raw max'), ('_p999', 'p99.9'), ('_p99', 'p99')):
        vals = [r[nm + key] for r in rows]
        print(f'   {lbl:8s} spread max/min-1 = {(max(vals)/min(vals)-1)*100:.2f} %')
