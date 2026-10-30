"""
Fluid mesh-independence 분석 (case 0).

입력
  wall/{phase}/wall_{phase}_{grid}.csv    x,y,z,Tx,Ty,Tz,Pressure   (solid 에 들어가는 하중 T)
  wall_raw/{phase}/raw_{phase}_{grid}.csv x,y,z,P,Rx,Ry,Rz          (vtp 원본: 압력 + 점성 traction)

주의: T = P*n + rinplane 인데 n 은 pyvista 가 재계산한 point normal 이라
      rinplane 과 정확히 직교하지 않는다. 따라서 |WSS| 를 sqrt(|T|^2 - P^2) 로
      역산하면 안 되고(50% 가 음수가 됨), 반드시 vtp 의 rinplane_traction 을 써야 한다.

지표
  |T|    = sqrt(Tx^2+Ty^2+Tz^2)   solid 하중 (압력 지배)
  |P|    = |pressure|             법선 성분
  |WSS|  = |rinplane_traction|    접선 성분 (메쉬에 가장 민감)

비교: 기준 격자의 벽면 점 위로 IDW(k=4) 보간 후
      상대 L2 error  e(fine,coarse) = ||f_coarse - f_fine|| / ||f_fine||

Usage: analyze_mi.py [--ref 0.01] [--grids 0.01,0.015,0.02]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
PHASES = ["peak", "low"]


def load(phase, grid):
    fT = BASE / "wall" / phase / f"wall_{phase}_{grid}.csv"
    fR = BASE / "wall_raw" / phase / f"raw_{phase}_{grid}.csv"
    if not (fT.exists() and fR.exists()):
        return None
    dT = pd.read_csv(fT)
    dR = pd.read_csv(fR)
    if len(dT) != len(dR):
        raise RuntimeError(f"{phase}/{grid}: 점 개수 불일치 {len(dT)} vs {len(dR)}")
    xyz = dT[["x", "y", "z"]].to_numpy()
    # 두 csv 는 같은 벽면 점집합이지만 순서가 다르다 -> 좌표로 1:1 매칭
    d, j = cKDTree(dR[["x", "y", "z"]].to_numpy()).query(xyz, k=1)
    if d.max() > 1e-6:
        raise RuntimeError(f"{phase}/{grid}: 좌표 매칭 실패 (max dist {d.max():.3e})")
    if len(np.unique(j)) != len(j):
        raise RuntimeError(f"{phase}/{grid}: 좌표 매칭이 1:1 이 아님")
    return {
        "xyz": xyz,
        "Tmag": np.linalg.norm(dT[["Tx", "Ty", "Tz"]].to_numpy(), axis=1),
        "P": np.abs(dR["P"].to_numpy()[j]),
        "WSS": np.linalg.norm(dR[["Rx", "Ry", "Rz"]].to_numpy()[j], axis=1),
        "n": len(dT),
    }


def interp(src, dst_xyz, key, k=4):
    tree = cKDTree(src["xyz"])
    dist, idx = tree.query(dst_xyz, k=k)
    w = 1.0 / np.maximum(dist, 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    return (src[key][idx] * w).sum(axis=1)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="0.01")
    ap.add_argument("--grids", default="0.01,0.015,0.02")
    args = ap.parse_args()
    grids = args.grids.split(",")

    for phase in PHASES:
        data = {g: load(phase, g) for g in grids}
        avail = [g for g in grids if data[g] is not None]
        if len(avail) < 2:
            print(f"\n### {phase}: csv 부족 ({avail})")
            continue

        print(f"\n{'='*78}")
        print(f"### {phase.upper()}    기준 격자 = {args.ref}")
        print(f"{'='*78}")
        print(f"{'grid':>7s} {'wall pts':>9s} {'mean|T|':>12s} {'mean|P|':>12s} "
              f"{'mean|WSS|':>10s} {'max|WSS|':>10s}")
        for g in avail:
            d = data[g]
            print(f"{g:>7s} {d['n']:>9,} {d['Tmag'].mean():>12,.1f} {d['P'].mean():>12,.1f} "
                  f"{d['WSS'].mean():>10.3f} {d['WSS'].max():>10.2f}")

        ref_xyz = data[args.ref]["xyz"]
        proj = {g: {k: interp(data[g], ref_xyz, k) for k in ("Tmag", "P", "WSS")} for g in avail}

        order = sorted(avail, key=float)      # 조밀 -> 성긴
        print(f"\n  상대 L2 error  (조밀한 쪽 기준으로 정규화)")
        print(f"  {'pair':>17s} {'|T|':>12s} {'|P|':>12s} {'|WSS|':>12s}")
        rows = {}
        for fine, coarse in zip(order[:-1], order[1:]):
            r = {k: rel_l2(proj[coarse][k], proj[fine][k]) for k in ("Tmag", "P", "WSS")}
            rows[(fine, coarse)] = r
            print(f"  {'e(%s,%s)' % (fine, coarse):>17s} " +
                  " ".join(f"{r[k]*100:>11.4f}%" for k in ("Tmag", "P", "WSS")))

        if len(order) >= 3:
            f1, f2 = (order[0], order[1]), (order[1], order[2])
            print()
            for k, lbl in (("Tmag", "|T|"), ("WSS", "|WSS|")):
                a, b = rows[f1][k], rows[f2][k]
                ratio = b / a if a else float("nan")
                print(f"  {lbl:>6s}:  e{f1} = {a*100:8.4f}%   e{f2} = {b*100:8.4f}%   -> {ratio:5.2f}x")


if __name__ == "__main__":
    main()
