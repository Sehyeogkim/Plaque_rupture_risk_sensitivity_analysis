#!/usr/bin/env python3
"""Batch LAP mesh CHECK (validation mode) over many cases.

Runs wsl_lap_check.py as a SUBPROCESS per case so a gmsh hard-crash kills only
that case, and parses its final `STATUS {json}` line. The worker discards the
.msh itself -- we keep only verdict + quality. Results are appended to the CSV
as each case finishes, so killing the run mid-way still leaves partial results
(and re-running skips what is already in the CSV).

Usage:
  lap_mesh_check.py <geo_root> <out_csv> [--cases 3,8,13] [--workers 4]
                   [--nproc 3] [--size 0.055] [--algo3d 10] [--timeout 1800]

Default cases = every case_<id> dir under geo_root that has all 4 STEP files.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / os.environ.get("LAP_WORKER", "wsl_lap_check.py")
PYTHON = "/home/jeff/miniconda3/bin/python3"
NEED = ["lumen.stp", "solid.stp", "lipid.stp", "fc.stp"]
FIELDS = ["case", "status", "mesh_size", "algo3d", "num_elements", "min_quality",
          "neg_jacobian", "seconds", "error"]
STATUS_RE = re.compile(r"^STATUS (\{.*\})\s*$", re.M)


def discover(geo_root):
    cases = []
    for d in sorted(geo_root.glob("case_*")):
        if all((d / f).exists() for f in NEED):
            try:
                cases.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return sorted(cases)


def run_one(geo_root, case, nproc, size, algo3d, timeout, log_dir):
    env = dict(os.environ, ALGO3D=str(algo3d), PYTHONPATH=str(HERE / "src"))
    cmd = [PYTHON, "-u", str(WORKER), str(geo_root), str(case), str(nproc), str(size)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = p.stdout or ""
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        (log_dir / f"case_{case}.log").write_text(out[-20000:])
        return {"case": case, "status": "timeout", "mesh_size": size, "algo3d": algo3d,
                "seconds": round(time.time() - t0, 1), "error": f"timeout>{timeout}s"}

    # keep the full gmsh log only when something went wrong -- passes are boring
    m = STATUS_RE.search(out)
    rec = {"case": case, "mesh_size": size, "algo3d": algo3d}
    if m:
        try:
            rec.update(json.loads(m.group(1)))
        except json.JSONDecodeError:
            rec.update({"status": "parse_error", "error": m.group(1)[:200]})
    else:
        rec.update({"status": "crashed", "error": f"no STATUS line (rc={p.returncode})",
                    "seconds": round(time.time() - t0, 1)})
    if rec.get("status") != "done":
        (log_dir / f"case_{case}.log").write_text(out[-20000:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geo_root", type=Path)
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nproc", type=int, default=3)
    ap.add_argument("--size", type=float, default=0.055)
    ap.add_argument("--algo3d", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    cases = ([int(x) for x in a.cases.replace(" ", "").split(",") if x]
             if a.cases else discover(a.geo_root))

    done = set()
    if a.out_csv.exists():
        with a.out_csv.open() as f:
            done = {int(r["case"]) for r in csv.DictReader(f) if r.get("case")}
    todo = [c for c in cases if c not in done]
    print(f"cases={len(cases)} already_done={len(done)} todo={len(todo)} "
          f"workers={a.workers} nproc={a.nproc} size={a.size} algo3d={a.algo3d}",
          flush=True)
    if not todo:
        return

    log_dir = a.out_csv.parent / (a.out_csv.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    new = not a.out_csv.exists()
    t0 = time.time()
    with a.out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
            f.flush()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_one, a.geo_root, c, a.nproc, a.size, a.algo3d,
                              a.timeout, log_dir): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                w.writerow(rec)
                f.flush()
                print(f"[{i}/{len(todo)}] case {rec['case']}: {rec.get('status')} "
                      f"el={rec.get('num_elements')} minq={rec.get('min_quality')} "
                      f"negJ={rec.get('neg_jacobian')} {rec.get('seconds')}s "
                      f"{rec.get('error', '')}", flush=True)

    with a.out_csv.open() as f:
        rows = list(csv.DictReader(f))
    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"\nDONE in {round(time.time() - t0)}s  ({len(rows)} cases total)")
    for k in sorted(tally):
        print(f"  {k:10s} {tally[k]}")
    bad = [r["case"] for r in rows if r["status"] != "done"]
    print(f"  not-done cases: {','.join(bad) if bad else 'none'}")
    print(f"  failure logs: {log_dir}")


if __name__ == "__main__":
    main()
