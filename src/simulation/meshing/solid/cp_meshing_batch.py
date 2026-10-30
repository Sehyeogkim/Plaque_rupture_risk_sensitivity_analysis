#!/usr/bin/env python3
"""Overnight driver: mesh the 227 CP cases that have no solid_CP_<id>.msh yet.

Runs cp_meshing.py per case as a subprocess (so a gmsh segfault costs one
case, not the run) and retries whatever is left with progressively coarser /
different-algorithm strategies. Never goes below size 0.05 and never above 5M
tets -- both are hard constraints from the user.

Stops as soon as TARGET_NEW meshes exist, or when a whole round adds nothing new
(no strategy left that helps).

Results stream to cp_batch_results.csv; per-case logs are kept only for failures.
"""
import csv
import json
import os
import re
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / 'cp_meshing.py'
PYTHON = '/home/jeff/miniconda3/bin/python3'
OUT_CSV = HERE / 'cp_batch_results.csv'
LOG_DIR = HERE / 'cp_batch_logs'
LOG_DIR.mkdir(exist_ok=True)
STATUS_RE = re.compile(r'^STATUS (\{.*\})\s*$', re.M)

TARGET_NEW = int(os.environ.get("TARGET_NEW", 400))
WORKERS = int(os.environ.get('WORKERS', 2))
NPROC = int(os.environ.get('NPROC', 7))
CASE_TIMEOUT = int(os.environ.get('CASE_TIMEOUT', 1500))

# (sizes, algos) per round -- coarser and/or different 3D algorithm each time
ROUNDS = [
    ('0.065,0.06,0.055,0.05', '10,1'),
    ('0.065,0.06', '1,10'),
]

FIELDS = ['round', 'case', 'status', 'algo', 'mesh_size', 'n_tet', 'min_quality',
          'mb', 'seconds', 'stage', 'error']


def made():
    return {int(p.stem.split('_')[-1]) for p in (HERE / 'CP_mesh').glob('solid_CP_*.msh')}


def run_case(case, sizes, algos, rnd):
    env = dict(os.environ, SIZES=sizes, ALGOS=algos,
               PATH='/home/jeff/miniconda3/bin:' + os.environ.get('PATH', ''),
               PYTHONPATH=str(HERE.parents[1] / 'lib'))
    cmd = [PYTHON, '-u', str(WORKER), str(case), str(NPROC)]
    t0 = time.time()
    try:
        # A single quadratic CP mesh peaked at 28 GB RSS and the kernel OOM-killed
        # it -- taking other workers down with it. Cap each worker's address space
        # so a runaway attempt dies as a normal MemoryError and the batch moves on.
        mem_cap = int(os.environ.get('MEM_CAP_GB', 13)) * 1024**3
        def _limit():
            resource.setrlimit(resource.RLIMIT_AS, (mem_cap, mem_cap))
        p = subprocess.run(cmd, cwd=str(HERE), env=env, timeout=CASE_TIMEOUT,
                           preexec_fn=_limit,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = p.stdout or ''
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ''
        (LOG_DIR / f'case_{case}_r{rnd}.log').write_text(out[-20000:])
        return {'round': rnd, 'case': case, 'status': 'timeout',
                'seconds': round(time.time() - t0, 1), 'error': f'>{CASE_TIMEOUT}s'}
    m = STATUS_RE.search(out)
    rec = {'round': rnd, 'case': case}
    if m:
        try:
            rec.update(json.loads(m.group(1)))
        except json.JSONDecodeError:
            rec.update({'status': 'parse_error'})
    else:
        rec.update({'status': 'crashed', 'error': f'no STATUS (rc={p.returncode})',
                    'seconds': round(time.time() - t0, 1)})
    if rec.get('status') != 'done':
        (LOG_DIR / f'case_{case}_r{rnd}.log').write_text(out[-20000:])
    rec.pop('tried', None)
    return rec


def main():
    todo_all = [int(x) for x in (HERE / 'missing_cases.txt').read_text().split(',')]
    start_made = made()
    print(f'missing={len(todo_all)}  already made in WSL={len(start_made)}  '
          f'target_new={TARGET_NEW}  workers={WORKERS} nproc={NPROC}', flush=True)

    new = not OUT_CSV.exists()
    f = OUT_CSV.open('a', newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    if new:
        w.writeheader(); f.flush()

    for rnd, (sizes, algos) in enumerate(ROUNDS, 1):
        have = made()
        if len(have) >= TARGET_NEW:
            print(f'target reached ({len(have)}/{TARGET_NEW}) — stopping', flush=True)
            break
        todo = [c for c in todo_all if c not in have]
        print(f'\n=== round {rnd}: sizes={sizes} algos={algos} — {len(todo)} cases '
              f'(have {len(have)}/{TARGET_NEW}) ===', flush=True)
        before = len(have)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(run_case, c, sizes, algos, rnd): c for c in todo}
            done_n = 0
            for fut in as_completed(futs):
                rec = fut.result()
                done_n += 1
                w.writerow(rec); f.flush()
                print(f'[r{rnd} {done_n}/{len(todo)}] case {rec["case"]}: {rec.get("status")} '
                      f'size={rec.get("mesh_size")} n_tet={rec.get("n_tet")} '
                      f'minq={rec.get("min_quality")} {rec.get("seconds")}s '
                      f'{rec.get("error", "")}', flush=True)
                if len(made()) >= TARGET_NEW:
                    print('target reached mid-round — letting running cases finish',
                          flush=True)
                    break

        gained = len(made()) - before
        print(f'=== round {rnd} done: +{gained} new (total {len(made())}/{TARGET_NEW}) ===',
              flush=True)
        if gained == 0:
            print('round added nothing — no strategy left, stopping', flush=True)
            break

    f.close()
    final = made()
    print(f'\nFINISHED. new meshes in WSL: {len(final)}  '
          f'(ws1/ws2 already hold 773 -> total {773 + len(final)}/1000)', flush=True)


if __name__ == '__main__':
    main()
