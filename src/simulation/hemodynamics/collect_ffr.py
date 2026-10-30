"""
Aggregate hyperemic FFR from every case's 1D result into a single FFR.csv.

Reads:  fluid_0723/case_<i>/pulsatile_1d_hyperemia/_1d_rst.json
        (structure: { "<i>": { "systolic":..., "diastolic":..., "mean":..., "FFR": <float> } })
Writes: FFR.csv  (case_id, FFR, status)
"""
import json
import csv
from pathlib import Path

ROOT = Path("/home/jeff/project/65_final_0723")
N = 384

rows = []
ok = 0
for i in range(N):
    j = ROOT / f"fluid_0723/case_{i}/pulsatile_1d_hyperemia/_1d_rst.json"
    ffr, status = "", "missing"
    if j.exists():
        try:
            d = json.load(open(j))
            d = d.get(str(i), d)          # unwrap {case_id: {...}}
            if "FFR" in d:
                ffr = round(float(d["FFR"]), 6)
                status = "ok"
                ok += 1
            else:
                status = "no_FFR_key"
        except Exception as e:
            status = f"ERR:{e}"
    rows.append((i, ffr, status))

out = ROOT / "FFR.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "FFR", "status"])
    w.writerows(rows)

print(f"wrote {out}  ({ok}/{N} cases have FFR)")
