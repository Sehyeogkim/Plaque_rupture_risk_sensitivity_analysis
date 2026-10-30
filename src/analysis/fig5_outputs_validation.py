"""
Figure 5 - simulated outputs and forward-solver checks (LAP dataset).
  A: PSS / del_PSS distributions   B: DoS vs FFR with clinical references
  C: circumferential index I_theta  D: axial index I_z
  E: PSS / del_PSS vs fibrous-cap thickness (auxiliary validation set)
Layout:
  Row 1: A (left)  | B (right, spans rows 1-2)
  Row 2: C (left)  | B cont.
  Row 3: D (left)  | E (right)
Output: outputs/figures/fig5_outputs_validation.{png,svg}
Run build_master.py first.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DESIGN_CSV, OUTPUTS, SIM, VALID

OUT = str(OUTPUTS / "figures"); os.makedirs(OUT, exist_ok=True)
FC_DIR = str(VALID / "fc_thickness")
REF_DIR = str(VALID / "ffr_literature")
MASTER = str(OUTPUTS / "master_LAP.csv")

STEEL = "steelblue"
plt.rcParams.update({"font.size": 22, "axes.linewidth": 1.3})
TIT, XL, YL, TK, ANN = 26, 24, 24, 20, 20

def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---- load data ----
pssd = pd.read_csv(MASTER)
PSS = pssd["PSS"].values/10000.0; DPSS = pssd["del_PSS"].values/10000.0

inp = pd.read_csv(DESIGN_CSV); ffr = pd.read_csv(SIM / "FFR.csv")
mB = pd.DataFrame({"DOS": inp["DOS"].values, "FFR": ffr["FFR"].values})
mB = mB[(mB.DOS>=0.25)&(mB.DOS<=0.75)].sort_values("DOS")

# ---- Rupture location data: read from the master table, not hard-coded ----
_loc = pd.read_csv(MASTER)
N_LOC = len(_loc)
CIRC_DATA = {k: {"Center": int((_loc[f"{p_}_circum_class"] == "center").sum()),
                 "Shoulder": int((_loc[f"{p_}_circum_class"] == "shoulder").sum())}
             for k, p_ in (("PSS", "PSS"), ("dPSS", "delPSS"))}
AXIAL_DATA = {k: {c: int((_loc[f"{p_}_axial_loc"] == c).sum())
                  for c in ("proximal", "mid", "distal")}
              for k, p_ in (("PSS", "PSS"), ("dPSS", "delPSS"))}

# fc thickness datasets
fc_levels = [
    (os.path.join(FC_DIR, "parameterD_level0.csv"),
     os.path.join(FC_DIR, "PSS_result_expD_stiff0_tag5_lumenFC.csv"),
     "tomato",    r"$E_{fc}$ = 1.2 MPa", "o"),
    (os.path.join(FC_DIR, "parameterD_level1.csv"),
     os.path.join(FC_DIR, "PSS_result_expD_stiff1_tag5_lumenFC.csv"),
     "steelblue", r"$E_{fc}$ = 1.6 MPa", "s"),
    (os.path.join(FC_DIR, "parameterD_level2.csv"),
     os.path.join(FC_DIR, "PSS_result_expD_stiff2_tag5_lumenFC.csv"),
     "seagreen",  r"$E_{fc}$ = 2.0 MPa", "^"),
]
fc_datasets = []
for inp_path, out_path, color, label, marker in fc_levels:
    inp_fc = pd.read_csv(inp_path).reset_index(drop=False).rename(columns={"index": "case_id"})
    pss_fc = pd.read_csv(out_path)
    df_fc = pss_fc.merge(inp_fc[["case_id", "fc_av_th", "E_fc"]], on="case_id").sort_values("fc_av_th")
    fc_datasets.append((df_fc, color, label, marker))

# ---- figure layout ----
# 6 rows x 2 cols; left side: A(row0-1), C(row2-3), D(row4-5); right: B(row0-3), E(row4-5)
fig = plt.figure(figsize=(26, 18))
gs = fig.add_gridspec(6, 2, hspace=2.00, wspace=0.30,
                      height_ratios=[1, 1, 1, 1, 1, 1],
                      width_ratios=[1, 1.2])

# A: top-left (rows 0-1)
gsA = gs[0:2, 0].subgridspec(1, 2, wspace=0.4)
axA1 = fig.add_subplot(gsA[0, 0])
axA2 = fig.add_subplot(gsA[0, 1])

# B: right side spanning rows 0-3 (DoS vs FFR)
axB = fig.add_subplot(gs[0:4, 1])

# C: middle-left (rows 2-3)
gsC = gs[2:4, 0].subgridspec(1, 2, wspace=0.4)
axC1 = fig.add_subplot(gsC[0, 0])
axC2 = fig.add_subplot(gsC[0, 1])

# D: bottom-left (rows 4-5)
gsD = gs[4:6, 0].subgridspec(1, 2, wspace=0.4)
axD1 = fig.add_subplot(gsD[0, 0])
axD2 = fig.add_subplot(gsD[0, 1])

# E: bottom-right (rows 4-5)
gsE = gs[4:6, 1].subgridspec(1, 2, wspace=0.5)
axE1 = fig.add_subplot(gsE[0, 0])
axE2 = fig.add_subplot(gsE[0, 1])

# ---- A: PSS / dPSS histograms ----
for a, vals, xl in [(axA1, PSS, "PSS [kPa]"),
                    (axA2, DPSS, r"$\Delta$PSS [kPa]")]:
    a.hist(vals, bins=30, color=STEEL, edgecolor="white", linewidth=0.6)
    a.set_xlabel(xl, fontsize=XL)
    a.set_ylabel("Count", fontsize=YL); a.tick_params(labelsize=TK); despine(a)

# ---- B: DoS vs FFR ----
edges = np.linspace(0.25, 0.75, 11); cent = 0.5*(edges[:-1]+edges[1:])
med=[]; q1=[]; q3=[]
for lo,hi in zip(edges[:-1], edges[1:]):
    s = mB[(mB.DOS>=lo)&(mB.DOS<hi)]["FFR"]
    med.append(s.median()); q1.append(s.quantile(.25)); q3.append(s.quantile(.75))
med=np.array(med); q1=np.array(q1); q3=np.array(q3)
axB.fill_between(cent, q1, q3, color="#7CB9E8", alpha=0.35, zorder=1, label="Simulation (IQR)")
axB.plot(cent, med, color="#1f6fb2", lw=3.0, marker="o", ms=12, zorder=4, label="Simulation (median)")
for fn,lab,col,mk in [("kristensen.csv","Kristensen et al.","red","s"),
                      ("dai.csv","Dai et al.","green","^"),
                      ("nijjer.csv","Nijjer et al.","orange","D"),
                      ("lee.csv","Lee et al.","purple","v")]:
    r = pd.read_csv(os.path.join(REF_DIR, fn)); r = r[(r.DOS>=0.25)&(r.DOS<=0.75)]
    axB.plot(r.DOS, r.FFR, lw=2.0, marker=mk, ms=10, markevery=2, color=col, label=lab,
             markeredgecolor="white", markeredgewidth=0.6, zorder=3)
axB.set_xlim(0.25, 0.75); axB.set_xlabel("DoS", fontsize=XL); axB.set_ylabel("FFR", fontsize=XL)
axB.legend(fontsize=20, loc="lower left", framealpha=0.9); axB.tick_params(labelsize=TK); despine(axB)

# ---- C: circumferential index distribution (threshold 0.5) ----
C_PSS, C_DPS = "#2a78d6", "#eb6834"
THIRD = 1.0 / 3.0


def _dist(a, vals, bins, thresholds, colr, xlabel, xticks, xticklabels,
          shade=None, edge=None):
    """Histogram of a location index with its classification threshold drawn."""
    if shade is not None:
        a.axvspan(*shade, color="#9a9994", alpha=0.06, lw=0, zorder=0)
    for th in thresholds:
        a.axvline(th, color="#6f6e6a", lw=1.8, zorder=1)
    if edge is not None:
        a.axvline(edge, color="#b8b7b3", lw=1.4, ls=(0, (4, 4)), zorder=1)
    ctr = (bins[:-1] + bins[1:]) / 2
    h, _ = np.histogram(vals, bins=bins)
    pct = h / len(vals) * 100
    a.step(ctr, pct, where="mid", color=colr, lw=2.6, zorder=3)
    a.fill_between(ctr, pct, step="mid", color=colr, alpha=0.18, lw=0, zorder=2)
    a.set_xlabel(xlabel, fontsize=XL)
    a.set_ylabel("Cases [%]", fontsize=YL)
    a.set_xticks(xticks)
    a.set_xticklabels(xticklabels, fontsize=TK - 2)
    a.set_xlim(bins[0], bins[-1])
    a.set_ylim(bottom=0)
    a.tick_params(labelsize=TK)
    despine(a)


# axis stops at the arc edge; the few peaks with I_theta > 1 sit outside the
# lipid arc and are left off the plot (percentages stay out of the full n)
CIRC_BINS = np.arange(0, 1.025, 0.025)
AX_BINS = np.linspace(-1, 1, 41)
for a, col, colr in [(axC1, "PSS_circum_index", C_PSS),
                     (axC2, "delPSS_circum_index", C_DPS)]:
    _dist(a, _loc[col].to_numpy(), CIRC_BINS, [0.5], colr,
          r"$I_\theta$", [0, 0.5, 1.0],
          ["0\ncenter", "0.5", "1.0\nshoulder"],
          shade=(0.5, 1.0))

# ---- D: axial index distribution (thresholds +/-1/3) ----
for a, col, colr in [(axD1, "PSS_loc_index", C_PSS),
                     (axD2, "delPSS_loc_index", C_DPS)]:
    _dist(a, _loc[col].to_numpy(), AX_BINS, [-THIRD, THIRD], colr,
          r"$I_z$", [-1, -THIRD, 0, THIRD, 1],
          ["$-1$\nprox.", "$-1/3$", "0\nMLA", "$+1/3$", "$+1$\ndistal"])

# ---- E: VI vs fibrous cap thickness (kPa) ----
for a, scol in [(axE1, "PSS_Principal"), (axE2, "del_PSS_Principal")]:
    for df_fc, color, label, marker in fc_datasets:
        x = df_fc["fc_av_th"].values * 10000
        y = df_fc[scol].values / 10000.0
        a.plot(x, y, marker=marker, linestyle="-", color=color, linewidth=2, markersize=9,
               markerfacecolor="white", markeredgewidth=2, label=label)
    a.set_xlabel(r"$t_{FC}$ [$\mu$m]", fontsize=XL)
    ylabel = "PSS" if scol == "PSS_Principal" else r"$\Delta$PSS"
    a.set_ylabel(rf"{ylabel} [kPa]", fontsize=YL)
    a.tick_params(labelsize=TK); despine(a)
    a.legend(fontsize=19, framealpha=0.9)
    a.ticklabel_format(axis="y", style="scientific", scilimits=(0,0))
    a.yaxis.get_offset_text().set_fontsize(TK)

fig.savefig(f"{OUT}/fig5_outputs_validation.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUT}/fig5_outputs_validation.svg", bbox_inches="tight")
plt.close(fig)
print(f"saved -> {OUT}/fig5_outputs_validation.png/.svg")
