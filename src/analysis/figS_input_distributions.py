"""
Supplementary figure - distribution of every design input, blocked by category.

One histogram per panel: the prescribed Latin-hypercube design (n = 1000).
x-axis carries the parameter symbol and its unit, y-axis the frequency.
Blocks: lesion morphology / calcification / material / hemodynamics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import DESIGN_CSV, OUTPUTS

OUT = str(OUTPUTS / "figures")
os.makedirs(OUT, exist_ok=True)

C_BAR = "#4A82C4"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b8b7b3"

design = pd.read_csv(DESIGN_CSV)
print(f"LHS design n={len(design)}")

# (column, symbol, unit, scale factor)
GROUPS = [
    ("Lesion morphology", [
        ('DOS',                  r'$DoS$',                          '',           1.0),
        ('lesion_length',        r'$L_{\mathrm{lesion}}$',          'cm',         1.0),
        ('lumen_axial_skewness', r'$\gamma_z$',                     '',           1.0),
        ('alpha',                r'$\theta_{\mathrm{lipid,max}}$',  r'$\degree$', 1.0),
        ('lipid_length_ratio',   r'$l_{\mathrm{lipid}}$',           '',           1.0),
        ('PI',                   r'$PI$',                           '',           1.0),
    ]),
    ("Calcification", [
        ('fraction',             r'$f_{\mathrm{calc}}$',            '',           1.0),
        ('d_fc_ca',              r'$d_{\mathrm{buf}}$',             'cm',         1.0),
        ('ca_axial_skewness',    r'$s_{\mathrm{ax}}$',              '',           1.0),
        ('ca_shoulder_skewness', r'$s_{\mathrm{circ}}$',            '',           1.0),
        ('ca_strength_ratio',    r'$w$',            '',           1.0),
        ('E_cal',                r'$E_{\mathrm{calc}}$',            'GPa',        1e-10),
    ]),
    ("Material", [
        ('E_vessel',             r'$E_{\mathrm{vessel}}$',          'MPa',        1e-7),
        ('E_fc',                 r'$E_{\mathrm{FC}}$',              'MPa',        1e-7),
        ('E_lipid',              r'$E_{\mathrm{lipid}}$',           'kPa',        1e-4),
    ]),
    ("Hemodynamics", [
        ('SBP',                  r'$P_{\mathrm{sys}}$',             'mmHg',       1.0),
        ('PP',                   r'$\Delta P$',                     'mmHg',       1.0),
        ('tau',                  r'$\tau$',                         's',          1.0),
    ]),
]

NB, NCOL = 25, 3
BLOCK_ROWS = [int(np.ceil(len(p) / NCOL)) for _, p in GROUPS]   # 2, 2, 1, 1

fig = plt.figure(figsize=(14.5, 19.5))
fig.patch.set_facecolor("white")
outer = fig.add_gridspec(len(GROUPS), 1, height_ratios=BLOCK_ROWS, hspace=0.42)

rows = []
for gi, ((gname, params), nrow) in enumerate(zip(GROUPS, BLOCK_ROWS)):
    inner = outer[gi].subgridspec(nrow, NCOL, hspace=0.62, wspace=0.30)
    axs = []
    for j, (col, sym, unit, k) in enumerate(params):
        ax = fig.add_subplot(inner[j // NCOL, j % NCOL])
        axs.append(ax)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c4")

        v = design[col].to_numpy(float) * k
        ax.hist(v, bins=NB, color=C_BAR, edgecolor="white", linewidth=0.6, zorder=2)
        ax.grid(axis="y", color="#dedddaa0", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.set_xlabel(f"{sym}  [{unit}]" if unit else sym,
                      fontsize=15, color=INK, labelpad=7)
        if j % NCOL == 0:
            ax.set_ylabel("Frequency", fontsize=13, color=INK2)
        ax.tick_params(colors=INK2, labelsize=11)

        rows.append({"group": gname, "parameter": col, "symbol": sym,
                     "unit": unit, "min": v.min(), "max": v.max(),
                     "mean": v.mean(), "std": v.std(ddof=1)})

    # block header: a rule across the figure with the category name on it
    fig.canvas.draw()
    top = max(a.get_position().y1 for a in axs)
    y = top + 0.019
    fig.add_artist(Line2D([0.055, 0.985], [y, y], color=RULE, lw=1.1,
                          transform=fig.transFigure))
    fig.text(0.055, y + 0.005, gname, ha="left", va="bottom",
             fontsize=16, color=INK, fontweight="semibold")

for ext in ("svg", "png"):
    p = os.path.join(OUT, f"figS_input_distributions.{ext}")
    fig.savefig(p, format=ext, dpi=190 if ext == "png" else None,
                facecolor="white", bbox_inches="tight")
    print("saved", p)
plt.close(fig)

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "figS_input_ranges.csv"), index=False)
print("saved", os.path.join(OUT, "figS_input_ranges.csv"))
pd.set_option("display.width", 200)
print("\n" + tab.round(3).to_string(index=False))
