"""
Figure 6 - clinical-consistency grid: VI (PSS / delta-PSS) against five independently
measured or prescribed quantities.  2 rows x 5 cols.

    col 0: PB(%)   col 1: delta_FFR   col 2: E_lipid   col 3: alpha   col 4: f_calc

Source: outputs/master_{LAP,CP}.csv (run build_master.py first); every
valid case is plotted.  The calcification column is drawn from CP, the only
cohort in which calcification exists; the other four columns are LAP.
Panel titles give Pearson r and Spearman rho; only the sign of rho is used
for the directional-consistency check.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from config import OUTPUTS

OUT_DIR = str(OUTPUTS / "figures")
os.makedirs(OUT_DIR, exist_ok=True)

lap = pd.read_csv(OUTPUTS / "master_LAP.csv")
cp = pd.read_csv(OUTPUTS / "master_CP.csv")
print(f"LAP n={len(lap)}   CP n={len(cp)}   (raw, no filtering)")

STRESS_ROWS = [('PSS', 'PSS', 'PSS [kPa]'),
               ('del_PSS', r'$\Delta$PSS', r'$\Delta$PSS [kPa]')]
DYN_TO_KPA = 1e4

# (xcol, xlabel, dataframe, cohort tag)
COLUMNS = [
    ('PB(%)',     'PB [%]',                                     lap, 'LAP'),
    ('delta_FFR', r'$\Delta$FFR',                               lap, 'LAP'),
    ('E_lipid',   r'$E_{lipid}$ [kPa]',                         lap, 'LAP'),
    ('alpha',     r'$\theta_{\mathrm{lipid,max}}$ [$\degree$]', lap, 'LAP'),
    ('fraction',  r'$f_{\mathrm{calc}}$',                       cp,  'CP'),
]

NROWS, NCOLS = len(STRESS_ROWS), len(COLUMNS)

fig = plt.figure(figsize=(25, NROWS * 4.5))
gs = gridspec.GridSpec(NROWS, NCOLS, wspace=0.35, hspace=0.50)
axes = np.empty((NROWS, NCOLS), dtype=object)
for _r in range(NROWS):
    for _c in range(NCOLS):
        axes[_r, _c] = fig.add_subplot(gs[_r, _c])

for r, (scol, sdisp, ylabel) in enumerate(STRESS_ROWS):
    print(f"\n=== row {r}: {sdisp} ===")
    for c, (xcol, xlabel, df_c, tag) in enumerate(COLUMNS):
        d = df_c[[xcol, scol]].dropna()
        x_vals = d[xcol].to_numpy(float)
        if xcol == 'E_lipid':
            x_vals = x_vals / 1e4
        vi = d[scol].to_numpy(float) / DYN_TO_KPA
        ax = axes[r, c]

        r_p, p_p = stats.pearsonr(x_vals, vi)
        r_s, p_s = stats.spearmanr(x_vals, vi)
        slope, intercept, _, _, _ = stats.linregress(x_vals, vi)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)

        ax.scatter(x_vals, vi, s=10, alpha=0.4)
        ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=2)

        ax.set_xlabel(xlabel, fontsize=31)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=31)
        ax.set_title(rf'r={r_p:.3f}, $\rho$={r_s:.3f}', fontsize=31)
        ax.tick_params(labelsize=28)
        ax.xaxis.get_offset_text().set_fontsize(26)
        ax.yaxis.get_offset_text().set_fontsize(26)

        if xcol == 'PB(%)':
            ax.set_xticks([60, 70, 80, 90])
        if scol == 'PSS':
            ax.set_yticks([100, 300, 500])
        else:
            ax.set_yticks([50, 150, 250])

        print(f"  X={xcol:12s} [{tag}]  n={len(d):4d}  "
              f"r={r_p:+.3f} (p={p_p:.2e})  rho={r_s:+.3f} (p={p_s:.2e})")

fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.06)
for ext in ("png", "svg"):
    p = os.path.join(OUT_DIR, f"fig6_consistency.{ext}")
    plt.savefig(p, dpi=150 if ext == "png" else None, bbox_inches='tight')
    print("saved", p)
plt.close()

# the calcification column, also computed on LAP, for reference
print("\n  f_calc on LAP (calcification is inert in this phenotype):")
for scol, sdisp in (('PSS', 'PSS'), ('del_PSS', 'dPSS')):
    d = lap[['fraction', scol]].dropna()
    r_p, p_p = stats.pearsonr(d.fraction, d[scol])
    r_s, p_s = stats.spearmanr(d.fraction, d[scol])
    print(f"    {sdisp:<6} r={r_p:+.3f} (p={p_p:.2e})  rho={r_s:+.3f} (p={p_s:.2e})")
