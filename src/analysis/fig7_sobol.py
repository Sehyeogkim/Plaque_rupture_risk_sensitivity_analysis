"""
Figure 7 - Sobol sensitivity indices.

Source: sobol_{ind,grp}.csv written by gpr_sobol.py, physical scale
    scale == 'orig'  (surrogate fitted on log PSS, predictions
                      inverse-transformed before Sobol)
By default the freshly computed outputs/sensitivity/ are plotted; pass
--reference to plot the shipped results/sensitivity/ instead.

Emits, into outputs/figures/:
    result2_individual_{LAP,CP}.{png,svg}   1x2  PSS | delta-PSS, horizontal
    result2_group_{LAP,CP}.{png,svg}        1x2  vertical grouped bars
    result2_combined_individual.{png,svg}   2x2  row1 LAP, row2 CP
    result2_combined_group.{png,svg}        2x2

Parameters are ordered by first-order index S1.  Bars carry bootstrap 95% CI whiskers (1,000 resamples), matching the Methods.
Indices are the unclipped values (S1_raw / ST_raw); a negative S1 is a
numerical artefact of a near-zero index and is drawn at zero.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUTS, RESULTS

HERE = str(OUTPUTS / "figures")
SENS = (RESULTS if "--reference" in sys.argv else OUTPUTS) / "sensitivity"
os.makedirs(HERE, exist_ok=True)

C_S1 = '#5B7B8A'   # slate     — first-order
C_ST = '#CC785C'   # terracotta — total-order
OUTPUTS = ['PSS', 'del_PSS']
OUT_DISP = {'PSS': 'PSS', 'del_PSS': r'$\Delta$PSS'}
COMMON_FIGSIZE = (18, 7)

SOURCES = {
    'LAP': SENS / "LAP",
    'CP':  SENS / "CP",
}
LABEL_OVERRIDES = {'lipid_length_ratio': r'$l_{\mathrm{lipid}}$'}
GROUP_ORDER = ['Morphology', 'Material', 'Hemodynamics']


def load(ds, kind):
    """kind in {'ind','grp'}; returns the raw/orig slice only."""
    d = pd.read_csv(os.path.join(SOURCES[ds], f"sobol_{kind}.csv"))
    d = d[d.scale == 'orig'].copy()
    if kind == 'ind':
        d['display_name'] = d['parameter'].map(LABEL_OVERRIDES).fillna(d['display_name'])
    return d


def sel(d, out):
    s = d[d.output == out].copy()
    for c in ('S1_raw', 'ST_raw'):
        s[c] = s[c].clip(lower=0)      # negative == numerical zero
    return s


def draw_individual(ax, s, tick_fs=19):
    s = s.sort_values('S1_raw').reset_index(drop=True)   # largest on top
    y = np.arange(len(s))
    h = 0.35
    ax.barh(y + h / 2, s['ST_raw'], h, xerr=s['ST_conf'], label='Total-order ($S_T$)',
            color=C_ST, hatch='///', edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='#3a3a3a', lw=1.1, capsize=2.5))
    ax.barh(y - h / 2, s['S1_raw'], h, xerr=s['S1_conf'], label='First-order ($S_1$)',
            color=C_S1, edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='#3a3a3a', lw=1.1, capsize=2.5))
    ax.set_yticks(y)
    ax.set_yticklabels(s['display_name'], fontsize=tick_fs)
    ax.set_xlabel('Sobol Index', fontsize=20)
    ax.legend(fontsize=18, loc='lower right')
    ax.tick_params(axis='x', labelsize=19)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.7, len(s) - 0.3)


def draw_group(ax, s, tick_fs=22):
    s = s.set_index('group').reindex(GROUP_ORDER).reset_index()
    x = np.arange(len(s))
    w = 0.35
    b1 = ax.bar(x - w / 2, s['S1_raw'], w, yerr=s['S1_conf'], label='$S_1$ (group)',
                color=C_S1, edgecolor='black', linewidth=0.5,
                error_kw=dict(ecolor='#3a3a3a', lw=1.2, capsize=4))
    b2 = ax.bar(x + w / 2, s['ST_raw'], w, yerr=s['ST_conf'], label='$S_T$ (group)',
                color=C_ST, hatch='///', edgecolor='black', linewidth=0.5,
                error_kw=dict(ecolor='#3a3a3a', lw=1.2, capsize=4))
    top = max(s['S1_raw'].max(), (s['ST_raw'] + s['ST_conf']).max())
    for bar, cf in zip(list(b1) + list(b2), list(s['S1_conf']) + list(s['ST_conf'])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cf + top * 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(s['group'], fontsize=tick_fs)
    ax.set_ylabel('Group Sobol Index', fontsize=22)
    ax.legend(fontsize=18)
    ax.tick_params(axis='y', labelsize=21)
    ax.set_ylim(0, top * 1.28)


def save(fig, stem, tight=True):
    for ext in ('png', 'svg'):
        p = os.path.join(HERE, f"{stem}.{ext}")
        fig.savefig(p, format=ext, dpi=150 if ext == 'png' else None,
                    bbox_inches='tight' if tight else None)
        print("saved", p)
    plt.close(fig)


TICK_FS = {'LAP': 19, 'CP': 16}

# ---- standalone figures, one per dataset ----------------------------------
for ds in ('LAP', 'CP'):
    ind, grp = load(ds, 'ind'), load(ds, 'grp')

    fig, axes = plt.subplots(1, 2, figsize=COMMON_FIGSIZE)
    for ax, out in zip(axes, OUTPUTS):
        draw_individual(ax, sel(ind, out), tick_fs=TICK_FS[ds])
    fig.tight_layout()
    save(fig, f"fig7_individual_{ds}", tight=False)

    fig, axes = plt.subplots(1, 2, figsize=COMMON_FIGSIZE)
    for ax, out in zip(axes, OUTPUTS):
        draw_group(ax, sel(grp, out))
    fig.tight_layout()
    save(fig, f"fig7_group_{ds}", tight=False)

# ---- stacked LAP over CP --------------------------------------------------
for kind, figsize in (('individual', (18, 15)), ('group', (18, 12))):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for r, ds in enumerate(('LAP', 'CP')):
        d = load(ds, 'ind' if kind == 'individual' else 'grp')
        for j, out in enumerate(OUTPUTS):
            if kind == 'individual':
                draw_individual(axes[r, j], sel(d, out), tick_fs=TICK_FS[ds])
            else:
                draw_group(axes[r, j], sel(d, out))
            if r == 0:
                axes[r, j].set_title(OUT_DISP[out], fontsize=26, pad=14)
    fig.tight_layout(rect=[0.035, 0, 1, 1], h_pad=3.0)
    for r, ds in enumerate(('LAP', 'CP')):
        pos = axes[r, 0].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, ds, rotation=90,
                 va='center', ha='center', fontsize=26, fontweight='bold')
    save(fig, f"fig7_combined_{kind}", tight=True)

# ---- console summary ------------------------------------------------------
print("\n" + "=" * 78)
for ds in ('LAP', 'CP'):
    ind, grp = load(ds, 'ind'), load(ds, 'grp')
    for out in OUTPUTS:
        s1 = sel(ind, out).nlargest(5, 'S1_raw')
        s = sel(ind, out).nlargest(5, 'ST_raw')
        g = sel(grp, out).set_index('group').reindex(GROUP_ORDER)
        print(f"  {ds:<4}{out:<9} top5 S1: "
              + " > ".join(f"{p} {v:.3f}" for p, v in zip(s1.parameter, s1.S1_raw)))
        print(f"  {'':<4}{'':<9} top5 ST: "
              + " > ".join(f"{p} {v:.3f}" for p, v in zip(s.parameter, s.ST_raw)))
        print(f"  {'':<4}{'':<9} groups : "
              + "  ".join(f"{k} S1={r.S1_raw:.3f}/ST={r.ST_raw:.3f}"
                          for k, r in g.iterrows())
              + f"   sum S1={sel(ind,out).S1_raw.sum():.2f}"
                f"  sum ST={sel(ind,out).ST_raw.sum():.2f}")
