"""Figure 1 (M2 versus D_min) from purity_records.json.
Usage: python code/plot_purity.py [DATA_DIR]   (default DATA_DIR = results/;
       use results/reference to plot the reference run without recomputing)"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]          # repository root
RESULTS = ROOT / "results"                           # experiment outputs
FIGURES = ROOT / "paper" / "figures"                 # figures used by the paper
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.linewidth': 0.6, 'legend.frameon': False})
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
rec = json.load(open(DATA / 'purity_records.json'))
def pts(pred):
    r = [x for x in rec if pred(x)]
    return np.array([-np.log2(x['F']) for x in r]), np.array([-np.log2(x['A2']) for x in r])

fig, ax = plt.subplots(figsize=(3.6, 2.9))
D = np.linspace(0, 2.1, 50)
ax.fill_between(D, 3*D, 4*D, color='0.9', lw=0)
ax.plot(D, 4*D, '--', color='0.35', lw=0.9, label=r'$M_2=4D_{\min}$ (all pure states)')
ax.plot(D, 3*D, '-', color='k', lw=1.0, label=r'$M_2=3D_{\min}$ (Theorem 5)')

x, y = pts(lambda r: r['label'].startswith('phase: random') or r['label'].startswith('phase: hyper'))
ax.scatter(x, y, s=5, color='tab:blue', alpha=0.45, lw=0, label='third-level phase states, $n\\leq 5$')
x, y = pts(lambda r: r['label'].startswith('general: stab'))
ax.scatter(x, y, s=9, marker='^', color='tab:orange', lw=0, label=r'$\sqrt{F}|0^n\rangle+\sqrt{1-F}|\phi_{\rm Haar}\rangle$, $n=4,5$')
x, y = pts(lambda r: r['label'] == 'general: Haar')
ax.scatter(x, y, s=8, marker='x', color='0.5', lw=0.6, label='Haar-random states, $n=4,5$')

# sharpness family psi_{4g_m}: exact purity (Eq. 12); D_min in [M2/3, 1]
ms = np.arange(1, 13)
M2 = -np.log2((1 + 7 * 4.0 ** (-ms)) / 8)
x1, y1 = pts(lambda r: r['label'] == 'phase: sharpness m=1')
x2, y2 = pts(lambda r: r['label'] == 'phase: sharpness m=2')
Dexact = {1: x1[0], 2: x2[0]}
for m, m2 in zip(ms, M2):
    if m in Dexact:
        ax.plot(Dexact[m], m2, '*', color='tab:red', ms=6, zorder=5)
    else:
        ax.errorbar(1.0, m2, xerr=[[1.0 - m2 / 3], [0]], fmt='*', color='tab:red', ms=4,
                    elinewidth=0.6, capsize=1.5, zorder=5)
ax.plot([], [], '*', color='tab:red', ms=6, label=r'$\psi_{4g_m}$, $m=1,\dots,12$')
ax.set_xlim(0, 2.05); ax.set_ylim(0, 4.3)
ax.text(1.16, 4.0, 'excluded for\nphase states', rotation=0, fontsize=6.3, color='0.3', ha='center', va='center')
ax.set_xlabel(r'$D_{\min}=-\log_2 F_S$'); ax.set_ylabel(r'$M_2=-\log_2 A_2$')
ax.legend(loc='lower right', fontsize=6.3, handlelength=1.6)
FIGURES.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(FIGURES / 'fig_purity_fidelity.pdf'); fig.savefig(FIGURES / 'fig_purity_fidelity.png', dpi=200)
print('wrote', FIGURES / 'fig_purity_fidelity.pdf')
ph = [x for x in rec if x['label'].startswith('phase')]
ph=[x for x in ph if x['F']<1-1e-9]; print('phase states (non-stabilizer):', len(ph), ' max M2/Dmin =', max(-np.log2(x['A2'])/-np.log2(x['F']) for x in ph if x['F'] < 1))
print('min A2 - F^3 over phase states:', min(x['A2'] - x['F']**3 for x in ph))
