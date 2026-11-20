"""Figure 2 from recovery_codim.jsonl and (optionally) recovery_time.jsonl.
Usage: python code/plot_recovery.py [DATA_DIR]   (default DATA_DIR = results/)
If recovery_time.jsonl is missing, only panels (a) and (b) are drawn."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]          # repository root
RESULTS = ROOT / "results"                           # experiment outputs
FIGURES = ROOT / "paper" / "figures"                 # figures used by the paper
import json, numpy as np, math, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.linewidth': 0.6, 'legend.frameon': False})
def load(f):
    return [json.loads(l) for l in open(f) if l.startswith('{')]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
C = load(DATA / 'recovery_codim.jsonl')
Tm = load(DATA / 'recovery_time.jsonl') if (DATA / 'recovery_time.jsonl').exists() else []
fig, axs = plt.subplots(1, 3 if Tm else 2, figsize=(7.2 if Tm else 5.0, 2.45))

# (a) certified codimension versus planted codimension
ax = axs[0]; ds = sorted(set(r['d'] for r in C))
first = [np.array([r['codim'] for r in C if r['d'] == d]) for d in ds]
merged = [np.array([r['codim_merged'] for r in C if r['d'] == d]) for d in ds]
thm = [np.array([5 * math.ceil(math.log2(1 / r['beta'])) + 14 for r in C if r['d'] == d]) for d in ds]
dd = np.array(ds)
ax.plot(dd, 15 * dd + 14, '-', color='0.75', lw=3.5, label=r'Cor. 11: $15d+14$')
ax.plot(dd, [t.max() for t in thm], '--', color='0.4', lw=1, label=r'Thm. 7: $5\lceil\log\frac{1}{\beta}\rceil+14$')
for i, d in enumerate(ds):
    ax.scatter(np.full(len(first[i]), d) - 0.08, first[i], s=12, color='tab:blue', lw=0, zorder=3)
    ax.scatter(np.full(len(merged[i]), d) + 0.08, merged[i], s=12, marker='s', color='tab:green', lw=0, zorder=3)
ax.scatter([], [], s=12, color='tab:blue', label='first certificate')
ax.scatter([], [], s=12, marker='s', color='tab:green', label='after merging')
ax.plot(dd, dd, '-', color='k', lw=0.6, label='planted $d$')
ax.set_yscale('log'); ax.set_xlabel('planted codimension $d$'); ax.set_ylabel('certified codimension')
ax.set_xticks(ds); ax.legend(fontsize=6, loc='upper left'); ax.set_title('(a)  $n=128$', fontsize=8.5)
ax.set_ylim(0.7, 300)

# (b) drawn pairs versus 1/beta
ax = axs[1]
b = np.array([1 / r['beta'] for r in C]); p = np.array([max(r['pairs'], 1) for r in C])
sc = ax.scatter(b, p, c=[r['d'] for r in C], cmap='viridis', s=12, lw=0, zorder=3)
xs = np.logspace(0, np.log10(b.max() * 2), 20)
ax.plot(xs, 2 * xs, '--', color='0.4', lw=1, label=r'$2/\beta$ (Lemma 8)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$1/\beta(T)$ (estimated)'); ax.set_ylabel('pairs drawn until success')
ax.legend(fontsize=6, loc='upper left'); ax.set_title('(b)  cost follows the bias', fontsize=8.5)
cb = fig.colorbar(sc, ax=ax, pad=0.02, ticks=ds); cb.set_label('$d$', fontsize=7); cb.ax.tick_params(labelsize=6)

# (c) wall-clock time versus n
for ax in ([axs[2]] if Tm else []):
    for d, mk, col in ((2, 'o', 'tab:purple'), (3, '^', 'tab:red')):
        ns = sorted(set(r['n'] for r in Tm if r['d'] == d))
        med = [np.median([r['time'] for r in Tm if r['d'] == d and r['n'] == n]) for n in ns]
        for n in ns:
            ts = [r['time'] for r in Tm if r['d'] == d and r['n'] == n]
            ax.scatter([n] * len(ts), ts, s=8, marker=mk, color=col, alpha=0.4, lw=0)
        ax.plot(ns, med, mk + '-', color=col, ms=3.5, lw=0.8, label=f'$d={d}$ (median)')
    ns = np.array(sorted(set(r['n'] for r in Tm)))
    ref = np.median([r['time'] for r in Tm if r['n'] == ns[-1]]) * (ns / ns[-1]) ** 4
    ax.plot(ns, ref, ':', color='0.4', lw=1, label=r'$\propto n^4$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off(); ax.set_xlabel('qubits $n$'); ax.set_ylabel('recovery time (s)')
    ax.legend(fontsize=6, loc='upper left'); ax.set_title('(c)  one CPU core', fontsize=8.5)
fig.tight_layout(w_pad=0.6)
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / 'fig_recovery.pdf'); fig.savefig(FIGURES / 'fig_recovery.png', dpi=200)
print('wrote', FIGURES / 'fig_recovery.pdf')
for i, d in enumerate(ds):
    print(d, 'first', sorted(first[i].tolist()), 'merged', sorted(merged[i].tolist()),
          'log2 beta mean %.2f' % np.mean([math.log2(r['beta']) for r in C if r['d'] == d]),
          'inside all:', all(r['inside'] for r in C if r['d'] == d))
