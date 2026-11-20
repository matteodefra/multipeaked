"""Exact M2 and D_min for random third-level phase states, the sharpness states
psi_{4g_m}, and general states (n <= 5). Usage: python code/exp_purity.py  ->  results/purity_records.json"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]          # repository root
RESULTS = ROOT / "results"                           # experiment outputs
FIGURES = ROOT / "paper" / "figures"                 # figures used by the paper
import numpy as np, time, json, magic
rng = np.random.default_rng(20261116)
w8 = np.exp(1j*np.pi/4)

def phase_state(n, gates):
    x = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    P = np.zeros(1 << n, dtype=int)
    for g in gates:
        typ, q = g[0], g[1:]
        if typ == 'T':   P += 1 * x[:, q[0]]
        if typ == 'Tdg': P += 7 * x[:, q[0]]
        if typ == 'S':   P += 2 * x[:, q[0]]
        if typ == 'CS':  P += 2 * x[:, q[0]] * x[:, q[1]]
        if typ == 'CSdg':P += 6 * x[:, q[0]] * x[:, q[1]]
        if typ == 'CZ':  P += 4 * x[:, q[0]] * x[:, q[1]]
        if typ == 'CCZ': P += 4 * x[:, q[0]] * x[:, q[1]] * x[:, q[2]]
    return w8 ** (P % 8) / np.sqrt(1 << n)

def random_gates(n, g, types):
    out = []
    for _ in range(g):
        t = types[rng.integers(len(types))]
        ar = {'T':1,'Tdg':1,'S':1,'CS':2,'CSdg':2,'CZ':2,'CCZ':3}[t]
        out.append((t,) + tuple(rng.choice(n, ar, replace=False)))
    return out

def sharp_state(m):
    n = 2*m + 1   # qubit 0 = z
    return phase_state(n, [('CCZ', 0, 2*j+1, 2*j+2) for j in range(m)])

def haar(n):
    v = rng.normal(size=1 << n) + 1j*rng.normal(size=1 << n)
    return v / np.linalg.norm(v)

def stab_plus_haar(n, F):
    phi = haar(n); phi[0] = 0; phi /= np.linalg.norm(phi)       # orthogonal to |0^n>
    v = np.sqrt(F) * np.eye(1 << n)[0] + np.sqrt(1 - F) * phi
    return v / np.linalg.norm(v)

records = []
def add(label, n, states):
    M = np.array(states).T
    t = time.time(); F = magic.F_S(M, n); A = np.array([magic.A2(s) for s in states])
    for f, a in zip(F, A): records.append(dict(label=label, n=n, F=float(f), A2=float(a)))
    print(f"{label:28s} n={n} count={len(states):4d}  {time.time()-t:6.1f}s  "
          f"max M2/Dmin={np.max(-np.log2(A)/np.maximum(-np.log2(F),1e-12)):.3f}", flush=True)

ALL = ['T','Tdg','S','CS','CSdg','CZ','CCZ']
for n in (3, 4, 5):
    cnt = {3: 300, 4: 300, 5: 200}[n]
    sts = [phase_state(n, random_gates(n, int(rng.integers(1, 3*n+1)), ALL)) for _ in range(cnt)]
    add('phase: random T/CS/CCZ', n, sts)
    sts = [phase_state(n, random_gates(n, int(rng.integers(1, 2*n+1)), ['CCZ','CZ'])) for _ in range(cnt//2)]
    add('phase: hypergraph', n, sts)
add('phase: sharpness m=1', 3, [sharp_state(1)])
add('phase: sharpness m=2', 5, [sharp_state(2)])
for n in (4, 5):
    Fs = np.linspace(0.08, 0.92, 22)
    add('general: stab + Haar', n, [stab_plus_haar(n, F) for F in Fs for _ in range(3)])
    add('general: Haar', n, [haar(n) for _ in range(40)])
RESULTS.mkdir(exist_ok=True)
json.dump(records, open(RESULTS / 'purity_records.json', 'w'))
print('wrote', RESULTS / 'purity_records.json')
