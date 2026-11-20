"""Recovery experiments for Theorem 7 / Corollary 11 on random planted instances.
Usage:  python code/exp_recovery.py codim  ->  results/recovery_codim.jsonl  (n = 128, d = 1..6)
        python code/exp_recovery.py time   ->  results/recovery_time.jsonl   (d = 2, 3; n = 64..256)
The planted subspace is used ONLY to generate the instance and to estimate beta(T)
for the plots; the recovery algorithm itself sees only the tensor T."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]          # repository root
RESULTS = ROOT / "results"                           # experiment outputs
FIGURES = ROOT / "paper" / "figures"                 # figures used by the paper
import numpy as np, math, time, json, sys
import recovery as R
from gf2 import rank, mm, rref

def merged_certificate(T, U, beta0, rng, extra=8):
    """Heuristic: merge further certified subspaces while the sum stays vanishing."""
    n = T.shape[0]
    for _ in range(extra):
        U2, _, _ = R.recover(T, beta0, rng)
        S = rref(np.vstack([U, U2]))[0]
        if not R.restrict(T, S).any():
            U = S
    return U

def run(n, d, seed, merge=True, extra=8):
    rng = np.random.default_rng(seed)
    T, Up, M = R.planted_instance(n, d, rng)
    beta0 = 2.0 ** (-3 * d)                          # Corollary 11
    t = time.time(); U, pairs, bound = R.recover(T, beta0, rng); dt = time.time() - t
    out = dict(n=n, d=d, seed=seed, pairs=int(pairs), time=dt, bound=bound,
               codim=int(n - rank(U)), inside=bool(not mm(M[:d], U.T).any()),
               beta=float(R.beta_estimate(T, Up, rng)))
    if merge:
        Um = merged_certificate(T, U, beta0, rng, extra=extra)
        out['codim_merged'] = int(n - rank(Um))
    return out

if __name__ == '__main__':
    mode = sys.argv[1]; res = []
    RESULTS.mkdir(exist_ok=True)
    out = open(RESULTS / f'recovery_{mode}.jsonl', 'w')        # one JSON record per line
    if mode == 'codim':
        n = 128
        for d in range(1, 7):
            reps = {1: 10, 2: 10, 3: 10, 4: 8, 5: 5, 6: 3}[d]
            for s in range(reps):
                r = run(n, d, 1000 * d + s, extra=(6 if d <= 4 else 2)); res.append(r)
                print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
    if mode == 'time':
        for d in (2, 3):
            for n in (64, 96, 128, 192, 256):
                for s in range(5):
                    r = run(n, d, 7000 + 100 * d + n + s, merge=False); res.append(r)
                    print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
    out.close()
