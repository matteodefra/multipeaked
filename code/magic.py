"""Exact stabilizer purity A2 and stabilizer fidelity F_S for small states.
F_S is computed by enumerating all stabilizer states in Dehaene--De Moor form:
  2^{-k/2} sum_{z in F_2^k} i^{l.z} (-1)^{q(z)} |s + B z>,
with B an RREF basis of a k-dim subspace, s a coset representative,
l in {0,1}^k and q an F_2 quadratic form (diagonal included)."""
import numpy as np, itertools, math

def fwht(a):
    """Unnormalized Walsh-Hadamard transform along the last axis."""
    a = a.copy(); h = 1; N = a.shape[-1]
    while h < N:
        a = a.reshape(a.shape[:-1] + (N // (2*h), 2, h))
        x, y = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :], a[..., 1, :] = x + y, x - y
        a = a.reshape(a.shape[:-3] + (N,)); h *= 2
    return a

def A2(psi):
    N = len(psi); idx = np.arange(N)
    V = np.conj(psi[idx[:, None] ^ idx[None, :]]) * psi[None, :]   # V[a,x]
    W = fwht(V)                                                      # <X^a Z^b>
    return float(np.sum(np.abs(W)**4) / N)

def subspaces(n, k):
    """All k-dim subspaces of F_2^n as RREF bases (lists of ints), with pivots."""
    out = []
    for piv in itertools.combinations(range(n), k):
        free = [(r, j) for r in range(k) for j in range(piv[r]+1, n) if j not in piv]
        for bits in itertools.product([0, 1], repeat=len(free)):
            rows = [1 << piv[r] for r in range(k)]
            for (r, j), b in zip(free, bits):
                if b: rows[r] |= 1 << j
            out.append((rows, piv))
    return out

def support_points(n, k):
    """Index arrays (num_affine_subspaces, 2^k) of all k-dim affine subspaces."""
    pts = []
    for rows, piv in subspaces(n, k):
        nonpiv = [j for j in range(n) if j not in piv]
        span = [0] * (1 << k)
        for z in range(1 << k):
            v = 0
            for r in range(k):
                if (z >> r) & 1: v ^= rows[r]
            span[z] = v
        span = np.array(span)
        for cbits in range(1 << len(nonpiv)):
            s = 0
            for t, j in enumerate(nonpiv):
                if (cbits >> t) & 1: s |= 1 << j
            pts.append(s ^ span)
    return np.array(pts)

_phase_cache = {}
def phase_tables(k):
    if k in _phase_cache: return _phase_cache[k]
    Z = np.array([[(z >> r) & 1 for r in range(k)] for z in range(1 << k)])  # 2^k x k
    mons = [(i, j) for i in range(k) for j in range(i, k)]
    mono = np.array([Z[:, i] * Z[:, j] for (i, j) in mons]).reshape(len(mons), 1 << k)
    nq = len(mons)
    if nq:
        bits = ((np.arange(1 << nq)[:, None] >> np.arange(nq)) & 1)
        S = (1 - 2 * ((bits @ mono) % 2)).astype(np.float64)                  # (2^nq, 2^k)
    else:
        S = np.ones((1, 1))
    L = np.array([[(1j) ** (-(int(bin(l & z).count("1")))) for z in range(1 << k)]
                  for l in range(1 << k)])                                     # i^{-l.z}
    _phase_cache[k] = (S, L)
    return S, L

_support_cache = {}
def F_S(psis, n, batch=8):
    """Stabilizer fidelity of each column of psis (shape 2^n x M)."""
    M = psis.shape[1]; best = np.zeros(M)
    for k in range(n + 1):
        key = (n, k)
        if key not in _support_cache: _support_cache[key] = support_points(n, k)
        P = _support_cache[key]; S, L = phase_tables(k)
        for b0 in range(0, M, batch):
            ps = psis[:, b0:b0+batch]
            vals = ps[P]                                  # (nsub, 2^k, m)
            for l in range(L.shape[0]):
                w = vals * L[l][None, :, None]           # (nsub, 2^k, m)
                ov = np.einsum('qz,szm->sqm', S, w, optimize=True)
                m = (np.abs(ov)**2).max(axis=(0, 1)) / (1 << k)
                best[b0:b0+batch] = np.maximum(best[b0:b0+batch], m)
    return best

def count_stabilizer_states(n):
    tot = 0
    for k in range(n + 1):
        tot += len(support_points(n, k)) * (1 << k) * (1 << (k*(k+1)//2))
    return tot
