import numpy as np, math, time
from gf2 import rref, rank, nullspace, mm

def contract1(T, v):              # T(v,.,.)
    n = T.shape[0]
    return mm(v[None, :], T.reshape(n, n*n)).reshape(n, n)

def restrict(T, B):               # T|_B in coordinates of the rows of B
    n = T.shape[0]; k = B.shape[0]
    X = mm(B, T.reshape(n, n*n)).reshape(k, n, n)                       # (k,n,n)
    X = mm(X.reshape(k*n, n), B.T).reshape(k, n, k)                      # (k,n,k)
    X = mm(B, X.transpose(1, 0, 2).reshape(n, k*k)).reshape(k, k, k)     # (k,k,k)
    return X.astype(np.uint8)

def planted_instance(n, d, rng):
    """Random symmetric trilinear T on F_2^n vanishing on a hidden subspace of codim d."""
    Ty = np.zeros((n, n, n), dtype=np.uint8)
    for i in range(d):
        for j in range(i, n):
            for k in range(j, n):
                if rng.integers(2):
                    for (a, b, c) in {(i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)}:
                        Ty[a, b, c] = 1
    while True:
        M = rng.integers(0, 2, (n, n)).astype(np.uint8)
        if rank(M) == n: break
    Tx = restrict(Ty, M.T.copy())     # T_x(a,b,c) = T_y(Ma, Mb, Mc)
    U_planted = nullspace(M[:d])      # x with (Mx)_1..d = 0
    return Tx, U_planted, M

def lemma_common(G):
    """G: (k,k,k) stack of symmetric forms on F_2^k. Returns basis K of a common
    vanishing subspace (Lemma 9), choosing any nonzero restricted generator."""
    k = G.shape[0]; K = np.eye(k, dtype=np.uint8)
    while K.shape[0] > 0:
        R = mm(mm(K, G.reshape(k, k*k).reshape(k, k, k).transpose(1, 0, 2).reshape(k, k*k)).reshape(K.shape[0], k, k).transpose(1, 0, 2).reshape(k*K.shape[0], k), K.T)
        R = R.reshape(k, K.shape[0], K.shape[0])     # R[i] = K G_i K^T
        nz = np.nonzero(R.reshape(k, -1).any(axis=1))[0]
        if len(nz) == 0: break
        rad = nullspace(R[nz[0]])                     # radical of the restricted form
        K = mm(rad, K).astype(np.uint8) if rad.shape[0] else np.zeros((0, k), np.uint8)
    return K

def recover(T, beta0, rng, batch=512, max_pairs=10**8):
    n = T.shape[0]; R = math.ceil(math.log2(1 / beta0)); bound = 5 * R + 14
    if bound >= n: return np.zeros((0, n), np.uint8), 0, bound
    T2 = T.reshape(n, n*n).astype(np.float32); drawn = 0
    while drawn < max_pairs:
        V = rng.integers(0, 2, (batch, n)).astype(np.float32)
        Uu = rng.integers(0, 2, (batch, n, 1)).astype(np.float32)
        X = np.mod(V @ T2, 2).reshape(batch, n, n)          # M_v for each v
        kept = np.nonzero(~np.mod(X @ Uu, 2).any(axis=(1, 2)))[0]
        for b in kept:
            pairs_used = drawn + b + 1
            Vb = nullspace(X[b].astype(np.uint8))           # V = ker M_v
            G = restrict(T, Vb)                             # forms M_w|_{VxV}, w in basis of V
            K = lemma_common(G)
            Ub = mm(K, Vb).astype(np.uint8) if K.shape[0] else np.zeros((0, n), np.uint8)
            dimU = rank(Ub) if Ub.shape[0] else 0
            if n - dimU <= bound and (dimU == 0 or not restrict(T, Ub).any()):
                return Ub, pairs_used, bound
        drawn += batch
    raise RuntimeError("no success")

def beta_estimate(T, U, rng, samples=400):
    """Stratified Monte Carlo estimate of beta(T) = E_a 2^{-rank M_a}, using the planted U
    only for variance reduction (inside U / outside U)."""
    n = T.shape[0]; k = U.shape[0]; d = n - k
    inside = np.mean([2.0 ** -rank(contract1(T, mm(rng.integers(0, 2, k)[None, :], U)[0]))
                      for _ in range(samples)])
    outside = np.mean([2.0 ** -rank(contract1(T, rng.integers(0, 2, n))) for _ in range(samples // 4)])
    return 2.0 ** -d * inside + (1 - 2.0 ** -d) * outside
