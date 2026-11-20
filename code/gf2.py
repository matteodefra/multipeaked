import numpy as np
def rref(A):
    A = (np.array(A, dtype=np.uint8) & 1).copy(); m, n = A.shape; piv = []; r = 0
    for c in range(n):
        if r == m: break
        rows = np.nonzero(A[r:, c])[0]
        if len(rows) == 0: continue
        p = r + rows[0]
        if p != r: A[[r, p]] = A[[p, r]]
        mask = A[:, c].astype(bool); mask[r] = False
        A[mask] ^= A[r]
        piv.append(c); r += 1
    return A[:r], piv
def rank(A): return len(rref(A)[1]) if np.size(A) else 0
def nullspace(A):
    """Basis (rows) of {x : A x = 0}."""
    A = np.atleast_2d(A); n = A.shape[1]
    R, piv = rref(A); free = [c for c in range(n) if c not in piv]
    N = np.zeros((len(free), n), dtype=np.uint8)
    for t, f in enumerate(free):
        N[t, f] = 1
        for i, p in enumerate(piv): N[t, p] = R[i, f]
    return N
def mm(A, B):  # matrix product mod 2 via float BLAS (exact for small sizes)
    return (np.asarray(A, np.float32) @ np.asarray(B, np.float32)).astype(np.int64) & 1
