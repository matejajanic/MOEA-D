from __future__ import annotations
import numpy as np

def dtlz2(x: np.ndarray, M: int = 3) -> np.ndarray:
    """
    DTLZ2 (minimization). Supports:
      - x shape (n_var,) -> returns (M,)
      - x shape (N, n_var) -> returns (N, M)
    Domain: x in [0,1]^n, n = (M-1)+k.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 2:
        N, n = x.shape
        if n < M:
            raise ValueError(f"DTLZ2 requires n_var >= M, got n_var={n}, M={M}.")

        g = np.sum((x[:, M-1:] - 0.5) ** 2, axis=1)
        F = np.empty((N, M), dtype=float)

        for m in range(M):
            val = 1.0 + g
            for j in range(M - m - 1):
                val = val * np.cos(x[:, j] * np.pi / 2.0)
            if m > 0:
                val = val * np.sin(x[:, M - m - 1] * np.pi / 2.0)
            F[:, m] = val
        return F

    if x.ndim == 1:
        n = x.shape[0]
        if n < M:
            raise ValueError(f"DTLZ2 requires n_var >= M, got n_var={n}, M={M}.")

        g = np.sum((x[M-1:] - 0.5) ** 2)
        f = np.empty(M, dtype=float)
        for m in range(M):
            val = 1.0 + g
            for j in range(M - m - 1):
                val *= np.cos(x[j] * np.pi / 2.0)
            if m > 0:
                val *= np.sin(x[M - m - 1] * np.pi / 2.0)
            f[m] = val
        return f

    raise ValueError("x must be 1D or 2D array.")


def sample_true_pareto_front_dtlz2(M: int = 3, K: int = 2000, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample reference points on the true Pareto front of DTLZ2 (g=0):
    Returns Z shape (K, M).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Z = rng.random((K, M))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / norms
    return Z