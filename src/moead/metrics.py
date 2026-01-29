from __future__ import annotations
import numpy as np

def igd(A: np.ndarray, Z: np.ndarray) -> float:
    """
    Inverted Generational Distance:
        A: (N, M) approx solution set in objective space
        Z: (K, M) reference set (e.g, sampled true Pareto front)
    IGD = average over z in Z of distance to nearest a in A
    """
    A = np.asarray(A, dtype = float)
    Z = np.asarray(Z, dtype = float)
    if A.ndim != 2 or Z.ndim != 2:
        raise ValueError("A and Z must be 2D arrays.")
    if A.shape[1] != Z.shape[1]:
        raise ValueError("Objective dimension mismatch between A and Z.")
    
    Z_norm = np.sum(Z * Z, axis = 1, keepdims = True)
    A_norm = np.sum(A * A, axis = 1, keepdims = True).T
    d2 = Z_norm + A_norm - 2.0 * (Z @ A.T)
    d2 = np.maximum(d2, 0.0)

    min_d = np.sqrt(np.min(d2, axis = 1))
    return float(np.mean(min_d))