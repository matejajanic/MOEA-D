from __future__ import annotations
import numpy as np

def zdt1(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(X < 0.0) or np.any(X > 1.0):
        raise ValueError("ZDT1 expects X in [0,1].")
    
    f1 = X[:, 0]
    # g(x) = 1 + 9/(n-1) * sum_{i=2..n} x_i
    n = X.shape[1]
    if n < 2:
        raise ValueError("ZDT1 typically uses n>=2.")
    g = 1.0 + 9.0 * np.sum(X[:, 1:], axis = 1) / (n - 1)
    # f2 = g * (1 - sqrt(f1/g))
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.stack([f1, f2], axis = 1)

def sample_true_pareto_front_zdt1(num_points: int = 200) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, num_points)
    f2 = 1.0 - np.sqrt(f1)
    return np.stack([f1, f2], axis = 1)
