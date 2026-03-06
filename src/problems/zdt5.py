from __future__ import annotations
import numpy as np


def zdt5(X: np.ndarray) -> np.ndarray:
    """
    ZDT5 binary benchmark.

    Structure:
    - First segment: 30 bits
    - Next 10 segments: 5 bits each
    Total bits = 30 + 5*10 = 80
    """

    if X.ndim != 2:
        raise ValueError("X must be 2D array")

    if X.shape[1] != 80:
        raise ValueError("ZDT5 expects 80 binary variables")

    u1 = np.sum(X[:, :30], axis=1)
    f1 = 1.0 + u1

    g = np.zeros_like(f1, dtype=float)

    for i in range(10):
        start = 30 + i * 5
        end = start + 5

        ui = np.sum(X[:, start:end], axis=1)

        vi = np.where(ui < 5, 2.0 + ui, 1.0)
        g += vi

    f2 = g / f1

    return np.column_stack([f1, f2])

def sample_true_pareto_front_zdt5() -> np.ndarray:
    """
    True Pareto front for the standard ZDT5 definition used here:
      - 1st block length 30 -> f1 in {1..31}
      - 10 blocks length 5 -> g_min = 10 (all blocks are all-ones)
    PF points: (f1, 10/f1)
    Returns shape (31, 2).
    """
    f1 = np.arange(1, 32, dtype=float)   # 1..31
    f2 = 10.0 / f1
    return np.column_stack([f1, f2])