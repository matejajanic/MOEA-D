from __future__ import annotations
import numpy as np


def variation_binary(
    rng: np.random.Generator,
    x1: np.ndarray,
    x2: np.ndarray,
    p_c: float = 0.9,
    p_m: float = 0.05,
) -> np.ndarray:

    n = x1.shape[0]

    if rng.random() < p_c:
        point = rng.integers(1, n)
        y = np.concatenate([x1[:point], x2[point:]])
    else:
        y = x1.copy()

    mut_mask = rng.random(size=n) < p_m
    y = np.where(mut_mask, 1 - y, y)

    return y