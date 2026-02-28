from __future__ import annotations
import numpy as np


def variation_binary(
    rng: np.random.Generator,
    x1: np.ndarray,
    x2: np.ndarray,
    p_c: float = 1.0,
    p_m: float = 0.01,
) -> np.ndarray:

    if rng.random() < p_c:
        mask = rng.random(size=x1.shape) < 0.5
        y = np.where(mask, x1, x2)
    else:
        y = x1.copy()

    mut_mask = rng.random(size=y.shape) < p_m
    y = np.where(mut_mask, 1 - y, y)

    return y