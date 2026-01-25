from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import comb
import numpy as np

def weights_2d_uniform(N: int) -> np.ndarray:
    if N < 2:
        raise ValueError("N must be >= 2 for 2D uniform weights.")
    w1 = np.linspace(0.0, 1.0, N)
    W = np.stack([w1, 1.0 - w1], axis = 1)
    return W

def simplex_lattice_weights(M: int, H: int) -> np.ndarray:
    '''
    Simplex-lattice design weights (Das and Dennis)
    Generates all weight vectors w in R^M such that:
    w_i = k_i / H, k_i are nonnegative integers, sum k_i = H

    Number of vectors = C(H+M-1, M-1)
    Returns: (N, M)
    '''

    if M < 2:
        raise ValueError("M must be >= 2.")
    if H < 1:
        raise ValueError("H must be >= 1.")
    
    N = comb(H + M - 1, M - 1)
    W = np.zeros((N, M), dtype = float)

    idx = 0
    slots = H + M - 1

    for bars in combinations(range(slots), M - 1):
        prev = -1
        ks = []
        for b in bars:
            ks.append(b - prev - 1)
            prev = b
        ks.append(slots - prev - 1)
        ks = np.array(ks, dtype = int)

        W[idx, :] = ks / float(H)
        idx += 1
    
    return W

def neighborhood_by_euclidean(W: np.ndarray, T: int) -> np.ndarray:
    if W.ndim != 2:
        raise ValueError("W must be 2D (N, M).")
    N = W.shape[0]
    if T < 1 or T > N:
        raise ValueError("T must be in [1, N].")
    
    norms = np.sum(W * W, axis = 1, keepdims = True)
    d2 = norms + norms.T - 2.0 * (W @ W.T)
    d2 = np.maximum(d2, 0.0)

    B = np.argsort(d2, axis = 1)[:, :T]
    return B

@dataclass(frozen = True)
class WeightSetup:
    W: np.ndarray
    B: np.ndarray

def build_weight_setup(n_obj: int, N: int | None = None, H: int | None = None, T: int = 20) -> WeightSetup:
    if n_obj == 2:
        if N is None:
            raise ValueError("For n_obj = 2, you must provide N.")
        W = weights_2d_uniform(N)
    else:
        if H is None:
            raise ValueError("For n_obj >= 3, you must provide H.")
        W = simplex_lattice_weights(n_obj, H)
    
    B = neighborhood_by_euclidean(W, T = min(T, W.shape[0]))
    return WeightSetup(W = W, B = B)