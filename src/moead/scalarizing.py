from __future__ import annotations
import numpy as np

def tchebyscheff(F: np.ndarray, w: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype = float)
    z = np.asarray(z, dtype = float)
    if F.ndim != 2:
        raise ValueError("F must be 2D (N, M).")
    if w.ndim != 1 or z.ndim != 1:
        raise ValueError("w and z must be 1D (M,).")
    if F.shape[1] != w.shape[0] or w.shape[0] != z.shape[0]:
        raise ValueError("Dimension mismatch among F, w, z.")
    
    ww = np.maximum(w, eps)
    return np.max(ww * np.abs(F - z[None, :]), axis = 1)

def tchebyscheff_one(f: np.ndarray, w: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> float:
    f = np.asarray(f, dtype = float)
    w = np.asarray(w, dtype = float)
    z = np.asarray(z, dtype = float)
    ww = np.maximum(w, eps)
    return float(np.max(ww * np.abs(f - z)))