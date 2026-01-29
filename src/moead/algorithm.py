from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from moead.scalarizing import tchebyscheff_one
from moead.variation import variation_sbx_poly
from moead.metrics import igd

@dataclass
class MOEADConfig:
    n_obj: int
    n_var: int
    pop_size: int
    T: int = 20
    n_gen: int = 200
    seed: int = 42
    nr: int = 2
    eta_c: float = 20.0
    eta_m: float = 20.0
    p_c: float = 1.0
    p_m: float | None = None
    variation: str = "sbx" #possible arguments are "placeholder" and "sbx"

def init_ideal_point(F: np.ndarray) -> np.ndarray:
    return np.min(F, axis = 0)

def update_ideal_point(z: np.ndarray, f_new: np.ndarray) -> np.ndarray:
    return np.minimum(z, f_new)

def update_neighbors_tchebyscheff(X: np.ndarray, F: np.ndarray, y: np.ndarray, f_y: np.ndarray, W: np.ndarray, B_i: np.ndarray, z: np.ndarray, nr: int) -> int:
    replaced = 0
    for j in B_i:
        g_old = tchebyscheff_one(F[j], W[j], z)
        g_new = tchebyscheff_one(f_y, W[j], z)
        if g_new <= g_old:
            X[j] = y
            F[j] = f_y
            replaced += 1
            if replaced >= nr:
                break
    return replaced

def simple_variation_placeholder(rng: np.random.Generator, x1: np.ndarray, x2: np.ndarray, xl: np.ndarray, xu: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    alpha = rng.random()
    y = alpha * x1 + (1.0 - alpha) * x2
    y = y + rng.normal(0.0, sigma, size = y.shape)
    return np.clip(y, xl, xu)

def moead_run(config: MOEADConfig, evaluate_fn, W: np.ndarray, B: np.ndarray, xl: np.ndarray, xu: np.ndarray, reference_Z = None) -> dict:
    rng = np.random.default_rng(config.seed)

    N = config.pop_size
    if W.shape[0] != N:
        raise ValueError("W must have pop_size rows.")
    if B.shape[0] != N:
        raise ValueError("B must have pop_size rows.")
    if xl.shape[0] != config.n_var or xu.shape[0] != config.n_var:
        raise ValueError("Bounds must be shape (n_var,).")
    
    X = rng.uniform(0.0, 1.0, size = (N, config.n_var))
    X = xl[None, :] + X * (xu - xl)[None, :]
    F = evaluate_fn(X)

    z = init_ideal_point(F)

    history_replaced = []
    history_igd = []
    for gen in range(config.n_gen):
        replaced_this_gen = 0
        for i in range(N):
            neigh = B[i]
            k, l = rng.choice(neigh, size = 2, replace = False) if len(neigh) >= 2 else (i, i)
            if config.variation == "placeholder":
                y = simple_variation_placeholder(rng, X[k], X[l], xl, xu)
            elif config.variation == "sbx":
                y = variation_sbx_poly(rng, X[k], X[l], xl, xu, eta_c = config.eta_c, eta_m = config.eta_m, p_c = config.p_c, p_m = config.p_m)
            else:
                raise ValueError(f"Unknown variation mode: {config.variation}")
            f_y = evaluate_fn(y[None, :])[0]
            z = update_ideal_point(z, f_y)

            replaced_this_gen += update_neighbors_tchebyscheff(X = X, F = F, y = y, f_y = f_y, W = W, B_i = neigh, z = z, nr = config.nr)
        history_replaced.append(replaced_this_gen)
        if reference_Z is not None:
            history_igd.append(igd(F, reference_Z))
    
    return {"X" : X, "F" : F, "z" : z, "history_replaced" : np.array(history_replaced, dtype = int), "history_igd" : np.array(history_igd, dtype = float) if reference_Z is not None else None}