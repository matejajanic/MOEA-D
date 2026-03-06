from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np

from moead.algorithm import moead_run, MOEADConfig
from moead.weights import weights_2d_uniform, simplex_lattice_weights, neighborhood_by_euclidean
from moead.metrics import filter_nondominated


def make_run_dir(base_out: str, tag: str | None) -> Path:
    base = Path(base_out)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = ts if tag is None else f"{ts}_{tag}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, cfg: MOEADConfig, extra: dict | None = None) -> None:
    payload = {"moead": asdict(cfg)}
    if extra:
        payload["extra"] = extra
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _choose_H_min(M: int, target_N: int) -> int:
    """
    Find smallest H such that C(H+M-1, M-1) >= target_N
    """
    from math import comb
    H = 1
    while comb(H + M - 1, M - 1) < target_N:
        H += 1
    return H


def build_WB(n_obj: int, pop_size: int, T: int, H: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build weight vectors W and neighborhoods B.

    - For 2 objectives: uniform line weights of size pop_size.
    - For >=3 objectives: simplex-lattice weights; if more than pop_size, subsample deterministically using seed.
      If fewer than pop_size (rare with our H-min), resample with replacement.
    """
    rng = np.random.default_rng(seed)

    if n_obj == 2:
        W = weights_2d_uniform(pop_size)
    else:
        if H is None:
            H = _choose_H_min(n_obj, pop_size)

        W_full = simplex_lattice_weights(n_obj, H)

        if W_full.shape[0] == pop_size:
            W = W_full
        elif W_full.shape[0] > pop_size:
            idx = rng.choice(W_full.shape[0], size=pop_size, replace=False)
            W = W_full[idx]
        else:
            idx = rng.choice(W_full.shape[0], size=pop_size, replace=True)
            W = W_full[idx]

    B = neighborhood_by_euclidean(W, T=min(T, W.shape[0]))
    return W, B


def run_single(cfg: MOEADConfig, evaluate_fn, W, B, xl, xu, reference_Z):
    return moead_run(
        config=cfg,
        evaluate_fn=evaluate_fn,
        W=W,
        B=B,
        xl=xl,
        xu=xu,
        reference_Z=reference_Z,
    )


def hv_ref_point_global_2d(F_list: list[np.ndarray], pad: float = 0.1) -> tuple[float, float]:
    """
    Compute a shared 2D HV reference point for comparing runs.

    We take the global maximum of each objective among all nondominated points
    (across all runs), then inflate by (1+pad).

    Minimization is assumed.
    """
    if len(F_list) == 0:
        raise ValueError("F_list is empty.")

    all_nd = []
    for F in F_list:
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or F.shape[1] != 2:
            raise ValueError("Each F must be (N,2) for hv_ref_point_global_2d.")
        nd = filter_nondominated(F)
        all_nd.append(nd)

    A = np.vstack(all_nd)
    worst = np.max(A, axis=0)  
    ref = (float(worst[0] * (1.0 + pad)), float(worst[1] * (1.0 + pad)))
    return ref