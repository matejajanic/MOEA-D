from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run
from experiments.core.config import ExperimentConfig
from experiments.core.problem_factory import get_problem_spec


def run_single(cfg: ExperimentConfig, seed: int, problem_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs ONE MOEA/D execution for a given config + seed.
    Returns `out` from moead_run plus extra metadata.
    """
    spec = get_problem_spec(cfg.problem, n_var=cfg.n_var, extra=problem_extra)

    # weights + neighborhood setup (future-proof)
    setup = build_weight_setup(n_obj=spec.n_obj, N=cfg.N, T=cfg.T)

    # bounds
    if spec.encoding == "real":
        xl, xu = spec.bounds_fn(cfg.n_var) if spec.bounds_fn else (None, None)
    else:
        xl, xu = None, None

    moead_cfg = MOEADConfig(
        n_obj=spec.n_obj,
        n_var=cfg.n_var,
        pop_size=cfg.N,
        T=cfg.T,
        n_gen=cfg.n_gen,
        seed=seed,
        nr=cfg.nr,
        variation=cfg.variation,

        # SBX/poly params (safe even if placeholder ignores)
        eta_c=cfg.eta_c,
        eta_m=cfg.eta_m,
        p_c=cfg.p_c,
        p_m=cfg.p_m,

        # binary flag
        encoding=spec.encoding,
    )

    out = moead_run(
        moead_cfg,
        evaluate_fn=spec.evaluate_fn,
        W=setup.W,
        B=setup.B,
        xl=xl,
        xu=xu,
    )

    # attach metadata
    out["_meta"] = {
        "problem": cfg.problem,
        "encoding": spec.encoding,
        "seed": seed,
        "variation": cfg.variation,
        "N": cfg.N,
        "T": cfg.T,
        "n_gen": cfg.n_gen,
    }
    return out