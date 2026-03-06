from __future__ import annotations
import numpy as np

from problems.dtlz2 import dtlz2, sample_true_pareto_front_dtlz2


def get_dtlz2(M: int, n_var: int, ref_points: int = 3000) -> dict:
    return {
        "n_obj": M,
        "evaluate_fn": (lambda X: dtlz2(X, M=M)),
        "xl": np.zeros(n_var),
        "xu": np.ones(n_var),
        "reference_Z": sample_true_pareto_front_dtlz2(M=M, K=ref_points),
    }