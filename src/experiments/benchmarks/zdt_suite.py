from __future__ import annotations

import numpy as np

from moead.algorithm import MOEADConfig
from problems.zdt import (
    zdt1, zdt2, zdt3, zdt4, zdt6,
    sample_true_pareto_front_zdt1,
    sample_true_pareto_front_zdt2,
    sample_true_pareto_front_zdt3,
    sample_true_pareto_front_zdt4,
    sample_true_pareto_front_zdt6,
)
from problems.zdt5 import zdt5, sample_true_pareto_front_zdt5


def get_zdt(problem: str, n_var: int) -> dict:
    """
    Returns dict with:
      evaluate_fn, n_obj, xl, xu, reference_Z (optional)
      encoding/variation suggestions handled in run.py
    """
    problem = problem.lower()

    if problem == "zdt1":
        return {
            "n_obj": 2,
            "evaluate_fn": zdt1,
            "xl": np.zeros(n_var),
            "xu": np.ones(n_var),
            "reference_Z": sample_true_pareto_front_zdt1(400),
        }
    if problem == "zdt2":
        return {
            "n_obj": 2,
            "evaluate_fn": zdt2,
            "xl": np.zeros(n_var),
            "xu": np.ones(n_var),
            "reference_Z": sample_true_pareto_front_zdt2(400),
        }
    if problem == "zdt3":
        return {
            "n_obj": 2,
            "evaluate_fn": zdt3,
            "xl": np.zeros(n_var),
            "xu": np.ones(n_var),
            "reference_Z": sample_true_pareto_front_zdt3(500),
        }
    if problem == "zdt4":
        xl = np.full(n_var, -5.0)
        xu = np.full(n_var, 5.0)
        xl[0] = 0.0
        xu[0] = 1.0
        return {
            "n_obj": 2,
            "evaluate_fn": zdt4,
            "xl": xl,
            "xu": xu,
            "reference_Z": sample_true_pareto_front_zdt4(400),
        }
    if problem == "zdt5":
        return {
            "n_obj": 2,
            "evaluate_fn": zdt5,
            "xl": None,
            "xu": None,
            "reference_Z": sample_true_pareto_front_zdt5(),  # usually not provided for ZDT5 here
        }
    if problem == "zdt6":
        return {
            "n_obj": 2,
            "evaluate_fn": zdt6,
            "xl": np.zeros(n_var),
            "xu": np.ones(n_var),
            "reference_Z": sample_true_pareto_front_zdt6(400),
        }

    raise ValueError(f"Unknown ZDT problem: {problem}")