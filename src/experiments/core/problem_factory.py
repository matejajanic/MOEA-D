from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

# Benchmarks
from problems.zdt import (
    zdt1, zdt2, zdt3, zdt4, zdt6,
    sample_true_pareto_front_zdt1,
    sample_true_pareto_front_zdt2,
    sample_true_pareto_front_zdt3,
    sample_true_pareto_front_zdt4,
    sample_true_pareto_front_zdt6,
)

# Binary / applications
from problems.zdt5 import zdt5
from problems.feature_selection import FeatureSelection
from problems.project_selection import ProjectSelection


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n_obj: int
    encoding: str  # "real" | "binary"

    # evaluation function
    evaluate_fn: Callable[[np.ndarray], np.ndarray]

    # bounds for real-coded problems (None for binary)
    bounds_fn: Optional[Callable[[int], Tuple[np.ndarray, np.ndarray]]] = None

    # true PF sampler for benchmarks (None for applications)
    pf_sampler: Optional[Callable[[int], np.ndarray]] = None

    # recommended HV reference point for 2D (None if not applicable)
    hv_ref_point: Optional[Tuple[float, float]] = None


def _bounds_zdt_standard(n_var: int):
    xl = np.zeros(n_var)
    xu = np.ones(n_var)
    return xl, xu


def _bounds_zdt4(n_var: int):
    xl = np.full(n_var, -5.0)
    xu = np.full(n_var, 5.0)
    xl[0] = 0.0
    xu[0] = 1.0
    return xl, xu


def get_problem_spec(name: str, *, n_var: int, extra: Optional[Dict[str, Any]] = None) -> ProblemSpec:
    """
    Returns a ProblemSpec for a given problem name.
    For applications, `extra` can contain constructor params (e.g. n_projects, budget_ratio).
    """
    extra = extra or {}

    # ---- BENCHMARK (real-coded) ----
    if name == "zdt1":
        return ProblemSpec(
            name="zdt1", n_obj=2, encoding="real",
            evaluate_fn=zdt1,
            bounds_fn=_bounds_zdt_standard,
            pf_sampler=lambda k: sample_true_pareto_front_zdt1(k),
            hv_ref_point=(1.1, 1.1),
        )
    if name == "zdt2":
        return ProblemSpec(
            name="zdt2", n_obj=2, encoding="real",
            evaluate_fn=zdt2,
            bounds_fn=_bounds_zdt_standard,
            pf_sampler=lambda k: sample_true_pareto_front_zdt2(k),
            hv_ref_point=(1.1, 1.1),
        )
    if name == "zdt3":
        return ProblemSpec(
            name="zdt3", n_obj=2, encoding="real",
            evaluate_fn=zdt3,
            bounds_fn=_bounds_zdt_standard,
            pf_sampler=lambda k: sample_true_pareto_front_zdt3(k),
            hv_ref_point=(1.1, 1.1),
        )
    if name == "zdt4":
        return ProblemSpec(
            name="zdt4", n_obj=2, encoding="real",
            evaluate_fn=zdt4,
            bounds_fn=_bounds_zdt4,
            pf_sampler=lambda k: sample_true_pareto_front_zdt4(k),
            hv_ref_point=(1.1, 1.1),
        )
    if name == "zdt6":
        return ProblemSpec(
            name="zdt6", n_obj=2, encoding="real",
            evaluate_fn=zdt6,
            bounds_fn=_bounds_zdt_standard,
            pf_sampler=lambda k: sample_true_pareto_front_zdt6(k),
            hv_ref_point=(1.1, 1.1),
        )

    # ---- COMBINATORIAL / APPLICATIONS (binary) ----
    if name == "zdt5":
        # NOTE: hv_ref_point here is a placeholder; you can set it after checking objective ranges.
        return ProblemSpec(
            name="zdt5", n_obj=2, encoding="binary",
            evaluate_fn=zdt5,
            bounds_fn=None,
            pf_sampler=None,          # you have your own true PF function; can be added later
            hv_ref_point=None,        # set later when you decide ref_point
        )

    if name == "feature":
        problem = FeatureSelection(**extra)
        return ProblemSpec(
            name="feature", n_obj=2, encoding="binary",
            evaluate_fn=problem.evaluate,
            bounds_fn=None,
            pf_sampler=None,
            hv_ref_point=None,
        )

    if name == "project":
        # defaults match your script
        n_projects = int(extra.get("n_projects", 100))
        budget_ratio = float(extra.get("budget_ratio", 0.4))
        problem = ProjectSelection(n_projects=n_projects, budget_ratio=budget_ratio)
        return ProblemSpec(
            name="project", n_obj=2, encoding="binary",
            evaluate_fn=problem.evaluate,
            bounds_fn=None,
            pf_sampler=None,
            hv_ref_point=None,
        )

    raise ValueError(f"Unknown problem name: {name}")