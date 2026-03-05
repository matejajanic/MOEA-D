from __future__ import annotations
from problems.project_selection import ProjectSelection


def get_project_problem(n_projects: int = 100, budget_ratio: float = 0.4, seed: int = 42) -> dict:
    ps = ProjectSelection(n_projects=n_projects, budget_ratio=budget_ratio, seed=seed)
    return {
        "n_obj": 2,
        "n_var": ps.n_projects,
        "evaluate_fn": ps.evaluate,
        "analyze_fn": ps.analyze,
        "budget": ps.budget,
        "n_projects": ps.n_projects,
        "budget_ratio": budget_ratio,
    }