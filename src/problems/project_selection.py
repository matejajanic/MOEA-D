from __future__ import annotations
import numpy as np


class ProjectSelection:

    def __init__(self, n_projects: int = 100, budget_ratio: float = 0.4, seed: int = 42):

        rng = np.random.default_rng(seed)

        self.n_projects = n_projects

        self.profit = rng.integers(10, 100, size=n_projects)
        self.cost = self.profit + rng.integers(5, 30, size=n_projects)
        self.risk = self.profit / 100 + rng.uniform(0, 0.2, size=n_projects)

        self.total_possible_cost = float(np.sum(self.cost))
        self.budget = float(budget_ratio * self.total_possible_cost)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        X: binary matrix (pop_size, n_projects)
        returns F matrix (pop_size, 2)
        """

        profits = X @ self.profit
        costs = X @ self.cost
        risks = X @ self.risk

        penalty = np.maximum(0, costs - self.budget)

        f1 = -profits + 1000 * penalty
        f2 = risks + 10 * penalty

        return np.column_stack([f1, f2])

    def analyze(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return raw, interpretable values for plotting/reporting.
        """
        profits = (X @ self.profit).astype(float)
        costs = (X @ self.cost).astype(float)
        risks = (X @ self.risk).astype(float)
        penalty = np.maximum(0.0, costs - self.budget)
        feasible = penalty <= 1e-12
        k = np.sum(X, axis=1).astype(int)

        return {
            "profit": profits,
            "cost": costs,
            "risk": risks,
            "penalty": penalty,
            "feasible": feasible,
            "k": k,
        }