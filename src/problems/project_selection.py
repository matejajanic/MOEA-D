from __future__ import annotations
import numpy as np


class ProjectSelection:

    def __init__(self, n_projects: int = 100, budget_ratio: float = 0.4, seed: int = 42):

        rng = np.random.default_rng(seed)

        self.n_projects = n_projects

        # Realistic correlated data:
        # Higher profit tends to have higher cost and risk
        self.profit = rng.integers(10, 100, size=n_projects)
        self.cost = self.profit + rng.integers(5, 30, size=n_projects)
        self.risk = self.profit / 100 + rng.uniform(0, 0.2, size=n_projects)

        self.total_possible_cost = np.sum(self.cost)
        self.budget = budget_ratio * self.total_possible_cost

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