from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run
from moead.problems.project_selection import ProjectSelection


def main():

    problem = ProjectSelection(n_projects=100, budget_ratio=0.4)

    setup = build_weight_setup(n_obj=2, N=101, T=20)

    cfg = MOEADConfig(
        n_obj=2,
        n_var=problem.n_projects,
        pop_size=101,
        n_gen=2000,
        encoding="binary",
        p_c=0.9,
        p_m=0.05,
    )

    out = moead_run(
        cfg,
        evaluate_fn=problem.evaluate,
        W=setup.W,
        B=setup.B,
        xl=None,
        xu=None,
    )

    F = out["F"]
    X = out["X"]

    profits = X @ problem.profit
    costs = X @ problem.cost
    risks = X @ problem.risk

    print("\n--- Budget Analysis ---")
    print("Budget:", problem.budget)
    print("Min cost:", costs.min())
    print("Max cost:", costs.max())
    print("Average cost:", costs.mean())
    print("Max violation:", np.max(np.maximum(0, costs - problem.budget)))
    num_selected = X.sum(axis=1)
    print("\nAverage selected projects:", num_selected.mean())
    print("Min selected projects:", num_selected.min())
    print("Max selected projects:", num_selected.max())

    plt.figure(figsize=(7, 5))
    plt.scatter(-F[:, 0], F[:, 1], s=18)  
    plt.xlabel("Total Profit")
    plt.ylabel("Total Risk")
    plt.title("Project Funding – Profit vs Risk")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig("project_selection_result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()