from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run
from moead.problems.zdt5 import zdt5


def true_pareto_front():
    f1 = np.arange(1, 32)  
    f2 = 10.0 / f1
    return np.column_stack([f1, f2])



def compute_igd(F: np.ndarray, PF: np.ndarray) -> float:
    distances = []
    for pf_point in PF:
        d = np.linalg.norm(F - pf_point, axis=1)
        distances.append(np.min(d))
    return np.mean(distances)




def count_optimal_g(X: np.ndarray) -> int:
    count = 0
    for x in X:
        g = 0
        for i in range(10):
            start = 30 + i * 5
            end = start + 5
            ui = np.sum(x[start:end])
            vi = 1.0 if ui == 5 else 2.0 + ui
            g += vi
        if g == 10:
            count += 1
    return count




def main():

    n_bits = 80

    setup = build_weight_setup(n_obj=2, N=101, T=20)

    cfg = MOEADConfig(
        n_obj=2,
        n_var=n_bits,
        pop_size=101,
        n_gen=4000,          
        encoding="binary",
        p_c=0.9,             
        p_m=0.05,            
    )

    out = moead_run(
        cfg,
        evaluate_fn=zdt5,
        W=setup.W,
        B=setup.B,
        xl=None,
        xu=None,
    )

    X = out["X"]
    F = out["F"]

    PF = true_pareto_front()

    igd_value = compute_igd(F, PF)
    optimal_count = count_optimal_g(X)

    print("\n--- ZDT5 ANALYSIS ---")
    print("IGD:", igd_value)
    print("Number of solutions with g = 10:", optimal_count)
    print("Population size:", len(X))


    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=18, label="MOEA/D")
    plt.plot(PF[:, 0], PF[:, 1], 'r--', linewidth=2, label="True Pareto Front")

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("ZDT5 - Binary MOEA/D vs True Pareto Front")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()