from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moead.problems.zdt import zdt1, sample_true_pareto_front_zdt1

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type = int, default = 200, help = "Population size")
    parser.add_argument("--n", type = int, default = 30, help = "Decision dimension")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--save", type = str, default = "", help = "Path to save plot (png)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.uniform(0.0, 1.0, size = (args.N, args.n))
    F = zdt1(X)

    PF = sample_true_pareto_front_zdt1(300)

    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], s = 12, label = "Random solutions")
    plt.plot(PF[:, 0], PF[:, 1], linewidth = 2.0, label = "True Pareto Front (ZDT1)")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("ZDT1: Random baseline vs true Pareto front")
    plt.legend()
    plt.grid(True, alpha = 0.3)

    if args.save:
        plt.savefig(args.save, dpi = 150, bbox_inches = "tight")
    plt.show()

if __name__ == "__main__":
    main()