from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moead.problems.zdt import (
    zdt1,
    zdt2,
    zdt3,
    zdt4,
    zdt6,
    sample_true_pareto_front_zdt1,
    sample_true_pareto_front_zdt2,
    sample_true_pareto_front_zdt3,
    sample_true_pareto_front_zdt4,
    sample_true_pareto_front_zdt6
)


def generate_random_population(problem: str, N: int, n: int, rng):
    if problem in ("zdt1", "zdt2", "zdt3"):
        return rng.uniform(0.0, 1.0, size=(N, n))
    elif problem == "zdt4":
        X = rng.uniform(-5.0, 5.0, size=(N, n))
        X[:, 0] = rng.uniform(0.0, 1.0, size=N)
        return X
    else:
        raise ValueError("Unknown problem")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--problem",
        type=str,
        default="zdt1",
        choices=["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
    )
    parser.add_argument("--N", type=int, default=200, help="Population size")
    parser.add_argument("--n", type=int, default=30, help="Decision dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="", help="Path to save plot (png)")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    X = generate_random_population(args.problem, args.N, args.n, rng)

    if args.problem == "zdt1":
        evaluate_fn = zdt1
        PF = sample_true_pareto_front_zdt1(300)
    elif args.problem == "zdt2":
        evaluate_fn = zdt2
        PF = sample_true_pareto_front_zdt2(300)
    elif args.problem == "zdt3":
        evaluate_fn = zdt3
        PF = sample_true_pareto_front_zdt3(300)
    elif args.problem == "zdt4":
        evaluate_fn = zdt4
        PF = sample_true_pareto_front_zdt4(300)
    elif args.problem == "zdt6":
        evaluate_fn = zdt6
        PF = sample_true_pareto_front_zdt6(300)
    else:
        raise ValueError("Unknown problem")

    F = evaluate_fn(X)

    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], s=12, label="Random solutions")
    plt.plot(PF[:, 0], PF[:, 1], linewidth=2.0, label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(f"{args.problem.upper()}: Random baseline vs true Pareto front")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()