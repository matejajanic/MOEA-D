from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moead.problems.zdt import (
    zdt1,
    zdt2,
    zdt3,
    zdt4,
    sample_true_pareto_front_zdt1,
    sample_true_pareto_front_zdt2,
    sample_true_pareto_front_zdt3,
    sample_true_pareto_front_zdt4,
)
from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run


def bounds_for_problem(problem: str, n_var: int):
    if problem in ("zdt1", "zdt2", "zdt3"):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
    elif problem == "zdt4":
        xl = np.full(n_var, -5.0)
        xu = np.full(n_var, 5.0)
        xl[0] = 0.0
        xu[0] = 1.0
    else:
        raise ValueError("Unknown problem")
    return xl, xu


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=101)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_var", type=int, default=30)
    parser.add_argument(
        "--variation",
        type=str,
        default="sbx",
        choices=["placeholder", "sbx"],
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="zdt1",
        choices=["zdt1", "zdt2", "zdt3", "zdt4"],
    )

    args = parser.parse_args()

    setup = build_weight_setup(n_obj=2, N=args.N, T=args.T)

    cfg = MOEADConfig(
        n_obj=2,
        n_var=args.n_var,
        pop_size=args.N,
        T=args.T,
        n_gen=args.n_gen,
        seed=args.seed,
        nr=2,
        variation=args.variation,
    )


    if args.problem == "zdt1":
        evaluate_fn = zdt1
        PF = sample_true_pareto_front_zdt1(400)
    elif args.problem == "zdt2":
        evaluate_fn = zdt2
        PF = sample_true_pareto_front_zdt2(400)
    elif args.problem == "zdt3":
        evaluate_fn = zdt3
        PF = sample_true_pareto_front_zdt3(400)
    elif args.problem == "zdt4":
        evaluate_fn = zdt4
        PF = sample_true_pareto_front_zdt4(400)
    else:
        raise ValueError("Unknown problem.")

    xl, xu = bounds_for_problem(args.problem, args.n_var)

    out = moead_run(
        cfg,
        evaluate_fn=evaluate_fn,
        W=setup.W,
        B=setup.B,
        xl=xl,
        xu=xu,
    )

    F = out["F"]

    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], s=12, label="MOEA/D")
    plt.plot(PF[:, 0], PF[:, 1], linewidth=2.0, label="True Pareto front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(f"{args.problem.upper()}: MOEA/D ({args.variation})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure()
    plt.plot(out["history_replaced"])
    plt.xlabel("generation")
    plt.ylabel("#replacements")
    plt.title("MOEA/D: replacements per generation")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()