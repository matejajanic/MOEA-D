from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run
from moead.metrics import igd

from problems.dtlz2 import dtlz2, sample_true_pareto_front_dtlz2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--H", type=int, default=10, help="simplex-lattice parameter for n_obj>=3")
    parser.add_argument("--k", type=int, default=10, help="DTLZ k parameter; n_var = (M-1)+k")
    parser.add_argument("--N", type=int, default=66, help="population/number of subproblems")
    parser.add_argument("--T", type=int, default=15, help="neighborhood size")
    parser.add_argument("--n_gen", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--nr", type=int, default=2)
    parser.add_argument("--eta_c", type=float, default=20.0)
    parser.add_argument("--eta_m", type=float, default=20.0)
    parser.add_argument("--p_c", type=float, default=1.0)

    args = parser.parse_args()

    M = args.M
    n_var = (M - 1) + args.k

    setup = build_weight_setup(n_obj=M, H=args.H, T=args.T)

    Zref = sample_true_pareto_front_dtlz2(M=M, K=4000, rng=np.random.default_rng(123))

    def eval_fn(x: np.ndarray) -> np.ndarray:
        return dtlz2(x, M=M)

    cfg = MOEADConfig(
        n_obj=M,
        n_var=n_var,
        pop_size=setup.W.shape[0],
        T=args.T,
        n_gen=args.n_gen,
        seed=args.seed,
        nr=args.nr,
        eta_c=args.eta_c,
        eta_m=args.eta_m,
        p_c=args.p_c,
        p_m=None,
        variation="sbx",
    )

    xl = np.zeros(n_var)
    xu = np.ones(n_var)

    out = moead_run(
        cfg,
        evaluate_fn=eval_fn,
        W=setup.W,
        B=setup.B,
        xl=xl,
        xu=xu,
        reference_Z=Zref,
    )

    F = out["F"]
    final_igd = float(out["history_igd"][-1]) if out["history_igd"] is not None else igd(F, Zref)
    print(f"Final IGD (DTLZ2, M={M}): {final_igd:.6f}")

    # 3D plot of final objective vectors (works for M=3)
    if M == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=10, label="MOEA/D (SBX)")
        ax.scatter(Zref[:, 0], Zref[:, 1], Zref[:, 2], s=2, alpha=0.2, label="True PF samples")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title("DTLZ2 (M=3): final objective vectors")
        ax.legend()
        plt.show()

    # IGD curve
    if out["history_igd"] is not None:
        plt.figure()
        plt.plot(out["history_igd"])
        plt.xlabel("generation")
        plt.ylabel("IGD (lower is better)")
        plt.title("DTLZ2 (M=3): IGD over generations")
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    main()