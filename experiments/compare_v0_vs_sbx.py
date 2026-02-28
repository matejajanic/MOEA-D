from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moead.problems.zdt import (
    zdt1,
    zdt2,
    sample_true_pareto_front_zdt1,
    sample_true_pareto_front_zdt2,
)
from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run

def run_mode(mode: str, args, setup, Zref, evaluate_fn):
    cfg = MOEADConfig(
        n_obj = 2,
        n_var = args.n_var,
        pop_size = args.N,
        T = args.T,
        n_gen = args.n_gen,
        seed = args.seed,
        nr = args.nr,
        eta_c = args.eta_c,
        eta_m = args.eta_m,
        p_c = args.p_c,
        p_m = None,
        variation = mode
    )

    xl = np.zeros(args.n_var)
    xu = np.ones(args.n_var)

    out = moead_run(
        cfg,
        evaluate_fn=evaluate_fn,
        W=setup.W,
        B=setup.B,
        xl=xl,
        xu=xu,
        reference_Z=Zref,
    )

    return out

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, default="zdt1", choices=["zdt1", "zdt2"])
    parser.add_argument("--N", type=int, default=101)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_var", type=int, default=30)
    parser.add_argument("--nr", type=int, default=2)
    parser.add_argument("--eta_c", type=float, default=20.0)
    parser.add_argument("--eta_m", type=float, default=20.0)
    parser.add_argument("--p_c", type=float, default=1.0)

    args = parser.parse_args()

    setup = build_weight_setup(n_obj=2, N=args.N, T=args.T)

    if args.problem == "zdt1":
        evaluate_fn = zdt1
        Zref = sample_true_pareto_front_zdt1(500)
    elif args.problem == "zdt2":
        evaluate_fn = zdt2
        Zref = sample_true_pareto_front_zdt2(500)
    else:
        raise ValueError("Unknown problem")

    out_v0 = run_mode("placeholder", args, setup, Zref, evaluate_fn)
    out_sbx = run_mode("sbx", args, setup, Zref, evaluate_fn)

    print(f"\nFinal IGD ({args.problem.upper()}):")
    print(f"  v0 / placeholder: {out_v0['history_igd'][-1]:.6f}")
    print(f"  SBX + poly:       {out_sbx['history_igd'][-1]:.6f}")

    plt.figure()
    plt.scatter(out_v0["F"][:, 0], out_v0["F"][:, 1], s=10, label="v0 (placeholder)")
    plt.scatter(out_sbx["F"][:, 0], out_sbx["F"][:, 1], s=10, label="SBX + poly")
    plt.plot(Zref[:, 0], Zref[:, 1], linewidth=2.0, label="True PF")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{args.problem.upper()} – v0 vs SBX (final Pareto fronts)")
    plt.show()

    plt.figure()
    plt.plot(out_v0["history_igd"], label="IGD v0")
    plt.plot(out_sbx["history_igd"], label="IGD SBX")
    plt.xlabel("generation")
    plt.ylabel("IGD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{args.problem.upper()} – IGD over generations")
    plt.show()

if __name__ == "__main__":
    main()