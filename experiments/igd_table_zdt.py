from __future__ import annotations
import argparse
import csv
import numpy as np
import os

from moead.problems.zdt import (
    zdt1,
    zdt2,
    zdt3,
    sample_true_pareto_front_zdt1,
    sample_true_pareto_front_zdt2,
    sample_true_pareto_front_zdt3
)
from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run


def run_once(mode: str, seed: int, args, setup, Zref, evaluate_fn) -> float:
    cfg = MOEADConfig(
        n_obj=2,
        n_var=args.n_var,
        pop_size=args.N,
        T=args.T,
        n_gen=args.n_gen,
        seed=seed,
        nr=args.nr,
        eta_c=args.eta_c,
        eta_m=args.eta_m,
        p_c=args.p_c,
        p_m=None,
        variation=mode,
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

    return float(out["history_igd"][-1])


def mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--problem", type=str, default="zdt1", choices=["zdt1", "zdt2", "zdt3"])
    p.add_argument("--N", type=int, default=101)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_gen", type=int, default=200)
    p.add_argument("--n_var", type=int, default=30)
    p.add_argument("--nr", type=int, default=2)
    p.add_argument("--eta_c", type=float, default=20.0)
    p.add_argument("--eta_m", type=float, default=20.0)
    p.add_argument("--p_c", type=float, default=1.0)

    p.add_argument("--seeds", type=int, default=20, help="number of runs")
    p.add_argument("--seed0", type=int, default=1, help="starting seed")
    p.add_argument("--csv", type=str, default=None)

    args = p.parse_args()

    # Select problem
    if args.problem == "zdt1":
        evaluate_fn = zdt1
        Zref = sample_true_pareto_front_zdt1(1000)
        default_csv = "report/igd_zdt1.csv"
    elif args.problem == "zdt2":
        evaluate_fn = zdt2
        Zref = sample_true_pareto_front_zdt2(1000)
        default_csv = "report/igd_zdt2.csv"
    elif args.problem == "zdt3":
        evaluate_fn = zdt3
        Zref = sample_true_pareto_front_zdt3(1000)
        default_csv = "report/igd_zdt3.csv"
    else:
        raise ValueError("Unknown problem")

    if args.csv is None:
        args.csv = default_csv

    setup = build_weight_setup(n_obj=2, N=args.N, T=args.T)
    seed_list = list(range(args.seed0, args.seed0 + args.seeds))

    results = []
    igd_v0 = []
    igd_sbx = []

    print(
        f"Running {args.problem.upper()} IGD table: "
        f"N={args.N}, T={args.T}, n_gen={args.n_gen}, runs={len(seed_list)}"
    )

    for s in seed_list:
        v0 = run_once("placeholder", s, args, setup, Zref, evaluate_fn)
        sbx = run_once("sbx", s, args, setup, Zref, evaluate_fn)

        igd_v0.append(v0)
        igd_sbx.append(sbx)
        results.append((s, v0, sbx))

        print(f"seed={s:3d} | IGD v0={v0:.6f} | IGD SBX={sbx:.6f}")

    m0, s0 = mean_std(igd_v0)
    m1, s1 = mean_std(igd_sbx)

    print("\n=== Summary (final IGD) ===")
    print(f"v0 (placeholder): mean={m0:.6f}, std={s0:.6f}")
    print(f"SBX+poly:         mean={m1:.6f}, std={s1:.6f}")

    print("\nLaTeX row example:")
    print(f"MOEA/D + placeholder & {m0:.6f} $\\pm$ {s0:.6f} \\\\")
    print(f"MOEA/D + SBX+poly    & {m1:.6f} $\\pm$ {s1:.6f} \\\\")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "igd_placeholder", "igd_sbx"])
        w.writerows(results)

    print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()