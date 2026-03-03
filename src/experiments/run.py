from __future__ import annotations
import argparse

from experiments.core.config import ExperimentConfig
from experiments.suites.benchmark import run_benchmark
from experiments.suites.applications import run_application
from experiments.suites.variation import run_variation_study
from experiments.suites.neighborhood import run_neighborhood_study


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--suite",
               required=True,
               choices=["benchmark", "applications", "variation", "neighborhood"])
    p.add_argument("--problem", required=True)

    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed0", type=int, default=1)

    p.add_argument("--N", type=int, default=101)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_gen", type=int, default=200)
    p.add_argument("--nr", type=int, default=2)

    p.add_argument("--n_var", type=int, default=30)
    p.add_argument("--encoding", default="real", choices=["real", "binary"])
    p.add_argument("--variation", default="sbx", choices=["placeholder", "sbx"])

    p.add_argument("--eta_c", type=float, default=20.0)
    p.add_argument("--eta_m", type=float, default=20.0)
    p.add_argument("--p_c", type=float, default=1.0)
    p.add_argument("--p_m", type=float, default=None)

    p.add_argument("--plot", default="show", choices=["show", "save", "none"])
    p.add_argument("--save_tag", default="")

    args = p.parse_args()

    cfg = ExperimentConfig(
        suite=args.suite,
        problem=args.problem,
        runs=args.runs,
        seed0=args.seed0,
        N=args.N,
        T=args.T,
        n_gen=args.n_gen,
        nr=args.nr,
        n_var=args.n_var,
        encoding=args.encoding,
        variation=args.variation,
        eta_c=args.eta_c,
        eta_m=args.eta_m,
        p_c=args.p_c,
        p_m=args.p_m,
        plot=args.plot,
        save_tag=args.save_tag,
    )

    if cfg.suite == "benchmark":
        res = run_benchmark(cfg)

    elif cfg.suite == "applications":
        res = run_application(cfg)

    elif cfg.suite == "variation":
        res = run_variation_study(cfg)

    elif cfg.suite == "neighborhood":
        res = run_neighborhood_study(cfg)

    print("\n=== SUMMARY ===")
    print(res)


if __name__ == "__main__":
    main()