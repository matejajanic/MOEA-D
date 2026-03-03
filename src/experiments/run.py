from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from moead.algorithm import MOEADConfig

from experiments.common import make_run_dir, save_config, build_WB, run_single
from experiments.plotting import plot_2d_front, plot_3d_front, plot_history
from experiments.benchmarks.zdt_suite import get_zdt
from experiments.benchmarks.dtlz_suite import get_dtlz2
from experiments.real_world.feature_suite import get_feature_problem
from experiments.real_world.project_suite import get_project_problem
from experiments.comparisons.variation_compare import run_variation_compare
from experiments.comparisons.metrics_compare import run_metrics_over_seeds


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def add_moead_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pop_size", type=int, required=True)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nr", type=int, default=2)

    p.add_argument("--encoding", type=str, default="real", choices=["real", "binary"])
    p.add_argument("--variation", type=str, default="sbx", choices=["sbx", "placeholder"])

    p.add_argument("--eta_c", type=float, default=20.0)
    p.add_argument("--eta_m", type=float, default=20.0)
    p.add_argument("--p_c", type=float, default=1.0)
    p.add_argument("--p_m", type=float, default=None)


def print_metrics_table(rows: list[dict], summary: dict) -> None:
    print("\n=== Metrics over seeds ===")
    print("seed\tIGD_final\tHV")
    for r in rows:
        igd = r["igd_final"]
        hv = r["hv"]
        igd_s = f"{igd:.6g}" if np.isfinite(igd) else "NA"
        hv_s = f"{hv:.6g}" if np.isfinite(hv) else "NA"
        print(f"{r['seed']}\t{igd_s}\t{hv_s}")

    print("-" * 40)
    igd_mean = summary["igd_mean"]
    igd_std = summary["igd_std"]
    hv_mean = summary["hv_mean"]
    hv_std = summary["hv_std"]

    igd_mean_s = f"{igd_mean:.6g}" if np.isfinite(igd_mean) else "NA"
    igd_std_s = f"{igd_std:.6g}" if np.isfinite(igd_std) else "NA"
    hv_mean_s = f"{hv_mean:.6g}" if np.isfinite(hv_mean) else "NA"
    hv_std_s = f"{hv_std:.6g}" if np.isfinite(hv_std) else "NA"

    print(f"IGD: mean={igd_mean_s}  std={igd_std_s}")
    print(f"HV : mean={hv_mean_s}  std={hv_std_s}")
    if summary.get("hv_ref_point") is not None:
        print(f"HV ref point (shared): {summary['hv_ref_point']}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser("experiments runner")

    parser.add_argument("--suite", type=str, required=True, choices=["benchmark", "realworld", "compare"])
    parser.add_argument("--problem", type=str, required=True)

    parser.add_argument("--H", type=int, default=None, help="Simplex-lattice resolution (needed for n_obj>=3)")

    parser.add_argument("--n_var", type=int, default=30, help="Decision variables for real-coded benchmarks (ZDT/DTLZ)")
    parser.add_argument("--M", type=int, default=3, help="Objectives for DTLZ2")

    parser.add_argument("--n_projects", type=int, default=100)
    parser.add_argument("--budget_ratio", type=float, default=0.4)

    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--hv_pad", type=float, default=0.1, help="Padding for shared HV ref point.")

    parser.add_argument("--save", action="store_true", help="Save artifacts (npz/png/csv/json). Default: show plots + print only.")
    parser.add_argument("--out", type=str, default="experiments/results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Less terminal output.")

    add_moead_args(parser)
    args = parser.parse_args()

    suite = args.suite
    prob = args.problem.lower()

    run_dir: Path | None = None
    if args.save:
        run_dir = make_run_dir(args.out, args.tag)

    # ---------- Build problem ----------
    evaluate_fn = None
    reference_Z = None
    xl = None
    xu = None
    n_obj = None
    n_var = None

    extra = {"suite": suite, "problem": prob}

    if suite == "benchmark":
        if prob.startswith("zdt"):
            if prob == "zdt5":
                pack = get_zdt(prob, n_var=80)
                n_obj = pack["n_obj"]
                n_var = 80
                evaluate_fn = pack["evaluate_fn"]
                xl, xu, reference_Z = pack["xl"], pack["xu"], pack["reference_Z"]
                args.encoding = "binary"
            else:
                pack = get_zdt(prob, n_var=args.n_var)
                n_obj = pack["n_obj"]
                n_var = args.n_var
                evaluate_fn = pack["evaluate_fn"]
                xl, xu, reference_Z = pack["xl"], pack["xu"], pack["reference_Z"]
                args.encoding = "real"

        elif prob == "dtlz2":
            pack = get_dtlz2(M=args.M, n_var=args.n_var)
            n_obj = pack["n_obj"]
            n_var = args.n_var
            evaluate_fn = pack["evaluate_fn"]
            xl, xu, reference_Z = pack["xl"], pack["xu"], pack["reference_Z"]
            args.encoding = "real"
        else:
            raise ValueError(f"Unknown benchmark: {prob}")

    elif suite == "realworld":
        if prob in ["feature", "feature_selection"]:
            pack = get_feature_problem(seed=args.seed)
            n_obj = pack["n_obj"]
            n_var = pack["n_var"]
            evaluate_fn = pack["evaluate_fn"]
            xl, xu, reference_Z = None, None, None
            args.encoding = "binary"
            extra["baseline_accuracy"] = pack["baseline_acc"]

        elif prob in ["project", "project_selection"]:
            pack = get_project_problem(n_projects=args.n_projects, budget_ratio=args.budget_ratio, seed=args.seed)
            n_obj = pack["n_obj"]
            n_var = pack["n_var"]
            evaluate_fn = pack["evaluate_fn"]
            xl, xu, reference_Z = None, None, None
            args.encoding = "binary"
            extra["budget"] = pack["budget"]
            extra["n_projects"] = args.n_projects
            extra["budget_ratio"] = args.budget_ratio
        else:
            raise ValueError(f"Unknown real-world problem: {prob}")

    elif suite == "compare":
        if prob.startswith("zdt") and prob != "zdt5":
            pack = get_zdt(prob, n_var=args.n_var)
            n_obj = pack["n_obj"]
            n_var = args.n_var
            evaluate_fn = pack["evaluate_fn"]
            xl, xu, reference_Z = pack["xl"], pack["xu"], pack["reference_Z"]
            args.encoding = "real"
        elif prob == "dtlz2":
            pack = get_dtlz2(M=args.M, n_var=args.n_var)
            n_obj = pack["n_obj"]
            n_var = args.n_var
            evaluate_fn = pack["evaluate_fn"]
            xl, xu, reference_Z = pack["xl"], pack["xu"], pack["reference_Z"]
            args.encoding = "real"
        else:
            raise ValueError("compare suite supports real-coded benchmarks (ZDT1-4,6 and DTLZ2).")
    else:
        raise ValueError(f"Unknown suite: {suite}")

    # ---------- Build config ----------
    cfg = MOEADConfig(
        n_obj=n_obj,
        n_var=n_var,
        pop_size=args.pop_size,
        T=args.T,
        n_gen=args.n_gen,
        seed=args.seed,
        nr=args.nr,
        encoding=args.encoding,
        eta_c=args.eta_c,
        eta_m=args.eta_m,
        p_c=args.p_c,
        p_m=args.p_m,
        variation=args.variation,
    )

    if run_dir is not None:
        save_config(run_dir, cfg, extra=extra)

    # ---------- Dispatch ----------
    if suite in ["benchmark", "realworld"]:
        W, B = build_WB(n_obj=cfg.n_obj, pop_size=cfg.pop_size, T=cfg.T, H=args.H)
        out = run_single(cfg, evaluate_fn, W, B, xl, xu, reference_Z)

        igd_final = None
        if out["history_igd"] is not None and len(out["history_igd"]) > 0:
            igd_final = float(out["history_igd"][-1])

        if not args.quiet:
            print(f"\n=== Run summary ===")
            print(f"suite={suite} problem={prob} encoding={cfg.encoding} variation={cfg.variation}")
            print(f"pop={cfg.pop_size} T={cfg.T} n_gen={cfg.n_gen} seed={cfg.seed} nr={cfg.nr}")
            print(f"final IGD: {igd_final if igd_final is not None else 'NA'}")
            print(f"total replacements: {int(np.sum(out['history_replaced']))}")

        # ---- ALWAYS visualize: save to run_dir if available, else show interactively ----
        viz_out = run_dir  # None -> show()

        if out["F"].shape[1] == 2:
            plot_2d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")
        elif out["F"].shape[1] == 3:
            plot_3d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")
        plot_history(viz_out, out["history_replaced"], out["history_igd"])

        # ---- Save artifacts only if --save ----
        if run_dir is not None:
            np.savez(
                run_dir / "result.npz",
                X=out["X"],
                F=out["F"],
                z=out["z"],
                history_replaced=out["history_replaced"],
                history_igd=out["history_igd"] if out["history_igd"] is not None else np.array([]),
            )
            print(f"[OK] Saved to: {run_dir}")
        else:
            print("[OK] (Nothing saved; use --save to write files.)")

    elif suite == "compare":
        seeds = parse_int_list(args.seeds)

        if not args.quiet:
            print(f"\n=== Compare mode ===")
            print(f"problem={prob} pop={cfg.pop_size} T={cfg.T} n_gen={cfg.n_gen}")
            print(f"seeds={seeds}")

        v_save_dir = (run_dir / "variation_compare") if run_dir is not None else None
        v_results = run_variation_compare(
            base_cfg=cfg,
            evaluate_fn=evaluate_fn,
            xl=xl,
            xu=xu,
            reference_Z=reference_Z,
            title_prefix=f"compare:{prob}",
            save_dir=v_save_dir,
            H=args.H,
        )

        if not args.quiet:
            for var in ["sbx", "placeholder"]:
                out = v_results[var]
                igd_final = None
                if out["history_igd"] is not None and len(out["history_igd"]) > 0:
                    igd_final = float(out["history_igd"][-1])
                print(f"{var}: final IGD = {igd_final if igd_final is not None else 'NA'}")

        m_save_dir = (run_dir / "metrics_compare") if run_dir is not None else None
        rows, summary = run_metrics_over_seeds(
            base_cfg=cfg,
            evaluate_fn=evaluate_fn,
            xl=xl,
            xu=xu,
            reference_Z=reference_Z,
            seeds=seeds,
            H=args.H,
            hv_pad=args.hv_pad,
            save_dir=m_save_dir,
        )
        print_metrics_table(rows, summary)

        if run_dir is not None:
            print(f"[OK] Compare saved to: {run_dir}")
        else:
            print("[OK] (Nothing saved; use --save to write files.)")

    else:
        raise RuntimeError("unreachable")


if __name__ == "__main__":
    main()