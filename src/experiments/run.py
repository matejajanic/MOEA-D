from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from moead.algorithm import MOEADConfig

from experiments.common import make_run_dir, save_config, build_WB, run_single
from experiments.plotting import (
    plot_2d_front,
    plot_3d_front,
    plot_history,
    plot_feature_accuracy_vs_k,
    plot_project_tradeoff,
    plot_textsum_3obj,
    plot_textsum_best_similarity_per_k,
    plot_textsum_3d,
)

from experiments.benchmarks.zdt_suite import get_zdt
from experiments.benchmarks.dtlz_suite import get_dtlz2

from experiments.real_world.feature_suite import get_feature_problem
from experiments.real_world.project_suite import get_project_problem
from experiments.real_world.text_suite import get_textsum_problem

from experiments.comparisons.variation_compare import run_variation_compare
from experiments.comparisons.metrics_compare import run_metrics_over_seeds


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def add_moead_args(p: argparse.ArgumentParser) -> None:
    
    p.add_argument("--pop_size", type=int, default=200)
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

    parser.add_argument("--H", type=int, default=None, help="Simplex-lattice resolution (n_obj>=3).")

    parser.add_argument("--n_var", type=int, default=30, help="Decision vars for real-coded benchmarks (ZDT/DTLZ).")
    parser.add_argument("--M", type=int, default=3, help="Objectives for DTLZ2.")

    parser.add_argument("--n_projects", type=int, default=100)
    parser.add_argument("--budget_ratio", type=float, default=0.4)

    parser.add_argument("--text_path", type=str, default=None, help="Path to UTF-8 text file for text summarization.")
    parser.add_argument("--max_sentences", type=int, default=40, help="Max sentences to consider (text summarization).")
    parser.add_argument("--tfidf_max_features", type=int, default=5000, help="TF-IDF max features (text summarization).")
    parser.add_argument(
        "--max_comp",
        type=float,
        default=None,
        help="Max allowed compression ratio in (0,1], e.g. 0.3 (text summarization).",
    )
    parser.add_argument("--print_summary", action="store_true", help="Print one selected summary to terminal.")

    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--hv_pad", type=float, default=0.1, help="Padding for shared HV reference point (compare).")

    parser.add_argument("--save", action="store_true", help="Save artifacts (npz/png/json). Default: show plots + print.")
    parser.add_argument("--out", type=str, default="experiments/results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Less terminal output.")
    parser.add_argument(
        "--no_show_replacements",
        action="store_true",
        help="Do not show/save the 'replacements per generation' plot.",
    )

    add_moead_args(parser)
    args = parser.parse_args()

    suite = args.suite
    prob = args.problem.lower()

    run_dir: Path | None = None
    if args.save:
        run_dir = make_run_dir(args.out, args.tag)

    evaluate_fn = None
    analyze_fn = None
    build_summary_fn = None
    reference_Z = None
    xl = None
    xu = None
    n_obj = None
    n_var = None

    extra: dict = {"suite": suite, "problem": prob}

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
            analyze_fn = pack.get("analyze_fn", None)
            args.encoding = "binary"
            extra["baseline_accuracy"] = pack.get("baseline_acc", None)

        elif prob in ["project", "project_selection"]:
            pack = get_project_problem(n_projects=args.n_projects, budget_ratio=args.budget_ratio, seed=args.seed)
            n_obj = pack["n_obj"]
            n_var = pack["n_var"]
            evaluate_fn = pack["evaluate_fn"]
            analyze_fn = pack.get("analyze_fn", None)
            args.encoding = "binary"
            extra["budget"] = pack.get("budget", None)
            extra["n_projects"] = pack.get("n_projects", args.n_projects)
            extra["budget_ratio"] = pack.get("budget_ratio", args.budget_ratio)

        elif prob in ["textsum", "summarization", "text_summarization"]:
            pack = get_textsum_problem(
                text_path=args.text_path,
                seed=args.seed,
                max_sentences=args.max_sentences,
                tfidf_max_features=args.tfidf_max_features,
                max_comp=args.max_comp,
            )
            n_obj = pack["n_obj"]  # 3
            n_var = pack["n_var"]
            evaluate_fn = pack["evaluate_fn"]
            analyze_fn = pack.get("analyze_fn", None)
            build_summary_fn = pack.get("build_summary_fn", None)
            args.encoding = "binary"
            extra["n_sent"] = pack.get("n_sent", None)
            extra["text_path"] = pack.get("text_path", None)
            extra["max_sentences"] = args.max_sentences
            extra["tfidf_max_features"] = args.tfidf_max_features
            extra["max_comp"] = args.max_comp
            extra["max_k"] = pack.get("max_k", None)

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

    if suite in ["benchmark", "realworld"]:
        W, B = build_WB(n_obj=cfg.n_obj, pop_size=cfg.pop_size, T=cfg.T, H=args.H, seed=cfg.seed)
        out = run_single(cfg, evaluate_fn, W, B, xl, xu, reference_Z)

        igd_final = None
        if out["history_igd"] is not None and len(out["history_igd"]) > 0:
            igd_final = float(out["history_igd"][-1])

        if not args.quiet:
            print("\n=== Run summary ===")
            if cfg.encoding == "real":
                print(f"suite={suite} problem={prob} encoding={cfg.encoding} variation={cfg.variation}")
            else:
                print(f"suite={suite} problem={prob} encoding={cfg.encoding}")
            print(f"pop={cfg.pop_size} T={cfg.T} n_gen={cfg.n_gen} seed={cfg.seed} nr={cfg.nr}")
            print(f"final IGD: {igd_final if igd_final is not None else 'NA'}")
            print(f"total replacements: {int(np.sum(out['history_replaced']))}")
            if prob in ["textsum", "summarization", "text_summarization"]:
                if extra.get("max_k") is not None:
                    print(f"max_comp={extra.get('max_comp')} => max_k={extra.get('max_k')}")

        viz_out = run_dir  

        if suite == "realworld" and prob in ["feature", "feature_selection"]:
            plot_feature_accuracy_vs_k(
                viz_out,
                X_bin=out["X"],
                F=out["F"],
                title="Feature selection: Accuracy vs #Selected Features",
            )

        elif suite == "realworld" and prob in ["project", "project_selection"]:
            if analyze_fn is not None:
                analysis = analyze_fn(out["X"])
                plot_project_tradeoff(
                    viz_out,
                    analysis=analysis,
                    title="Project selection: Profit vs Risk",
                    budget=extra.get("budget", None),
                )
            else:
                if out["F"].shape[1] == 2:
                    plot_2d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")

        elif suite == "realworld" and prob in ["textsum", "summarization", "text_summarization"]:
            analysis = analyze_fn(out["X"]) if analyze_fn is not None else None
            if analysis is not None:
                try:
                    plot_textsum_3d(viz_out, analysis, out["F"], title="Text summarization")
                except Exception:
                    pass

                plot_textsum_3obj(viz_out, analysis, title="Text summarization")
                plot_textsum_best_similarity_per_k(viz_out, analysis)

                if args.print_summary and build_summary_fn is not None and not args.quiet:
                    sim = np.asarray(analysis["sim"], dtype=float)
                    k = np.asarray(analysis["k"], dtype=int)

                    max_k = extra.get("max_k", None)
                    if max_k is not None:
                        feasible = np.where(k <= int(max_k))[0]
                    else:
                        feasible = np.arange(k.shape[0])

                    if feasible.size == 0:
                        feasible = np.arange(k.shape[0])

                    best_local = feasible[np.lexsort((k[feasible], -sim[feasible]))[0]]

                    print("\n--- Selected summary (best similarity, tie-break smaller k) ---")
                    print(f"similarity={sim[best_local]:.4f}, k={k[best_local]}")
                    max_show = min(10, int(k[best_local])) if int(k[best_local]) > 0 else 10
                    print(build_summary_fn(out["X"][best_local], max_sentences=max_show))
                    print("--------------------------------------------------------------\n")

            else:
                if out["F"].shape[1] == 2:
                    plot_2d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")
                elif out["F"].shape[1] == 3:
                    plot_3d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")

        else:
            if out["F"].shape[1] == 2:
                plot_2d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")
            elif out["F"].shape[1] == 3:
                plot_3d_front(viz_out, out["F"], reference_Z, f"{suite}:{prob} front")

        plot_history(
            viz_out,
            out["history_replaced"],
            out["history_igd"],
            show_replacements=not args.no_show_replacements,
        )

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
            print("\n=== Compare mode ===")
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
                outv = v_results[var]
                igd_v = None
                if outv["history_igd"] is not None and len(outv["history_igd"]) > 0:
                    igd_v = float(outv["history_igd"][-1])
                print(f"{var}: final IGD = {igd_v if igd_v is not None else 'NA'}")

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