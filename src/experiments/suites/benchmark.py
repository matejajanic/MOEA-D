from __future__ import annotations
from experiments.analysis.metrics import compute_igd, compute_hv
from experiments.analysis.statistics import mean_std
from experiments.analysis.plots import plot_pareto, plot_curve
from experiments.core.runner import run_single


def run_benchmark(cfg):

    igd_vals = []
    hv_vals = []

    last_out = None
    spec = None

    for i in range(cfg.runs):
        seed = cfg.seed0 + i
        out, spec = run_single(cfg, seed)
        last_out = out

        F = out["F"]

        if spec.pf_sampler is not None:
            PF = spec.pf_sampler(800)
            igd_vals.append(compute_igd(F, PF))

        if spec.hv_ref is not None:
            hv_vals.append(compute_hv(F, spec.hv_ref))

    summary = {
        "IGD": mean_std(igd_vals) if igd_vals else None,
        "HV": mean_std(hv_vals) if hv_vals else None,
    }

    if cfg.plot != "none":
        F = last_out["F"]
        PF = spec.pf_sampler(800) if spec.pf_sampler else None

        plot_pareto(F, PF, f"{cfg.problem} Pareto", cfg.plot)
        if "history_igd" in last_out:
            plot_curve(last_out["history_igd"], "IGD over generations", "IGD", cfg.plot)

    return summary