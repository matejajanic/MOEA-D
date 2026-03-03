from __future__ import annotations
from experiments.analysis.metrics import compute_hv
from experiments.analysis.statistics import mean_std
from experiments.analysis.plots import plot_pareto
from experiments.core.runner import run_single


def run_application(cfg):

    hv_vals = []
    last_out = None
    spec = None

    for i in range(cfg.runs):
        seed = cfg.seed0 + i
        out, spec = run_single(cfg, seed)
        last_out = out

        if spec.hv_ref is not None:
            hv_vals.append(compute_hv(out["F"], spec.hv_ref))

    summary = {
        "HV": mean_std(hv_vals) if hv_vals else None,
    }

    if cfg.plot != "none":
        plot_pareto(last_out["F"], None, f"{cfg.problem} Pareto", cfg.plot)

    return summary