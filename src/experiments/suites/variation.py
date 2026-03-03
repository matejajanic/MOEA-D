from __future__ import annotations
import numpy as np

from experiments.core.runner import run_single
from experiments.analysis.metrics import compute_igd, compute_hv
from experiments.analysis.plots import plot_two_curves, plot_bar
from experiments.analysis.statistics import mean_std


def run_variation_study(cfg):

    igd_v0 = []
    igd_sbx = []

    hv_v0 = []
    hv_sbx = []

    for i in range(cfg.runs):
        seed = cfg.seed0 + i

        # placeholder
        cfg.variation = "placeholder"
        out_v0, spec = run_single(cfg, seed)

        # sbx
        cfg.variation = "sbx"
        out_sbx, _ = run_single(cfg, seed)

        if spec.pf_sampler is not None:
            PF = spec.pf_sampler(800)
            igd_v0.append(compute_igd(out_v0["F"], PF))
            igd_sbx.append(compute_igd(out_sbx["F"], PF))

        if spec.hv_ref is not None:
            hv_v0.append(compute_hv(out_v0["F"], spec.hv_ref))
            hv_sbx.append(compute_hv(out_sbx["F"], spec.hv_ref))

    summary = {
        "IGD_placeholder": mean_std(igd_v0) if igd_v0 else None,
        "IGD_sbx": mean_std(igd_sbx) if igd_sbx else None,
        "HV_placeholder": mean_std(hv_v0) if hv_v0 else None,
        "HV_sbx": mean_std(hv_sbx) if hv_sbx else None,
    }

    if cfg.plot != "none" and igd_v0:
        plot_bar(
            ["placeholder", "sbx"],
            [mean_std(igd_v0)["mean"], mean_std(igd_sbx)["mean"]],
            "IGD Comparison",
            "IGD",
            cfg.plot,
        )

    return summary