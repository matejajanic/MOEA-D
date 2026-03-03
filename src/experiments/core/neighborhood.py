from __future__ import annotations
from experiments.core.runner import run_single
from experiments.analysis.metrics import compute_igd
from experiments.analysis.statistics import mean_std
from experiments.analysis.plots import plot_bar


def run_neighborhood_study(cfg):

    igd_euclid = []
    igd_cosine = []

    for i in range(cfg.runs):
        seed = cfg.seed0 + i

        # euclidean
        cfg.neighborhood = "euclidean"
        out_e, spec = run_single(cfg, seed)

        # cosine
        cfg.neighborhood = "cosine"
        out_c, _ = run_single(cfg, seed)

        if spec.pf_sampler is not None:
            PF = spec.pf_sampler(800)
            igd_euclid.append(compute_igd(out_e["F"], PF))
            igd_cosine.append(compute_igd(out_c["F"], PF))

    summary = {
        "IGD_euclidean": mean_std(igd_euclid),
        "IGD_cosine": mean_std(igd_cosine),
    }

    if cfg.plot != "none":
        plot_bar(
            ["euclidean", "cosine"],
            [mean_std(igd_euclid)["mean"], mean_std(igd_cosine)["mean"]],
            "Neighborhood Comparison",
            "IGD",
            cfg.plot,
        )

    return summary