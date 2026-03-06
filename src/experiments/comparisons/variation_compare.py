from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from experiments.common import build_WB, run_single
from experiments.plotting import plot_2d_front, plot_history


def run_variation_compare(
    base_cfg,
    evaluate_fn,
    xl,
    xu,
    reference_Z,
    title_prefix: str = "",
    save_dir: Path | None = None,
    H: int | None = None,
) -> dict:
    """
    Compares SBX vs placeholder under the same settings.

    Default behavior: just returns results dict.
    Optional saving: if save_dir is provided, saves npz + png per variation.
    """

    results: dict[str, dict] = {}

    for var in ["sbx", "placeholder"]:
        cfg = replace(base_cfg, variation=var)
        W, B = build_WB(cfg.n_obj, cfg.pop_size, cfg.T, H=H, seed=cfg.seed)
        out = run_single(cfg, evaluate_fn, W, B, xl, xu, reference_Z)
        results[var] = out

        if save_dir is not None:
            sub = save_dir / f"variation_{var}"
            sub.mkdir(parents=True, exist_ok=True)

            np.savez(
                sub / "result.npz",
                X=out["X"],
                F=out["F"],
                z=out["z"],
                history_replaced=out["history_replaced"],
                history_igd=out["history_igd"] if out["history_igd"] is not None else np.array([]),
            )

            if out["F"].shape[1] == 2:
                plot_2d_front(sub, out["F"], reference_Z, f"{title_prefix} {var} (2D Front)")
            plot_history(sub, out["history_replaced"], out["history_igd"])

    return results