from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from experiments.common import build_WB, run_single, hv_ref_point_global_2d
from experiments.plotting import compute_2d_hv


def run_metrics_over_seeds(
    base_cfg,
    evaluate_fn,
    xl,
    xu,
    reference_Z,
    seeds: list[int],
    H: int | None,
    hv_pad: float = 0.1,
    save_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """
    Runs repeated experiments across seeds and summarizes final IGD/HV.

    Default behavior: returns rows + summary (for terminal print).
    Optional saving: if save_dir is provided, saves CSV + summary.txt.

    HV:
      - computed only if n_obj == 2
      - uses ONE shared reference point across seeds (fair).
    """

    runs: list[dict] = []
    F_list: list[np.ndarray] = []

    for s in seeds:
        cfg = replace(base_cfg, seed=s)
        W, B = build_WB(cfg.n_obj, cfg.pop_size, cfg.T, H=H, seed=cfg.seed)
        out = run_single(cfg, evaluate_fn, W, B, xl, xu, reference_Z)

        igd_final = np.nan
        if out["history_igd"] is not None and len(out["history_igd"]) > 0:
            igd_final = float(out["history_igd"][-1])

        row = {
            "seed": int(s),
            "igd_final": float(igd_final),
            "F": out["F"],  
        }
        runs.append(row)

        if out["F"].shape[1] == 2:
            F_list.append(out["F"])

    hv_ref = None
    if F_list:
        hv_ref = hv_ref_point_global_2d(F_list, pad=hv_pad)

    rows: list[dict] = []
    for r in runs:
        hv = np.nan
        if hv_ref is not None and r["F"].shape[1] == 2:
            hv = float(compute_2d_hv(r["F"], hv_ref))
        rows.append({"seed": r["seed"], "igd_final": r["igd_final"], "hv": hv})

   
    igd_vals = np.array([x["igd_final"] for x in rows], dtype=float)
    hv_vals = np.array([x["hv"] for x in rows], dtype=float)

    summary = {
        "n_runs": len(rows),
        "igd_mean": float(np.nanmean(igd_vals)),
        "igd_std": float(np.nanstd(igd_vals)),
        "hv_mean": float(np.nanmean(hv_vals)),
        "hv_std": float(np.nanstd(hv_vals)),
        "hv_ref_point": hv_ref,  
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        arr = np.array([[x["seed"], x["igd_final"], x["hv"]] for x in rows], dtype=float)
        np.savetxt(
            save_dir / "metrics_by_seed.csv",
            arr,
            delimiter=",",
            header="seed,igd_final,hv",
            comments="",
        )
        lines = [
            f"n_runs: {summary['n_runs']}",
            f"igd_mean: {summary['igd_mean']}",
            f"igd_std: {summary['igd_std']}",
            f"hv_mean: {summary['hv_mean']}",
            f"hv_std: {summary['hv_std']}",
            f"hv_ref_point: {summary['hv_ref_point']}",
        ]
        (save_dir / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    return rows, summary