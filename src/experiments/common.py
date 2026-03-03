from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from moead.algorithm import MOEADConfig, moead_run
from moead.weights import build_weight_setup


def make_run_dir(base: str | Path = "experiments/results", tag: str | None = None) -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = ts if tag is None else f"{ts}_{tag}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, cfg: MOEADConfig, extra: dict | None = None) -> None:
    payload = {"moead": asdict(cfg)}
    if extra:
        payload.update(extra)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_WB(n_obj: int, pop_size: int, T: int, H: int | None) -> tuple[np.ndarray, np.ndarray]:
    ws = build_weight_setup(n_obj=n_obj, N=pop_size if n_obj == 2 else None, H=H, T=T)
    return ws.W, ws.B


def run_single(
    cfg: MOEADConfig,
    evaluate_fn,
    W: np.ndarray,
    B: np.ndarray,
    xl: np.ndarray | None,
    xu: np.ndarray | None,
    reference_Z: np.ndarray | None,
) -> dict:
    return moead_run(
        config=cfg,
        evaluate_fn=evaluate_fn,
        W=W,
        B=B,
        xl=xl,
        xu=xu,
        reference_Z=reference_Z,
    )


def hv_ref_point_2d(F: np.ndarray, pad: float = 0.1) -> tuple[float, float]:
    fmax = np.max(F, axis=0)
    return (float(fmax[0] * (1.0 + pad) + 1e-9), float(fmax[1] * (1.0 + pad) + 1e-9))


def hv_ref_point_global_2d(F_list: list[np.ndarray], pad: float = 0.1) -> tuple[float, float]:
    """
    One shared HV reference point for multiple runs (fair comparisons).
    """
    if not F_list:
        raise ValueError("F_list is empty.")
    fmax = np.max(np.vstack(F_list), axis=0)
    return (float(fmax[0] * (1.0 + pad) + 1e-9), float(fmax[1] * (1.0 + pad) + 1e-9))