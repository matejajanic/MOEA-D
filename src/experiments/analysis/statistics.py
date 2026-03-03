from __future__ import annotations
import numpy as np
from typing import Sequence, Dict


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    return {"mean": mean, "std": std}