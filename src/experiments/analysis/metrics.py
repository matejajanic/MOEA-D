from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from moead.metrics import igd, hypervolume_2d


def compute_igd(F: np.ndarray, PF: np.ndarray) -> float:
    return float(igd(F, PF))


def compute_hv(F: np.ndarray, ref_point: Optional[Tuple[float, float]]) -> Optional[float]:
    if ref_point is None:
        return None
    return float(hypervolume_2d(F, ref_point))