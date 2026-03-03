from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any


VariationType = Literal["placeholder", "sbx"]
EncodingType = Literal["real", "binary"]
SuiteType = Literal["benchmark", "applications"]
PlotMode = Literal["show", "save", "none"]


@dataclass(frozen=True)
class ExperimentConfig:
    # high-level
    suite: SuiteType
    problem: str

    # repetition
    runs: int = 1
    seed0: int = 1  # seeds are seed0..seed0+runs-1

    # MOEA/D core
    N: int = 101
    T: int = 20
    n_gen: int = 200
    nr: int = 2

    # decision space
    n_var: int = 30
    encoding: EncodingType = "real"

    # variation params
    variation: VariationType = "sbx"
    eta_c: float = 20.0
    eta_m: float = 20.0
    p_c: float = 1.0
    p_m: Optional[float] = None  # if None, algorithm picks default

    # neighborhood (future-proof)
    neighborhood: str = "euclidean"  # "euclidean" | "cosine" | "semantic"
    weight_setup: str = "simplex_lattice"  # future-proof

    # plotting / saving
    plot: PlotMode = "show"  # show/save/none
    save_tag: str = ""       # optional identifier for filenames

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)