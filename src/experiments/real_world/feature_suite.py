from __future__ import annotations
from problems.feature_selection import FeatureSelection


def get_feature_problem(seed: int = 42) -> dict:
    fs = FeatureSelection(seed=seed)
    return {
        "n_obj": 2,
        "n_var": fs.n_features,
        "evaluate_fn": fs.evaluate,
        "baseline_acc": fs.baseline_accuracy(),
    }