from __future__ import annotations
import numpy as np

def zdt1(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(X < 0.0) or np.any(X > 1.0):
        raise ValueError("ZDT1 expects X in [0,1].")
    
    f1 = X[:, 0]
    # g(x) = 1 + 9/(n-1) * sum_{i=2..n} x_i
    n = X.shape[1]
    if n < 2:
        raise ValueError("ZDT1 typically uses n>=2.")
    g = 1.0 + 9.0 * np.sum(X[:, 1:], axis = 1) / (n - 1)
    # f2 = g * (1 - sqrt(f1/g))
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.stack([f1, f2], axis = 1)

def sample_true_pareto_front_zdt1(num_points: int = 200) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, num_points)
    f2 = 1.0 - np.sqrt(f1)
    return np.stack([f1, f2], axis = 1)

def zdt3(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(X < 0.0) or np.any(X > 1.0):
        raise ValueError("ZDT3 expects X in [0,1].")

    f1 = X[:, 0]
    n = X.shape[1]
    if n < 2:
        raise ValueError("ZDT3 typically uses n>=2.")

    g = 1.0 + 9.0 * np.sum(X[:, 1:], axis=1) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
    f2 = g * h
    return np.stack([f1, f2], axis=1)


def sample_true_pareto_front_zdt3(num_points: int = 500) -> np.ndarray:
    """
    ZDT3 true Pareto front is disconnected (5 regions).
    We sample f1 only from the known intervals and compute f2 for g=1.
    """
    # Known f1 intervals for the Pareto-optimal front
    intervals = [
        (0.0, 0.0830015349),
        (0.1822287280, 0.2577623634),
        (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038),
        (0.8233317983, 0.8518328654)
    ]

    # distribute points roughly evenly across intervals
    per = max(2, num_points // len(intervals))
    f1_list = []
    for a, b in intervals:
        f1_list.append(np.linspace(a, b, per, endpoint=True))
    f1 = np.concatenate(f1_list)

    if f1.shape[0] > num_points:
        f1 = f1[:num_points]
    elif f1.shape[0] < num_points:
        extra = num_points - f1.shape[0]
        f1 = np.concatenate([f1, np.linspace(intervals[0][0], intervals[0][1], extra, endpoint=True)])

    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    return np.stack([f1, f2], axis=1)