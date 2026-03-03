from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


# ==========================================================
# Internal helper
# ==========================================================

def _finalize(mode: str, save_path: Optional[str]):
    if mode == "save":
        if save_path is None:
            raise ValueError("save_path required when mode='save'")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
    elif mode == "show":
        plt.show()
    elif mode == "none":
        plt.close()
    else:
        raise ValueError(f"Unknown plot mode: {mode}")


# ==========================================================
# Pareto plot
# ==========================================================

def plot_pareto(F, PF, title, mode, save_path=None):
    """
    Plot approximation set F and optional true PF.
    """
    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], s=14, label="Approximation")

    if PF is not None:
        plt.plot(PF[:, 0], PF[:, 1], linewidth=2.0, label="True PF")

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    _finalize(mode, save_path)


# ==========================================================
# Single curve plot
# ==========================================================

def plot_curve(y, title, ylabel, mode, save_path=None):
    """
    Plot single curve (e.g. IGD over generations).
    """
    plt.figure()
    plt.plot(y)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    _finalize(mode, save_path)


# ==========================================================
# Two curve comparison
# ==========================================================

def plot_two_curves(y1, y2, label1, label2, title, ylabel, mode, save_path=None):
    """
    Plot two curves (e.g. IGD placeholder vs SBX).
    """
    plt.figure()
    plt.plot(y1, label=label1)
    plt.plot(y2, label=label2)

    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    _finalize(mode, save_path)


# ==========================================================
# Bar chart
# ==========================================================

def plot_bar(labels, values, title, ylabel, mode, save_path=None):
    """
    Simple bar chart (e.g. IGD comparison).
    """
    plt.figure()
    x = np.arange(len(labels))

    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    _finalize(mode, save_path)