from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from moead.metrics import hypervolume_2d, filter_nondominated


def _finalize_figure(out_path: Path | None, filename: str) -> None:
    plt.tight_layout()
    if out_path is None:
        # Interactive display (no saving)
        plt.show()
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / filename, dpi=200)
        plt.close()


def plot_2d_front(out_dir: Path | None, F: np.ndarray, Z: np.ndarray | None, title: str) -> None:
    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], s=12, alpha=0.8, label="Approx (F)")
    if Z is not None and Z.shape[1] == 2:
        plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.5, label="Reference (Z)")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(title)
    plt.legend()
    _finalize_figure(out_dir, "front_2d.png")


def plot_3d_front(out_dir: Path | None, F: np.ndarray, Z: np.ndarray | None, title: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=10, alpha=0.8, label="Approx (F)")
    if Z is not None and Z.shape[1] == 3:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=6, alpha=0.4, label="Reference (Z)")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if out_dir is None:
        plt.show()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "front_3d.png", dpi=200)
        plt.close(fig)


def plot_history(out_dir: Path | None, history_replaced: np.ndarray, history_igd: np.ndarray | None) -> None:
    # replacements
    plt.figure()
    plt.plot(history_replaced)
    plt.xlabel("Generation")
    plt.ylabel("#Replacements")
    plt.title("Replacements per generation")
    _finalize_figure(out_dir, "history_replaced.png")

    if history_igd is not None and len(history_igd) > 0:
        plt.figure()
        plt.plot(history_igd)
        plt.xlabel("Generation")
        plt.ylabel("IGD")
        plt.title("IGD over generations")
        _finalize_figure(out_dir, "history_igd.png")


def compute_2d_hv(F: np.ndarray, ref_point: tuple[float, float]) -> float:
    A = filter_nondominated(F)
    return hypervolume_2d(A, ref_point=ref_point)