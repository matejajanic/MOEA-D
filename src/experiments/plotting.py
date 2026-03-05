from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from moead.metrics import hypervolume_2d, filter_nondominated


def _finalize_figure(out_dir, filename: str) -> None:
    import matplotlib.pyplot as plt
    from pathlib import Path

    plt.tight_layout()
    if out_dir is None:
        plt.show()
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def plot_history(out_dir, history_replaced, history_igd, show_replacements: bool = True) -> None:
    import matplotlib.pyplot as plt

    history_replaced = np.asarray(history_replaced, dtype=float)
    history_igd = None if history_igd is None else np.asarray(history_igd, dtype=float)

    if show_replacements:
        plt.figure()
        plt.plot(np.arange(len(history_replaced)), history_replaced)
        plt.xlabel("Generation")
        plt.ylabel("# replacements")
        plt.title("Replacements per generation")
        _finalize_figure(out_dir, "history_replacements.png")

    if history_igd is not None and history_igd.size > 0:
        plt.figure()
        plt.plot(np.arange(len(history_igd)), history_igd)
        plt.xlabel("Generation")
        plt.ylabel("IGD")
        plt.title("IGD per generation")
        _finalize_figure(out_dir, "history_igd.png")


def plot_2d_front(out_dir: Path | None, F: np.ndarray, Z: np.ndarray | None, title: str) -> None:
    plt.figure()
    if Z is not None and Z.shape[1] == 2:
        plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.5, label="Reference (Z)")
    plt.scatter(F[:, 0], F[:, 1], s=12, alpha=0.8, label="Approx (F)")
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


def plot_feature_accuracy_vs_k(out_dir: Path | None, X_bin: np.ndarray, F: np.ndarray, title: str) -> None:
    if X_bin.ndim != 2:
        raise ValueError("X_bin must be 2D (N, n_features).")
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError("F must be (N,2) at least.")

    k = np.sum(X_bin, axis=1).astype(int)
    acc = 1.0 - F[:, 0]

    plt.figure()
    plt.scatter(k, acc, s=14, alpha=0.75, label="Solutions")

    if k.size > 0:
        k_max = int(np.max(k))
        best = np.full(k_max + 1, np.nan, dtype=float)
        for kk, aa in zip(k, acc):
            if np.isnan(best[kk]) or aa > best[kk]:
                best[kk] = aa

        xs = np.where(~np.isnan(best))[0]
        ys = best[xs]
        if xs.size > 0:
            plt.plot(xs, ys, linewidth=2.0, label="Best accuracy per k")

    plt.xlabel("# selected features (k)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    _finalize_figure(out_dir, "feature_accuracy_vs_k.png")


def plot_project_tradeoff(
    out_dir: Path | None,
    analysis: dict[str, np.ndarray],
    title: str,
    budget: float | None = None,
) -> None:
    """
    Projects visualization:
      - profit vs risk scatter
      - feasible vs infeasible (budget violation) shown differently
    """
    profit = np.asarray(analysis["profit"], dtype=float)
    risk = np.asarray(analysis["risk"], dtype=float)
    feasible = np.asarray(analysis["feasible"], dtype=bool)
    penalty = np.asarray(analysis["penalty"], dtype=float)

    plt.figure()
    plt.scatter(profit[feasible], risk[feasible], s=14, alpha=0.8, label="Feasible (within budget)")
    if np.any(~feasible):
        plt.scatter(profit[~feasible], risk[~feasible], s=14, alpha=0.5, label="Infeasible (budget violated)")

    plt.xlabel("Total profit")
    plt.ylabel("Total risk")
    t = title
    if budget is not None:
        t = f"{title} (budget={budget:.2f})"
    plt.title(t)
    plt.legend()
    _finalize_figure(out_dir, "project_profit_vs_risk.png")

    # Optional: #projects vs profit (helps intuition)
    if "k" in analysis:
        k = np.asarray(analysis["k"], dtype=int)
        plt.figure()
        plt.scatter(k[feasible], profit[feasible], s=14, alpha=0.8, label="Feasible")
        if np.any(~feasible):
            plt.scatter(k[~feasible], profit[~feasible], s=14, alpha=0.5, label="Infeasible")
        plt.xlabel("# selected projects (k)")
        plt.ylabel("Total profit")
        plt.title("Projects: Profit vs #Selected Projects")
        plt.legend()
        _finalize_figure(out_dir, "project_profit_vs_k.png")

def plot_textsum_similarity_vs_compression(
    out_dir: Path | None,
    analysis: dict[str, np.ndarray],
    title: str,) -> None:
    """
    Text summarization visualization:
      x-axis: compression (#selected / #sentences)
      y-axis: cosine similarity to document
    """
    sim = np.asarray(analysis["sim"], dtype=float)
    comp = np.asarray(analysis["compression"], dtype=float)
    k = np.asarray(analysis["k"], dtype=int)

    plt.figure()
    plt.scatter(comp, sim, s=14, alpha=0.75, label="Solutions")
    plt.xlabel("Compression (k / n_sent)")
    plt.ylabel("Cosine similarity (summary vs doc)")
    plt.title(title)
    plt.legend()
    _finalize_figure(out_dir, "textsum_similarity_vs_compression.png")

    # Optional: best similarity per k
    if k.size > 0:
        plt.figure()
        kmax = int(np.max(k))
        best = np.full(kmax + 1, np.nan, dtype=float)
        for kk, ss in zip(k, sim):
            if np.isnan(best[kk]) or ss > best[kk]:
                best[kk] = ss
        xs = np.where(~np.isnan(best))[0]
        ys = best[xs]
        plt.plot(xs, ys, linewidth=2.0, label="Best similarity per k")
        plt.xlabel("# selected sentences (k)")
        plt.ylabel("Cosine similarity")
        plt.title("Text summarization: Best similarity vs #selected sentences")
        plt.legend()
        _finalize_figure(out_dir, "textsum_best_similarity_vs_k.png")

def plot_textsum_3obj(out_dir, analysis: dict, title: str) -> None:
    """
    3-objective text summarization plots (pairwise):
      (compression, similarity), (redundancy, similarity), (compression, redundancy)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    comp = np.asarray(analysis["compression"], dtype=float)
    sim = np.asarray(analysis["sim"], dtype=float)
    red = np.asarray(analysis["redundancy"], dtype=float)

    # similarity vs compression
    plt.figure()
    plt.scatter(comp, sim, s=14, alpha=0.75, label="Solutions")
    plt.xlabel("Compression (k / n_sent)")
    plt.ylabel("Cosine similarity (summary vs doc)")
    plt.title(f"{title}: Similarity vs Compression")
    plt.legend()
    _finalize_figure(out_dir, "textsum_sim_vs_comp.png")

    # similarity vs redundancy
    plt.figure()
    plt.scatter(red, sim, s=14, alpha=0.75, label="Solutions")
    plt.xlabel("Redundancy (avg pairwise cosine)")
    plt.ylabel("Cosine similarity (summary vs doc)")
    plt.title(f"{title}: Similarity vs Redundancy")
    plt.legend()
    _finalize_figure(out_dir, "textsum_sim_vs_red.png")

    # redundancy vs compression
    plt.figure()
    plt.scatter(comp, red, s=14, alpha=0.75, label="Solutions")
    plt.xlabel("Compression (k / n_sent)")
    plt.ylabel("Redundancy (avg pairwise cosine)")
    plt.title(f"{title}: Redundancy vs Compression")
    plt.legend()
    _finalize_figure(out_dir, "textsum_red_vs_comp.png")


def plot_textsum_best_similarity_per_k(out_dir, analysis: dict) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    k = np.asarray(analysis["k"], dtype=int)
    sim = np.asarray(analysis["sim"], dtype=float)

    if k.size == 0:
        return

    kmax = int(np.max(k))
    best = np.full(kmax + 1, np.nan, dtype=float)
    for kk, ss in zip(k, sim):
        if np.isnan(best[kk]) or ss > best[kk]:
            best[kk] = ss

    xs = np.where(~np.isnan(best))[0]
    ys = best[xs]

    plt.figure()
    plt.plot(xs, ys, linewidth=2.0, label="Best similarity per k")
    plt.xlabel("# selected sentences (k)")
    plt.ylabel("Cosine similarity")
    plt.title("Text summarization: Best similarity vs #selected sentences")
    plt.legend()
    _finalize_figure(out_dir, "textsum_best_sim_vs_k.png")

def plot_textsum_3d(out_dir, analysis: dict, F: np.ndarray, title: str) -> None:
    """
    3D visualization for text summarization (3 objectives).
    Pareto front is colored orange.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from moead.metrics import filter_nondominated

    comp = np.asarray(analysis["compression"], dtype=float)
    sim = np.asarray(analysis["sim"], dtype=float)
    red = np.asarray(analysis["redundancy"], dtype=float)

    F = np.asarray(F, dtype=float)

    # --- Pareto mask ---
    F_nd = filter_nondominated(F)

    # mark which rows are nondominated
    nd_mask = np.zeros(F.shape[0], dtype=bool)
    for i in range(F.shape[0]):
        for f in F_nd:
            if np.allclose(F[i], f):
                nd_mask[i] = True
                break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # all solutions
    ax.scatter(
        comp[~nd_mask],
        red[~nd_mask],
        sim[~nd_mask],
        s=14,
        alpha=0.4,
        label="Dominated",
    )

    # Pareto front (orange)
    ax.scatter(
        comp[nd_mask],
        red[nd_mask],
        sim[nd_mask],
        s=30,
        alpha=0.9,
        color="orange",
        label="Pareto front",
    )

    ax.set_xlabel("Compression (k / n_sent)")
    ax.set_ylabel("Redundancy (avg pairwise cosine)")
    ax.set_zlabel("Cosine similarity")
    ax.set_title(f"{title}: 3D trade-off")

    ax.view_init(elev=20, azim=45)
    ax.legend()

    plt.tight_layout()

    if out_dir is None:
        plt.show()
    else:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "textsum_3d.png", dpi=200)
        plt.close(fig)

def compute_2d_hv(F: np.ndarray, ref_point: tuple[float, float]) -> float:
    A = filter_nondominated(F)
    return hypervolume_2d(A, ref_point=ref_point)