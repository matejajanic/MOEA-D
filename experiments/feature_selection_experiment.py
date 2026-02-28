from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run
from moead.problems.feature_selection import FeatureSelection


def main():

    problem = FeatureSelection()

    setup = build_weight_setup(n_obj=2, N=61, T=15)

    cfg = MOEADConfig(
        n_obj=2,
        n_var=problem.n_features,
        pop_size=61,
        n_gen=200,
        encoding="binary",
        p_c=0.9,
        p_m=0.1,
    )

    out = moead_run(
        cfg,
        evaluate_fn=problem.evaluate,
        W=setup.W,
        B=setup.B,
        xl=None,
        xu=None,
    )

    F = out["F"]
    X = out["X"]
    accuracies = 1 - F[:, 0]
    fractions = F[:, 1]
    num_features = X.sum(axis=1)

    print("\n==============================")
    print("FEATURE SELECTION ANALYSIS")
    print("==============================")

    print("\n--- Accuracy Statistics ---")
    print("Min accuracy:", round(accuracies.min(), 4))
    print("Max accuracy:", round(accuracies.max(), 4))
    print("Average accuracy:", round(accuracies.mean(), 4))
    print("Std accuracy:", round(accuracies.std(), 4))

    print("\n--- Model Complexity ---")
    print("Min selected features:", int(num_features.min()))
    print("Max selected features:", int(num_features.max()))
    print("Average selected features:", round(num_features.mean(), 2))
    print("Std selected features:", round(num_features.std(), 2))

    best_acc_idx = accuracies.argmax()
    print("\n--- Highest Accuracy Solution ---")
    print("Accuracy:", round(accuracies[best_acc_idx], 4))
    print("Selected features:", int(num_features[best_acc_idx]))
    print("Feature fraction:", round(fractions[best_acc_idx], 4))

    min_feat_idx = num_features.argmin()
    print("\n--- Smallest Model Solution ---")
    print("Accuracy:", round(accuracies[min_feat_idx], 4))
    print("Selected features:", int(num_features[min_feat_idx]))
    print("Feature fraction:", round(fractions[min_feat_idx], 4))

    print("\n--- Pareto Spread ---")
    print("Accuracy range:", round(accuracies.max() - accuracies.min(), 4))
    print("Feature count range:", int(num_features.max() - num_features.min()))

    baseline_acc = problem.baseline_accuracy()

    print("\n--- Baseline (All Features) ---")
    print("Baseline accuracy:", round(baseline_acc, 4))
    print("Best MOEA/D accuracy:", round(accuracies.max(), 4))
    print("Accuracy difference:", round(baseline_acc - accuracies.max(), 4))

    efficiency = accuracies.max() / (num_features[best_acc_idx] / problem.n_features)
    print("\n--- Efficiency Indicator ---")
    print("Accuracy per feature ratio (best solution):", round(efficiency, 4))

    print("\n==============================\n")

   

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 1], 1 - F[:, 0], s=20)
    plt.xlabel("Fraction of Selected Features")
    plt.ylabel("Accuracy")
    plt.title("Feature Selection – Accuracy vs Model Complexity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig("result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()