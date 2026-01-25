from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moead.weights import build_weight_setup

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type = int, default = 2, help = "Number of objectives")
    parser.add_argument("--N", type = int, default = 21, help = "(Only for m = 2) number of weight vectors")
    parser.add_argument("--H", type = int, default = 10, help = "(Only for m >= 3) simplex lattice parameter")
    parser.add_argument("--T", type = int, default = 5, help = "Neighborhood size")
    args = parser.parse_args()

    setup = build_weight_setup(n_obj = args.m, N = args.N, H = args.H, T = args.T)
    W, B = setup.W, setup.B

    print(f"W shape: {W.shape}")
    print(f"B shape: {B.shape}")
    print("First 5 weight vectors:")
    print(np.round(W[:5], 3))
    print("Neighbors of first 3 vectors (indices):")
    print(B[:3])

    if args.m == 2:
        plt.figure()
        plt.scatter(W[:, 0], W[:, 1], s = 25)
        for i in range(min(5, W.shape[0])):
            plt.annotate(str(i), (W[i, 0], W[i, 1]))
        plt.xlabel("lambda_1")
        plt.ylabel("lambda_2")
        plt.title("2D weight vectors on simplex")
        plt.grid(True, alpha = 0.3)
        plt.show()

if __name__ == "__main__":
    main()