from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from moead.weights import build_weight_setup
from moead.algorithm import MOEADConfig, moead_run
from moead.problems.zdt5 import zdt5


def main():

    n_bits = 80

    setup = build_weight_setup(n_obj=2, N=101, T=20)

    cfg = MOEADConfig(
        n_obj=2,
        n_var=n_bits,
        pop_size=101,
        n_gen=500,
        encoding="binary",
        p_c=1.0,
        p_m=1.0 / n_bits
    )

    out = moead_run(
        cfg,
        evaluate_fn=zdt5,
        W=setup.W,
        B=setup.B,
        xl=None,
        xu=None,
    )

    F = out["F"]

    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], s=15)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("ZDT5 – Binary MOEA/D")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()