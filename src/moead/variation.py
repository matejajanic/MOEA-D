from __future__ import annotations
import numpy as np

def sbx_crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta_c: float = 20.0, p_c: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n = p1.shape[0]
    if rng.random() > p_c:
        return p1.copy(), p2.copy()
    
    c1 = p1.copy()
    c2 = p2.copy()

    for i in range(n):
        if rng.random() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                x1 = min(p1[i], p2[i])
                x2 = max(p1[i], p2[i])

                u = rng.random()

                #child 1
                beta = 1.0 + (2.0 * (x1 - xl[i]) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                if u <= 1.0 / alpha:
                    betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
                child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                #child 2
                beta = 1.0 + (2.0 * (xu[i] - x2) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                if u <= 1.0 / alpha:
                    betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
                child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                child1 = np.clip(child1, xl[i], xu[i])
                child2 = np.clip(child2, xl[i], xu[i])
                if rng.random() <= 0.5:
                    c1[i] = child2
                    c2[i] = child1
                else:
                    c1[i] = child1
                    c2[i] = child2

            else:
                c1[i] = p1[i]
                c2[i] = p2[i]

        else:
            c1[i] = p1[i]
            c2[i] = p2[i]

    return c1, c2

def polynomial_mutation(rng: np.random.Generator, x: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta_m: float = 20.0, p_m: float | None = None) -> np.ndarray:
    n = x.shape[0]
    if p_m is None:
        p_m = 1.0 / n
    
    y = x.copy()
    for i in range(n):
        if rng.random() <= p_m:
            if xu[i] - xl[i] <= 0:
                continue

            delta1 = (y[i] - xl[i]) / (xu[i] - xl[i])
            delta2 = (xu[i] - y[i]) / (xu[i] - xl[i])
            u = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if u < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            
            y[i] = y[i] + deltaq * (xu[i] - xl[i])
            y[i] = np.clip(y[i], xl[i], xu[i])
        
    return y

def variation_sbx_poly(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta_c: float = 20.0, eta_m: float = 20.0, p_c: float = 1.0, p_m: float | None = None) -> np.ndarray:
    c1, c2 = sbx_crossover(rng, p1, p2, xl, xu, eta_c = eta_c, p_c = p_c)
    y = c1 if rng.random() < 0.5 else c2
    y = polynomial_mutation(rng, y, xl, xu, eta_m = eta_m, p_m = p_m)
    return y