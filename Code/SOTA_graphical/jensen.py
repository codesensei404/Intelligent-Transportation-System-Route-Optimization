# jensen.py

import numpy as np
from scipy.stats import norm

def jensen_analysis(mean, var, budget):
    sigma = np.sqrt(var)

    if var == 0:
        prob = 1.0 if mean <= budget else 0.0
    else:
        prob = norm.cdf(budget, loc=mean, scale=sigma)

    deterministic = 1 if mean <= budget else 0

    print("\n===== Jensen Inequality Analysis =====")
    print(f"E[T]: {mean:.2f}")
    print(f"Var(T): {var:.2f}")
    print(f"Budget D: {budget}")

    print(f"\nDeterministic f(E[T]): {deterministic}")
    print(f"Probabilistic E[f(T)] = P(T<=D): {prob:.4f}")

    gap = abs(prob - deterministic)
    print(f"Jensen Gap: {gap:.4f}")