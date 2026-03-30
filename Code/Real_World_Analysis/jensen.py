# jensen.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def jensen_analysis(mean, var, budget):
    sigma = np.sqrt(var)

    # ---- Compute probability ----
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

    return prob


# ---------- Gaussian PDF + CDF ----------
def plot_gaussian(mean, var, budget):
    sigma = np.sqrt(var)

    x = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)
    pdf = norm.pdf(x, mean, sigma)
    cdf = norm.cdf(x, mean, sigma)

    # ---- PDF ----
    plt.figure()
    plt.plot(x, pdf, label="PDF")

    x_fill = np.linspace(mean - 4*sigma, budget, 500)
    y_fill = norm.pdf(x_fill, mean, sigma)
    plt.fill_between(x_fill, y_fill, alpha=0.5, label="P(T ≤ D)")

    plt.axvline(x=budget, linestyle='--', label="Deadline D")

    plt.title("Travel Time Distribution (PDF)")
    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig("jensen_pdf.png")
    plt.close()

    # ---- CDF ----
    plt.figure()
    plt.plot(x, cdf, label="CDF")

    plt.axvline(x=budget, linestyle='--', label="Deadline D")
    plt.axhline(y=norm.cdf(budget, mean, sigma), linestyle='--', label="P(T ≤ D)")

    plt.title("Cumulative Distribution (CDF)")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.savefig("jensen_cdf.png")
    plt.close()

    print("Saved: jensen_pdf.png, jensen_cdf.png")


# ---------- Jensen Curve ----------
def plot_jensen_curve(mean, var, budget):
    sigma = np.sqrt(var)

    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 500)

    # Nonlinear function (probability-like)
    def f(t):
        return norm.cdf(budget, loc=t, scale=sigma)

    fx = f(x)

    # Jensen values
    f_E = f(mean)

    samples = np.random.normal(mean, sigma, 10000)
    E_f = np.mean(f(samples))

    # Plot
    plt.figure()
    plt.plot(x, fx, label="f(x)")

    # Mark points
    plt.scatter(mean, f_E, color='red')
    plt.text(mean, f_E, " f(E[T])", fontsize=9)

    plt.scatter(mean, E_f, color='green')
    plt.text(mean, E_f, " E[f(T)]", fontsize=9)

    plt.axvline(mean, linestyle='--', label="E[T]")

    plt.title("Jensen's Inequality Visualization")
    plt.xlabel("Travel Time")
    plt.ylabel("Probability Function")
    plt.legend()
    plt.grid(True)

    plt.savefig("jensen_curve.png")
    plt.close()

    print("Saved: jensen_curve.png")