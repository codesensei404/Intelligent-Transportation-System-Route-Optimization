# gaussian_plot.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gaussian(mean, var, budget):
    sigma = np.sqrt(var)

    x = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)
    y = norm.pdf(x, mean, sigma)

    plt.figure()
    plt.plot(x, y)

    # Shade probability region
    x_fill = np.linspace(mean - 4*sigma, budget, 500)
    y_fill = norm.pdf(x_fill, mean, sigma)
    plt.fill_between(x_fill, y_fill)

    # Deadline line
    plt.axvline(x=budget)

    plt.title("Travel Time Distribution")
    plt.xlabel("Time")
    plt.ylabel("Density")

    plt.savefig("gaussian_plot.png")
    plt.close()

    print("Gaussian plot saved as gaussian_plot.png")