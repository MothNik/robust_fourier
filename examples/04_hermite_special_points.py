"""
This script shows how to evaluate special points of the Hermite functions, namely

- the x-position of their largest zero (= outermost root where y = 0)
- the x-position at which the outermost oscillation fades below machine precision
- the x- and y-position of the maximum of the Hermite functions in their outermost
    oscillation
- the Gaussian approximation of the outermost oscillation

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from robust_fourier import hermite_approx, single_hermite_function

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_fourier.mplstyle")
)

# === Constants ===

# the x-values to evaluate the Hermite functions (as offset of the center mu)
X_FROM = -320.0
X_TO = 320.0
NUM_X = 50_001

# the scaling factor alpha and center mu to use
ALPHA = 20.0
MU = 150.0
# the order of the Hermite function to evaluate
ORDER = 25

# the path where to store the plot (only for developers)
PLOT_FILEPATH = "../docs/hermite_functions/EX-04-HermiteFunctions_SpecialPoints.svg"

# === Main ===

if __name__ == "__main__":
    fig, ax = plt.subplots(
        figsize=(12, 8),
    )

    # the Hermite function of interest is evaluated in full to visualize its shape
    x_values = np.linspace(start=X_FROM + MU, stop=X_TO + MU, num=NUM_X)
    hermite_function = single_hermite_function(
        x=x_values,
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )
    # its special points are evaluated
    # 1) the x-positions at which the outermost oscillation fades below machine
    # precision
    x_fadeout = hermite_approx.x_fadeout(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )
    # 2) the x-positions of the largest zeros
    x_largest_zero = hermite_approx.x_largest_zeros(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )
    # 3) the x-positions of the largest extrema
    x_largest_extremum, y_largest_extremum = hermite_approx.x_and_y_largest_extrema(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )

    # 4) the Gaussian approximation of the outermost oscillation ...
    left_gaussian, right_gaussian = hermite_approx.tail_gauss_fit(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )
    # ... which is solved for the 50% level
    x_left_fifty_percent = left_gaussian.solve_for_y_fraction(y_fraction=0.5)
    x_right_fifty_percent = right_gaussian.solve_for_y_fraction(y_fraction=0.5)

    # the Hermite function and its special points are plotted
    ax.axvline(
        x=MU,
        color="black",
        linewidth=0.5,
        zorder=2,
    )
    ax.axhline(
        y=0.0,
        color="black",
        linewidth=0.5,
        zorder=2,
    )
    ax.plot(
        x_values,
        hermite_function,
        label="Hermite function",
        color="#FF8000",
        linewidth=2.0,
        zorder=3,
    )
    ax.plot(
        x_values,
        left_gaussian(x=x_values),
        label="Gaussian approximation",
        color="#990000",
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )
    ax.plot(
        x_values,
        right_gaussian(x=x_values),
        color="#990000",
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )

    ax.scatter(
        x_fadeout,
        np.zeros_like(x_fadeout),
        marker="D",
        facecolor="none",
        edgecolors="red",
        linewidths=3.0,
        s=150,
        label="Numerical Fadeouts",
        zorder=5,
    )
    ax.scatter(
        x_largest_zero,
        np.zeros_like(x_largest_zero),
        marker="o",
        facecolor="none",
        edgecolors="purple",
        linewidths=3.0,
        s=200,
        label="Largest Zeros",
        zorder=6,
    )
    ax.scatter(
        x_largest_extremum,
        y_largest_extremum,
        marker="X",
        facecolor="none",
        edgecolors="blue",
        linewidths=3.0,
        label="Largest Extrema",
        s=200,
        zorder=7,
    )
    ax.scatter(
        np.array([x_left_fifty_percent, x_right_fifty_percent]),
        np.array([0.5, 0.5]) * y_largest_extremum,
        marker="P",
        facecolor="none",
        edgecolors="#FF007F",
        linewidths=3.0,
        s=200,
        label="Gaussian 50% Level Approximation",
        zorder=8,
    )

    psi_label = (
        r"$\psi_{"
        + f"{ORDER}"
        + r"}^{\left("
        + f"{ALPHA:.0f}; {MU:.0f}"
        + r"\right)}\left(x\right)$"
    )
    ax.set_title(
        "Special Points of the Hermite function " + psi_label,
        fontsize=18,
    )
    ax.set_xlabel("x")
    ax.set_ylabel(psi_label)
    ax.legend(
        ncol=2,
        loc=8,
        bbox_to_anchor=(0.35, 0.85),
        fontsize=12,
    )

    ax.set_xlim(X_FROM + MU, X_TO + MU)

    # the plot is stored
    if os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":
        plt.savefig(os.path.join(os.path.dirname(__file__), PLOT_FILEPATH))

    plt.show()
