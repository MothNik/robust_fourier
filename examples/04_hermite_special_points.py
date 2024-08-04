"""
This script shows how to evaluate special points of the Hermite functions, namely

- the x-position of their largest zero (= outermost root where y = 0)
- the x-position at which the outermost oscillation fades below machine precision
- the x-position of the maximum of the Hermite functions in their outermost oscillation

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from robust_hermite_ft import single_hermite_function
from robust_hermite_ft.hermite_functions import (
    approximate_hermite_funcs_fadeout_x,
    approximate_hermite_funcs_largest_extrema_x,
    approximate_hermite_funcs_largest_zeros_x,
)

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_hermite_ft.mplstyle")
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
PLOT_FILEPATH = "../docs/hermite_functions/EX-04-HermiteFunctions_SpecialPoints.png"

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
    x_fadeout = approximate_hermite_funcs_fadeout_x(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )
    # 2) the x-positions of the largest zeros
    x_largest_zero = approximate_hermite_funcs_largest_zeros_x(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )
    # 3) the x-positions of the largest extrema
    x_largest_extremum = approximate_hermite_funcs_largest_extrema_x(
        n=ORDER,
        alpha=ALPHA,
        x_center=MU,
    )

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
        color="red",
        linewidth=2.0,
        zorder=3,
    )
    ax.scatter(
        x_fadeout,
        np.zeros_like(x_fadeout),
        marker="D",
        facecolor="none",
        edgecolors="#00CCCC",
        linewidths=3.0,
        s=150,
        label="Numerical Fadeouts",
        zorder=4,
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
        zorder=5,
    )
    ax.scatter(
        x_largest_extremum,
        single_hermite_function(
            x=x_largest_extremum,
            n=ORDER,
            alpha=ALPHA,
            x_center=MU,
        ),
        marker="X",
        facecolor="none",
        edgecolors="blue",
        linewidths=3.0,
        label="Largest Extrema",
        s=200,
        zorder=6,
    )

    psi_label = (
        r"$\psi_{"
        + f"{ORDER}"
        + r"}^{\left("
        + f"{ALPHA:.0f}; {MU:.0f}"
        + r"\right)}\left(x\right)$"
    )
    ax.set_title("Special Points of the Hermite function " + psi_label)
    ax.set_xlabel("x")
    ax.set_ylabel(psi_label)
    ax.legend(
        ncol=2,
        loc=8,
        bbox_to_anchor=(0.2925, 0.845),
    )

    ax.set_xlim(X_FROM + MU, X_TO + MU)

    # the plot is stored
    if os.getenv("ROBHERMFT_DEVELOPER", "false").lower() == "true":
        plt.savefig(os.path.join(os.path.dirname(__file__), PLOT_FILEPATH))

    plt.show()
