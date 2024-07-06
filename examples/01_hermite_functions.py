"""
This script generates plots of different Hermite functions with different scales.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from src.hermite_functions import _dilated_hermite_function_basis

# === Constants ===

# the x-values to evaluate the Hermite functions
X_FROM = -5.0
X_TO = 5.0
NUM_X = 1_001

# the scaling factors alpha to use
ALPHAS = [0.5, 1.0, 2.0]
# the orders of the Hermite functions to plot
ORDERS = 6
# the offset between the individual Hermite functions
OFFSET = -2.0

# the path where to store the plot and its resolution
PLOT_FILEPATH = "../docs//hermite_functions/01_DilatedHermiteFunctions.png"
DPI = 300

# === Main ===

if __name__ == "__main__":

    fig, ax = plt.subplots(
        ncols=len(ALPHAS),
        sharex=True,
        sharey=True,
        figsize=(12, 8),
    )

    # the Hermite functions are evaluated and plotted for each scaling factor alpha
    x_values = np.linspace(start=X_FROM, stop=X_TO, num=NUM_X)

    colors = plt.cm.winter_r(np.linspace(0, 1, ORDERS + 1))  # type: ignore
    for idx_alpha, alpha in enumerate(ALPHAS):
        # the Hermite functions are computed and plotted
        hermite_basis = _dilated_hermite_function_basis(
            x=x_values,
            n=ORDERS,
            alpha=alpha,
        )

        # NOTE: x-axis are plotted for orientation
        for idx_order in range(0, ORDERS + 1):
            ax[idx_alpha].axhline(
                y=idx_order * OFFSET,
                color="black",
                linewidth=0.5,
                zorder=idx_order * 2,
            )
            ax[idx_alpha].plot(
                x_values,
                hermite_basis[::, idx_order] + idx_order * OFFSET,
                color=colors[idx_order],
                zorder=(idx_order + 1) * 2,
                label=f"n = {idx_order}",
            )

        # the title, grid, x-labels, and ticks are set
        ax[idx_alpha].set_title(
            r"$\alpha$" + f"= {alpha:.1f}",
            fontsize=16,
        )
        ax[idx_alpha].set_xlabel(
            r"$x$",
            fontsize=16,
            labelpad=10,
        )
        ax[idx_alpha].tick_params(axis="both", which="major", labelsize=14)
        ax[idx_alpha].grid(which="major", axis="both")
        ax[idx_alpha].set_xlim(X_FROM, X_TO)

        # for the first plot, a y-label and a legend are added
        if idx_alpha == 0:
            ax[idx_alpha].set_ylabel(
                r"$\psi_{n}^{\left(\alpha\right)}\left(x\right)$",
                fontsize=16,
                labelpad=10,
            )
            ax[idx_alpha].legend(
                loc="upper left",
                fontsize=14,
                frameon=False,
            )

    plt.tight_layout()

    # the plot is saved ...
    plt.savefig(
        os.path.join(os.path.dirname(__file__), PLOT_FILEPATH),
        dpi=DPI,
        bbox_inches="tight",
    )

    # ... and shown
    plt.show()
