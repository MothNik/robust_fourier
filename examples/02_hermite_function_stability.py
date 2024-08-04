"""
This script generates plots of different Hermite functions and shows how the computation
with in the logarithmic space keeps them stable.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from robust_hermite_ft import hermite_function_basis

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_hermite_ft.mplstyle")
)

# === Constants ===

# the x-values to evaluate the Hermite functions
X_FROM = -65.0
X_TO = 65.0
NUM_X = 100_001

# the scaling factor alpha to use
ALPHA = 1.0
# the orders of the Hermite functions to plot and their colours
ORDERS = [100, 200, 300, 500, 1_000, 2_000]
COLORS = ["red", "#FF8000", "#00CCCC", "blue", "purple", "#FF007F"]
SPECIAL_COLOR = "#00CC00"
# the offset between the individual Hermite functions
OFFSET = -0.5

# the path where to store the plot (only for developers)
PLOT_FILEPATH = "../docs/hermite_functions/EX-02-DilatedHermiteFunctions_Stability.png"

# === Main ===

if __name__ == "__main__":

    fig, ax = plt.subplots(
        figsize=(12, 8),
    )

    # all Hermite basis functions are evaluated ...
    x_values = np.linspace(start=X_FROM, stop=X_TO, num=NUM_X)
    hermite_basis = hermite_function_basis(
        x=x_values,
        n=max(ORDERS),
        alpha=ALPHA,
        workers=-1,
    )

    # ... and the individual Hermite functions of interest are plotted
    ax.axvline(
        x=0.0,
        color="black",
        linewidth=0.5,
        zorder=2,
    )

    for idx_order, order in enumerate(ORDERS):
        ax.axhline(
            y=idx_order * OFFSET,
            color="black",
            linewidth=0.5,
            zorder=2 + idx_order * 2,
        )
        ax.plot(
            x_values,
            hermite_basis[::, order] + idx_order * OFFSET,
            label=f"n={order}",
            color=COLORS[idx_order],
            zorder=2 + 2 * idx_order + 1,
        )

    # the title, grid, labels, and ticks are set
    psi_label = r"$\psi_{n}^{\left(" + f"{ALPHA:.0f}; 0" + r"\right)}\left(x\right)$"
    ax.set_title("Numerical Stability of the Dilated Hermite Functions " + psi_label)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(psi_label)
    ax.set_xlim(X_FROM, X_TO)

    # finally, a legend is added
    ax.legend(
        ncol=2,
        loc=8,
        bbox_to_anchor=(0.175, 0.81),
    )

    # the plot is saved
    if os.getenv("ROBHERMFT_DEVELOPER", "false").lower() == "true":
        plt.savefig(
            os.path.join(os.path.dirname(__file__), PLOT_FILEPATH),
            bbox_inches="tight",
        )

    # the plot is shown
    plt.show()
