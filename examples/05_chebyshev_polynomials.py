"""
This script shows how to generate plots of different Chebyshev polynomials with
different scales and centers.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from robust_fourier import chebyshev_polyvander

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_fourier.mplstyle")
)

# === Constants ===

# the x-values to evaluate the Chebyshev polynomials
X_FROM = -1.0
X_TO = 1.0
NUM_X = 10_001

# the x boundaries for plotting
PLOT_X_FROM = -1.0
PLOT_X_TO = 1.0

# the scaling factors alpha and centers mu to use
ALPHAS_AND_MUS = [(1.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
# the orders of the Chebyshev polynomials to plot
ORDERS = 6

# the path where to store the plot (only for developers)
PLOT_FILEPATH = (
    "../docs/chebyshev_polynomials"
    "/EX-05-DilatedChebyshevPolynomials_DifferentScales.svg"
)

# === Main ===

if __name__ == "__main__":

    fig, ax = plt.subplots(
        ncols=len(ALPHAS_AND_MUS),
        sharex=True,
        sharey=True,
        figsize=(12, 8),
    )

    # the Chebyshev polynomials are evaluated and plotted for each scaling factor alpha
    colors = plt.cm.cool_r(np.linspace(0, 1, ORDERS + 1))  # type: ignore
    for idx_alpha, (alpha, mu) in enumerate(ALPHAS_AND_MUS):
        # a grid and vertical y-axis line is plotted for orientation
        ax[idx_alpha].axvline(  # type: ignore
            x=0.0,
            color="black",
            linewidth=0.5,
            zorder=2,
        )
        ax[idx_alpha].axhline(  # type: ignore
            y=0.0,
            color="black",
            linewidth=0.5,
            zorder=2,
        )

        # the Chebyshev polynomials are computed and plotted
        x_values = np.linspace(start=mu - alpha, stop=mu + alpha, num=NUM_X)
        chebyshev_basis = chebyshev_polyvander(
            x=x_values,
            n=ORDERS,
            alpha=alpha,
            x_center=mu,
            kind="second",
            jit=True,
        )

        # NOTE: x-axis are plotted for orientation
        for idx_order in range(0, ORDERS + 1):
            ax[idx_alpha].plot(  # type: ignore
                x_values,
                chebyshev_basis[::, ORDERS - idx_order],
                color=colors[ORDERS - idx_order],
                zorder=2 + (idx_order + 1) * 2,
            )

        # the title, grid, x-labels, and ticks are set
        title = r"$\alpha$ = " + f"{alpha:.1f}\n" + r"$\mu$ = " + f"{mu:.1f}"
        ax[idx_alpha].set_title(title)  # type: ignore
        ax[idx_alpha].set_xlabel(r"$x$")  # type: ignore
        ax[idx_alpha].set_xlim(PLOT_X_FROM, PLOT_X_TO)  # type: ignore

        # for the first plot, a y-label and a legend are added
        if idx_alpha == 0:
            ax[idx_alpha].set_ylabel(  # type: ignore
                r"$U_{n}^{\left(\alpha;\mu\right)}\left(x\right)$",
            )

    # a colorbar is added for the orders
    sm = plt.cm.ScalarMappable(
        cmap="cool_r",
        norm=plt.Normalize(vmin=0, vmax=ORDERS),  # type: ignore
    )
    fig.colorbar(
        sm,
        ax=ax.ravel().tolist(),  # type: ignore
        label=r"Order $n$",
        orientation="horizontal",
    )

    fig.suptitle(
        "Dilated Chebyshev Polynomials with Different Scales and Centers",
        fontsize=18,
        y=1.05,
    )

    # the plot is saved ...
    if os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":
        plt.savefig(os.path.join(os.path.dirname(__file__), PLOT_FILEPATH))

    # ... and shown
    plt.show()
