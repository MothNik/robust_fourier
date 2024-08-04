"""
This script shows how to generate plots of the Hermite functions and their Continuous
Fourier transforms.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from robust_hermite_ft import HermiteFunctionBasis, approximate_hermite_funcs_fadeout_x
from robust_hermite_ft.fourier_transform import (
    TimeSpaceSignal,
    convert_discrete_to_continuous_ft,
    discrete_ft,
)

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_hermite_ft.mplstyle")
)

# === Constants ===

# the x-values to evaluate the Hermite functions (without any centering)
X_FROM = -25.0
X_TO = 25.0
NUM_X = 5_001

# the scaling factors alpha for the time/space domain
ALPHAS = [
    0.5,
    0.5,
    0.5,
]
# the centers mu for the time/space domain
TIME_SPACE_X_CENTER_MU = [
    0.0,
    0.0,
    5.0,
]
# the order n so Hermite functions to plot
ORDERS_N = [
    10,
    15,
    15,
]
# the colors for each order for the analytical and the numerical Fourier transform
COLORS_ANALYTICAL = [
    "#CC0000",
    "#00CCCC",
    "#00FF00",
]
COLORS_NUMERICAL = [
    "#FF8000",
    "purple",
    "#666600",
]

# the path where to store the plot (only for developers)
PLOT_FILEPATH = (
    "../docs/hermite_functions/EX-05-DilatedHermiteFunctions_FourierTransforms.svg"
)

assert (
    len(ALPHAS)
    == len(TIME_SPACE_X_CENTER_MU)
    == len(ORDERS_N)
    == len(COLORS_ANALYTICAL)
    == len(COLORS_NUMERICAL)
), "The number of scaling factors, centers, orders, and colors must be the same."

# === Main ===

if __name__ == "__main__":

    # the canvas is set up with as many columns as there are parameter sets, each with
    # one row for the time/space domain, and two rows for the real and imaginary part of
    # the Continuous Fourier Transform, respectively
    fig = plt.figure(figsize=(12, 10))
    gridspec = GridSpec(nrows=19, ncols=len(ALPHAS), figure=fig)
    ax = np.empty((3, len(ALPHAS)), dtype=object)
    for col_idx in range(len(ALPHAS)):
        ax[0, col_idx] = fig.add_subplot(gridspec[0:4, col_idx])
        ax[1, col_idx] = fig.add_subplot(gridspec[9:13, col_idx])
        ax[2, col_idx] = fig.add_subplot(gridspec[14:19, col_idx])

    # the x-values for the Hermite functions are generated
    x_values = np.linspace(start=X_FROM, stop=X_TO, num=NUM_X)

    for col_idx, (
        alpha,
        time_space_x_center,
        n,
        color_analytical,
        color_numerical,
    ) in enumerate(
        zip(
            ALPHAS,
            TIME_SPACE_X_CENTER_MU,
            ORDERS_N,
            COLORS_ANALYTICAL,
            COLORS_NUMERICAL,
        )
    ):

        # the second and third row share the x-axis (angular frequency axis)
        ax[2, col_idx].sharex(ax[1, col_idx])  # type: ignore

        # an x- and y-axis line are plotted for orientation
        for row_idx in range(0, 3):
            ax[row_idx, col_idx].axhline(  # type: ignore
                y=0.0,
                color="black",
                linewidth=0.5,
                zorder=2,
            )
            ax[row_idx, col_idx].axvline(  # type: ignore
                x=0.0 if row_idx > 0 else time_space_x_center,
                color="black",
                linewidth=0.5,
                zorder=2,
            )

        # the y-axis labels are set for the first column
        if col_idx == 0:
            ax[0, col_idx].set_ylabel(  # type: ignore
                r"$\psi_{n}^{\left(\alpha;\mu\right)}\left(x\right)$"
            )
            ax[1, col_idx].set_ylabel(  # type: ignore
                "Real part of\n"
                r"$\mathcal{F}\left\{\psi_{n}^{\left(\alpha;\mu\right)}"
                r"\left(x\right)\right\}$"
            )
            ax[2, col_idx].set_ylabel(  # type: ignore
                "Imaginary part of\n"
                r"$\mathcal{F}\left\{\psi_{n}^{\left(\alpha;\mu\right)}"
                r"\left(x\right)\right\}$"
            )

        # the Hermite functions are computed in the time/space domain
        hermite_basis = HermiteFunctionBasis(
            n=n,
            alpha=alpha,
            time_space_x_center=time_space_x_center,
        )

        hermite_function = hermite_basis(x=x_values)[::, -1]

        # from this, the numerical Continuous Fourier Transform is computed
        time_space_signal = TimeSpaceSignal(
            x=x_values,
            y=hermite_function,
        )
        numerical_discrete_ft = discrete_ft(signal=time_space_signal, norm="ortho")
        numerical_continuous_ft = convert_discrete_to_continuous_ft(
            dft=numerical_discrete_ft,
        )

        # besides, the analytical Continuous Fourier Transform is computed
        analytical_conti_ft = hermite_basis(
            omega=numerical_discrete_ft.angular_frequencies
        )[::, -1]

        # the Hermite function and its Continuous Fourier Transform are plotted
        ax[0, col_idx].plot(
            x_values,
            hermite_function,
            color=color_analytical,
            linewidth=2.0,
            label="Hermite function",
            zorder=3,
        )

        ax[1, col_idx].plot(
            np.fft.fftshift(numerical_discrete_ft.angular_frequencies),
            np.fft.fftshift(numerical_continuous_ft.real),
            color=color_numerical,
            linewidth=4.0,
            label="Numerical CFT",
            zorder=3,
        )
        ax[1, col_idx].plot(
            np.fft.fftshift(numerical_discrete_ft.angular_frequencies),
            np.fft.fftshift(analytical_conti_ft.real),
            color=color_analytical,
            linewidth=1.0,
            label="Analytical CFT",
            zorder=3,
        )

        ax[2, col_idx].plot(
            np.fft.fftshift(numerical_discrete_ft.angular_frequencies),
            np.fft.fftshift(numerical_continuous_ft.imag),
            label="Numerical CFT",
            color=color_numerical,
            linewidth=4.0,
            zorder=3,
        )
        ax[2, col_idx].plot(
            np.fft.fftshift(numerical_discrete_ft.angular_frequencies),
            np.fft.fftshift(analytical_conti_ft.imag),
            color=color_analytical,
            linewidth=1.0,
            label="Analytical CFT",
            zorder=3,
        )

        # the titles and x-axis labels are set
        ax[0, col_idx].set_title(  # type: ignore
            r"$n = " + f"{n}$"
            r", $\alpha = "
            + f"{alpha:.1f}$"
            + r", $\mu = "
            + f"{time_space_x_center:.1f}$\n\n"
            "Time/Space Domain"
        )
        ax[0, col_idx].set_xlabel(r"$x$")  # type: ignore

        ax[1, col_idx].set_title("Frequency Domain", y=1.40)  # type: ignore
        ax[2, col_idx].set_xlabel(r"$\omega$")  # type: ignore

        # the plots for the Continuous Fourier Transform also have a legend
        ax[1, col_idx].legend(loc=8, bbox_to_anchor=(0.5, 1.05))  # type: ignore

        # the x-axis limits are set dynamically to make sure only the relevant parts are
        # shown
        x_fadeouts = approximate_hermite_funcs_fadeout_x(
            n=n,
            alpha=alpha,
            x_center=time_space_x_center,
        )
        ax[0, col_idx].set_xlim(*x_fadeouts)  # type: ignore

        frequencies_fadeout = approximate_hermite_funcs_fadeout_x(
            n=n,
            alpha=1.0 / alpha,
            x_center=0.0,
        )
        ax[2, col_idx].set_xlim(*frequencies_fadeout)  # type: ignore

        # tiny red triangles are made at the bottom of the time/space domain plots to
        # indicate the center of the Hermite functions
        ax[0, col_idx].scatter(
            time_space_x_center,
            ax[0, col_idx].get_ylim()[0] + 0.02 * np.diff(ax[0, col_idx].get_ylim())[0],
            color="red",
            marker="v",
            s=100,
            zorder=2,
        )

    # the figure is finalised with a title
    fig.suptitle(
        "Dilated Hermite Functions and their Continuous Fourier Transforms",
        fontsize=18,
        y=1.05,
    )

    # # the plot is saved ...
    if os.getenv("ROBHERMFT_DEVELOPER", "false").lower() == "true":
        plt.savefig(os.path.join(os.path.dirname(__file__), PLOT_FILEPATH))

    # # ... and shown
    plt.show()
