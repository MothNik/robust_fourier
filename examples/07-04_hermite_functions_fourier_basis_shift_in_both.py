"""
This script serves as a demonstration of the Fourier basis functions that are based on
the Hermite functions.

It demonstrates the most general case where both the time/space and frequency domains
are shifted in the x-direction.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from robust_fourier import hermite_approx, single_hermite_function
from robust_fourier.fourier_transform import (
    TimeSpaceSignal,
    convert_discrete_to_continuous_ft,
    discrete_ft,
)

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_fourier.mplstyle")
)

# === Example 1: Hermite functions centered at the origin in both domains ===

# the orders of the Hermite functions
ORDERS = [
    9,
    10,
]
# the scaling factor gamma for the x-direction in the frequency domain
FREQUENCY_GAMMA = 2.0
# the center in the frequency domain
FREQUENCY_CENTER = 15.0
# the center in the time/space domain
TIME_SPACE_CENTER = 2.0
# the number of points in the time/space domain
TIME_SPACE_NUM_POINTS = 10_001

# the colours for the different orders as (colour for real, colour for imaginary)
COLORS = [
    ("#36A07F", "#962446"),
    ("#DB9807", "#FF3399"),
]

# the path where to store the plot (only for developers)
PLOT_FILEPATH = (
    "../docs/hermite_functions/EX-07-02-{index:02d}-HermiteFunctionsFourierBasis_"
    "Frequency_at_Origin_Time_Space_Shifted_Order_{order:02d}.png"
)


# all orders are processed and plotted by means of a loop
for index, order in enumerate(ORDERS):

    # first, the scaling factor beta for the x-direction in the time/space domain is
    # computed as the reciprocal of gamma
    time_space_beta = 1.0 / FREQUENCY_GAMMA

    # then, the point where the Hermite functions drop below 1% of their maximum value
    # is determined ...
    t_fadeout = hermite_approx.x_tail_drop_to_fraction(
        n=order,
        y_fraction=0.01,
        alpha=time_space_beta,
        x_center=TIME_SPACE_CENTER,
    ).ravel()
    # ... and the sampling points are set to cover the range from -x_fadeout to
    # x_fadeout and beyond (for better resolution in the frequency domain)
    fadeout_difference = t_fadeout[1] - t_fadeout[0]
    t_values = np.linspace(
        start=t_fadeout[0] - 2.0 * fadeout_difference,
        stop=t_fadeout[1] + 2.0 * fadeout_difference,
        num=TIME_SPACE_NUM_POINTS,
    )

    # the Hermite functions are evaluated at the given points
    time_space_hermite_function_original = single_hermite_function(
        x=t_values,
        n=order,
        alpha=time_space_beta,
        x_center=TIME_SPACE_CENTER,
    )

    # the pre-factors to account for the shift in the frequency domain and the
    # symmetrization are computed
    time_space_prefactor = 2.0 * np.cos(
        FREQUENCY_CENTER * (t_values - TIME_SPACE_CENTER)
    )

    time_space_hermite_function = TimeSpaceSignal(
        y=time_space_prefactor * time_space_hermite_function_original,
        x=t_values,
    )

    # the continuous Fourier transform is computed
    hermite_function_cft = convert_discrete_to_continuous_ft(
        dft=discrete_ft(signal=time_space_hermite_function),
    )

    # aside from this numerical computation, the analytical Fourier transform is
    # computed for comparison
    # a prefactor is computed to account for the shift in the time/space domain
    ft_prefactor = ((-1.0j) ** order) * np.exp(
        -1.0j
        * TIME_SPACE_CENTER
        * np.fft.fftshift(hermite_function_cft.angular_frequencies)
    )

    hermite_function_ft_analytical = ft_prefactor * single_hermite_function(
        x=np.fft.fftshift(hermite_function_cft.angular_frequencies),
        n=order,
        alpha=FREQUENCY_GAMMA,
        x_center=FREQUENCY_CENTER,
    )

    # a mirrored version of the Hermite functions has to be added to the analytical
    # Fourier transform to account for the symmetrization
    hermite_function_ft_analytical += ft_prefactor * single_hermite_function(
        x=np.fft.fftshift(hermite_function_cft.angular_frequencies),
        n=order,
        alpha=FREQUENCY_GAMMA,
        x_center=-FREQUENCY_CENTER,
    )

    # the Hermite function and its Fourier transform are plotted

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 12),
    )
    ax[0, 0].sharex(ax[1, 0])  # type: ignore
    ax[0, 1].sharex(ax[1, 1])  # type: ignore

    for iter_i in range(0, ax.size):  # type: ignore
        row_i, col_j = divmod(iter_i, 2)
        ax[row_i, col_j].axvline(  # type: ignore
            x=0.0,
            color="black",
            linewidth=0.5,
            zorder=2,
        )
        ax[row_i, col_j].axhline(  # type: ignore
            y=0.0,
            color="black",
            linewidth=0.5,
            zorder=2,
        )

    # the time/space domain is plotted
    ax[0, 0].plot(  # type: ignore
        time_space_hermite_function.x,
        time_space_hermite_function.y.real,
        color=COLORS[index][0],
        linewidth=3.0,
        zorder=3,
    )
    ax[1, 0].plot(  # type: ignore
        time_space_hermite_function.x,
        time_space_hermite_function.y.imag,
        color=COLORS[index][1],
        linewidth=3.0,
        zorder=3,
    )

    # the frequency domain is plotted
    ax[0, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        np.fft.fftshift(hermite_function_cft.real),
        color="black",
        label="Numerical",
        linewidth=5.0,
        alpha=0.25,
        zorder=3,
    )
    ax[0, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        hermite_function_ft_analytical.real,
        color=COLORS[index][0],
        label="Analytical",
        linewidth=3.0,
        linestyle="--",
        zorder=4,
    )

    ax[1, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        np.fft.fftshift(hermite_function_cft.imag),
        color="black",
        label="Numerical",
        linewidth=5.0,
        alpha=0.25,
        zorder=3,
    )
    ax[1, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        hermite_function_ft_analytical.imag,
        color=COLORS[index][1],
        label="Analytical",
        linewidth=3.0,
        linestyle="--",
        zorder=4,
    )

    # the labels and titles are set
    ax[0, 0].set_ylabel("Real part")  # type: ignore
    ax[1, 0].set_ylabel("Imaginary part")  # type: ignore
    ax[1, 0].set_xlabel(r"Time $t$")  # type: ignore
    ax[1, 1].set_xlabel(r"Angular frequency $\omega$")  # type: ignore

    ax[0, 0].set_title("Time/Space domain")  # type: ignore
    ax[0, 1].set_title("Frequency domain")  # type: ignore

    fig.suptitle(
        "Hermite Function Basis Fourier Pair with Shifts in BOTH domains\n\n"
        + r"$n = "
        + f"{order}"
        + r"$"
        + r", $\beta = "
        + f"{time_space_beta:.1f}"
        + r"$"
        + r", $\gamma = "
        + f"{FREQUENCY_GAMMA:.0f}"
        + r"$"
        + r", $t_{0}="
        + f"{TIME_SPACE_CENTER:.0f}"
        + r"$"
        + r", $\omega_{0}="
        + f"{FREQUENCY_CENTER:.0f}"
        + r"$",
        fontsize=18,
        y=1.05,
    )

    # a legend is added for the analytical and numerical Fourier transform
    ax[0, 1].legend(loc=8, bbox_to_anchor=(1.25, 0.4))  # type: ignore
    ax[1, 1].legend(loc=8, bbox_to_anchor=(1.25, 0.4))  # type: ignore

    # finally, the x-axis limits are set
    ax[0, 0].set_xlim(  # type: ignore
        t_fadeout[0] - 0.2 * fadeout_difference,
        t_fadeout[1] + 0.2 * fadeout_difference,
    )
    omega_fadeout = hermite_approx.x_tail_drop_to_fraction(
        n=order,
        y_fraction=0.01,
        alpha=FREQUENCY_GAMMA,
        x_center=FREQUENCY_CENTER,
    ).ravel()
    omega_fadeout_difference = omega_fadeout[1] - omega_fadeout[0]
    ax[0, 1].set_xlim(  # type: ignore
        -(omega_fadeout[1] + 0.2 * omega_fadeout_difference),
        omega_fadeout[1] + 0.2 * omega_fadeout_difference,
    )

    # the plot is stored
    # if os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":
    #     plot_file_path = PLOT_FILEPATH.format(
    #         index=index + 1,
    #         order=order,
    #     )
    #     plt.savefig(
    #         os.path.join(os.path.dirname(__file__), plot_file_path),
    #     )

plt.show()
