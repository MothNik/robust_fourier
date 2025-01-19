"""
This script serves as a demonstration of the Fourier basis functions that are based on
the Hermite functions.

It demonstrates the where the time/space domain is shifted in the x-direction while the
frequency domain remains centered at the origin.

"""

# === Imports ===

import os

import _07_utils as ut
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

# === Main ===

# the orders of the Hermite functions
ORDERS = [
    9,
    10,
]
# the scaling factor gamma for the x-direction in the frequency domain
FREQUENCY_GAMMA = 2.0
# the center in the time/space domain
TIME_SPACE_CENTER = 2.0
# the number of points in the time/space domain
TIME_SPACE_NUM_POINTS = 10_001

# the colours for the different orders as (colour for real, colour for imaginary)
COLORS = [
    ("#FF6B3F", "#962446"),
    ("#36A07F", "#FF3399"),
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
    time_space_hermite_function = TimeSpaceSignal(
        y=time_space_hermite_function_original,
        x=t_values,
    )

    # the continuous Fourier transform is computed for the highest order
    hermite_function_cft = convert_discrete_to_continuous_ft(
        dft=discrete_ft(signal=time_space_hermite_function),
    )

    # aside from this numerical computation, the analytical Fourier transform is
    # computed for comparison
    hermite_function_ft_analytical = (
        ((-1.0j) ** order)
        * np.exp(
            -1.0j
            * TIME_SPACE_CENTER
            * np.fft.fftshift(hermite_function_cft.angular_frequencies)
        )
        * single_hermite_function(
            x=np.fft.fftshift(hermite_function_cft.angular_frequencies),
            n=order,
            alpha=FREQUENCY_GAMMA,
            x_center=None,
        )
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
            linewidth=1.0,
            zorder=2,
        )
        ax[row_i, col_j].axhline(  # type: ignore
            y=0.0,
            color="black",
            linewidth=1.0,
            zorder=3,
        )

    # the time/space domain is plotted
    ax[0, 0].plot(  # type: ignore
        time_space_hermite_function.x,
        time_space_hermite_function.y,
        color=COLORS[index][0],
        linewidth=3.0,
        zorder=4,
    )
    ax[1, 0].plot(  # type: ignore
        time_space_hermite_function.x,
        np.zeros_like(time_space_hermite_function.y),
        color=COLORS[index][1],
        linewidth=3.0,
        zorder=4,
    )

    # the frequency domain is plotted
    ax[0, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        np.fft.fftshift(hermite_function_cft.real),
        color="black",
        label="Numerical",
        linewidth=5.0,
        alpha=0.15,
        zorder=4,
    )
    ax[0, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        hermite_function_ft_analytical.real,
        color=COLORS[index][0],
        label="Analytical",
        linewidth=3.0,
        linestyle="--",
        zorder=5,
    )

    ax[1, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        np.fft.fftshift(hermite_function_cft.imag),
        color="black",
        label="Numerical",
        linewidth=5.0,
        alpha=0.15,
        zorder=4,
    )
    ax[1, 1].plot(  # type: ignore
        np.fft.fftshift(hermite_function_cft.angular_frequencies),
        hermite_function_ft_analytical.imag,
        color=COLORS[index][1],
        label="Analytical",
        linewidth=3.0,
        linestyle="--",
        zorder=5,
    )

    # the labels and titles are set
    ax[0, 0].set_ylabel("Real part")  # type: ignore
    ax[1, 0].set_ylabel("Imaginary part")  # type: ignore
    ax[1, 0].set_xlabel(r"Time $t$")  # type: ignore
    ax[1, 1].set_xlabel(r"Angular frequency $\omega$")  # type: ignore

    ax[0, 0].set_title("Time/Space domain")  # type: ignore
    ax[0, 1].set_title("Frequency domain")  # type: ignore

    fig.suptitle(
        "Hermite Function Basis Fourier Pair with Shift in the TIME/SPACE Domain\n\n"
        + r"$j = "
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
        + r", $\omega_{0}=0$",
        fontsize=18,
        y=1.05,
    )

    # a legend is added for the analytical and numerical Fourier transform
    ax[0, 1].legend(loc=8, bbox_to_anchor=(1.25, 0.4))  # type: ignore
    ax[1, 1].legend(loc=8, bbox_to_anchor=(1.25, 0.4))  # type: ignore

    # the x-axis limits are set
    ax[0, 0].set_xlim(  # type: ignore
        t_fadeout[0] - 0.33 * fadeout_difference,
        t_fadeout[1] + 0.33 * fadeout_difference,
    )
    omega_fadeout = hermite_approx.x_tail_drop_to_fraction(
        n=order,
        y_fraction=0.01,
        alpha=FREQUENCY_GAMMA,
        x_center=None,
    ).ravel()
    omega_fadeout_difference = omega_fadeout[1] - omega_fadeout[0]
    ax[0, 1].set_xlim(  # type: ignore
        omega_fadeout[0] - 0.33 * omega_fadeout_difference,
        omega_fadeout[1] + 0.33 * omega_fadeout_difference,
    )

    # finally, the respective centers are highlighted
    for iter_i in range(0, ax.size):  # type: ignore
        row_i, col_j = divmod(iter_i, 2)
        is_time_space_axis = ut.is_time_space_axis(iter_i)
        is_ax_for_center_label = ut.is_desired_axis(
            iter_i,
            domain="time_space",
            complex_axis="imaginary",
        )
        x_center = TIME_SPACE_CENTER if is_time_space_axis else 0.0
        # an arrow is added to indicate the center on the x-axis
        x_lims, y_lims = ut.get_and_freeze_both_axis_limits(
            ax=ax[row_i, col_j],  # type: ignore
        )
        x_diff_sign = -1.0 if is_time_space_axis else 1.0
        arrow_rad = -0.1 if is_time_space_axis else 0.1
        center_text = "\n"
        if is_ax_for_center_label:
            center_text = ut.CENTER_ARROW_LABEL
            x_diff_sign = 2.2 * x_diff_sign

        ax[row_i, col_j].annotate(  # type: ignore
            text=center_text,
            xy=(
                x_center + x_lims.get_scaled_width(0.01 * x_diff_sign),
                y_lims.lower + y_lims.get_scaled_width(0.01),
            ),
            xytext=(
                x_center + x_lims.get_scaled_width(0.1 * x_diff_sign),
                y_lims.lower + y_lims.get_scaled_width(0.15),
            ),
            fontsize=ut.LABEL_FONTSIZE,
            horizontalalignment="center",
            verticalalignment="center",
            arrowprops=ut.CENTER_ARROWPROPS.as_arrow_properties(rad=arrow_rad),
            bbox=ut.NORMAL_TEXTBOX_BBOX_PROPS_ASSIGNMENT[center_text],
            zorder=6,
        )

    # the fadeout in both the time/space and frequency domain is highlighted
    for iter_i in range(0, ax.size):  # type: ignore
        row_i, col_j = divmod(iter_i, 2)
        is_time_space_axis = ut.is_time_space_axis(iter_i)
        # an arrow is added to indicate the fadeout on the x-axis
        x_lims, y_lims = ut.get_both_axis_limits(ax=ax[row_i, col_j])  # type: ignore
        fadeout_text = "\n"
        if ut.is_real_axis(iter_i):
            fadeout_text = (
                ut.FADEOUT_TIME_SPACE_LABEL
                if is_time_space_axis
                else ut.FADEOUT_FREQUENCY_LABEL
            )

        for fadeout, x_diff_sign in zip(
            t_fadeout if is_time_space_axis else omega_fadeout,
            [-1.0, 1.0],
        ):
            xy = (fadeout, y_lims.get_scaled_width(0.025))
            xytext = [
                fadeout + x_lims.get_scaled_width(0.07 * x_diff_sign),
                y_lims.upper - y_lims.get_scaled_width(0.2),
            ]
            ax[row_i, col_j].annotate(  # type: ignore
                text=fadeout_text,
                xy=xy,
                xytext=xytext,
                fontsize=ut.LABEL_FONTSIZE,
                horizontalalignment="center",
                verticalalignment="center",
                arrowprops=ut.FADEOUT_ARROWPROPS.as_arrow_properties(),
                bbox=ut.NORMAL_TEXTBOX_BBOX_PROPS_ASSIGNMENT[fadeout_text],
                zorder=7,
            )

    # the plot is stored
    if os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":
        plot_file_path = PLOT_FILEPATH.format(
            index=index + 1,
            order=order,
        )
        plt.savefig(
            os.path.join(os.path.dirname(__file__), plot_file_path),
        )

plt.show()
