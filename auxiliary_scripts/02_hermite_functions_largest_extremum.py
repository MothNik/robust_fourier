"""
This script aims to evaluate a simple approximation formula for the xy-position of the
largest extremum (= outermost maximum/minimum) of the Hermite functions.
Since the extrema are symmetric around the origin, only the positive extrema are
computed and the negative extrema are obtained by symmetry.

The script is based on an approximation of the largest zero (= outermost root) of the
Hermite functions as well as their numerical fadeout point (where the tail fades below
machine precision).
The largest extremum has to be located between these two points and can be found via
numerical optimisation.

In the end, it was found that a quintic B-spline with only a few knots is sufficient to
represent the largest extrema of the Hermite functions with a decent accuracy.
Therefore, this script auto-generates the B-spline coefficients for the largest
extrema of the Hermite functions and stores them in the Python file that will then be
available within ``robust_fourier``.

NOTE: THIS SCRIPT CAN ONLY BE RUN IF THE DEVELOPER MODE IS ENABLED BY SETTING THE
      ENVIRONMENT VARIABLE ``ROBFT_DEVELOPER`` TO ``true``.

"""

# === Imports ===

import json
import os
import subprocess
from math import sqrt as pysqrt
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.optimize import minimize
from tqdm import tqdm

from robust_fourier.hermite_functions import hermite_approx, single_hermite_function

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_fourier.mplstyle")
)

# === Constants ===

# the path where the reference data is stored (relative to the current file)
REFERENCE_DATA_FILE_PATH = "./files/02-01_hermite_functions_largest_extrema.npy"
# whether to overwrite the reference data (will trigger a computation of up to a few
# minutes)
OVERWRITE_REFERENCE_DATA = False

# the path where the diagnostic plot is stored (relative to the current file)
X_DIAGNOSTIC_PLOT_FILE_PATH = "./files/02-02_hermite_functions_X_largest_extrema.svg"
Y_DIAGNOSTIC_PLOT_FILE_PATH = "./files/02-03_hermite_functions_Y_largest_extrema.svg"

# the path where to store the spline specifications (relative to the current file)
SPLINE_SPECS_FILE_PATH = (
    "../src/robust_fourier/hermite_functions/_hermite_largest_extrema_spline.py"
)
# the template for storing the spline specifications in the Python file
SPLINE_SPECS_TEMPLATE = """
\"\"\"
Module :mod:`hermite_functions.{file_name}`

This file is auto-generated by the script ``{generator_file}``.

This module stores the B-spline specifications for the xy-positions of the largest
extrema (= outermost maximum/minimum) of the first ``{xy_order_stop}`` Hermite functions.

For both x and y, the splines are quintic B-splines with ``{x_num_knots}`` and ``{y_num_knots}`` knots
and their maximum absolute relative error is ``{max_abs_rel_error:.2e}`` with respect to a numerical
optimisation which itself is the limiting factor for the accuracy.

For diagnostic plots that show the fit quality, please see
``auxiliary_scripts/{x_diagnostic_plot_file_path}`` and
``auxiliary_scripts/{y_diagnostic_plot_file_path}``.

\"\"\"  # noqa: E501

# === Imports ===

import numpy as np

# === Constants ===

# the specifications of the B-spline for the positions of the largest extrema of the
# Hermite functions

# --- The X-positions of the largest extrema ---

HERMITE_LARGEST_EXTREMA_MAX_ORDER = {xy_order_stop}
X_HERMITE_LARGEST_EXTREMA_SPLINE_TCK = (
    np.array({x_knots}),
    np.array({x_coefficients}),
    {x_degree},
)

# --- the Y-positions of the largest extrema ---

Y_HERMITE_LARGEST_EXTREMA_SPLINE_TCK = (
    np.array({y_knots}),
    np.array({y_coefficients}),
    {y_degree},
)

"""  # noqa: E501

# whether to overwrite the spline coefficients file
OVERWRITE_SPLINE_SPECS = True

# the number of grid points for the initial bracketing of the extremum
NUM_EVAL = 100
# the x and gradient tolerances for the optimisation
OPT_XTOL = 1e-13
OPT_GTOL = 1e-15
# the maximum number of iterations for the optimisation
MAX_ITER = 100_000

# the orders and spacings of the Hermite functions to evaluate
ORDER_START = 1  # for order 0 the extremum is exactly x = 0 and y = pi ** (-0.25)
ORDERS_AND_SPACINGS = [
    (100, 1),  # order to, spacing
    (250, 2),
    (500, 4),
    (1_000, 8),
    (2_500, 16),
    (5_000, 32),
    (10_000, 64),
    (25_000, 128),
    (50_000, 256),
    (100_512, 512),
]
# the degree of the B-spline
SPLINE_DEGREE = 5
# the maximum relative tolerance for the extrema evaluated by the spline
XY_MAX_RTOL = 1e-12

# === Functions ===


def _hermite_func_first_derivative(
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Evaluates the first derivative of the Hermite function of order ``n`` at the
    position ``x``.

    """

    # the first derivative is given as a weighted sum of the Hermite functions of order
    # ``n-1`` and ``n+1``
    return pysqrt(n / 2) * single_hermite_function(x=x, n=n - 1) - pysqrt(
        (n + 1) / 2
    ) * single_hermite_function(x=x, n=n + 1)


def _hermite_func_second_derivative(
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Evaluates the second derivative of the Hermite function of order ``n`` at the
    position ``x``.

    """

    # the second derivative is given as a weighted sum of the first derivatives of the
    # Hermite functions of order ``n-1`` and ``n+1``
    return pysqrt(n / 2) * _hermite_func_first_derivative(x=x, n=n - 1) - pysqrt(
        (n + 1) / 2
    ) * _hermite_func_first_derivative(x=x, n=n + 1)


def find_hermite_functions_largest_extremum_xy(n: int) -> Tuple[float, float]:
    """
    Finds the xy-position of the largest extremum of the Hermite function of order
    ``n``.

    """

    # an initial guess for the location of the largest extremum is made by bracketing it
    # between the largest zero and the fadeout point
    x_largest_zero = hermite_approx.x_largest_zeros(n=n)[-1]
    x_fadeout = hermite_approx.x_fadeout(n=n)[-1]

    # the extremum is bracketed between the largest zero and the fadeout point;
    # over this range, the first derivative of the Hermite function is evaluated and
    # checked for a sign change
    # NOTE: the fadeout point is way too conservative and therefore the bracketing is
    #       done with a smaller range
    x_values_initial = np.linspace(
        start=x_largest_zero,
        stop=x_largest_zero + 0.2 * (x_fadeout - x_largest_zero),
        num=NUM_EVAL,
    )
    hermite_derivative_values = _hermite_func_first_derivative(
        x=x_values_initial,
        n=n,
    )

    # the sign change is used to find the initial interval of the extremum
    sign_change_index = np.where(
        np.sign(hermite_derivative_values[:-1])
        != np.sign(hermite_derivative_values[1:])
    )[0][0]

    # a bounded interval around the sign change is used to find the extremum
    if sign_change_index.size < 1:
        raise RuntimeError("No sign change found for the Hermite function derivative.")

    lower_bound, upper_bound = x_values_initial[
        sign_change_index : sign_change_index + 2
    ]

    # the extremum is found via numerical optimisation
    # NOTE: for the positive x-values, the extremum is a maximum
    # NOTE: for order 1, the second derivative is not available and the optimisation
    #       method is changed to "Newton-CG" with a numerical approximation of the
    #       Hessian
    if n > 1:
        opt_method = "trust-exact"
        opt_kwargs = dict(
            hess=lambda x: -_hermite_func_second_derivative(x=x, n=n)[0],
            options=dict(maxiter=MAX_ITER, gtol=OPT_GTOL),
        )
    else:
        opt_method = "Newton-CG"
        opt_kwargs = dict(
            hess="3-point",
            options=dict(maxiter=MAX_ITER, xtol=OPT_XTOL),
        )

    result = minimize(
        fun=lambda x: -single_hermite_function(x=x, n=n)[0],
        jac=lambda x: -_hermite_func_first_derivative(x=x, n=n)[0],
        x0=(lower_bound + upper_bound) / 2,
        method=opt_method,
        **opt_kwargs,
    )

    # a final sanity check is made to ensure that the unbounded optimisation did not
    # leave its bounds
    if not (x_largest_zero <= result.x[0] <= x_fadeout):
        raise RuntimeError("Optimisation result out of bounds.")

    return result.x[0], (-1.0) * result.fun


# === Main ===

if __name__ == "__main__" and os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":

    # --- Reference data loading / computation ---

    # if available and enabled, the reference data is loaded
    reference_file_path = os.path.join(
        os.path.dirname(__file__), REFERENCE_DATA_FILE_PATH
    )
    try:
        if OVERWRITE_REFERENCE_DATA:
            raise FileNotFoundError()

        reference_data = np.load(reference_file_path, allow_pickle=False)
        (
            orders,
            x_outermost_extremum,
            y_outermost_extremum,
        ) = (
            reference_data[::, 0],
            reference_data[::, 1],
            reference_data[::, 2],
        )

    # otherwise, the reference data is computed
    except (FileNotFoundError, NotADirectoryError):
        order_start = ORDER_START
        orders = []
        for order_end, spacing in ORDERS_AND_SPACINGS:
            orders.extend(range(order_start, order_end, spacing))
            order_start = order_end

        orders = np.array(orders, dtype=np.int64)
        x_outermost_extremum = np.empty_like(orders, dtype=np.float64)
        y_outermost_extremum = np.empty_like(orders, dtype=np.float64)

        progress_bar = tqdm(total=len(orders), desc="Computing outermost extrema")
        for idx, n in enumerate(orders):
            (
                x_outermost_extremum[idx],
                y_outermost_extremum[idx],
            ) = find_hermite_functions_largest_extremum_xy(n=n)
            progress_bar.update(1)

        # the reference data is stored
        np.save(
            reference_file_path,
            np.column_stack((orders, x_outermost_extremum, y_outermost_extremum)),
            allow_pickle=False,
        )

    # --- Spline fitting ---

    # 1) the x-positions

    # the spline is fitted with an ever decreasing smoothing value s until the
    # maximum absolute error drops below the threshold
    max_abs_rel_error = np.inf
    s_value = 1e-10
    x_weights = np.reciprocal(x_outermost_extremum)  # all > 0
    x_tck = None
    x_outermost_extremum_approx = None
    while max_abs_rel_error > XY_MAX_RTOL and s_value > 1e-30:
        x_tck = splrep(
            x=orders,
            y=x_outermost_extremum,
            w=x_weights,
            k=SPLINE_DEGREE,
            s=s_value,
        )
        x_outermost_extremum_approx = splev(x=orders, tck=x_tck)

        max_abs_error = np.abs(
            (x_outermost_extremum - x_outermost_extremum_approx) * x_weights
        ).max()
        s_value /= 10.0**0.25

    assert x_tck is not None and x_outermost_extremum_approx is not None, (
        "No spline was fitted for the X-positions, please re-adjust the tolerances and "
        "smoothing values."
    )
    print(
        f"\nFinal number of spline knots for the X-positions: {len(x_tck[0])} for "
        f"smoothing value {s_value=:.2e}"
    )

    # 2) the y-positions

    # the y-positions of the largest extrema are fitted with their own spline
    max_abs_rel_error = np.inf
    s_value = 1e-10
    y_weights = np.reciprocal(y_outermost_extremum)  # all > 0
    y_tck = None
    y_outermost_extremum_approx = None
    while max_abs_rel_error > XY_MAX_RTOL and s_value > 1e-30:
        y_tck = splrep(
            x=orders,
            y=y_outermost_extremum,
            w=y_weights,
            k=SPLINE_DEGREE,
            s=s_value,
        )
        y_outermost_extremum_approx = splev(x=orders, tck=y_tck)

        max_abs_error = np.abs(
            (y_outermost_extremum - y_outermost_extremum_approx) * y_weights
        ).max()
        s_value /= 10.0**0.25

    assert y_tck is not None and y_outermost_extremum_approx is not None, (
        "No spline was fitted for the Y-positions, please re-adjust the tolerances and "
        "smoothing values."
    )
    print(
        f"Final number of spline knots for the Y-positions: {len(y_tck[0])} for "
        f"smoothing value {s_value=:.2e}"
    )

    # the spline coefficients are stored (if enabled)
    if OVERWRITE_SPLINE_SPECS:
        spline_specs_file_path = os.path.join(
            os.path.dirname(__file__), SPLINE_SPECS_FILE_PATH
        )

        # the Python-file is created from the template ...
        with open(spline_specs_file_path, "w") as spline_specs_file:
            spline_specs_file.write(
                SPLINE_SPECS_TEMPLATE.format(
                    generator_file="/".join(
                        __file__.replace("\\", "/").split("/")[-2::]
                    ),
                    file_name=SPLINE_SPECS_FILE_PATH.split("/")[-1].split(".")[0],
                    xy_order_stop=round(orders[-1]),
                    x_num_knots=len(x_tck[0]),
                    y_num_knots=len(y_tck[0]),
                    max_abs_rel_error=XY_MAX_RTOL,
                    x_diagnostic_plot_file_path=X_DIAGNOSTIC_PLOT_FILE_PATH.split("/")[
                        -1
                    ],
                    y_diagnostic_plot_file_path=Y_DIAGNOSTIC_PLOT_FILE_PATH.split("/")[
                        -1
                    ],
                    x_knots=json.dumps(x_tck[0].tolist()),  # type: ignore
                    x_coefficients=json.dumps(x_tck[1].tolist()),
                    x_degree=SPLINE_DEGREE,
                    y_knots=json.dumps(y_tck[0].tolist()),  # type: ignore
                    y_coefficients=json.dumps(y_tck[1].tolist()),
                    y_degree=SPLINE_DEGREE,
                )
            )

        # ... and formatted
        subprocess.run(["black", spline_specs_file_path])

    # --- Diagnostic plot ---

    # 1) the x-positions

    fig_x, ax_x = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(12, 8),
    )

    ax_x[0].plot(  # type: ignore
        orders,
        x_outermost_extremum,
        color="red",
        label="Optimised X-Extrema",
    )
    ax_x[0].plot(  # type: ignore
        orders,
        x_outermost_extremum_approx,
        color="#00CCCC",
        label="Spline Approximation",
    )

    ax_x[1].axhline(0.0, color="black", linewidth=0.5)  # type: ignore
    ax_x[1].plot(  # type: ignore
        orders,
        100.0 * x_weights * (x_outermost_extremum - x_outermost_extremum_approx),
        color="#00CCCC",
        zorder=2,
        label="Difference",
    )
    ax_x[1].axhline(  # type: ignore
        100.0 * XY_MAX_RTOL,
        color="#FF007F",
        linestyle="--",
        label="Threshold",
        linewidth=2.0,
        zorder=0,
    )
    ax_x[1].axhline(  # type: ignore
        -100.0 * XY_MAX_RTOL,
        color="#FF007F",
        linestyle="--",
        linewidth=2.0,
        zorder=1,
    )
    ax_x[1].scatter(  # type: ignore
        x_tck[0],
        np.zeros_like(x_tck[0]),
        s=60,
        marker=6,
        color="purple",
        label="Knots",
        zorder=3,
    )

    ax_x[0].set_ylabel("Largest Extremum X-Position")  # type: ignore
    ax_x[1].set_ylabel(r"Approximation Error $\left(\%\right)$")  # type: ignore
    ax_x[1].set_xlabel("Hermite Function Order")  # type: ignore

    ax_x[1].set_xlim(orders[0], orders[-1])  # type: ignore

    ax_x[0].legend()  # type: ignore
    ax_x[1].legend()  # type: ignore

    fig_x.suptitle(
        "X-Positions of the Largest Extrema of the Hermite Functions",
        fontsize=18,
    )

    # 2) the y-positions

    fig_y, ax_y = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(12, 8),
    )

    ax_y[0].plot(  # type: ignore
        orders,
        y_outermost_extremum,
        color="red",
        label="Optimised Y-Extrema",
    )
    ax_y[0].plot(  # type: ignore
        orders,
        y_outermost_extremum_approx,
        color="#00CCCC",
        label="Spline Approximation",
    )

    ax_y[1].axhline(0.0, color="black", linewidth=0.5)  # type: ignore
    ax_y[1].plot(  # type: ignore
        orders,
        100.0 * y_weights * (y_outermost_extremum - y_outermost_extremum_approx),
        color="#00CCCC",
        zorder=2,
        label="Difference",
    )
    ax_y[1].axhline(  # type: ignore
        100.0 * XY_MAX_RTOL,
        color="#FF007F",
        linestyle="--",
        label="Threshold",
        linewidth=2.0,
        zorder=0,
    )
    ax_y[1].axhline(  # type: ignore
        -100.0 * XY_MAX_RTOL,
        color="#FF007F",
        linestyle="--",
        linewidth=2.0,
        zorder=1,
    )
    ax_y[1].scatter(  # type: ignore
        y_tck[0],
        np.zeros_like(y_tck[0]),
        s=60,
        marker=6,
        color="purple",
        label="Knots",
        zorder=3,
    )

    ax_y[0].set_ylabel("Largest Extremum Y-Position")  # type: ignore
    ax_y[1].set_ylabel(r"Approximation Error $\left(\%\right)$")  # type: ignore
    ax_y[1].set_xlabel("Hermite Function Order")  # type: ignore

    ax_y[1].set_xlim(orders[0], orders[-1])  # type: ignore

    ax_y[0].legend()  # type: ignore
    ax_y[1].legend()  # type: ignore

    fig_y.suptitle(
        "Y-Positions of the Largest Extrema of the Hermite Functions",
        fontsize=18,
    )

    # 3) Storing the diagnostic plots

    # the plot is stored (if the spline coefficients were stored)
    if OVERWRITE_SPLINE_SPECS:
        for path, fig in zip(
            [X_DIAGNOSTIC_PLOT_FILE_PATH, Y_DIAGNOSTIC_PLOT_FILE_PATH],
            [fig_x, fig_y],
        ):
            fig.savefig(os.path.join(os.path.dirname(__file__), path))

    plt.show()


elif __name__ == "__main__":
    print(
        "This script can only be run if the developer mode is enabled by setting the "
        "environment variable 'ROBFT_DEVELOPER' to 'true'."
    )
