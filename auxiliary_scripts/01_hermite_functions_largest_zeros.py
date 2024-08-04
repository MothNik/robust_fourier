"""
This script aims to evaluate a simple approximation formula for the location of the
largest zeros (= outermost roots) of the Hermite functions.
Since the roots are symmetric around the origin, only the positive roots are computed
and the negative roots are obtained by symmetry.
Furthermore, the roots of the Hermite functions - being the product of an always
positive Gaussian and a Hermite polynomial - are simply the roots of the Hermite
polynomials that can be computed accurately with the SciPy function
:func:`scipy.special.roots_hermite`.

However, the computation with the SciPy function takes too long and therefore, a simple
approximation formula should be derived to speed up the computation.

In the end, it was found that a quintic B-spline with only a few knots is sufficient to
represent the largest zeros of the Hermite functions with a decent accuracy.
Therefore, this script auto-generates the B-spline coefficients for the largest zeros
of the Hermite functions and stores them in the Python file that will then be available
within ``robust_hermite_ft``.

NOTE: THIS SCRIPT CAN ONLY BE RUN IF THE DEVELOPER MODE IS ENABLED BY SETTING THE
      ENVIRONMENT VARIABLE ``ROBHERMFT_DEVELOPER`` TO ``true``.

"""

# === Imports ===

import json
import os
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.special import roots_hermite
from tqdm import tqdm

plt.style.use(
    os.path.join(os.path.dirname(__file__), "../docs/robust_hermite_ft.mplstyle")
)


# === Constants ===

# the path where the reference data is stored (relative to the current file)
REFERENCE_DATA_FILE_PATH = "./files/01-01_hermite_functions_largest_zeros.npy"
# whether to overwrite the reference data (will trigger a massive computation)
OVERWRITE_REFERENCE_DATA = False

# the path where the diagnostic plot is stored (relative to the current file)
DIAGNOSTIC_PLOT_FILE_PATH = "./files/01-02_hermite_functions_largest_zeros.png"

# the path where to store the spline specifications (relative to the current file)
SPLINE_SPECS_FILE_PATH = (
    "../src/robust_hermite_ft/hermite_functions/_hermite_largest_roots_spline.py"
)
# the template for storing the spline specifications in the Python file
SPLINE_SPECS_TEMPLATE = """
\"\"\"
Module :mod:`hermite_functions._hermite_largest_roots_spline`

This file is auto-generated by the script ``auxiliary_scripts/01_hermite_functions_largest_zeros.py``.

This module stores the B-spline specifications for the largest zeros (= outermost roots)
of the first ``{order_stop}`` Hermite functions.
The spline is a quintic B-spline with ``{num_knots}`` knots and its maximum absolute relative
error is ``{max_abs_rel_error:.2e}`` with respect to ``scipy.special.roots_hermite``.
It should be noted that ``scipy.special.roots_hermite`` only gives roots whose y-values
have an absolute tolerance of roughly ``1e-8``, so the approximation error is limited
by this reference value and not by the spline itself.

For a diagnostic plot that shows the fit quality, please see
``auxiliary_scripts/{diagnostic_plot_file_path}``.

\"\"\"  # noqa: E501

# === Imports ===

import numpy as np

# === Constants ===

# the specifications of the B-spline for the largest zeros of the Hermite functions
HERMITE_LARGEST_ZEROS_MAX_ORDER = {order_stop}
HERMITE_LARGEST_ZEROS_SPLINE_TCK = (
    np.array({knots}),
    np.array({coefficients}),
    {degree},
)

"""  # noqa: E501

# whether to overwrite the spline coefficients file
OVERWRITE_SPLINE_SPECS = True

# the orders and spacings of the Hermite functions to evaluate
ORDER_START = 2  # for order 0 there is no root and for order 1 it's a exactly 0
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
# the maximum relative tolerance for the roots evaluated by the spline
# NOTE: since ``scipy.special.roots_hermite`` does not give roots that are that accurate
#       it is set as a soft limit because extra accuracy on the spline level is not
#       meaningful
X_MAX_RTOL = 1e-11


# === Main ===

if (
    __name__ == "__main__"
    and os.getenv("ROBHERMFT_DEVELOPER", "false").lower() == "true"
):

    # --- Reference data loading / computation ---

    # if available and enabled, the reference data is loaded
    reference_file_path = os.path.join(
        os.path.dirname(__file__), REFERENCE_DATA_FILE_PATH
    )
    try:
        if OVERWRITE_REFERENCE_DATA:
            raise FileNotFoundError()

        reference_data = np.load(reference_file_path, allow_pickle=False)
        orders, outerm_root_x_positions = (
            reference_data[::, 0],
            reference_data[::, 1],
        )

    # otherwise, the reference data is computed
    except (FileNotFoundError, NotADirectoryError):
        order_start = ORDER_START
        orders = []
        for order_end, spacing in ORDERS_AND_SPACINGS:
            orders.extend(range(order_start, order_end, spacing))
            order_start = order_end

        orders = np.array(orders, dtype=np.int64)
        outerm_root_x_positions = np.empty_like(orders, dtype=np.float64)

        progress_bar = tqdm(total=len(orders), desc="Computing outermost roots")
        for idx, n in enumerate(orders):
            outerm_root_x_positions[idx] = roots_hermite(n=n)[0][-1]
            progress_bar.update(1)

        # the reference data is stored
        np.save(
            reference_file_path,
            np.column_stack((orders, outerm_root_x_positions)),
            allow_pickle=False,
        )

    # --- Spline fitting ---

    # the spline is fitted with an ever decreasing smoothing value s until the
    # maximum absolute error drops below the threshold
    max_abs_rel_error = np.inf
    s_value = 1e-10
    weights = np.reciprocal(outerm_root_x_positions)  # all > 0
    tck = None
    outerm_root_x_positions_approx = None
    while max_abs_rel_error > X_MAX_RTOL and s_value > 1e-30:
        tck = splrep(
            x=orders,
            y=outerm_root_x_positions,
            w=weights,
            k=SPLINE_DEGREE,
            s=s_value,
        )
        outerm_root_x_positions_approx = splev(x=orders, tck=tck)

        max_abs_rel_error = np.abs(
            (outerm_root_x_positions - outerm_root_x_positions_approx) * weights
        ).max()
        s_value /= 10.0**0.25

    assert (
        tck is not None and outerm_root_x_positions_approx is not None
    ), "No spline was fitted, please re-adjust the tolerances and smoothing values."
    print(
        f"\nFinal number of spline knots: {len(tck[0])} for smoothing value "
        f"{s_value=:.2e}"
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
                    order_stop=round(orders[-1]),
                    num_knots=len(tck[0]),
                    max_abs_rel_error=X_MAX_RTOL,
                    diagnostic_plot_file_path=DIAGNOSTIC_PLOT_FILE_PATH,
                    knots=json.dumps(tck[0].tolist()),  # type: ignore
                    coefficients=json.dumps(tck[1].tolist()),
                    degree=SPLINE_DEGREE,
                )
            )

        # ... and formatted
        subprocess.run(["black", spline_specs_file_path])

    # --- Diagnostic plot ---

    fig, ax = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(12, 8),
    )

    ax[0].plot(  # type: ignore
        orders,
        outerm_root_x_positions,
        color="red",
        label="SciPy Accurate Roots",
    )
    ax[0].plot(  # type: ignore
        orders,
        outerm_root_x_positions_approx,
        color="#00CCCC",
        label="Spline Approximation",
    )

    ax[1].axhline(0.0, color="black", linewidth=0.5)  # type: ignore
    ax[1].plot(  # type: ignore
        orders,
        100.0 * weights * (outerm_root_x_positions - outerm_root_x_positions_approx),
        color="#00CCCC",
        zorder=2,
        label="Difference",
    )
    ax[1].axhline(  # type: ignore
        100.0 * X_MAX_RTOL,
        color="#FF007F",
        linestyle="--",
        label="Threshold",
        linewidth=2.0,
        zorder=0,
    )
    ax[1].axhline(  # type: ignore
        -100.0 * X_MAX_RTOL,
        color="#FF007F",
        linestyle="--",
        linewidth=2.0,
        zorder=1,
    )
    ax[1].scatter(  # type: ignore
        tck[0],
        np.zeros_like(tck[0]),
        s=60,
        marker=6,
        color="purple",
        label="Knots",
        zorder=3,
    )

    ax[0].set_ylabel("Largest Zero Position")  # type: ignore
    ax[1].set_ylabel(r"Approximation Error $\left(\%\right)$")  # type: ignore
    ax[1].set_xlabel("Hermite Function Order")  # type: ignore

    ax[0].set_ylim(0.0, None)  # type: ignore
    ax[1].set_xlim(orders[0], orders[-1])  # type: ignore

    ax[0].legend()  # type: ignore
    ax[1].legend()  # type: ignore

    # the plot is stored (if the spline coefficients were stored)
    if OVERWRITE_SPLINE_SPECS:
        diagnostic_plot_file_path = os.path.join(
            os.path.dirname(__file__), DIAGNOSTIC_PLOT_FILE_PATH
        )
        fig.savefig(diagnostic_plot_file_path)

    plt.show()


elif __name__ == "__main__":
    print(
        "This script can only be run if the developer mode is enabled by setting the "
        "environment variable 'ROBHERMFT_DEVELOPER' to 'true'."
    )
