"""
This test suite implements the tests for the module :mod:`hermite_functions._approximations`.

"""  # noqa: E501

# === Imports ===

from math import sqrt as pysqrt
from typing import Optional

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from robust_hermite_ft import (
    approximate_hermite_funcs_fadeout_x,
    approximate_hermite_funcs_largest_extrema_x,
    approximate_hermite_funcs_largest_zeros_x,
    single_hermite_function,
)

# === Constants ===

# the absolute and relative x-width for testing the largest zero
LARGEST_ZERO_X_ABS_WIDTH = 1e-9
LARGEST_ZERO_X_REL_WIDTH = 1e-9

# the y-tolerance for the fadeout points as a multiple of the machine epsilon
# NOTE: to avoid numerical issues, the tolerance is slightly increased
FADEOUT_Y_TOL_EPS_MULTIPLIER = 10.0

# the width of the x-interval spanned for testing the largest extrema via numerical
# minimisation
LARGEST_EXTREMUM_NUMMINIM_X_WIDTH = 1e-3
# the absolute tolerance of the numerical minimisation for the largest extrema
LARGEST_EXTREMUM_NUMMINIM_X_ATOL = 1e-13
# the maximum number of iterations for the numerical minimisation for the largest
# extrema
LARGEST_EXTREMUM_NUMMINIM_MAX_ITER = 100_000
# the relative y-tolerance for testing the largest extrema
LARGEST_EXTREMUM_Y_RTOL = 1e-10

# the scales alpha to test
TEST_SCALES_ALPHA = [0.05, 0.5, 1.0, 2.0, 20.0]

# the centers mu to test
TEST_X_CENTERS_MU = [-10.0, 0.0, None, 10.0]

# === Auxiliary Functions ===


def _hermite_func_largest_extremum_objective(
    x: float,
    n: int,
    alpha: float,
    x_center: Optional[float],
) -> float:
    """
    This function is the objective function for the minimisation of the Hermite function
    to find the largest extremum by the function :func:`minimize_scalar` from
    :mod:`scipy.optimize`.

    """

    return -abs(
        single_hermite_function(
            x=x,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )[0]
    )


# === Tests ===


@pytest.mark.parametrize("x_center", TEST_X_CENTERS_MU)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
@pytest.mark.parametrize(
    "n, num_roots",
    [
        (0, 0),  # special case of no root
        (1, 1),  # special case of only 1 root
        (2, 2),  # small even
        (3, 2),  # small odd
        (4, 2),  # small even
        (5, 2),  # small odd
        (10, 2),  # small even
        (11, 2),  # small odd
        (999, 2),  # medium odd
        (1_000, 2),  # medium even
        (9_999, 2),  # large odd
        (10_000, 2),  # large even
        (99_999, 2),  # very large odd
        (100_000, 2),  # very large even
    ],
)
def test_hermite_funcs_largest_zero_approximation(
    n: int,
    num_roots: int,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test checks the approximation of the largest zero of the Hermite functions
    via :func:`approximate_hermite_funcs_largest_zeros_x`.

    It does so by comparing checking if the Hermite functions change their sign at two
    points that are located around the estimated largest zero and only a small numerical
    tolerance apart.

    """

    # the largest zeros are estimated
    x_largest_zeros = approximate_hermite_funcs_largest_zeros_x(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # before proceeding, it is checked that the correct number of largest zeros is
    # returned
    assert x_largest_zeros.size == num_roots, (
        f"Expected {num_roots=} largest zeros for {n=}, {alpha=}, {x_center=}, "
        f"but got {x_largest_zeros.size}."
    )

    # then, the sign change at each estimated largest zero is checked
    x_center_for_ref_tolerance = x_center if x_center is not None else 0.0
    for x_lgz in x_largest_zeros:
        # NOTE: for the special case that the largest zero is exactly zero, the sign
        #       change is not checked with a relative width, but with an absolute width
        # NOTE: ``x_center`` has to be subtracted for the relative width to avoid it
        #       from getting bigger than it truly is just because of the offset
        x_reference_distance = max(
            LARGEST_ZERO_X_ABS_WIDTH / alpha,
            LARGEST_ZERO_X_REL_WIDTH * (abs(x_lgz) - abs(x_center_for_ref_tolerance)),
        )
        x_zero_reference = np.array(
            [
                x_lgz - x_reference_distance,
                x_lgz + x_reference_distance,
            ]
        )

        # the Hermite function is evaluated at the reference points ...
        hermite_around_zero_values = single_hermite_function(
            x=x_zero_reference,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

        # ... and checked for the sign change that has to happen at the largest zero
        # NOTE: the ``!=`` operator is used to check for a sign change because it could
        #       happen that the Hermite function is exactly zero and then a check for a
        #       flip of the sign between -1 and +1 or vice versa would not detect this
        assert np.sign(hermite_around_zero_values[0]) != np.sign(
            hermite_around_zero_values[1]
        ), (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"does not change its sign at the approximated largest zero {x_lgz:.10f}."
        )


@pytest.mark.parametrize("x_center", TEST_X_CENTERS_MU)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
@pytest.mark.parametrize(
    "n", [0, 1, 2, 3, 4, 5, 10, 11, 999, 1_000, 9_999, 10_000, 99_999, 100_000]
)
def test_hermite_funcs_fadeout_approximation(
    n: int,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test checks the approximation of the fadeout point of the Hermite functions
    via :func:`approximate_hermite_funcs_fadeout_x`.

    It does so by checking if the Hermite functions are numerically zero at the
    estimated fadeout points.

    """

    # the fadeout points are estimated
    x_fadeouts = approximate_hermite_funcs_fadeout_x(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # the Hermite functions are evaluated at the fadeout points ...
    hermite_fadeout_values = single_hermite_function(
        x=x_fadeouts,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # ... and checked if they are close to zero
    # NOTE: that different alpha-values scale the Hermite functions to preserve
    #       orthonormality, so this scaling (by the square root of alpha) is taken
    #       into account in the tolerance
    y_fadeout_tolerance = (
        pysqrt(alpha) * FADEOUT_Y_TOL_EPS_MULTIPLIER * np.finfo(np.float64).eps
    )
    assert np.abs(hermite_fadeout_values <= y_fadeout_tolerance).all(), (
        f"The Hermite function of order {n} with {alpha=} and {x_center=} "
        f"does not fade out at the approximated fadeout point {x_fadeouts:.10f}."
    )


@pytest.mark.parametrize("x_center", TEST_X_CENTERS_MU)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
@pytest.mark.parametrize(
    "n, num_extrema",
    [
        (0, 1),  # special case of only 1 extremum
        (1, 2),  # small odd
        (2, 2),  # small even
        (3, 2),  # small odd
        (4, 2),  # small even
        (5, 2),  # small odd
        (10, 2),  # small even
        (11, 2),  # small odd
        (999, 2),  # medium odd
        (1_000, 2),  # medium even
        (9_999, 2),  # large odd
        (10_000, 2),  # large even
        (99_999, 2),  # very large odd
        (100_000, 2),  # very large even
    ],
)
def test_hermite_funcs_largest_extrema_approximation(
    n: int,
    num_extrema: int,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test checks the approximation of the largest extrema of the Hermite functions
    via :func:`approximate_hermite_funcs_largest_extrema_x`.

    It does so by spanning an interval around the estimated largest extrema and checking
    that ``scipy.optimize.minimize_scalar`` does not find a minimum or maximum within
    this interval that is stronger than the estimated extremum (with a small tolerance).

    """

    # the largest extrema are estimated
    x_largest_extrema = approximate_hermite_funcs_largest_extrema_x(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # before proceeding, it is checked that the correct number of largest extrema is
    # returned
    assert x_largest_extrema.size == num_extrema, (
        f"Expected {num_extrema=} largest extrema for {n=}, {alpha=}, {x_center=}, "
        f"but got {x_largest_extrema.size}."
    )

    # the Hermite function is evaluated at the estimated largest extrema because they
    # will be compared to the results of the numerical minimisation
    hermite_extrema_values = single_hermite_function(
        x=x_largest_extrema,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # then, the largest extrema are checked
    for x_lge, herm_lge in zip(x_largest_extrema, hermite_extrema_values):
        # the interval around the extremum is spanned
        x_lower_bound = x_lge - LARGEST_EXTREMUM_NUMMINIM_X_WIDTH / alpha
        x_upper_bound = x_lge + LARGEST_EXTREMUM_NUMMINIM_X_WIDTH / alpha

        # the objective function for the extremum is minimised
        reference_result = minimize_scalar(
            _hermite_func_largest_extremum_objective,
            bounds=(x_lower_bound, x_upper_bound),
            args=(n, alpha, x_center),
            method="bounded",
            options=dict(
                xatol=LARGEST_EXTREMUM_NUMMINIM_X_ATOL / alpha,
                maxiter=LARGEST_EXTREMUM_NUMMINIM_MAX_ITER,
            ),
        )

        # the resulting extremum values is checked against the Hermite function
        # values at the estimated extremum (with the numerical minimisation as reference
        # for the tolerance)
        tolerance = LARGEST_EXTREMUM_Y_RTOL * np.abs(reference_result.fun)
        extremum_y_difference = abs(herm_lge) - abs(reference_result.fun)
        assert extremum_y_difference <= tolerance, (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"does not have the largest extremum at the approximated position "
            f"{x_lge:.10f}, but at {reference_result.x:.10f}."
        )
