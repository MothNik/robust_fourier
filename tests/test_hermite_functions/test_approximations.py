"""
This test suite implements the tests for the module :mod:`hermite_functions._approximations`.

"""  # noqa: E501

# === Imports ===

from math import sqrt as pysqrt
from typing import Optional

import numpy as np
import pytest

from robust_hermite_ft.hermite_functions import (
    approximate_hermite_funcs_fadeout_x,
    approximate_hermite_funcs_largest_extrema_x,
    approximate_hermite_funcs_largest_zeros_x,
    single_hermite_function,
)

# === Constants ===

# the absolute x-tolerance for testing the largest zero
LARGEST_ZERO_X_ATOL = 1e-7

# the y-tolerance for the fadeout points as a multiple of the machine epsilon
# NOTE: to avoid numerical issues, the tolerance is slightly increased
FADEOUT_Y_TOL_EPS_MULTIPLIER = 10.0

# the absolute x-tolerance for testing the largest extrema
LARGEST_EXTREMUM_X_ATOL = 1e-3
# the number of points for the extremum test
NUM_EXTREMUM_POINTS = 1_001
# the relative y-tolerance for the largest extrema
# NOTE: this is slightly worse than when the approximations were created
LARGEST_EXTREMUM_Y_RTOL = 1e-12

# the scales alpha to test
TEST_SCALES_ALPHA = [0.05, 0.5, 1.0, 2.0, 20.0]

# the centers mu to test
TEST_X_CENTERS_MU = [-10.0, 0.0, None, 10.0]

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
    for x_lgz in x_largest_zeros:
        x_zero_reference = np.array(
            [
                x_lgz - LARGEST_ZERO_X_ATOL / alpha,
                x_lgz + LARGEST_ZERO_X_ATOL / alpha,
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
            f"does not change its sign at the approximated largest zero {x_lgz:.5f}."
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
        f"does not fade out at the approximated fadeout points."
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

    It does so by spanning an interval in the close neighborhood of the estimated
    largest extrema and checking that no other point in this interval has a larger
    absolute value (with a certain tolerance) while some points have a smaller absolute
    value.

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

    # then, the absolute values of the Hermite functions are checked around the
    # estimated largest extrema
    for x_lge in x_largest_extrema:
        # the Hermite function is evaluated at the estimated largest extremum ...
        x_extremum_reference = np.linspace(
            start=x_lge - LARGEST_EXTREMUM_X_ATOL / alpha,
            stop=x_lge + LARGEST_EXTREMUM_X_ATOL / alpha,
            num=NUM_EXTREMUM_POINTS,
        )

        # ... and within the neighborhood of the extremum
        largest_extremum_absolute_value = np.abs(
            single_hermite_function(
                x=x_lge,
                n=n,
                alpha=alpha,
                x_center=x_center,
            )
        )[0]
        hermite_around_extremum_absolute_values = np.abs(
            single_hermite_function(
                x=x_extremum_reference,
                n=n,
                alpha=alpha,
                x_center=x_center,
            )
        )

        # the points are checked for the absence of larger absolute values ...
        # NOTE: for this the tolerance has to be scaled with the maximum absolute value
        #       value at the extremum to account for the scaling of the Hermite
        #       functions
        y_tolerance = LARGEST_EXTREMUM_Y_RTOL * largest_extremum_absolute_value
        any_higher_values_present = np.any(
            hermite_around_extremum_absolute_values
            > largest_extremum_absolute_value + y_tolerance
        )
        assert not any_higher_values_present, (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"has larger absolute values than the largest extremum {x_lge:.5f}."
        )

        # ... and the presence of smaller absolute values
        any_lower_values_present = np.any(
            hermite_around_extremum_absolute_values
            < largest_extremum_absolute_value - y_tolerance
        )
        assert any_lower_values_present, (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"does not have smaller absolute values than the largest extremum "
            f"{x_lge:.5f}."
        )
