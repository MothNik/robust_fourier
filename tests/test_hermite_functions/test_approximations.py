"""
This test suite implements the tests for the module :mod:`hermite_functions._approximations`.

"""  # noqa: E501

# === Imports ===

from math import isclose
from math import sqrt as pysqrt
from typing import Optional

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from robust_fourier import hermite_approx, single_hermite_function

# === Constants ===

# the absolute and relative x-width for testing the largest zero
LARGEST_ZERO_TEST_X_ABS_WIDTH = 1e-9
LARGEST_ZERO_TEST_X_REL_WIDTH = 1e-9

# the y-tolerance for the fadeout points as a multiple of the machine epsilon
# NOTE: to avoid numerical issues, the tolerance is slightly increased
FADEOUT_TEST_Y_TOL_EPS_MULTIPLIER = 10.0

# the width of the x-interval spanned for testing the largest extrema via numerical
# minimisation
LARGEST_EXTREMUM_TEST_NUMMINIM_X_WIDTH = 1e-3
# the absolute tolerance of the numerical minimisation for the largest extrema
LARGEST_EXTREMUM_TEST_NUMMINIM_X_ATOL = 1e-13
# the maximum number of iterations for the numerical minimisation for the largest
# extrema
LARGEST_EXTREMUM_TEST_NUMMINIM_MAX_ITER = 100_000
# the relative y-tolerance for testing the largest extrema for the true evaluation ...
LARGEST_EXTREMUM_TEST_TRUE_EVAL_Y_RTOL = 1e-10
# ... and the spline approximation
LARGEST_EXTREMUM_TEST_SPLINE_APPROX_Y_RTOL = 1e-9

# the relative tolerance for the amplitude comparison of the Gaussian approximations
# of the outermost oscillation
GAUSSIAN_APPROX_TEST_AMPLITUDE_RTOL = 1e-15
# the percentages to test the Gaussian approximation of the outermost oscillation in %
GAUSSIAN_APPROX_TEST_PERCENTAGES = [1.0, 5.0, 10.0, 50.0]
# the relative y-tolerance for testing the Gaussian approximation of the outermost
# oscillation against the Hermite function
# NOTE: the tolerance is really crude because the Gaussian approximation was not meant
#       to be very accurate
GAUSSIAN_APPROX_TEST_AGAINST_HERMITE_Y_RTOL = 1e-3
# the relative y-tolerance for testing the Gaussian approximation solution against the
# evaluation of the Gaussian itself
# NOTE: this is fairly strict because the Gaussian should be consistent in its own
#       evaluation
GAUSSIAN_APPROX_TEST_AGAINST_GAUSS_Y_RTOL = 1e-11

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
    via :func:`hermite_approx.x_largest_zeros`.

    It does so by comparing checking if the Hermite functions change their sign at two
    points that are located around the estimated largest zero and only a small numerical
    tolerance apart.

    """

    # the largest zeros are estimated
    x_largest_zeros = hermite_approx.x_largest_zeros(
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
            alpha * LARGEST_ZERO_TEST_X_ABS_WIDTH,
            LARGEST_ZERO_TEST_X_REL_WIDTH
            * (abs(x_lgz) - abs(x_center_for_ref_tolerance)),
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

    return


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
    via :func:`hermite_approx.x_fadeout`.

    It does so by checking if the Hermite functions are numerically zero at the
    estimated fadeout points.

    """

    # the fadeout points are estimated
    x_fadeouts = hermite_approx.x_fadeout(
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
    # NOTE: different alpha-values require scaling the Hermite functions to preserve
    #       orthonormality, so this scaling (by the reciprocal square root of alpha) is
    #       taken into account in the tolerance
    y_fadeout_tolerance = FADEOUT_TEST_Y_TOL_EPS_MULTIPLIER * (
        np.finfo(np.float64).eps / pysqrt(alpha)
    )
    assert np.abs(hermite_fadeout_values <= y_fadeout_tolerance).all(), (
        f"The Hermite function of order {n} with {alpha=} and {x_center=} "
        f"does not fade out at the approximated fadeout point {x_fadeouts:.10f}."
    )

    return


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
    via :func:`hermite_approx.x_and_y_largest_extrema`.

    It does so by spanning an interval around the estimated largest extrema and checking
    that ``scipy.optimize.minimize_scalar`` does not find a minimum or maximum within
    this interval that is stronger than the estimated extremum (with a small tolerance).

    """

    # the largest extrema are estimated
    x_largest_extrema, y_largest_extrema = hermite_approx.x_and_y_largest_extrema(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # before proceeding, it is checked that the correct number of largest extrema is
    # returned
    assert x_largest_extrema.size == num_extrema, (
        f"Expected {num_extrema=} largest extrema for {n=}, {alpha=}, {x_center=}, "
        f"but got {x_largest_extrema.size} (for x)."
    )
    assert y_largest_extrema.size == num_extrema, (
        f"Expected {num_extrema=} largest extrema for {n=}, {alpha=}, {x_center=}, "
        f"but got {y_largest_extrema.size} (for y)."
    )

    # the Hermite function is evaluated at the estimated largest extrema because they
    # will be compared to the results of the numerical minimisation
    hermite_extrema_values = single_hermite_function(
        x=x_largest_extrema,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )
    # the sign is checked because it will not be considered anymore for the following
    # checks
    assert np.array_equal(
        np.sign(hermite_extrema_values),
        np.sign(y_largest_extrema),
    ), (
        f"The Hermite function of order {n} with {alpha=} and {x_center=} "
        f"have sign deviations between the estimated and the numerical largest extrema."
    )

    # then, the largest extrema are checked
    for x_lge, herm_lge in zip(x_largest_extrema, hermite_extrema_values):
        # the interval around the extremum is spanned
        x_lower_bound = x_lge - alpha * LARGEST_EXTREMUM_TEST_NUMMINIM_X_WIDTH
        x_upper_bound = x_lge + alpha * LARGEST_EXTREMUM_TEST_NUMMINIM_X_WIDTH

        # the objective function for the extremum is minimised
        reference_result = minimize_scalar(
            _hermite_func_largest_extremum_objective,
            bounds=(x_lower_bound, x_upper_bound),
            args=(n, alpha, x_center),
            method="bounded",
            options=dict(
                xatol=alpha * LARGEST_EXTREMUM_TEST_NUMMINIM_X_ATOL,
                maxiter=LARGEST_EXTREMUM_TEST_NUMMINIM_MAX_ITER,
            ),
        )

        # the resulting extremum values is checked against the Hermite function
        # values at the estimated extremum (with the numerical minimisation as reference
        # for the tolerance)
        tolerance = LARGEST_EXTREMUM_TEST_TRUE_EVAL_Y_RTOL * abs(reference_result.fun)
        extremum_y_difference = abs(herm_lge) - abs(reference_result.fun)
        assert abs(extremum_y_difference) <= tolerance, (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"does not have the largest extremum at the approximated position "
            f"{x_lge:.10f}, but at {reference_result.x:.10f}."
        )

        del tolerance, extremum_y_difference

        # the resulting extremum is also checked against the spline approximation
        tolerances = LARGEST_EXTREMUM_TEST_SPLINE_APPROX_Y_RTOL * abs(
            reference_result.fun
        )
        extremum_y_differences = np.abs(y_largest_extrema) - abs(reference_result.fun)
        assert (np.abs(extremum_y_differences) <= tolerances).all(), (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"spline approximation for the y-position of the largest extremum failed."
        )

    return


@pytest.mark.parametrize("x_center", TEST_X_CENTERS_MU)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
@pytest.mark.parametrize(
    "n",
    [
        0,  # special case of only 1 extremum
        1,  # small odd
        2,  # small even
        3,  # small odd
        4,  # small even
        5,  # small odd
        10,  # small even
        11,  # small odd
        999,  # medium odd
        1_000,  # medium even
        9_999,  # large odd
        10_000,  # large even
        99_999,  # very large odd
        100_000,  # very large even
    ],
)
def test_hermite_funcs_tail_gauss_approximation_scalar_fractions(
    n: int,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test checks the approximation of the Gaussian tail of the Hermite functions
    via :func:`hermite_approx.get_tail_gauss_fit` and :func:`x_tail_drop_to_fraction`.

    It does so by checking if the Hermite functions drop below a percentage of their
    respective maxima at the points estimated by the Gaussian approximation because they
    were meant to be a conservative estimate.

    The test is performed for scalar fractions because they require different handling
    of the results.
    An equivalent test for array fractions is implemented in the next test.

    """

    # the Gaussian approximations of the outermost oscillation are estimated
    left_gaussian, right_gaussian = hermite_approx.get_tail_gauss_fit(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # both amplitudes are checked for being equal
    # NOTE: since the left Gaussian can have a negative amplitude, the absolute value is
    #       taken for the comparison and compared with a relative tolerance to avoid
    #       floating point issues
    assert isclose(
        abs(left_gaussian.amplitude),  # type: ignore
        abs(right_gaussian.amplitude),  # type: ignore
        abs_tol=0.0,
        rel_tol=GAUSSIAN_APPROX_TEST_AMPLITUDE_RTOL,
    ), (
        f"The left and right Gaussian approximations of the Hermite function of order "
        f"{n} with {alpha=} and {x_center=} have different amplitudes."
    )

    # the largest extrema are estimated as well
    _, y_largest_extrema = hermite_approx.x_and_y_largest_extrema(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # they are compared to be equal to the amplitude values because they were derived
    # from them (again with a relative tolerance)
    assert np.allclose(
        np.array([left_gaussian.amplitude, right_gaussian.amplitude]),
        y_largest_extrema,
        atol=0.0,
        rtol=GAUSSIAN_APPROX_TEST_AMPLITUDE_RTOL,
    ), (
        f"The Gaussian approximations of the Hermite function of order {n} with "
        f"{alpha=} and {x_center=} have amplitudes that differ from the largest "
        f"extrema."
    )

    y_largest_extrema = np.abs(y_largest_extrema)

    # now, the approximations for all the percentages are tested INDIVIDUALLY
    for percentage in GAUSSIAN_APPROX_TEST_PERCENTAGES:
        # the Gaussian approximation is solved for the percentage level
        y_fraction = percentage / 100.0
        x_drop_to_percentage = hermite_approx.x_tail_drop_to_fraction(
            n=n,
            y_fraction=y_fraction,
            alpha=alpha,
            x_center=x_center,
        )

        # the Gaussians are evaluated at the drop points and compared against their
        # maximum to check if the Gaussian-internal solution is consistent
        left_gaussian_value = left_gaussian(x=x_drop_to_percentage[0, 0])
        right_gaussian_value = right_gaussian(x=x_drop_to_percentage[0, 1])

        assert isclose(
            left_gaussian_value,
            y_fraction * left_gaussian.amplitude,
            abs_tol=0.0,
            rel_tol=GAUSSIAN_APPROX_TEST_AGAINST_GAUSS_Y_RTOL,
        ), (
            f"The left Gaussian approximation of the Hermite function of order {n} "
            f"with {alpha=} and {x_center=} has inconsistent Gaussian solution at "
            f"{percentage}% level."
        )
        assert isclose(
            right_gaussian_value,
            y_fraction * right_gaussian.amplitude,
            abs_tol=0.0,
            rel_tol=GAUSSIAN_APPROX_TEST_AGAINST_GAUSS_Y_RTOL,
        ), (
            f"The right Gaussian approximation of the Hermite function of order {n} "
            f"with {alpha=} and {x_center=} has inconsistent Gaussian solution at "
            f"{percentage}% level."
        )

        # then, the solution is checked against the Hermite function evaluation
        hermite_values_drop_to_percentage = single_hermite_function(
            x=x_drop_to_percentage.ravel(),
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

        # first, the signs are checked because it will not be considered anymore for the
        # following checks
        assert np.array_equal(
            np.sign(hermite_values_drop_to_percentage),
            np.sign(np.array([left_gaussian_value, right_gaussian_value])),
        ), (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"have sign deviations between the estimated and the numerical drop points."
        )

        # finally, it is checked that the Hermite functions are below the percentage
        # level estimated by the Gaussian approximation because this was the purpose of
        # the approximation to be more conservative
        y_drop_to_percentage_target = (
            y_fraction + GAUSSIAN_APPROX_TEST_AGAINST_HERMITE_Y_RTOL
        ) * y_largest_extrema
        assert (
            np.abs(hermite_values_drop_to_percentage) <= y_drop_to_percentage_target
        ).all(), (
            f"The Hermite function of order {n} with {alpha=} and {x_center=} "
            f"does not drop below {percentage}% of its maximum at the approximated "
            f"drop points {x_drop_to_percentage}."
        )

    return


@pytest.mark.parametrize("x_center", TEST_X_CENTERS_MU)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
@pytest.mark.parametrize(
    "n",
    [
        0,  # special case of only 1 extremum
        1,  # small odd
        2,  # small even
        3,  # small odd
        4,  # small even
        5,  # small odd
        10,  # small even
        11,  # small odd
        999,  # medium odd
        1_000,  # medium even
        9_999,  # large odd
        10_000,  # large even
        99_999,  # very large odd
        100_000,  # very large even
    ],
)
def test_hermite_funcs_tail_gauss_approximation_array_fractions(
    n: int,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test is equivalent to the previous test
    :func:`test_hermite_funcs_tail_gauss_approximation_scalar_fractions`, but it is
    implemented for Array fractions.


    """

    # the Gaussian approximations of the outermost oscillation are estimated
    left_gaussian, right_gaussian = hermite_approx.get_tail_gauss_fit(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # the largest extrema are estimated as well
    _, y_largest_extrema = hermite_approx.x_and_y_largest_extrema(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # NOTE: no amplitude tests are performed because this happened in the scalar test
    #       already

    # afterwards, all the percentages are tested TOGETHER
    y_fractions = np.array(GAUSSIAN_APPROX_TEST_PERCENTAGES) / 100.0
    x_drop_to_percentage = hermite_approx.x_tail_drop_to_fraction(
        n=n,
        y_fraction=y_fractions,
        alpha=alpha,
        x_center=x_center,
    )

    # a shape check is carried out to ensure that the correct number of drop points is
    # returned
    assert x_drop_to_percentage.shape == (len(GAUSSIAN_APPROX_TEST_PERCENTAGES), 2), (
        f"Expected {len(GAUSSIAN_APPROX_TEST_PERCENTAGES)} drop points for {n=}, "
        f"{alpha=}, {x_center=}, but got {x_drop_to_percentage.shape}."
    )

    # the Gaussians are evaluated at the drop points and compared against their maximum
    # to check if the Gaussian-internal solution is consistent
    left_gaussian_values = left_gaussian(x=x_drop_to_percentage[::, 0])
    right_gaussian_values = right_gaussian(x=x_drop_to_percentage[::, 1])

    assert np.allclose(
        left_gaussian_values,
        y_fractions * left_gaussian.amplitude,
        atol=0.0,
        rtol=GAUSSIAN_APPROX_TEST_AGAINST_GAUSS_Y_RTOL,
    ), (
        f"The left Gaussian approximation of the Hermite function of order {n} with "
        f"{alpha=} and {x_center=} has inconsistent Gaussian solution at "
        f"{GAUSSIAN_APPROX_TEST_PERCENTAGES}% levels."
    )
    assert np.allclose(
        right_gaussian_values,
        y_fractions * right_gaussian.amplitude,
        atol=0.0,
        rtol=GAUSSIAN_APPROX_TEST_AGAINST_GAUSS_Y_RTOL,
    ), (
        f"The right Gaussian approximation of the Hermite function of order {n} with "
        f"{alpha=} and {x_center=} has inconsistent Gaussian solution at "
        f"{GAUSSIAN_APPROX_TEST_PERCENTAGES}% levels."
    )

    # then, the evaluation is checked against the Hermite function evaluation
    hermite_values_drop_to_percentage = single_hermite_function(
        x=x_drop_to_percentage.ravel(),
        n=n,
        alpha=alpha,
        x_center=x_center,
    ).reshape((-1, 2))

    # the sign is checked because it will not be considered anymore for the following
    # checks
    assert np.array_equal(
        np.sign(hermite_values_drop_to_percentage),
        np.sign(np.column_stack([left_gaussian_values, right_gaussian_values])),
    ), (
        f"The Hermite function of order {n} with {alpha=} and {x_center=} "
        f"have sign deviations between the estimated and the numerical drop points."
    )

    # finally, it is checked that the Hermite functions are below the percentage level
    # estimated by the Gaussian approximation because this was the purpose of the
    # approximation to be more conservative
    y_drop_to_percentage_targets = (
        y_fractions[::, np.newaxis] + GAUSSIAN_APPROX_TEST_AGAINST_HERMITE_Y_RTOL
    ) * np.abs(y_largest_extrema)[np.newaxis, ::]
    assert (
        np.abs(hermite_values_drop_to_percentage) <= y_drop_to_percentage_targets
    ).all(), (
        f"The Hermite function of order {n} with {alpha=} and {x_center=} "
        f"does not drop below {GAUSSIAN_APPROX_TEST_PERCENTAGES}% of its maximum at "
        f"the approximated drop points {x_drop_to_percentage}."
    )

    return
