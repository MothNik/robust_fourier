"""
Module :mod:`hermite_functions._numpy_funcs`

This module provides NumPy-based implementations of the Hermite functions.

"""

# === Imports ===

from math import ceil as py_ceil
from math import sqrt as py_sqrt

import numpy as np
from numpy import abs as np_abs
from numpy import exp, log, sqrt, square
from scipy.integrate import trapezoid

# === Functions ===


def _hermite_function_basis(
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Evaluates the complete basis of Hermite functions that are given by the product of a
    scaled Gaussian with a Hermite polynomial and can be written as follows:

    .. image:: docs/hermite_functions/equations/HF-05-Hermite_Functions_Basic_Definition.svg

    .. image:: docs/hermite_functions/equations/HF-06-Hermite_Polynomials_Basic_Definition.svg

    Please refer to the Notes section for further details.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (m,)
        The points at which the Hermite functions are evaluated.
    n : :class:`int`
        The order of the Hermite functions.

    Returns
    -------
    hermite_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the Hermite functions.

    References
    ----------
    The implementation is an adaption of the Appendix in [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    Notes
    -----
    Direct evaluation of the Hermite polynomials becomes ill-conditioned in finite
    precision since it involves the division of a polynomial by the square root of
    a factorial of ``n`` which is prone to overflow.
    To avoid this, the recurrence relation of the Hermite functions is used to compute
    the values in a numerically stable way.
    However, there will still be underflow of the Gaussian part of the Hermite function
    for large values of ``x``. Therefore, a special scaling strategy is employed that
    keeps the values of the Hermite functions at a scale of 1 during the recursion while
    tracking a correction term that is added to the exponent of the Gaussian part of the
    Hermite function.

    """  # noqa: E501

    # the recurrence relation here is started from the virtual -1-th order Hermite
    # function which is defined as h_{-1} = 0
    # this is done to make the recursion easier and to avoid the need for handling
    # too many special cases
    h_i_minus_1 = np.zeros_like(x)

    # a result Array for the results is initialised
    hermite_functions = np.empty(shape=(x.size, n + 1))

    # the 0-th order Hermite function is defined as
    # h_{0} = pi ** (-1/4) * exp(-x ** 2 / 2)
    # NOTE: here it is kept as h_{0} = exp(phi) * 1 where phi is the exponent
    #       "correction" term phi = (ln(pi) / 4) - 0.5 * (x ** 2)
    log_fourth_root_of_pi = -0.28618247146235  # ln(pi) / 4
    h_i = np.ones_like(x)
    exponent_corrections = log_fourth_root_of_pi - 0.5 * square(x)

    hermite_functions[::, 0] = exp(exponent_corrections)

    # if only the 0-th order is requested, the function can exit early here
    if n < 1:
        return hermite_functions

    # if higher orders are requested, a recursion is entered to compute the remaining
    # Hermite functions
    # the recursion is given by
    # h_{i+1} = sqrt(2 / (i + 1)) * x * h_{i} - sqrt(i / (i + 1)) * h_{i-1}
    # this is done in a numerically stable way by keeping the Hermite functions at a
    # scale of one and keeping track of the updated correction factors phi

    # the pre-factors for h_{i} and h_{i-1} are pre-computed
    # the pre-factor for h_{i} is sqrt(2 / (i + 1)) * x (here without the x-part)
    iterators = np.arange(0, n, 1, np.int64)
    prefactors_i = sqrt(2.0 / (iterators + 1.0))
    # the pre-factor for h_{i-1} is sqrt(i / (i + 1))
    prefactors_i_minus_1 = sqrt(iterators / (iterators + 1.0))

    for iter_i in iterators:
        # the new Hermite function is computed ...
        h_i_plus_1 = (
            prefactors_i[iter_i] * x * h_i - prefactors_i_minus_1[iter_i] * h_i_minus_1
        )
        # ... and stored after the correction factor is applied
        hermite_functions[::, iter_i + 1] = exp(exponent_corrections) * h_i_plus_1

        # afterwards, the correction factors are updated
        # NOTE: special care must be taken for values that are zero to avoid division by
        #       zero; they will not be updated
        scale_factors = np.where(h_i_plus_1 != 0.0, np_abs(h_i_plus_1), 1.0)
        h_i_minus_1 = h_i / scale_factors
        # NOTE: theoretically, h_{i+1} would be divided by its absolute value here, but
        #       a / |a| = sign(a) so the expensive division can be stated as a sign
        #       evaluation; here, everything relies on a sign definition that gives 0
        #       for a value of 0 and not +1 or -1 and ``np.sign`` meets this requirement
        h_i = np.sign(h_i_plus_1)
        exponent_corrections += log(scale_factors)

    # finally, the Hermite functions are returned
    return hermite_functions


def _single_hermite_function(
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Computes a single Hermite function of order ``n`` at the given points ``x`` by
    making use of a complex integral which is way faster than the recursion if only
    a specific Hermite function is needed.

    The Hermite functions are defined as

    .. image:: docs/hermite_functions/equations/HF-05-Hermite_Functions_Basic_Definition.svg

    .. image:: docs/hermite_functions/equations/HF-06-Hermite_Polynomials_Basic_Definition.svg

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (m,)
        The points at which the Hermite functions are evaluated.
    n : :class:`int`
        The order of the Hermite functions.

    Returns
    -------
    hermite_function : :class:`numpy.ndarray` of shape (m,)
        The values of the ``n``-th order Hermite function.

    References
    ----------
    The implementation is based on [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """  # noqa: E501

    # for the special case of n = 0, the Hermite function is just the Gaussian
    if n == 0:
        log_fourth_root_of_pi = -0.28618247146235  # ln(pi) / 4
        return np.exp(log_fourth_root_of_pi - 0.5 * np.square(x))

    # the integration of the circular integral from -pi to pi is set up
    # the number of required data points to evaluate this integral properly is
    # determined by the order as ``num_integ_points = ceil(12.42 * sqrt(n))``
    # NOTE: here, a bit more than half of this number is chosen because the function
    #       is symmetric around the y-axis
    # NOTE: the second formula comes from a custom regression with the aim to keep the
    #       error bounded to the machine precision for all the small orders where the
    #       approximation made in the publication is invalid; it will naturally fade
    #       out for larger orders where the approximation is valid; the turnover point
    #       is around n = 150
    sqrt_n = py_sqrt(n)
    num_integ_points = max(
        py_ceil(6.5 * sqrt_n),
        py_ceil(-0.173858887140972 * n + 8.03260560462478 * sqrt_n + 6.85116548584779),
    )
    # the points are evenly distributed over the positive half of the circle
    # NOTE: we will exploit mirroring later
    integ_points = np.linspace(0.0, np.pi, num_integ_points)
    delta_angle = np.pi / (num_integ_points - 1)

    # then, a correction factor gamma needs to be evaluated from a value k which is
    # sqrt(n / 2)
    k_value = 0.7071067811865476 * sqrt_n
    log_n_factorial = np.sum(np.log(np.arange(1, max(2, n + 1), 1, np.int64)))
    log_gamma = (
        0.5 * log_n_factorial
        - 0.5 * n * np.log(2.0)
        - n * np.log(k_value)
        - 0.25 * np.log(np.pi)
    )

    # now, all the values of the integrand are evaluated by making use of potentially
    # reduced computations that consider only some angles where the integrand exceeds
    # the machine precision
    log_eps = np.log(np.finfo(x.dtype).eps)
    one_over_sqrt_two_n = 1.0 / np.sqrt(2 * n)
    k_value_squared = k_value * k_value
    # for computing the interval bounds, the intersection of a quadratic cosine
    # polynomial with the logarithm of the machine precision are required
    # the constant part of the cosine value of this intersection value is pre-computed
    bound_value_part_two = (
        np.sqrt(n + 2.0 * (log_gamma - log_eps)) * one_over_sqrt_two_n
    )
    hermite_function = np.zeros_like(x)

    # each input point is evaluated separately
    # NOTE: this is probably only really fast for Numba-compiled code
    for iter_i, x_value in enumerate(x):
        # the x-value is made positive
        # NOTE: axis/point symmetry of the Hermite functions is exploited here, i.e.,
        #       for even orders, the function is symmetric around the y-axis and for
        #       odd orders, the function is anti-symmetric around the y-axis
        x_value_internal = abs(x_value)
        # the variable part of the bound cosine values is computed ...
        bound_value_part_one = x_value_internal * one_over_sqrt_two_n
        # ... followed by the values that the cosine has to yield for the intersections
        bound_value_left = bound_value_part_one - bound_value_part_two
        bound_value_right = bound_value_part_one + bound_value_part_two

        # if the left cosine value is within the bounds of -1 and 1, its arccos is
        # computed
        # otherwise, the bound angle is either pi (for < -1.0) or 0.0 (for > 1.0)
        if -1.0 <= bound_value_left <= 1.0:
            bound_value_left = np.arccos(bound_value_left)
        elif bound_value_left < -1.0:
            bound_value_left = np.pi
        else:
            bound_value_left = 0.0

        # the right bound is computed in almost the same way except for the fact that
        # the right cosine value cannot drop below 0.0 due to the absolute x-value
        # NOTE: the right bound value cannot drop below 0.0
        if bound_value_right <= 1.0:
            bound_value_right = np.arccos(bound_value_right)
        else:
            bound_value_right = 0.0

        # if the interval is degenerate, the integral is zero
        if bound_value_left == 0.0 and bound_value_right == 0.0:
            continue

        # now, the 2 sub-intervals for the integral are evaluated
        # the go from [bound_value_left, bound_value_right] for the negative angles and
        # [bound_value_right, bound_value_left] for the positive angles
        # NOTE: here, only the positive angles are considered because the target
        #       function's real part can be proven to be symmetric around the y-axis
        sub_integ_points = integ_points[
            np.where(
                np.logical_and(
                    integ_points >= bound_value_right,
                    integ_points <= bound_value_left,
                )
            )
        ]

        integrand_values_constant_log = log_gamma - 0.5 * np.square(x_value_internal)
        integrand_values_x_prefactor = 2 * x_value_internal * k_value
        # NOTE: the prefactor 2 exploits the symmetry of the target function, so only
        #       1 side has to be integrated
        hermite_function[iter_i] = 2.0 * trapezoid(
            y=np.exp(
                -1.0j * n * sub_integ_points
                + integrand_values_x_prefactor * np.exp(1.0j * sub_integ_points)
                - k_value_squared * np.exp(2.0j * sub_integ_points)
                + integrand_values_constant_log
            ).real,
            dx=delta_angle,
        )

        # finally, the symmetry of the Hermite functions is exploited
        # for even orders, the function is symmetric around the y-axis, so nothing has
        # to be done
        # for odd orders, the function is anti-symmetric around the y-axis, so the sign
        # has to be flipped
        if x_value < 0.0 and n % 2 == 1:
            hermite_function[iter_i] *= -1.0

    # finally, the hermite functions are properly scaled and returned
    return hermite_function / (2.0 * np.pi)
