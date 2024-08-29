"""
Module :mod:`hermite_functions._approximations`

This module offers approximation functions for special points of the Hermite functions,
namely

- the x-position of their largest zero (= outermost root where y = 0)
- the x-position at which the outermost tail fades below machine precision
- the x- and y-position of the maximum of the Hermite functions in their outermost
    oscillation

"""

# === Imports ===

from math import log as pylog
from math import sqrt as pysqrt
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import splev

from .._utils import (
    IntScalar,
    RealScalar,
    get_validated_alpha,
    get_validated_grid_points,
    get_validated_offset_along_axis,
    get_validated_order,
)
from ._hermite_largest_extrema_spline import (
    HERMITE_LARGEST_EXTREMA_MAX_ORDER,
    X_HERMITE_LARGEST_EXTREMA_SPLINE_TCK,
    Y_HERMITE_LARGEST_EXTREMA_SPLINE_TCK,
)
from ._hermite_largest_roots_spline import (
    HERMITE_LARGEST_ZEROS_MAX_ORDER,
    HERMITE_LARGEST_ZEROS_SPLINE_TCK,
)
from ._hermite_tail_gauss_sigma_spline import (
    HERMITE_TAIL_GAUSS_APPROX_MAX_ORDER,
    HERMITE_TAIL_GAUSS_SIGMA_SPLINE_TCK,
)

# === Constants ===

# the logarithm of the machine precision for float64
_LOG_DOUBLE_EPS = pylog(np.finfo(np.float64).eps)


# === Models ===


# a class that represents a squared exponential peak that can be evaluated at a given
# x-position


class TailSquaredExponentialApprox:
    """
    A class that represents a squared exponential or scaled Gaussian peak to approximate
    the outermost tail of the Hermite functions. It can be evaluated at a given
    x-position and also solve for the x-position at which the peak reaches a given
    y-value.

    Parameters
    ----------
    center_mu : :class:`float` or :class:`int`
        The center at which the y-value of the peak is ``amplitude``.
    stddev_sigma : :class:`float` or :class:`int`
        The standard deviation of the peak.
    amplitude : :class:`float` or :class:`int`
        The amplitude of the peak that is reached at the ``center_mu``.
    side: ``{"left", "right"}``
        The side of the Hermite function for which the peak is approximated.
        The fadeout point of the Hermite functions will be located at this side from
        the ``center_mu``.

    Attributes
    ----------
    center_mu, stddev_sigma, amplitude : :class:`float`
        The center, standard deviation, and amplitude of the peak.
    side : ``{"left", "right"}``
        The side of the Hermite function for which the peak is approximated.

    Methods
    -------
    __call__(x)
        Evaluates the peak at the given x-position.
    solve_for_y_fraction(y_fraction)
        Solves for the x-position at which the peak reaches the given y-value as a
        fraction of the amplitude.

    """

    def __init__(
        self,
        center_mu: RealScalar,
        stddev_sigma: RealScalar,
        amplitude: RealScalar,
        side: Literal["left", "right"],
    ) -> None:
        self.center_mu: RealScalar = center_mu
        self.stddev_sigma: RealScalar = stddev_sigma
        self.amplitude: RealScalar = amplitude
        self.side: Literal["left", "right"] = side

    def __call__(
        self,
        x: Union[RealScalar, ArrayLike],
    ) -> Union[float, np.ndarray]:
        """
        Evaluates the peak at the given x-position.

        Parameters
        ----------
        x : :class:`float` or :class:`int` or Array-like of shape (m,)
            The points at which the peak is evaluated.
            Internally, it will be promoted to ``np.float64``.
            It has to contain at least one element.

        Returns
        -------
        peak_values : :class:`float` or :class:`numpy.ndarray` of shape (m,)
            The y-values of the peak at the given x-positions.
            For scalar ``x``, the result is a also a scalar while for array-like ``x``,
            the result is an Array as well.

        Raises
        ------
        TypeError
            If ``x`` is not of the expected type.
        ValueError
            If ``x`` is not 1-dimensional after conversion to a NumPy array.

        """
        # the x-values are validated
        x_internal = get_validated_grid_points(grid_points=x, dtype=np.float64)

        # the peak is evaluated at the given x-values
        peak_values = self.amplitude * np.exp(
            np.negative(
                0.5 * np.square((x_internal - self.center_mu) / self.stddev_sigma)
            )
        )

        if np.isscalar(x):
            return float(peak_values[0])

        return peak_values

    def solve_for_y_fraction(
        self,
        y_fraction: Union[RealScalar, ArrayLike],
    ) -> Union[float, np.ndarray]:
        """
        Solves for the x-position at which the peak reaches the given y-value as a
        fraction of the amplitude.
        For a ``"left"`` side peak, the x-position will be the leftmost of both possible
        solutions while for a ``"right"`` side peak, it will be the rightmost.

        Parameters
        ----------
        y_fraction : :class:`float` or :class:`int` or Array-like of shape (m,)
            The y-values for which the x-positions are solved as a fraction of the
            amplitude.
            Internally, it will be promoted to ``np.float64``.
            It has to contain at least one element.

        Returns
        -------
        x_positions : :class:`float` or :class:`numpy.ndarray` of shape (m,)
            The x-positions at which the peak reaches the fraction of the amplitude.
            It will be the left or right solution for the ``"left"`` or ``"right"``
            side, respectively.
            For scalar ``y``, the result is a also a scalar while for array-like ``y``,
            the result is an Array as well.
            It will contain ``NaN``-values for y-values that are not in the range of the
            peak.

        Raises
        ------
        TypeError
            If ``y`` is not of the expected type.
        ValueError
            If ``y`` is not 1-dimensional after conversion to a NumPy array.

        """

        # the y-values are validated
        y_internal = get_validated_grid_points(grid_points=y_fraction, dtype=np.float64)

        # the x-positions are solved for the given y-values
        sign = -1.0 if self.side == "left" else 1.0
        x_positions = self.center_mu + sign * self.stddev_sigma * np.sqrt(
            -2.0 * np.log(y_internal)
        )

        if np.isscalar(y_fraction):
            return float(x_positions[0])

        return x_positions


# === Auxiliary Functions ===


def _apply_centering_and_scaling(
    values: np.ndarray,
    alpha: float,
    x_center: float,
) -> np.ndarray:
    """
    Applies centering and scaling to the given values.

    """

    # the scaling is applied (if required)
    if alpha != 1.0:
        values *= alpha

    # the centering is applied (if required)
    if x_center != 0.0:
        values += x_center

    return values


# === Functions ===


def x_largest_zeros(
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
) -> np.ndarray:
    """
    Approximates the x-position of the largest zero (= outermost root) of the Hermite
    functions for a given order and scaling factor.
    Please refer to the Notes for further details on the approximation.

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite functions.
        It must be a non-negative integer ``>= 0`` and less than or equal to the maximum
        order for the spline interpolation (roughly 100 000).
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.

    Returns
    -------
    x_largest_zeros : :class:`numpy.ndarray` of shape (0,) or (1,) or (2,)
        The x-positions of the left and right largest zeros of the Hermite functions
        (in that order) for the given order, scaling factor, and center.
        It will hold two entries except for

        - order 0 where there is no zero and
        - order 1 where the zero is exactly at ``x_center``.

    Raises
    ------
    NotImplementedError
        If the order is larger than the maximum order for the spline interpolation.

    Notes
    -----
    The approximation is based on a spline that mimics the results of
    ``scipy.special.roots_hermite``, but this function gives roots whose y-values are in
    the order of ``1e-8``. Thus, the reference values are the limiting factor for
    accuracy and not the spline itself.
    Given that it's based on a spline, the approximation is limited to a maximum order
    of roughly 100 000, but this should be sufficient for most applications.

    """

    # --- Input Validation ---

    n = get_validated_order(n=n)
    alpha = get_validated_alpha(alpha=alpha)
    x_center = get_validated_offset_along_axis(offset=x_center, which_axis="x")

    # --- Computation ---

    # if the order exceeds the maximum order for the spline interpolation, an error is
    # raised
    if n > HERMITE_LARGEST_ZEROS_MAX_ORDER:  # pragma: no cover
        raise NotImplementedError(
            f"Order {n} exceeds the maximum order {HERMITE_LARGEST_ZEROS_MAX_ORDER} "
            f"for the spline interpolation for the largest zeros."
        )

    # if the order exceeds 1, the spline interpolation is used to approximate the
    # largest zeros
    if n > 1:
        x_largest_zero_positive = splev(
            x=n,
            tck=HERMITE_LARGEST_ZEROS_SPLINE_TCK,
        )
        x_largest_zeros = np.array(
            [
                -x_largest_zero_positive,  # type: ignore
                x_largest_zero_positive,
            ]
        )

    # for order 1, the zero is exactly at 0
    elif n == 1:
        x_largest_zeros = np.array([0.0])

    # for order 0, there is no zero
    else:
        x_largest_zeros = np.array([])

    # the centering and scaling are applied
    return _apply_centering_and_scaling(
        values=x_largest_zeros,
        alpha=alpha,
        x_center=x_center,
    )


# TODO: make all output equally sized independent of the order
def x_largest_extrema(
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
) -> np.ndarray:
    """
    Approximates the x-position of the maximum of the Hermite functions in their
    outermost oscillation for a given order and scaling factor.
    Please refer to the Notes for further details on the approximation.

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite functions.
        It must be a non-negative integer ``>= 0`` and less than or equal to the maximum
        order for the spline interpolation (roughly 100 000).
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.

    Returns
    -------
    x_largest_extrema : :class:`numpy.ndarray` of shape (1,) or (2,)
        The x-positions of the maximum of the Hermite functions in their outermost
        oscillation (in that order) for the given order, scaling factor, and center.
        It will hold two entries except for

        - order 0 where the maximum is exactly at ``x_center``.

    Raises
    ------
    NotImplementedError
        If the order is larger than the maximum order for the spline interpolation.

    Notes
    -----
    The approximation is based on a spline interpolation of the results of a numerical
    optimisation to find the maximum of the Hermite functions.
    Given that it's based on a spline, the approximation is limited to a maximum order
    of roughly 100 000, but this should be sufficient for most applications.

    """

    # --- Input Validation ---

    n = get_validated_order(n=n)
    alpha = get_validated_alpha(alpha=alpha)
    x_center = get_validated_offset_along_axis(offset=x_center, which_axis="x")

    # --- Computation ---

    # if the order exceeds the maximum order for the spline interpolation, an error is
    # raised
    if n > HERMITE_LARGEST_EXTREMA_MAX_ORDER:  # pragma: no cover
        raise NotImplementedError(
            f"Order {n} exceeds the maximum order {HERMITE_LARGEST_EXTREMA_MAX_ORDER} "
            f"for the spline interpolation for the largest extrema."
        )

    # if the order is not 0, the spline interpolation is used to approximate the
    # largest extrema
    if n > 0:
        x_largest_extrema_positive = splev(
            x=n,
            tck=X_HERMITE_LARGEST_EXTREMA_SPLINE_TCK,
        )
        x_largest_extrema = np.array(
            [
                -x_largest_extrema_positive,  # type: ignore
                x_largest_extrema_positive,
            ]
        )

    # for order 0, the maximum is exactly at the center
    else:
        x_largest_extrema = np.array([0.0])

    # the centering and scaling are applied
    return _apply_centering_and_scaling(
        values=x_largest_extrema,
        alpha=alpha,
        x_center=x_center,
    )


# TODO: make all output equally sized independent of the order
def y_largest_extrema(
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
) -> np.ndarray:
    """
    Approximates the y-position of the maximum of the Hermite functions in their
    outermost oscillation for a given order and scaling factor.
    Please refer to the Notes for further details on the approximation.

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite functions.
        It must be a non-negative integer ``>= 0`` and less than or equal to the maximum
        order for the spline interpolation (roughly 100 000).
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.
        This value has no effect on the result.

    Returns
    -------
    y_largest_extrema : :class:`numpy.ndarray` of shape (1,) or (2,)
        The y-positions of the maximum of the Hermite functions in their outermost
        oscillation (in that order) for the given order, scaling factor, and center.
        It will hold two entries except for

        - order 0 where the maximum is exactly at
            :math:`\\frac{1}{\\sqrt[4]{\\pi\\cdot\\alpha^{2}}}`

    Raises
    ------
    NotImplementedError
        If the order is larger than the maximum order for the spline interpolation.

    Notes
    -----
    The approximation is based on a spline interpolation of the results of a the
    evaluation of the Hermite functions at the x-position provided by
    :func:`x_largest_extrema`.

    """

    # --- Input Validation ---

    n = get_validated_order(n=n)
    alpha = get_validated_alpha(alpha=alpha)

    # --- Computation ---

    # if the order exceeds the maximum order for the spline interpolation, an error is
    # raised
    if n > HERMITE_LARGEST_EXTREMA_MAX_ORDER:  # pragma: no cover
        raise NotImplementedError(
            f"Order {n} exceeds the maximum order {HERMITE_LARGEST_EXTREMA_MAX_ORDER} "
            f"for the spline interpolation for the largest extrema."
        )

    # if the order is exactly zero, the maximum is returned directly
    if n == 0:
        return np.array([1.0 / pysqrt(pysqrt(np.pi * alpha * alpha))])

    # otherwise, the spline is evaluated to get the y-position of the maximum
    y_largest_extrema_positive = splev(
        x=n,
        tck=Y_HERMITE_LARGEST_EXTREMA_SPLINE_TCK,
    )
    # it still needs to be scaled by the square root of alpha
    y_largest_extrema_positive /= pysqrt(alpha)  # type: ignore

    # for even orders, both extrema are the same and positive
    if n % 2 == 0:
        return np.array([y_largest_extrema_positive, y_largest_extrema_positive])

    # for odd orders, the extrema are the same, but the left one is positive while the
    # right one is negative
    return np.array(
        [
            -y_largest_extrema_positive,
            y_largest_extrema_positive,
        ]
    )


# TODO: make all output equally sized independent of the order
def x_and_y_largest_extrema(
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximates the x- and y-positions of the maximum of the Hermite functions in their
    outermost oscillation for a given order and scaling factor.
    It is a convenience function that combines the results of :func:`x_largest_extrema`
    and :func:`y_largest_extrema`.

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite functions.
        It must be a non-negative integer ``>= 0`` and less than or equal to the maximum
        order for the spline interpolation (roughly 100 000).
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.

    Returns
    -------
    x_largest_extrema, y_largest_extrema : :class:`numpy.ndarray` of shape (1,) or (2,)
        The x- and y-positions of the maximum of the Hermite functions in their outermost
        oscillation (in that order) for the given order, scaling factor, and center.
        They will hold two entries each except for

        - order 0 where the maximum is exactly at the x-position ``x_center`` and the
            y-position is :math:`\\frac{1}{\\sqrt[4]{\\pi\\cdot\\alpha^{2}}}`

    Raises
    ------
    NotImplementedError
        If the order is larger than the maximum order for the spline interpolations.

    """  # noqa: E501

    return (
        x_largest_extrema(
            n=n,
            alpha=alpha,
            x_center=x_center,
        ),
        y_largest_extrema(
            n=n,
            alpha=alpha,
            x_center=x_center,
        ),
    )


def x_fadeout(
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
) -> np.ndarray:
    """
    Approximates the x-position at which the outermost tail of the dilated Hermite
    functions drops below machine precision when compared to the maximum value for
    ``float64``.
    Please refer to the Notes for further details on the approximation.

    Parameters
    ----------
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.

    Returns
    -------
    x_fadeouts : :class:`numpy.ndarray` of shape (2,)
        The x-positions of the left and right fadeouts of the outermost tail of the
        Hermite functions (in that order) for the given order, scaling factor, and
        center.

    Notes
    -----
    At the fadeout point, there is no point in a complex integral representation of the
    Hermite functions that adds any numerically non-zero contribution. This
    approximation might be a bit conservative, but it provides consistent fadeout
    points.

    References
    ----------
    The implementation is derived from [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """

    # --- Input Validation ---

    n = get_validated_order(n=n)
    alpha = get_validated_alpha(alpha=alpha)
    x_center = get_validated_offset_along_axis(offset=x_center, which_axis="x")

    # --- Computation ---

    # for orders > 0, the gamma-value is computed which is required for the estimation
    if n > 0:
        sqrt_n = pysqrt(n)
        k_value = 0.7071067811865476 * sqrt_n  # k = sqrt(n / 2)
        log_n_factorial = np.log(  # log(n!)
            np.arange(
                start=1,
                stop=max(2, n + 1),
                step=1,
                dtype=np.int64,
            )
        ).sum()
        log_gamma = (
            0.5 * log_n_factorial
            - 0.5 * n * pylog(2.0)  # log(2 ** (n / 2))
            - n * pylog(k_value)  # log(k ** n)
            - 0.25 * pylog(np.pi)  # log(pi ** 0.25)
        )

        # then, the approximation is made by finding the point where the arccosine of
        # the argument of the complex integral representation of the Hermite functions
        # exceeds 1.0; beyond this point, the contribution of any part of the complex
        # integral representation is numerically zero
        x_fadeout_positive = pysqrt(2.0) * sqrt_n + pysqrt(
            n + 2.0 * (log_gamma - _LOG_DOUBLE_EPS - pylog(6.5 * sqrt_n))
        )

    # for order 0, the fadeout is taken as the point where the Gaussian term drops
    # below machine precision
    else:
        # the argument of the Gaussian term is -0.5 * x ** 2, and the fadeout is
        # computed by solving for the point where the exponent exceeds the logarithm of
        # the machine precision
        x_fadeout_positive = pysqrt(-2.0 * _LOG_DOUBLE_EPS)

    # finally, the fadeouts are stored as an Array
    x_fadeouts = np.array(
        [
            -x_fadeout_positive,  # type: ignore
            x_fadeout_positive,
        ]
    )

    # the centering and scaling are applied
    return _apply_centering_and_scaling(
        values=x_fadeouts,
        alpha=alpha,
        x_center=x_center,
    )


def tail_gauss_fit(
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
) -> Tuple[TailSquaredExponentialApprox, TailSquaredExponentialApprox]:
    """
    Approximates the outermost tail of the dilated Hermite functions with a squared
    exponential or scaled Gaussian peak.
    This is a very crude approximation that is only valid for the respective side that
    goes towards the fadeout point. One of its characteristics is that the Gaussian peak
    is too wide in most of the points to get conservative estimates for the tail
    behaviour.

    Parameters
    ----------
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.

    Returns
    -------
    left_tail, right_tail : :class:`TailSquaredExponentialApprox`
        The squared exponential or scaled Gaussian peaks that approximate the outermost
        tail of the Hermite functions on the left and right side of the maximum,
        respectively.
        They can be called directly to evaluate the peak at a given x-position or to
        solve for the x-position at which the peak reaches a given y-value.
        Please refer to the documentation of the class
        :class:`TailSquaredExponentialApprox` for further details.

    Raises
    ------
    NotImplementedError
        If the order is larger than the maximum order for the spline interpolations.

    """

    # if the order is larger than the maximum order for the spline interpolation of the
    # Gaussian standard deviation, an error is raised
    if n > HERMITE_TAIL_GAUSS_APPROX_MAX_ORDER:  # pragma: no cover
        raise NotImplementedError(
            f"Order {n} exceeds the maximum order "
            f"{HERMITE_TAIL_GAUSS_APPROX_MAX_ORDER} for the spline interpolation for "
            f"the tail Gaussian approximation."
        )

    # first, the xy-coordinates of the largest extrema are computed
    # NOTE: this also performs the input validation
    x_extrema, y_extrema = x_and_y_largest_extrema(
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # if the order is 0, the there would only be one fit that fits perfectly over the
    # whole curve; yet 2 peaks are required, so the extrema are simply repeated
    if n == 0:
        x_extrema = np.repeat(x_extrema, repeats=2)
        y_extrema = np.repeat(y_extrema, repeats=2)

    # afterwards, the standard deviation sigma is computed by evaluating the
    # pre-computed spline curve at the natural logarithm of the order
    # the spline is only valid for nonzero orders
    if n > 0:
        stddev_sigma = splev(
            x=pylog(n),
            tck=HERMITE_TAIL_GAUSS_SIGMA_SPLINE_TCK,
        )
        # for proper scaling, the standard deviation is multiplied with the alpha value
        stddev_sigma *= alpha  # type: ignore

    # for order zero, the standard deviation is simply the alpha value
    else:
        stddev_sigma = alpha

    # the peaks are created
    left_tail = TailSquaredExponentialApprox(
        center_mu=x_extrema[0],
        stddev_sigma=stddev_sigma,  # type: ignore
        amplitude=y_extrema[0],
        side="left",
    )
    right_tail = TailSquaredExponentialApprox(
        center_mu=x_extrema[1],
        stddev_sigma=stddev_sigma,  # type: ignore
        amplitude=y_extrema[1],
        side="right",
    )

    return left_tail, right_tail
