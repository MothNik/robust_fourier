"""
Module :mod:`hermite_functions._approximations`

This module offers approximation functions for special points of the Hermite functions,
namely

- the x-position of their largest zero (= outermost root where y = 0)
- the x-position at which the outermost tail fades below machine precision
- the x-position of the maximum of the Hermite functions in their outermost
    oscillation

"""

# === Imports ===

from math import log as pylog
from math import sqrt as pysqrt
from typing import Optional, Union

import numpy as np
from scipy.interpolate import splev

from ._hermite_largest_extrema_spline import (
    HERMITE_LARGEST_EXTREMA_MAX_ORDER,
    HERMITE_LARGEST_EXTREMA_SPLINE_TCK,
)
from ._hermite_largest_roots_spline import (
    HERMITE_LARGEST_ZEROS_MAX_ORDER,
    HERMITE_LARGEST_ZEROS_SPLINE_TCK,
)
from ._validate import (
    _get_validated_alpha,
    _get_validated_order,
    _get_validated_x_center,
)

# === Constants ===

# the logarithm of the machine precision for float64
_LOG_DOUBLE_EPS = pylog(np.finfo(np.float64).eps)

# === Auxiliary Functions ===


def _apply_centering_and_scaling(
    values: np.ndarray,
    alpha: float,
    x_center: Optional[float],
) -> np.ndarray:
    """
    Applies centering and scaling to the given values.

    """

    # the scaling is applied (if required)
    if alpha != 1.0:
        values /= alpha

    # the centering is applied (if required)
    if x_center is not None and x_center != 0.0:
        values += x_center

    return values


# === Functions ===


def hermite_funcs_largest_zeros_x(
    n: int,
    alpha: Union[float, int] = 1.0,
    x_center: Union[float, int, None] = None,
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
        ``x_scaled = alpha * x``.
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

    n = _get_validated_order(n=n)
    alpha = _get_validated_alpha(alpha=alpha)
    x_center = _get_validated_x_center(x_center=x_center)

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


def hermite_funcs_largest_extrema_x(
    n: int,
    alpha: Union[float, int] = 1.0,
    x_center: Union[float, int, None] = None,
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
        ``x_scaled = alpha * x``.
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

    n = _get_validated_order(n=n)
    alpha = _get_validated_alpha(alpha=alpha)
    x_center = _get_validated_x_center(x_center=x_center)

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
            tck=HERMITE_LARGEST_EXTREMA_SPLINE_TCK,
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


def hermite_funcs_fadeout_x(
    n: int,
    alpha: Union[float, int] = 1.0,
    x_center: Union[float, int, None] = None,
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
        ``x_scaled = alpha * x``.
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

    n = _get_validated_order(n=n)
    alpha = _get_validated_alpha(alpha=alpha)
    x_center = _get_validated_x_center(x_center=x_center)

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
