"""
Module :mod:`hermite_functions._func_interface`

This module implements the interface to the either NumPy-based or Numba-based
implementations of the Hermite functions.

It augments them with an additional input validation which is better done in Python
and also handles the incorporation of the Numba-functions if Numba is available at
runtime.

"""

# === Imports ===

from math import sqrt as pysqrt
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils import (
    IntScalar,
    RealScalar,
    get_validated_chebpoly_or_hermfunc_input,
    get_validated_grid_points,
    normalise_x_values,
)
from ._numba_funcs import nb_hermite_function_vander as _nb_hermite_function_vander
from ._numpy_funcs import _hermite_function_vander as _np_hermite_function_vander
from ._numpy_funcs import _single_hermite_function as _np_single_hermite_function

# === Main Functions ===


def hermite_function_vander(
    x: Union[RealScalar, ArrayLike],
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
    jit: bool = True,
    validate_parameters: bool = True,
) -> NDArray[np.float64]:
    """
    Computes the complete basis (Vandermonde matrix) of dilated Hermite functions up to
    order ``n`` for the given points ``x``. It makes use of a recursion formula to
    compute all Hermite basis functions in one go.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or Array-like of shape (m,)
        The points at which the dilated Hermite functions are evaluated.
        Internally, it will be promoted to ``np.float64``.
        It has to contain at least one element.
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite functions.
        If ``None`` or ``0``, the functions are centered at the origin.
        Otherwise, the center is shifted to the given value.
    jit : :class:`bool`, default=``True``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.
    validate_parameters : :class:`bool`, default=``True``
        Whether to validate all the input parameters (``True``) or only ``x``
        (``False``).
        Disabling the input checks is not recommended and was only implemented for
        internal use.

    Returns
    -------
    hermite_func_vander : :class:`numpy.ndarray` of shape (m, n + 1) of dtype ``np.float64``
        The values of the dilated Hermite functions at the points ``x`` represented as
        a Vandermonde matrix.
        It will always be 2D even if ``x`` is a scalar.

    Raises
    ------
    TypeError
        If any of the input arguments is not of the expected type.
    ValueError
        If ``x`` is not 1-dimensional after conversion to a NumPy array.
    ValueError
        If ``n`` is not a non-negative integer or ``alpha`` is not a positive number.

    Notes
    -----
    The dilated Hermite functions are defined as

    .. image:: docs/hermite_functions/equations/HF-01-Hermite_Functions_TimeSpace_Domain.svg

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/HF-02-Hermite_Polynomials_TimeSpace_Domain.svg

    Internally, they are computed in a numerically stable way that relies on a
    logarithmic scaling trick to avoid over- and underflow in the recursive calculation
    of the Hermite polynomials and the Gaussians. This allows for arbitrary large orders
    ``n`` to be evaluated.

    References
    ----------
    The implementation is an adaption of the Appendix in [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """  # noqa: E501

    # --- Input validation ---

    if validate_parameters:
        (
            x_internal,
            n,
            alpha,
            x_center,
        ) = get_validated_chebpoly_or_hermfunc_input(
            x=x,
            x_dtype=np.float64,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

    else:  # pragma: no cover
        x_internal = get_validated_grid_points(grid_points=x, dtype=np.float64)

    # --- Computation ---

    # if required, the x-values are centered
    x_internal = normalise_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,  # type: ignore
        alpha=alpha,  # type: ignore
    )

    # if requested, the Numba-accelerated implementation is used
    # NOTE: this does not have to necessarily involve Numba because it can also be
    #       the NumPy-based implementation under the hood
    func = _nb_hermite_function_vander if jit else _np_hermite_function_vander
    hermite_basis = func(  # type: ignore
        x=x_internal,  # type: ignore
        n=n,  # type: ignore
    )

    # to preserve orthonormality, the Hermite functions are scaled with the square root
    # of the scaling factor alpha (if required)
    if alpha != 1.0:
        hermite_basis *= 1.0 / pysqrt(alpha)

    # NOTE: the Array has to be transposed because the low level functions return the
    #       transposed basis because it is more efficient for the computation
    return np.moveaxis(
        hermite_basis,
        source=0,
        destination=1,
    )


def single_hermite_function(
    x: Union[RealScalar, ArrayLike],
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
    validate_parameters: bool = True,
) -> NDArray[np.float64]:
    """
    Computes a single dilated Hermite function of order ``n`` for the given points
    ``x``. It offers a fast alternative for the computation of only a single high order
    Hermite function (``n >= 1000``), but not a full basis of them.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or Array-like of shape (m,)
        The points at which the dilated Hermite function is evaluated.
        Internally, it will be promoted to ``np.float64``.
        It has to contain at least one element.
    n : :class:`int`
        The order of the dilated Hermite function.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite function.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.
    validate_parameters : :class:`bool`, default=``True``
        Whether to validate all the input parameters (``True``) or only ``x``
        (``False``).
        Disabling the input checks is not recommended and was only implemented for
        internal use.

    Returns
    -------
    hermite_function : :class:`numpy.ndarray` of shape (m,) of dtype ``np.float64``
        The values of the dilated Hermite function at the points ``x``.
        It will always be 1D even if ``x`` is a scalar.

    Raises
    ------
    TypeError
        If any of the input arguments is not of the expected type.
    ValueError
        If ``x`` is not 1-dimensional after conversion to a NumPy array.
    ValueError
        If ``n`` is not a non-negative integer or ``alpha`` is not a positive number.

    Notes
    -----
    The dilated Hermite functions are defined as

    .. image:: docs/hermite_functions/equations/HF-01-Hermite_Functions_TimeSpace_Domain.svg

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/HF-02-Hermite_Polynomials_TimeSpace_Domain.svg

    For their computation, the function does not rely on recursion, but a direct
    evaluation of the Hermite functions via a complex integral.
    While this may be way faster than using the recursion for a single function for
    ``n >= 1000``, it is not suitable for computing a full basis of them.

    References
    ----------
    The implementation is based on [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """  # noqa: E501

    # --- Input validation ---

    if validate_parameters:
        (
            x_internal,
            n,
            alpha,
            x_center,
        ) = get_validated_chebpoly_or_hermfunc_input(
            x=x,
            x_dtype=np.float64,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

    else:  # pragma: no cover
        x_internal = get_validated_grid_points(grid_points=x, dtype=np.float64)

    # --- Computation ---

    # if required, the x-values are centered
    x_internal = normalise_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,  # type: ignore
        alpha=alpha,  # type: ignore
    )

    hermite_function = _np_single_hermite_function(
        x=x_internal,
        n=n,  # type: ignore
    )

    # to preserve orthonormality, the Hermite function is scaled with the square root of
    # the scaling factor alpha (if required)
    if alpha != 1.0:
        hermite_function *= 1.0 / pysqrt(alpha)

    return hermite_function
