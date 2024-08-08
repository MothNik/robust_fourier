"""
Module :mod:`hermite_functions._func_interface`

This module implements the interface to the either NumPy-based or Numba-based
implementations of the Hermite functions.

It augments them with an additional input validation which is better done in Python
and also handles the compilation of the Numba-functions if Numba is available at
runtime.

"""

# === Imports ===

from math import sqrt as pysqrt
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._numba_funcs import nb_hermite_function_basis as _nb_hermite_function_basis
from ._numpy_funcs import _hermite_function_basis as _np_hermite_function_basis
from ._numpy_funcs import _single_hermite_function as _np_single_hermite_function
from ._validate import (
    IntScalar,
    RealScalar,
    get_validated_hermite_function_input,
    get_validated_x_values,
)

# === Auxiliary Functions ===


def _is_data_linked(
    arr: np.ndarray,
    original: Union[RealScalar, ArrayLike],
) -> bool:
    """
    Strictly checks for ``arr`` not sharing any data with ``original``, under the
    assumption that ``arr = atleast_1d(original)`` followed by a potential type
    conversion.
    If ``arr`` is a view of ``original``, this function returns ``False``.

    Was copied from the SciPy utility function ``scipy.linalg._misc._datacopied``, but
    the name and the docstring were adapted to make them clearer. Besides, the check for
    scalar ``original``s was added.

    """

    if np.isscalar(original):
        return False
    if arr is original:
        return True
    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return original.__array__().dtype is arr.dtype  # type: ignore

    return arr.base is not None


def _normalise_x_values(
    x_internal: NDArray[np.float64],
    x: Union[float, int, ArrayLike],
    x_center: float,
    alpha: float,
) -> NDArray[np.float64]:
    """
    Centers the given x-values around the given center value by handling potential
    copies and the special case where the center is the origin (0).
    Afterwards, the x-values are scaled with the scaling factor alpha, also handling
    the special case where alpha is 1.0.

    """

    # when the x-values are centered around the origin and not scaled, the x-values are
    # not modified at all
    center_is_origin = x_center == 0.0
    alpha_is_unity = alpha == 1.0
    if center_is_origin and alpha_is_unity:
        return x_internal

    # if x is a view of the original x-Array, a copy is made to avoid modifying
    # the x-values
    if _is_data_linked(arr=x_internal, original=x):
        x_internal = x_internal.copy()

    # the x-values are centered around the given center value (if required)
    if not center_is_origin:
        x_internal -= x_center

    # if required, the x-values are scaled with the scaling factor alpha (if required)
    if not alpha_is_unity:
        x_internal /= alpha

    return x_internal


# === Main Functions ===


def hermite_function_basis(
    x: Union[RealScalar, ArrayLike],
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
    jit: bool = True,
    validate_parameters: bool = True,
) -> NDArray[np.float64]:
    """
    DEPRECATED: ONLY KEPT FOR COMPARISON PURPOSES

    Computes the basis of dilated Hermite functions up to order ``n`` for the given
    points ``x``. It makes use of a recursion formula to compute all Hermite basis
    functions in one go.

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
    hermite_function_basis : :class:`numpy.ndarray` of shape (m, n + 1) of dtype ``np.float64``
        The values of the dilated Hermite functions at the points ``x``.
        It will always be 2D even if ``x`` is a scalar.

    Raises
    ------
    TypeError
        If either ``x``, ``n``, or ``alpha`` is not of the expected type.
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
        ) = get_validated_hermite_function_input(
            x=x,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

    else:  # pragma: no cover
        x_internal = get_validated_x_values(x=x)

    # --- Computation ---

    # if requested, the Numba-accelerated implementation is used
    # NOTE: this does not have to necessarily involve Numba because it can also be
    #       the NumPy-based implementation under the hood
    # if required, the x-values are centered
    x_internal = _normalise_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,  # type: ignore
        alpha=alpha,  # type: ignore
    )

    func = _nb_hermite_function_basis if jit else _np_hermite_function_basis
    hermite_basis = func(  # type: ignore
        x=x_internal,  # type: ignore
        n=n,  # type: ignore
    )

    # to preserve orthonormality, the Hermite functions are scaled with the square root
    # of the scaling factor alpha (if required)
    if alpha != 1.0:
        hermite_basis *= 1.0 / pysqrt(alpha)

    return hermite_basis


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
        If either ``x``, ``n``, or ``alpha`` is not of the expected type.
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
        ) = get_validated_hermite_function_input(
            x=x,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

    else:  # pragma: no cover
        x_internal = get_validated_x_values(x=x)

    # --- Computation ---

    # if required, the x-values are centered
    x_internal = _normalise_x_values(
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
