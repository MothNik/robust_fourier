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
from numpy.typing import ArrayLike

from .._utils import _get_num_workers
from ._numba_funcs import nb_hermite_function_basis as _nb_hermite_function_basis
from ._numpy_funcs import _hermite_function_basis as _np_hermite_function_basis
from ._numpy_funcs import _single_hermite_function as _np_single_hermite_function
from ._validate import _get_validated_hermite_function_input

from ._c_hermite import (  # pyright: ignore[reportMissingImports]; fmt: skip; isort: skip   # noqa: E501
    hermite_function_basis as _c_hermite_function_basis,
)

# === Auxiliary Functions ===


def _is_data_linked(arr, original) -> bool:
    """
    Strictly checks for ``arr`` not sharing any data with ``original``, under the
    assumption that ``arr = atleast_1d(original)``.
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
        return True

    return arr.base is not None


def _center_x_values(
    x_internal: np.ndarray,
    x: Union[float, int, ArrayLike],
    x_center: Optional[float],
) -> np.ndarray:
    """
    Centers the given x-values around the given center value by handling potential
    copies and the special case where the center is the origin (0).

    """

    # when the x-values are centered around the origin, the x-values are not modified
    if x_center is None or x_center == 0.0:
        return x_internal

    # if x is a view of the original x-Array, a copy is made to avoid modifying
    # the x-values
    if _is_data_linked(arr=x_internal, original=x):
        x_internal = x_internal.copy()

    # the x-values are centered around the given center value
    x_internal -= x_center
    return x_internal


# === Main Functions ===


def hermite_function_basis(
    x: Union[float, int, ArrayLike],
    n: int,
    alpha: Union[float, int] = 1.0,
    x_center: Union[float, int, None] = None,
    workers: int = 1,
) -> np.ndarray:
    """
    Computes the basis of dilated Hermite functions up to order ``n`` for the given
    points ``x``. It makes use of a recursion formula to compute all Hermite basis
    functions in one go. For maximum speed, the computation across the different
    x-values is implemented in a parallelized Cython function.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or Array-like of shape (m,)
        The points at which the dilated Hermite functions are evaluated.
        Internally, it will be promoted to ``np.float64``.
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = alpha * x``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite functions.
        If ``None`` or ``0``, the functions are centered at the origin.
        Otherwise, the center is shifted to the given value.
    workers : :class:`int`, default=``1``
        The number of parallel workers to use for the computation.
        If ``workers=1``, the computation is done in a single thread.
        If ``workers=-1``, the number of workers is set to the number of threads
        available for the process calling this function (not necessarily the number of
        threads available in the whole system).
        Values that exceed the number of available threads are silently clipped to the
        maximum number available.

    Returns
    -------
    hermite_function_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the dilated Hermite functions at the points ``x``.
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

    .. image:: docs/hermite_functions/equations/Dilated_Hermite_Functions_Of_Generic_X.png

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/Dilated_Hermite_Polynomials_Of_Generic_X.png

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

    (
        x_internal,
        n,
        alpha,
        x_center,
    ) = _get_validated_hermite_function_input(
        x=x,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # the number of workers is determined
    workers = _get_num_workers(workers)

    # --- Computation ---

    # if required, the x-values are centered
    x_internal = _center_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,
    )

    # the computation is done in serial and parallel fashion using the
    # Cython-accelerated implementation
    return pysqrt(alpha) * _c_hermite_function_basis(
        x=alpha * x_internal,
        n=n,
        workers=workers,
    )


def slow_hermite_function_basis(
    x: Union[float, int, ArrayLike],
    n: int,
    alpha: Union[float, int] = 1.0,
    x_center: Union[float, int, None] = None,
    jit: bool = False,
) -> np.ndarray:
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
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = alpha * x``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite functions.
        If ``None`` or ``0``, the functions are centered at the origin.
        Otherwise, the center is shifted to the given value.
    jit : :class:`bool`, default=``False``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.

    Returns
    -------
    hermite_function_basis : :class:`numpy.ndarray` of shape (m, n + 1)
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

    .. image:: docs/hermite_functions/equations/Dilated_Hermite_Functions_Of_Generic_X.png

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/Dilated_Hermite_Polynomials_Of_Generic_X.png

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

    (
        x_internal,
        n,
        alpha,
        x_center,
    ) = _get_validated_hermite_function_input(
        x=x,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # --- Computation ---

    # if requested, the Numba-accelerated implementation is used
    # NOTE: this does not have to necessarily involve Numba because it can also be
    #       the NumPy-based implementation under the hood
    # if required, the x-values are centered
    x_internal = _center_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,
    )

    func = _nb_hermite_function_basis if jit else _np_hermite_function_basis
    return pysqrt(alpha) * func(  # type: ignore
        x=alpha * x_internal,  # type: ignore
        n=n,  # type: ignore
    )


def single_hermite_function(
    x: Union[float, int, ArrayLike],
    n: int,
    alpha: Union[float, int] = 1.0,
    x_center: Union[float, int, None] = None,
) -> np.ndarray:
    """
    Computes a single dilated Hermite function of order ``n`` for the given points
    ``x``. It offers a fast alternative for the computation of only a single high order
    Hermite function (``n >= 1000``), but not a full basis of them.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or Array-like of shape (m,)
        The points at which the dilated Hermite function is evaluated.
        Internally, it will be promoted to ``np.float64``.
    n : :class:`int`
        The order of the dilated Hermite function.
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
    hermite_function : :class:`numpy.ndarray` of shape (m,)
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

    .. image:: docs/hermite_functions/equations/Dilated_Hermite_Functions_Of_Generic_X.png

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/Dilated_Hermite_Polynomials_Of_Generic_X.png

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

    (
        x_internal,
        n,
        alpha,
        x_center,
    ) = _get_validated_hermite_function_input(
        x=x,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # --- Computation ---

    # if required, the x-values are centered
    x_internal = _center_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,
    )

    return pysqrt(alpha) * _np_single_hermite_function(
        x=alpha * x_internal,
        n=n,
    )