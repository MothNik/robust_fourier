"""
Module :mod:`hermite_functions._interface`

This module implements the interface to the either NumPy-based or Numba-based
implementations of the Hermite functions.

It augments them with an additional input validation which is better done in Python
and also handles the compilation of the Numba-functions if Numba is available at
runtime.

"""

# === Imports ===

from math import sqrt
from typing import Tuple, Union

import numpy as np
import psutil

from ._numba_funcs import nb_hermite_function_basis as _nb_hermite_function_basis
from ._numpy_funcs import _hermite_function_basis as _np_hermite_function_basis
from ._numpy_funcs import _single_hermite_function as _np_single_hermite_function

from ._c_hermite import (  # pyright: ignore[reportMissingImports]; fmt: skip; isort: skip   # noqa: E501
    hermite_function_basis as _c_hermite_function_basis,
)

# === Auxiliary Functions ===


def _get_num_workers(workers: int) -> int:
    """
    Gets the number of available workers for the process calling this function.

    Parameters
    ----------
    workers : :class:`int`
        Number of workers requested.

    Returns
    -------
    workers : :class:`int`
        Number of workers available.

    """

    # the number of workers may not be less than -1
    if workers < -1:
        raise ValueError(
            f"Expected 'workers' to be greater or equal to -1 but got {workers}."
        )

    # then, the maximum number of workers is determined ...
    # NOTE: the following does not count the number of total threads, but the number of
    #       threads available to the process calling this function
    process = psutil.Process()
    max_workers = len(process.cpu_affinity())  # type: ignore
    del process

    # ... and overwrites the number of workers if it is set to -1
    workers = max_workers if workers == -1 else workers

    # the number of workers is limited between 1 and the number of available threads
    return max(1, min(workers, max_workers))


def _get_validated_hermite_function_input(
    x: Union[float, int, np.ndarray],
    n: int,
    alpha: Union[float, int] = 1.0,
) -> Tuple[np.ndarray, int, float]:
    """
    Validates the input for the Hermite functions and returns the validated input.

    """

    # the input is validated according to the requirements of the higher level caller
    # functions
    if not isinstance(x, (float, int, np.ndarray)):
        raise TypeError(
            f"Expected 'x' to be a float, int, or np.ndarray but got type {type(x)}."
        )

    if not isinstance(n, int):
        raise TypeError(f"Expected 'n' to be an integer but got type {type(n)}.")

    if not isinstance(alpha, (float, int)):
        raise TypeError(
            f"Expected 'alpha' to be a float or integer but got type {type(alpha)}."
        )

    # the x-values are converted to a 1D NumPy array for checking
    x_inter = np.atleast_1d(x)
    if not x_inter.dtype == np.float64:
        x_inter = x_inter.astype(np.float64)

    if x_inter.ndim != 1:
        raise ValueError(
            f"Expected 'x' to be 1-dimensional but it is {x_inter.ndim}-dimensional."
        )

    if n < 0:
        raise ValueError(f"Expected 'n' to be a non-negative integer but got {n}.")

    if alpha <= 0.0:
        raise ValueError(f"Expected 'alpha' to be a positive number but got {alpha}.")

    # the validated input is returned
    return x_inter, n, alpha


# === Main Functions ===


def hermite_function_basis(
    x: Union[float, int, np.ndarray],
    n: int,
    alpha: Union[float, int] = 1.0,
    workers: int = 1,
) -> np.ndarray:
    """
    Computes the basis of dilated Hermite functions up to order ``n`` for the given
    points ``x``. It makes use of a recursion formula to compute all Hermite basis
    functions in one go. For maximum speed, the computation across the different
    x-values is implemented in a parallelized Cython function.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or :class:`numpy.ndarray` of shape (m,)
        The points at which the dilated Hermite functions are evaluated.
        Internally, it will be promoted to ``np.float64``.
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
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
        If either ``x``, ``n``, ``alpha``, or ``workers`` is not of the expected type.
    ValueError
        If ``x`` is not 1-dimensional after conversion to a NumPy array.
    ValueError
        If ``n`` is not a non-negative integer or ``alpha`` is not a positive number.

    Notes
    -----
    The dilated Hermite functions are defined as

    .. image:: docs/hermite_functions/equations/DilatedHermiteFunctions.png

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    Internally, they are computed in a numerically stable way that relies on a
    logarithmic scaling trick to avoid over- and underflow in the recursive calculation
    of the Hermite polynomials and the Gaussians. This allows for arbitrary large orders
    ``n`` to be evaluated.

    References
    ----------
    The implementation is an adaption of the Appendix in [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """

    # --- Input validation ---

    x_inter, n, alpha = _get_validated_hermite_function_input(x=x, n=n, alpha=alpha)

    # the number of workers is determined
    workers = _get_num_workers(workers)

    # --- Computation ---

    # the computation is done in parallel using the Cython-accelerated implementation
    reciprocal_alpha = 1.0 / alpha
    return sqrt(reciprocal_alpha) * _c_hermite_function_basis(
        x=reciprocal_alpha * x_inter,
        n=n,
        workers=workers,
    )


def slow_hermite_function_basis(
    x: Union[float, int, np.ndarray],
    n: int,
    alpha: Union[float, int] = 1.0,
    jit: bool = False,
) -> np.ndarray:
    """
    DEPRECATED: ONLY KEPT FOR COMPARISON PURPOSES

    Computes the basis of dilated Hermite functions up to order ``n`` for the given
    points ``x``. It makes use of a recursion formula to compute all Hermite basis
    functions in one go.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or :class:`numpy.ndarray` of shape (m,)
        The points at which the dilated Hermite functions are evaluated.
        Internally, it will be promoted to ``np.float64``.
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
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

    .. image:: docs/hermite_functions/equations/DilatedHermiteFunctions.png

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    Internally, they are computed in a numerically stable way that relies on a
    logarithmic scaling trick to avoid over- and underflow in the recursive calculation
    of the Hermite polynomials and the Gaussians. This allows for arbitrary large orders
    ``n`` to be evaluated.

    References
    ----------
    The implementation is an adaption of the Appendix in [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """

    # --- Input validation ---

    x_inter, n, alpha = _get_validated_hermite_function_input(x=x, n=n, alpha=alpha)

    # --- Computation ---

    # if requested, the Numba-accelerated implementation is used
    # NOTE: this does not have to necessarily involve Numba because it can also be
    #       the NumPy-based implementation under the hood
    reciprocal_alpha = 1.0 / alpha
    if jit:
        return sqrt(reciprocal_alpha) * _nb_hermite_function_basis(  # type: ignore
            x=reciprocal_alpha * x_inter,  # type: ignore
            n=n,  # type: ignore
        )

    # if Numba is not requested, the NumPy-based implementation is used
    return sqrt(reciprocal_alpha) * _np_hermite_function_basis(
        x=reciprocal_alpha * x_inter,
        n=n,
    )


def single_hermite_function(
    x: Union[float, int, np.ndarray],
    n: int,
    alpha: Union[float, int] = 1.0,
    jit: bool = False,
) -> np.ndarray:
    """
    Computes a single dilated Hermite function of order ``n`` for the given points
    ``x``. It offers a fast alternative for the computation of only a single high order
    Hermite function (``n >= 1000``), but not a full basis of them.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or :class:`numpy.ndarray` of shape (m,)
        The points at which the dilated Hermite function is evaluated.
        Internally, it will be promoted to ``np.float64``.
    n : :class:`int`
        The order of the dilated Hermite function.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.

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

    .. image:: docs/hermite_functions/equations/DilatedHermiteFunctions.png

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    For their computation, the function does not rely on recursion, but a direct
    evaluation of the Hermite functions via a complex integral.
    While this may be way faster than using the recursion for a single function for
    ``n >= 1000``, it is not suitable for computing a full basis of them.

    References
    ----------
    The implementation is based on [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """

    # --- Input validation ---

    x_inter, n, alpha = _get_validated_hermite_function_input(x=x, n=n, alpha=alpha)

    # --- Computation ---

    reciprocal_alpha = 1.0 / alpha
    return sqrt(reciprocal_alpha) * _np_single_hermite_function(
        x=reciprocal_alpha * x_inter,
        n=n,
    )
