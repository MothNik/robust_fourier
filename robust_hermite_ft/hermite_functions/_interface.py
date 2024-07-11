"""
Module :mod:`hermite_functions._interface`

This module implements the interface to the either NumPy-based or Numba-based
implementations of the Hermite functions.

It augments them with an additional input validation which is better done in Python
and also handles the compilation of the Numba-functions if Numba is available at
runtime.

"""

# === Imports ===

from typing import Union

import numpy as np

from ._numba_funcs import (
    nb_dilated_hermite_function_basis as _nb_dilated_hermite_function_basis,
)
from ._numpy_funcs import (
    _dilated_hermite_function_basis as _dilated_hermite_function_basis,
)

# === Functions ===


def hermite_function_basis(
    x: Union[float, int, np.ndarray],
    n: int,
    alpha: Union[float, int] = 1.0,
    jit: bool = False,
) -> np.ndarray:
    """
    Computes the basis of dilated Hermite functions up to order ``n`` for the given
    points ``x``.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or :class:`numpy.ndarray` of shape (m,)
        The points at which the dilated Hermite functions are evaluated.
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
        If Numba is not available, function silently falls back to the NumPy-based
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

    Internally, they are computed in a numerically stable way that relies on the
    logsumexp trick to avoid over- and underflow in the evaluation of the Hermite
    polynomials and thus allows for arbitrary large orders ``n``.

    """

    # --- Input validation ---

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

    if x_inter.ndim != 1:
        raise ValueError(
            f"Expected 'x' to be 1-dimensional but it is {x_inter.ndim}-dimensional."
        )

    if n < 0:
        raise ValueError(f"Expected 'n' to be a non-negative integer but got {n}.")

    if alpha <= 0.0:
        raise ValueError(f"Expected 'alpha' to be a positive number but got {alpha}.")

    # --- Functionality ---

    # if requested, the Numba-accelerated implementation is used
    # NOTE: this does not have to necessarily involve Numba because it can also be
    #       the NumPy-based implementation under the hood
    if jit:
        with np.errstate(divide="ignore", invalid="ignore"):
            hermite_function_basis = _nb_dilated_hermite_function_basis(
                x=x_inter,
                n=n,
                alpha=alpha,
            )

        return hermite_function_basis

    # if Numba is not requested, the NumPy-based implementation is used
    with np.errstate(divide="ignore", invalid="ignore"):
        hermite_function_basis = _dilated_hermite_function_basis(
            x=x_inter,
            n=n,
            alpha=alpha,
        )

    return hermite_function_basis
