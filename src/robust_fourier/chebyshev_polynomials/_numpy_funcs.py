"""
Module :mod:`chebyshev_polynomials._numpy_funcs`

This module provides NumPy-based implementations of the Chebyshev polynomials of the
first and second kind.

"""

# === Imports ===

from typing import Tuple

import numpy as np

# === Functions ===


def _chebyshev_polyvander(
    x: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simultaneously evaluates the complete basis (Vandermonde matrix) of Chebyshev
    polynomials of the first and second kind by making use of a numerically stabilised
    combined recurrence relation.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (m,)
        The points at which the Chebyshev polynomials are evaluated.
    n : :class:`int`
        The order of the Chebyshev polynomials.

    Returns
    -------
    chebyshev_t1_vander : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the Chebyshev polynomials of the first kind evaluated at the
        points ``x`` represented as a Vandermonde matrix.

    chebyshev_u2_vander : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the Chebyshev polynomials of the second kind evaluated at the
        points ``x`` represented as a Vandermonde matrix.

    References
    ----------
    The implementation is taken from [1]_.

    .. [1] Hrycak T., Schmutzhard S., Accurate evaluation of Chebyshev polynomials in
       floating-point arithmetic, BIT Numer Math, (2019) 59, pp. 403–416,
       DOI: 10.1007/s10543-018-0738-5

    Notes
    -----
    The naive three-term recurrence relations for Chebyshev polynomials of the first and
    second kind have bad numerical properties because the error is increasing
    proportional to ``n * n``, i.e., quadratically with the order of the polynomial
    ``n``.
    However, a combined recurrence relation that links both kinds of Chebyshev
    polynomials has an error proportional to ``n``, i.e., a linear increase with the
    order of the polynomial ``n``. Already for ``n`` as little as ``4``, the combined
    recurrence relation outperforms the naive three-term recurrence relations. So, the
    roughly doubled computational cost of the combined recurrence relation is justified.
    On top of that, a combined evaluation offers the opportunity to easily compute the
    derivatives of the Chebyshev polynomials because the derivatives of the first kind
    are related to the polynomials of the second kind and vice versa.

    """

    # some pre-computations are done to avoid redundant computations
    # the combined recurrence relation is started with the initial value
    # - 1 for the Chebyshev polynomial of the first kind T_0(x)
    # - 0 for the Chebyshev polynomial of the second kind U_{-1}(x)
    chebyshev_t1_vander = np.empty(shape=(n + 1, x.size))
    chebyshev_u2_vander = np.empty_like(chebyshev_t1_vander)
    t_i_minus_1 = np.ones_like(x)
    u_i_minus_2 = np.zeros_like(x)
    one_minus_x_squared = 1.0 - (x * x)

    # then, the combined recurrence relation is iteratively applied by means of a loop
    # the recurrence relation is given by
    # T_{n}(x) = x * T_{n-1}(x) - (1 - x * x) * U_{n-2}(x)
    # U_{n-1}(x) = x * U_{n-2}(x) + T_{n-1}(x)
    for iter_j in range(0, n + 1):
        chebyshev_t1_vander[iter_j] = t_i_minus_1  # NOTE: is not a view
        u_i_minus_1 = x * u_i_minus_2 + t_i_minus_1
        chebyshev_u2_vander[iter_j] = u_i_minus_1  # NOTE: is not a view
        t_i_minus_1 = x * t_i_minus_1 - one_minus_x_squared * u_i_minus_2
        u_i_minus_2 = u_i_minus_1

    return chebyshev_t1_vander, chebyshev_u2_vander
