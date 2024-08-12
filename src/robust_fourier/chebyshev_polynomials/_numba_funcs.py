"""
Module :mod:`chebyshev_polynomials._numba_funcs`

This module provides Numba-based implementations of the Chebyshev polynomials of the
first and second kind.

Depending on the runtime availability of Numba, the functions are either compiled or
imported from the NumPy-based implementation.

"""

# === Imports ===

from typing import Tuple

import numpy as np

from src.robust_fourier._utils.numba_helpers import do_numba_normal_jit_action

# === Functions and Compilation ===

try:
    # if available/enabled, the functions are re-defined and compiled by Numba

    # --- Imports ---

    if do_numba_normal_jit_action:  # pragma: no cover
        from numba import jit
    else:
        from .._utils import no_jit as jit

    # --- Basic Implementation ---

    def _chebyshev_poly_bases_implementation(
        x: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simultaneously evaluates the complete basis of Chebyshev polynomials of the
        first and second kind by making use of a numerically stabilised combined
        recurrence relation.

        Parameters
        ----------
        x : :class:`numpy.ndarray` of shape (m,)
            The points at which the Chebyshev polynomials are evaluated.
        n : :class:`int`
            The order of the Chebyshev polynomials.

        Returns
        -------
        chebyshev_t1_basis : :class:`numpy.ndarray` of shape (m, n + 1)
            The values of the Chebyshev polynomials of the first kind.

        chebyshev_u2_basis : :class:`numpy.ndarray` of shape (m, n + 1)
            The values of the Chebyshev polynomials of the second kind.

        References
        ----------
        The implementation is taken from [1]_.

        .. [1] Hrycak T., Schmutzhard S., Accurate evaluation of Chebyshev polynomials
           in floating-point arithmetic, BIT Numer Math, (2019) 59, pp. 403–416,
           DOI: 10.1007/s10543-018-0738-5

        Notes
        -----
        The naive three-term recurrence relations for Chebyshev polynomials of the first
        and second kind have bad numerical properties because the error is increasing
        proportional to ``n * n``, i.e., quadratically with the order of the polynomial
        ``n``.
        However, a combined recurrence relation that links both kinds of Chebyshev
        polynomials has an error proportional to ``n``, i.e., a linear increase with the
        order of the polynomial ``n``. Already for ``n`` as little as ``4``, the
        combined recurrence relation outperforms the naive three-term recurrence
        relations. So, the roughly doubled computational cost of the combined recurrence
        relation is justified.
        On top of that, a combined evaluation offers the opportunity to easily compute
        the derivatives of the Chebyshev polynomials because the derivatives of the
        first kind are related to the polynomials of the second kind and vice versa.

        """

        # the two output arrays are initialised
        chebyshev_t1_basis = np.empty(shape=(x.size, n + 1))
        chebyshev_u2_basis = np.empty(shape=(x.size, n + 1))

        # the parallel loop over all the points is started
        for iter_i in range(0, x.size):
            # some pre-computations are done to avoid redundant computations
            # the combined recurrence relation is started with the initial value
            # - 1 for the Chebyshev polynomial of the first kind T_0(x)
            # - 0 for the Chebyshev polynomial of the second kind U_{-1}(x)
            x_value = x[iter_i]
            one_minus_x_squared = -(x_value * x_value) + 1
            t_i_minus_1 = 1.0
            u_i_minus_2 = 0.0

            # then, the combined recurrence relation is iteratively applied by means of
            # a loop
            # the recurrence relation is given by
            # T_{n}(x) = x * T_{n-1}(x) - (1 - x * x) * U_{n-2}(x)
            # U_{n-1}(x) = x * U_{n-2}(x) + T_{n-1}(x)
            for iter_j in range(0, n + 1):
                chebyshev_t1_basis[iter_i, iter_j] = t_i_minus_1
                u_i_minus_1 = (x_value * u_i_minus_2) + t_i_minus_1
                chebyshev_u2_basis[iter_i, iter_j] = u_i_minus_1
                t_i_minus_1 = (
                    x_value * t_i_minus_1
                ) - one_minus_x_squared * u_i_minus_2
                u_i_minus_2 = u_i_minus_1

        return chebyshev_t1_basis, chebyshev_u2_basis

    # --- Compilation ---

    nb_chebyshev_poly_bases = jit(
        nopython=True,
        cache=True,
    )(_chebyshev_poly_bases_implementation)


# otherwise, the NumPy-based implementation of the Hermite functions is declared as the
# Numba-based implementation
except ImportError:  # pragma: no cover

    from ._numpy_funcs import (  # noqa: F401, E501 # isort:skip
        _chebyshev_poly_bases as nb_chebyshev_poly_bases,
    )
