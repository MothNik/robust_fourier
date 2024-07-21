"""
Module :mod:`hermite_functions._numba_funcs`

This module provides Numba-based implementations of the Hermite functions.

Depending on the runtime availability of Numba, the functions are either compiled or
imported from the NumPy-based implementation.

"""

# === Imports ===

import numpy as np
from numpy import abs as np_abs
from numpy import exp, log, sqrt, square

from .._utils.numba_helpers import do_numba_normal_jit_action

# === Functions ===


def _hermite_function_basis(
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Evaluates the complete basis of Hermite functions that are given by the product of a
    scaled Gaussian with a Hermite polynomial and can be written as follows:

    .. image:: docs/hermite_functions/equations/HermiteFunctions.png

    .. image:: docs/hermite_functions/equations/HermitePolynomials.png

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
    keeps the values of the Hermite functions at a scale of roughly 1 during the
    recursion while tracking a correction term that is added to the exponent of the
    Gaussian part of the Hermite function.

    """

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


# === Compilation ===

# if available, the functions are compiled by Numba
try:
    if do_numba_normal_jit_action:  # pragma: no cover
        from numba import jit
    else:
        from .._utils import no_jit as jit

    # if it is enabled, the functions are compiled
    nb_hermite_function_basis = jit(
        nopython=True,
        cache=True,
    )(_hermite_function_basis)


# otherwise, the NumPy-based implementation of the Hermite functions is declared as the
# Numba-based implementation
except ImportError:  # pragma: no cover
    from ._numpy_funcs import _hermite_function_basis

    nb_hermite_function_basis = _hermite_function_basis
