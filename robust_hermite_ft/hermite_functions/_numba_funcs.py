"""
Module :mod:`hermite_functions._numba_funcs`

This module provides Numba-based implementations of the Hermite functions.

Depending on the runtime availability of Numba, the functions are either compiled or
imported from the NumPy-based implementation.

"""

# === Imports ===

from typing import Tuple

import numpy as np

# === Functions ===


def _nb_logsumexp(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the log of the sum of exponentials of the real input elements in a Numba-
    compatible way.
    For efficiency, the function only accepts two 2-column arrays that are implicitly
    treated with the ``axis```-parameter set to ``1``.

    Parameters
    ----------
    a : :class:`numpy.ndarray` of shape (n, 2)
        The input array.
    b : :class:`numpy.ndarray` of shape (n, 2)
        Scaling factor for exp(`a`) must be of the same shape as `a`.
    Returns
    -------
    res : :class:`numpy.ndarray` of shape (n,)
        The natural logarithm of the sum of exponentials.
    sgn : :class:`numpy.ndarray` of shape (n,)
        The sign of the sum of exponentials after exponentiation.

    """

    # all entries where the scaling factor is zero are set to -inf
    a_internal = np.where(b != 0.0, a, -np.inf)

    # then, the maximum value along the specified axis is evaluated together with the
    # scaled reduced exponentials ``b * np.exp(a - a_max)``
    a_max = np.empty(shape=(a_internal.shape[0]), dtype=a_internal.dtype)
    for iter_i in range(0, a_internal.shape[0]):
        a_max[iter_i] = np.max(a_internal[iter_i, ::])

    a_max[~np.isfinite(a_max)] = 0.0
    tmp = b * np.exp(a_internal - a_max.reshape((-1, 1)))

    # then, the logsumexp is calculated from the scaled reduced exponentials
    res = np.sum(tmp, axis=1)

    return np.log(np.abs(res)) + a_max, np.sign(res)


def _nb_slogabs_dilated_hermite_polynomial_basis(
    x: np.ndarray,
    n: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the natural logarithm of complete basis of dilated Hermite polynomials up
    to order ``n`` for the given points ``x``.

    The dilated Hermite polynomials are defined as

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (m,)
        The points at which the dilated Hermite polynomials are evaluated.
    n : :class:`int`
        The order of the dilated Hermite polynomials.
    alpha : :class:`float`
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.

    Returns
    -------
    logsabs_hermpoly_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The natural logarithms of the absolute values of the dilated Hermite
        polynomials (see Notes).
    signs_hermpoly_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The signs of the dilated Hermite polynomials after exponentiation of the
        logarithms (see Notes).

    Notes
    -----
    The function makes use of the logsumexp trick to avoid overflow in the evaluation
    of the Hermite polynomials. During the recursive calculations (which are written as
    a loop), the logarithms and signs are tracked for all the multiplications and also
    additions and subtractions.
    To obtain the values of the Hermite polynomials, the exponentiation has to be
    applied like

    .. code-block:: python

        values = np.exp(logsabs_hermpoly_basis) * signs_hermpoly_basis

    although this is highly discouraged due to the potential of overflow. Actually,
    the function was never intended to be used for this purpose, and its actual use
    case is described below.

    Having the logarithms is crucial for the evaluation of the Hermite functions that
    are used in the robust Fourier transform. Since their computation involves the
    division of a Hermite polynomial (whose value can overflow) by the square root of
    a factorial and a multiplication by a Gaussian (both values can easily underflow),
    working in the logarithmic space is the only possible way. In the end, the
    multiplication of the arbitrarily large polynomial with an arbitrarily small scaled
    Gaussian will yield values that are bounded between
    :math:`\\frac{\\pm\\pi^{-\\frac{1}{4}}{\\sqrt{\\alpha}}`, which means that it can
    always be evaluated when over- and underflow are avoided.

    """

    # the zero-th dilated Hermite polynomial is defined as h_0 = 1, so its logarithm is
    # 0 and its sign is +1
    logabs_prefactor = np.log(2) - 2 * np.log(alpha)  # 2 / (alpha ** 2) in normal space
    signs_h_n_minus_1 = np.ones_like(x)
    logs_h_n_minus_1 = np.zeros_like(x)  # 1 in normal space
    signs_x = np.where(x >= 0.0, 1.0, -1.0)

    logsabs_hermpoly_basis = np.zeros(shape=(x.size, n + 1))
    signs_hermpoly_basis = np.zeros(shape=(x.size, n + 1))
    logsabs_hermpoly_basis[::, 0] = logs_h_n_minus_1
    signs_hermpoly_basis[::, 0] = signs_h_n_minus_1

    # if only the 0-th order is requested, the function can exit early here
    if n < 1:
        return logsabs_hermpoly_basis, signs_hermpoly_basis

    # if higher orders are requested, the first order is also calculated explicitly
    # for either a direct return or a recursion
    # the first Hermite polynomial is given by h_1 = 2 * x / alpha ** 2
    # h_1 inherits the shifted logarithm and the sign of x
    logsabs_x = np.log(np.abs(x))
    signs_h_n = signs_x.copy()
    logs_h_n = logabs_prefactor + logsabs_x  # 2 * x / (alpha ** 2) in normal space
    logsabs_hermpoly_basis[::, 1] = logs_h_n
    signs_hermpoly_basis[::, 1] = signs_h_n

    # if only the first order is requested, the function can exit early here
    if n <= 1:
        return logsabs_hermpoly_basis, signs_hermpoly_basis

    # the remaining polynomials are calculated using the recursion relation that is
    # written as a loop to keep the complexity at bay
    # some pre-computations for signs and pre-factors need to be made to make the
    # following loop efficient
    tmp_logsabs_add = np.empty(shape=(x.size, 2))
    tmp_logsabs_add[::, 1] = logsabs_x
    tmp_sign_multipliers = np.concatenate(
        (
            np.full(shape=(x.size, 1), fill_value=-1.0),
            signs_x.reshape((-1, 1)),
        ),
        axis=1,
    )

    for iter_i in range(2, n + 1):
        # the logsumexp trick is used, but this requires the signs to be computed
        # properly
        # for the first term (2 / (alpha ** 2)) * x * h_n, the sign is the product of
        # the signs of x and h_n
        # for the second term -(2 / (alpha ** 2)) * n * h_{n-1}, the sign is the
        # opposite of the sign of h_{n-1} since the (2 / (alpha ** 2)) * n is always
        # positive
        # besides, the prefactors x and n need to be incorporated into the logsabs
        # values; for the x-values, this is already done in the ``tmp_logsabs_add``, but
        # for the n-values, this needs to be done here
        tmp_logsabs_add[::, 0] = np.log(iter_i - 1)

        # with those, the logs need to be calculated using the logsumexp trick
        # NOTE: to make it Numba-compatible, the ``logsumexp`` function was replaced by
        #       a custom implementation
        logsabs_hermpoly_basis[::, iter_i], signs_hermpoly_basis[::, iter_i] = (
            _nb_logsumexp(
                a=logsabs_hermpoly_basis[::, iter_i - 2 : iter_i] + tmp_logsabs_add,
                b=signs_hermpoly_basis[::, iter_i - 2 : iter_i] * tmp_sign_multipliers,
            )
        )
        logsabs_hermpoly_basis[::, iter_i] += logabs_prefactor

    return logsabs_hermpoly_basis, signs_hermpoly_basis


def nb_dilated_hermite_function_basis(
    x: np.ndarray,
    n: int,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates the complete basis of dilated Hermite functions that are given by the
    product of a scaled dilated Gaussian with a dilated Hermite polynomial and can be
    written as follows:

    .. image:: docs/hermite_functions/equations/DilatedHermiteFunctions.png

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    Please refer to the Notes section for further details.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (m,)
        The points at which the dilated Hermite functions are evaluated.
    n : :class:`int`
        The order of the dilated Hermite functions.
    alpha : :class:`float`
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.

    Returns
    -------
    hermite_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the dilated Hermite functions.

    Notes
    -----
    Direct evaluation of the Hermite polynomials becomes ill-conditioned in finite
    precision since it involves the division of a polynomial by the square root of
    a factorial of ``n`` which is prone to overflow.
    To avoid this, the Hermite polynomials, the Gaussians, and the factorials are
    evaluated in the logarithmic space which gives

    .. image:: docs/hermite_functions/equations/LogDilatedHermiteFunctions.png

    With this, Hermite functions of all orders ``n`` can be evaluated without any
    numerical issues and it can be ensured that they are always bounded between
    :math:`\\frac{\\pm\\pi^{-\\frac{1}{4}}{\\sqrt{\\alpha}}`.

    """

    # first, some constants are pre-defined
    negative_log_fourth_root_pi = -0.28618247146235004  # -0.25 * np.log(np.pi)
    negative_half_log_two = -0.34657359027997264  # -0.5 * np.log(2)

    # then, the natural logarithms of the absolute values and the signs of the dilated
    # Hermite polynomials are calculated
    logsabs_hermpoly_basis, signs_hermpoly_basis = (
        _nb_slogabs_dilated_hermite_polynomial_basis(
            x=x,
            n=n,
            alpha=alpha,
        )
    )

    # afterwards, the natural logarithms of the prefactors individually for the
    # Gaussians and their scaling factor (here ``prefactors``)
    # NOTE: the Gaussian and the prefactor are guaranteed to be positive, so the signs
    #       can be skipped here
    logs_gaussian = -(0.5 / alpha / alpha) * np.square(x)
    # NOTE: the function can also be jitted by Numba, but Numba does not support
    #       keywords ``start``, ``stop``, ``step``, and ``dtype`` for ``np.arange``
    orders = np.arange(0, n + 1, 1, np.int64)

    # NOTE: the logarithm of the factorial could be computed by the ``gammaln`` function
    #       but since only integer arguments are used, it is directly computed as the
    #       cumulative sum of the logarithms of the integers from 1 to ``n``
    #       (inclusive); this also makes it Numba-compatible
    orders_for_log_factorial = orders.copy()
    orders_for_log_factorial[0] = 1.0  # 0! = 1
    logs_prefactors = (
        negative_log_fourth_root_pi
        + negative_half_log_two * orders
        + (orders - 0.5) * np.log(alpha)
        - 0.5 * np.cumsum(np.log(orders_for_log_factorial))
    )

    # finally, the Hermite functions are evaluated by adding the logarithms of the
    # Gaussians and the pre-factors to the logarithms of the Hermite polynomials by
    # leveraging NumPy's broadcasting capabilities
    return signs_hermpoly_basis * np.exp(
        logs_gaussian.reshape((-1, 1))
        + logs_prefactors.reshape((1, -1))
        + logsabs_hermpoly_basis
    )


# === Compilation ===

# if available, the functions are compiled by Numba
try:
    from numba import jit

    # if it is enabled, the functions are compiled
    _nb_logsumexp = jit(
        nopython=True,
        inline="always",
        cache=True,
    )(_nb_logsumexp)

    _nb_slogabs_dilated_hermite_polynomial_basis = jit(
        nopython=True,
        cache=True,
    )(_nb_slogabs_dilated_hermite_polynomial_basis)

    nb_dilated_hermite_function_basis = jit(
        nopython=True,
        cache=True,
    )(nb_dilated_hermite_function_basis)

# otherwise, the NumPy-based implementation of the Hermite functions is declared as the
# Numba-based implementation
except ImportError:
    from ._numpy_funcs import _dilated_hermite_function_basis

    nb_dilated_hermite_function_basis = _dilated_hermite_function_basis