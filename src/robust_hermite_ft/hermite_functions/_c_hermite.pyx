# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
Cython module :mod:`_hermite.pyx` for the evaluation of the Hermite function basis.

"""

# === Imports ===

from libc.math cimport exp, log, sqrt
import numpy as np
cimport numpy as np
from cython.parallel import prange

# === Constants ===

cdef double LOG_FOURTH_ROOT_OF_PI = -0.28618247146235  # ln(pi) / 4


# === Interface ===

def hermite_function_basis(
    double[::1] x,
    int n,
    int workers,
):
    """
    Python wrapper for the evaluation of the Hermite function basis up to order ``n``
    at the points ``x``.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape ``(num_x,)`` and dtype ``numpy.float64``
        The points at which the Hermite function basis should be evaluated.
    n : int
        The order up to which the Hermite function basis should be evaluated.
    workers : int
        The number of workers to use for parallelisation.

    Returns
    -------
    hermite_basis : :class:`numpy.ndarray` of shape ``(num_x, n + 1)`` and dtype ``numpy.float64``
        The values of the Hermite functions.

    """

    cdef int num_x = x.shape[0]
    cdef double[::, ::1] hermite_basis = np.empty(
        shape=(num_x, n + 1),
        dtype=np.float64,
    )

    _eval_hermite_function_basis(
        &x[0],
        num_x,
        n,
        &hermite_basis[0, 0],
        workers,
    )

    return np.asarray(hermite_basis)

# === Auxiliary Functions ===


cdef int _eval_hermite_function_basis(
    double* x,
    int num_x,
    int n,
    double* hermite_basis,
    int workers,
):
    """
    Evaluates the Hermite function basis up to order ``n`` at the points ``x``.
    It overwrites the input array ``hermite_basis`` with the results in a parallelized
    fashion.

    """

    # --- Variables ---

    cdef int iter_x

    # --- Main Loop ---

    for iter_x in prange(
        num_x,
        nogil=True,
        num_threads=workers,
    ):
        _eval_hermite_function_basis_core(
            x,
            num_x,
            n,
            hermite_basis,
            iter_x,
        )

    return 0  # dummy return value


cdef int _eval_hermite_function_basis_core(
    double* x,
    int num_x,
    int n,
    double* hermite_basis,
    int iter_x,
) noexcept nogil:
    """
    Core function for the evaluation of the Hermite function basis.
    For numerical stability, the Hermite functions are kept scaled to a value of 1
    while the exponent of the correction factor phi is tracked separately to
    compensate for this permanent re-scaling.

    """

    # --- Variables ---

    # to avoid the need for handling too many early exit cases, the virtual
    # -1-th Hermite function h_{-1} is set to 0.0
    # the 0-th Hermite function h_0 is (pi ** -0.25) * exp(-0.5 * x * x) but due to the
    # correction scheme applied, it will be stored as 1 while the exponent of the
    # correction factor phi is tracked separately
    cdef int iter_j
    cdef int base_index = iter_x * (n + 1)
    cdef double iter_j_plus_1
    cdef double xi = x[iter_x]
    cdef double h_i_minus_1 = 0.0
    cdef double h_i = 1.0
    cdef double h_i_plus_1
    cdef double exponent_correction = LOG_FOURTH_ROOT_OF_PI - 0.5 * xi * xi
    cdef double prefactor_i_minus_1, prefactor_i
    cdef double scale_factor

    # --- Zero-th order ---

    # for the zero-th order, the exponent correction just needs to be exponentiated
    hermite_basis[base_index] = exp(exponent_correction)

    # in case n == 0, the function evaluation is done
    if n < 1:
        return 0  # dummy return value

    # --- Main Loop ---

    # if higher orders are requested, a recursion is entered to compute the remaining
    # Hermite functions
    # the recursion is given by
    # h_{i+1} = sqrt(2 / (i + 1)) * x * h_{i} - sqrt(i / (i + 1)) * h_{i-1}
    # for a numerically stable calculation, the Hermite functions are kept scaled to a
    # value of 1 while the exponent of the correction factor phi is tracked separately
    # to compensate for this permanent re-scaling
    # NOTE: the loop tackles all but the last iteration to avoid the need for a
    #       conditional check in each iteration
    base_index += 1
    for iter_j in range(0, n - 1):
        # first, the prefactors are calculated ...
        # NOTE: the 1.0 has to be double to avoid integer division
        iter_j_plus_1 = iter_j + 1.0
        prefactor_i = sqrt(2.0 / iter_j_plus_1)
        prefactor_i_minus_1 = sqrt(iter_j / iter_j_plus_1)

        # ... followed by the actual recursion step ...
        h_i_plus_1 = prefactor_i * xi * h_i - prefactor_i_minus_1 * h_i_minus_1
        # ... after which the exponent correction is applied for storing the result
        hermite_basis[base_index + iter_j] = exp(exponent_correction) * h_i_plus_1

        # afterwards, the Hermite functions are scaled to a value of 1 and the exponent
        # correction is updated to prepare for the next iteration
        scale_factor = abs(h_i_plus_1) if h_i_plus_1 != 0.0 else 1.0
        h_i_minus_1 = h_i / scale_factor
        # NOTE: theoretically, h_{i+1} would be divided by its absolute value here,
        #       but a / |a| = sign(a) so the expensive division can be stated as a
        #       sign evaluation; here, everything relies on a sign definition that gives
        #       0 for a value of 0 and not +1 or -1
        if h_i_plus_1 != 0.0:
            h_i = +1.0 if h_i_plus_1 > 0.0 else -1.0
        else:
            h_i = 0.0

        exponent_correction += log(scale_factor)

    # --- Last Iteration ---

    # here, only the pre-factors and the recursion step are calculated
    # NOTE: the 1.0 has to be double to avoid integer division
    prefactor_i = sqrt(2.0 / n)
    prefactor_i_minus_1 = sqrt((n - 1.0) / n)
    hermite_basis[base_index + n - 1] = exp(exponent_correction) * (
        prefactor_i * xi * h_i - prefactor_i_minus_1 * h_i_minus_1
    )

    return 0  # dummy return value
