"""
Module :mod:`chebyshev_polynomials._numba_funcs`

This module provides Numba-based implementations of the Chebyshev polynomials of the
first and second kind.

Depending on the runtime availability of Numba, the functions are either compiled or
imported from the NumPy-based implementation.

"""

# === Imports ===

from .._utils._numba_helpers import do_numba_normal_jit_action
from ._numpy_funcs import _chebyshev_poly_bases

# === Functions ===

# NOTE: here are no functions because the NumPy-based implementation was written to be
#       compatible with Numba ``jit``-compilation


# === Compilation ===

# if available/enabled, the functions are compiled by Numba
try:
    if do_numba_normal_jit_action:  # pragma: no cover
        from numba import jit
    else:
        from .._utils import no_jit as jit

    # if it is enabled, the functions are compiled
    nb_chebyshev_poly_bases = jit(
        nopython=True,
        cache=True,
    )(_chebyshev_poly_bases)


# otherwise, the NumPy-based implementation of the Hermite functions is declared as the
# Numba-based implementation
except ImportError:  # pragma: no cover

    nb_chebyshev_poly_bases = _chebyshev_poly_bases
