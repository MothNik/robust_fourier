"""
Module :mod:`hermite_functions._numba_funcs`

This module provides Numba-based implementations of the Hermite functions.

Depending on the runtime availability of Numba, the functions are either compiled or
imported from the NumPy-based implementation.

"""

# === Imports ===

from .._utils._numba_helpers import do_numba_normal_jit_action
from ._numpy_funcs import _hermite_function_basis

# === Functions ===

# NOTE: here are not functions because the NumPy-based implementation was written to be
#       compatible with Numba ``jit``-compilation


# === Compilation ===

# if available/enabled, the functions are compiled by Numba
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

    nb_hermite_function_basis = _hermite_function_basis
