"""
Module :mod:`_utils`

This module provides utility functionalities that are used throughout the package, e.g.,

- handling of Numba-related tasks

"""

# === Imports ===

from .numba_helpers import (  # noqa: F401
    NUMBA_NO_JIT_ARGV,
    NUMBA_NO_JIT_ENV_KEY,
    NumbaJitActions,
    do_numba_normal_jit_action,
    no_jit,
)
