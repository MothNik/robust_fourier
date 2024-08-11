"""
Module :mod:`_utils`

This module provides utility functionalities that are used throughout the package, e.g.,

- handling of Numba-related tasks

"""

# === Imports ===

from ._validate import (  # noqa: F401
    IntScalar,
    RealScalar,
    get_validated_alpha,
    get_validated_chebpoly_or_hermfunc_input,
    get_validated_offset_along_axis,
    get_validated_order,
    get_validated_x_values,
)
from .numba_helpers import (  # noqa: F401
    NUMBA_NO_JIT_ARGV,
    NUMBA_NO_JIT_ENV_KEY,
    NumbaJitActions,
    do_numba_normal_jit_action,
    no_jit,
)
