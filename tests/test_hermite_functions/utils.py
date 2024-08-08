"""
Utility classes and functions for testing the module :mod:`hermite_functions`.

"""

# === Imports ===

from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple

import numpy as np

from robust_hermite_ft.hermite_functions import (
    HermiteFunctionBasis,
    hermite_function_basis,
    slow_hermite_function_basis,
)

# === Models ===

# an Enum class for the different implementations of the Hermite function basis


class HermiteFunctionBasisImplementations(str, Enum):
    FUNCTION_CYTHON_SINGLE = auto()
    FUNCTION_CYTHON_PARALLEL = auto()
    FUNCTION_NUMPY_SINGLE = auto()
    FUNCTION_NUMBA_SINGLE = auto()
    CLASS_INTERFACE = auto()


# === Constants ===

# a list of all Hermite function basis implementations
ALL_HERMITE_IMPLEMENTATIONS = [
    HermiteFunctionBasisImplementations.FUNCTION_CYTHON_SINGLE,
    HermiteFunctionBasisImplementations.FUNCTION_CYTHON_PARALLEL,
    HermiteFunctionBasisImplementations.FUNCTION_NUMPY_SINGLE,
    HermiteFunctionBasisImplementations.FUNCTION_NUMBA_SINGLE,
    HermiteFunctionBasisImplementations.CLASS_INTERFACE,
]

# === Functions ===

# a function to set up the Hermite function basis implementations for calling them
# directly


def setup_hermite_function_basis_implementations(
    implementation: HermiteFunctionBasisImplementations,
    n: Any,
    alpha: Any,
    x_center: Any,
) -> Tuple[
    Callable[..., np.ndarray],
    Dict[str, Any],
]:
    """
    Sets up the Hermite function basis implementations for calling them directly by
    selecting the correct function and pre-setting some keyword arguments.

    """

    if implementation == HermiteFunctionBasisImplementations.CLASS_INTERFACE:
        return (
            HermiteFunctionBasis(
                n=n,
                alpha=alpha,
                x_center=x_center,
            ),
            dict(),
        )

    base_kwargs = dict(n=n, alpha=alpha, x_center=x_center)

    if implementation == HermiteFunctionBasisImplementations.FUNCTION_CYTHON_SINGLE:
        kwargs = dict(workers=1, **base_kwargs)
        return hermite_function_basis, kwargs

    elif implementation == HermiteFunctionBasisImplementations.FUNCTION_CYTHON_PARALLEL:
        kwargs = dict(workers=-1, **base_kwargs)
        return hermite_function_basis, kwargs

    elif implementation == HermiteFunctionBasisImplementations.FUNCTION_NUMPY_SINGLE:
        kwargs = dict(jit=False, **base_kwargs)
        return slow_hermite_function_basis, kwargs

    elif implementation == HermiteFunctionBasisImplementations.FUNCTION_NUMBA_SINGLE:
        kwargs = dict(jit=True, **base_kwargs)
        return slow_hermite_function_basis, kwargs

    else:
        raise AssertionError(f"Unknown implementation: {implementation}")
