"""
Utility classes and functions for testing the module :mod:`hermite_functions`.

"""

# === Imports ===

from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple

import numpy as np

from robust_fourier.hermite_functions import (
    HermiteFunctionBasis,
    hermite_function_basis,
)

# === Models ===

# an Enum class for the different implementations of the Hermite function basis


class HermiteFunctionBasisImplementations(str, Enum):
    FUNCTION_NUMPY = auto()
    FUNCTION_NUMBA = auto()
    CLASS_NUMPY = auto()
    CLASS_NUMBA = auto()


# === Constants ===

# a list of all Hermite function basis implementations
ALL_HERMITE_IMPLEMENTATIONS = [
    HermiteFunctionBasisImplementations.FUNCTION_NUMPY,
    HermiteFunctionBasisImplementations.FUNCTION_NUMBA,
    HermiteFunctionBasisImplementations.CLASS_NUMPY,
    HermiteFunctionBasisImplementations.CLASS_NUMBA,
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

    if implementation in {
        HermiteFunctionBasisImplementations.CLASS_NUMPY,
        HermiteFunctionBasisImplementations.CLASS_NUMBA,
    }:
        jit = implementation == HermiteFunctionBasisImplementations.CLASS_NUMBA
        return (
            HermiteFunctionBasis(
                n=n,
                alpha=alpha,
                x_center=x_center,
                jit=jit,
            ),
            dict(),
        )

    base_kwargs = dict(n=n, alpha=alpha, x_center=x_center)

    if implementation == HermiteFunctionBasisImplementations.FUNCTION_NUMPY:
        kwargs = dict(jit=False, **base_kwargs)
        return hermite_function_basis, kwargs

    if implementation == HermiteFunctionBasisImplementations.FUNCTION_NUMBA:
        kwargs = dict(jit=True, **base_kwargs)
        return hermite_function_basis, kwargs

    raise AssertionError(f"Unknown implementation: {implementation}")
