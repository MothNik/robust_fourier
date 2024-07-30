"""
Utility classes and functions for testing the module :mod:`hermite_functions`.

"""

# === Imports ===

from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

from robust_hermite_ft.hermite_functions import (
    hermite_function_basis,
    slow_hermite_function_basis,
)

# === Models ===

# an Enum class for the different implementations of the Hermite function basis


class HermiteFunctionBasisImplementations(str, Enum):
    CYTHON_SINGLE = auto()
    CYTHON_PARALLEL = auto()
    NUMPY_SINGLE = auto()
    NUMBA_SINGLE = auto()


# === Constants ===

# a list of all Hermite function basis implementations
ALL_HERMITE_IMPLEMENTATIONS = [
    HermiteFunctionBasisImplementations.CYTHON_SINGLE,
    HermiteFunctionBasisImplementations.CYTHON_PARALLEL,
    HermiteFunctionBasisImplementations.NUMPY_SINGLE,
    HermiteFunctionBasisImplementations.NUMBA_SINGLE,
]

# === Functions ===

# a function to set up the Hermite function basis implementations for calling them
# directly


def setup_hermite_function_basis_implementations(
    implementation: HermiteFunctionBasisImplementations,
) -> Tuple[
    Union[
        Callable[
            [np.ndarray, int, Union[float, int], Union[float, int, None], int],
            np.ndarray,
        ],
        Callable[
            [np.ndarray, int, Union[float, int], Union[float, int, None], bool],
            np.ndarray,
        ],
    ],
    Dict[str, Any],
]:
    """
    Sets up the Hermite function basis implementations for calling them directly by
    selecting the correct function and pre-setting some keyword arguments.

    """

    if implementation == HermiteFunctionBasisImplementations.CYTHON_SINGLE:
        return hermite_function_basis, dict(workers=1)

    elif implementation == HermiteFunctionBasisImplementations.CYTHON_PARALLEL:
        return hermite_function_basis, dict(workers=-1)

    elif implementation == HermiteFunctionBasisImplementations.NUMPY_SINGLE:
        return slow_hermite_function_basis, dict(jit=False)

    elif implementation == HermiteFunctionBasisImplementations.NUMBA_SINGLE:
        return slow_hermite_function_basis, dict(jit=True)

    else:
        raise AssertionError(f"Unknown implementation: {implementation}")
