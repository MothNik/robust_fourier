"""
Utility classes and functions for testing the module :mod:`chebyshev_polynomials`.

"""

# === Imports ===

from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple

import numpy as np

from robust_fourier import ChebyshevPolynomialBasis, chebyshev_polyvander

# === Models ===

# an Enum class for the different implementations of the Chebyshev polynomial basis


class ChebyshevPolyBasisImplementations(str, Enum):
    FUNCTION_NUMPY = auto()
    FUNCTION_NUMBA = auto()
    CLASS_NUMPY = auto()
    CLASS_NUMBA = auto()


# === Constants ===

# a list of all Chebyshev polynomial basis implementations
ALL_CHEBYSHEV_IMPLEMENTATIONS = [
    ChebyshevPolyBasisImplementations.FUNCTION_NUMPY,
    ChebyshevPolyBasisImplementations.FUNCTION_NUMBA,
    ChebyshevPolyBasisImplementations.CLASS_NUMPY,
    ChebyshevPolyBasisImplementations.CLASS_NUMBA,
]

# === Functions ===

# a function to set up the Chebyshev polynomial basis implementations for calling them
# directly


def setup_chebyshev_poly_basis_implementations(
    implementation: ChebyshevPolyBasisImplementations,
    n: Any,
    alpha: Any,
    x_center: Any,
    kind: Any,
) -> Tuple[
    Callable[..., np.ndarray],
    Dict[str, Any],
]:
    """
    Sets up the Chebyshev polynomial basis implementations for calling them directly by
    selecting the correct function and pre-setting some keyword arguments.

    """

    if implementation in {
        ChebyshevPolyBasisImplementations.CLASS_NUMPY,
        ChebyshevPolyBasisImplementations.CLASS_NUMBA,
    }:
        jit = implementation == ChebyshevPolyBasisImplementations.CLASS_NUMBA
        return (
            ChebyshevPolynomialBasis(
                n=n,
                alpha=alpha,
                x_center=x_center,
                kind=kind,
                jit=jit,
            ),
            dict(),
        )

    base_kwargs = dict(
        n=n,
        alpha=alpha,
        x_center=x_center,
        kind=kind,
    )

    if implementation == ChebyshevPolyBasisImplementations.FUNCTION_NUMPY:
        kwargs = dict(jit=False, **base_kwargs)
        return chebyshev_polyvander, kwargs

    if implementation == ChebyshevPolyBasisImplementations.FUNCTION_NUMBA:
        kwargs = dict(jit=True, **base_kwargs)
        return chebyshev_polyvander, kwargs

    raise AssertionError(f"Unknown implementation: {implementation}")
