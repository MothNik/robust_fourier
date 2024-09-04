"""
Module :mod:`chebyshev_polynomials`

This module provides implementations of the Chebyshev polynomials of the first and
second kind for fitting a basis for robust Fourier transforms.

"""

# === Imports ===

from ._class_interface import ChebyshevPolynomialBasis  # noqa: F401
from ._func_interface import chebyshev_polyvander  # noqa: F401
