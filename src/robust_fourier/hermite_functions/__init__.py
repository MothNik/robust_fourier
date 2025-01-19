"""
Module :mod:`hermite_functions`

This module provides  implementations of the Hermite functions which are essential for
the robust Fourier transform.
The ``n``-th dilated and shifted Hermite function is given by

.. image:: docs/hermite_functions/equations/HF-01-Hermite_Functions_OfGenericX.svg

where :math:`H_{n}` is the :math:`n`-th Hermite polynomial

.. image:: docs/hermite_functions/equations/HF-02-Hermite_Polynomials_OfGenericX.svg

They have two nice properties:
- they are orthogonal
- they are eigenfunctions of the Fourier transform which makes them suitable for the
    robust Fourier transform by Least Squares Fits.

"""  # noqa: E501

# === Imports ===

from . import _approximations as hermite_approx  # noqa: F401
from ._class_interface import HermiteFunctionBasis  # noqa: F401
from ._func_interface import (  # noqa: F401
    hermite_function_vander,
    single_hermite_function,
)
