"""
Module :mod:`hermite_functions`

This module provides  implementations of the Hermite functions which are essential for
the robust Fourier transform.
The ``n``-th Hermite function is given by

.. image:: docs/hermite_functions/equations/HermiteFunctions.png

where :math:`H_{n}` is the :math:`n`-th Hermite polynomial.

They have two nice properties:
- they are orthogonal
- they are eigenfunctions of the Fourier transform which makes them suitable for the
    robust Fourier transform by Least Squares Fits.

Here, a scaled version of the Hermite functions is used that introduces a scaling
factor :math:`{\\alpha}`:

"""

# === Imports ===

from ._interface import (  # noqa: F401
    hermite_function_basis,
    single_hermite_function,
    slow_hermite_function_basis,
)
