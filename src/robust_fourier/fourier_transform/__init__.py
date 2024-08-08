"""
Module :mod:`fourier_transform`

This module provides the implementation of the robust Fourier transform by Least Squares
Fits with Hermite functions as well as general utility functions for dealing with
Discrete Fourier transforms.

"""

# === Imports ===

from ._fft_utils import (  # noqa: F401
    ContinuousFourierTransform,
    DiscreteFourierTransform,
    TimeSpaceSignal,
    angular_frequency_grid,
    convert_continuous_to_discrete_ft,
    convert_discrete_to_continuous_ft,
    discrete_ft,
    grid_spacing,
)
