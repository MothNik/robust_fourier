"""
Module :mod:`_utils._x_preprocessing`

This module implements the input preprocessing for the independent variable ``x`` that
is shared by the Chebyshev polynomial and Hermite function classes, namely

- checking if ``x`` is still linked to the original data it was derived from and
- normalising ``x`` by shifting it with a center ``x_center`` and scaling it with a
    scaling factor ``alpha``

"""

# === Imports ===

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validate import RealScalar

# === Functions ===


def _is_data_linked(
    arr: np.ndarray,
    original: Union[RealScalar, ArrayLike],
) -> bool:
    """
    Strictly checks for ``arr`` not sharing any data with ``original``, under the
    assumption that ``arr = atleast_1d(original)`` followed by a potential type
    conversion.
    If ``arr`` is a view of ``original``, this function returns ``False``.

    Was copied from the SciPy utility function ``scipy.linalg._misc._datacopied``, but
    the name and the docstring were adapted to make them clearer. Besides, the check for
    scalar ``original``s was added.

    """

    if np.isscalar(original):
        return False
    if arr is original:
        return True
    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return original.__array__().dtype is arr.dtype  # type: ignore

    return arr.base is not None


def normalise_x_values(
    x_internal: NDArray[np.float64],
    x: Union[float, int, ArrayLike],
    x_center: float,
    alpha: float,
) -> NDArray[np.float64]:
    """
    Centers the given x-values around the given center value by handling potential
    copies and the special case where the center is the origin (0).
    Afterwards, the x-values are scaled with the scaling factor alpha, also handling
    the special case where alpha is 1.0.

    """

    # when the x-values are centered around the origin and not scaled, the x-values are
    # not modified at all
    center_is_origin = x_center == 0.0
    alpha_is_unity = alpha == 1.0
    if center_is_origin and alpha_is_unity:
        return x_internal

    # if x is a view of the original x-Array, a copy is made to avoid modifying
    # the x-values
    if _is_data_linked(arr=x_internal, original=x):
        x_internal = x_internal.copy()

    # the x-values are centered around the given center value (if required)
    if not center_is_origin:
        x_internal -= x_center

    # if required, the x-values are scaled with the scaling factor alpha (if required)
    if not alpha_is_unity:
        x_internal /= alpha

    return x_internal
