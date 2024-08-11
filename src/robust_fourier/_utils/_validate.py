"""
Module :mod:`_utils._validate`

This module implements the input validations for the Chebyshev polynomial and Hermite
function parameters:

- ``x``: the independent variable
- ``n``: the order
- ``alpha``: the scaling factor of the independent variable
- ``x_center``: the center

"""

# === Imports ===

from typing import Any, Tuple, Union

import numpy as np

# === Types ===

# a Python or NumPy float or integer
RealScalar = Union[float, int, np.floating, np.integer]
_real_scalar_types_no_pyfloat = (int, np.floating, np.integer)

# a Python or NumPy integer
IntScalar = Union[int, np.integer]

# === Constants ===

# the required dtype for the x-values
_X_DTYPE = np.float64

# === Exceptions ===


class _GivesNoRealArrayError(Exception):
    """
    Exception raised when the input does not result in a real-valued NumPy array.

    """

    pass


# === Functions ===


def get_validated_x_values(x: Any) -> np.ndarray:
    """
    Validates the input for the x-values and returns the validated input.

    """

    # the x-values are converted to a 1D NumPy array for checking
    try:
        x_internal = np.atleast_1d(x)
        if not np.isreal(x_internal).all():
            raise _GivesNoRealArrayError()

    except _GivesNoRealArrayError:
        x_type_str = f"{type(x)}"
        if hasattr(x, "dtype"):  # pragma: no cover
            x_type_str += f" with dtype {x.dtype}"  # type: ignore

        raise TypeError(
            f"Expected 'x' to be a real scalar or a real-value Array-like but got type "
            f"{x_type_str}."
        )

    # if the converted x-values are not 1-dimensional, an error is raised
    if x_internal.ndim != 1:
        raise ValueError(
            f"Expected 'x' to be 1-dimensional but it is {x_internal.ndim}-dimensional."
        )

    # if required, the dtype is converted to the target dtype
    if not x_internal.dtype == _X_DTYPE:
        x_internal = x_internal.astype(_X_DTYPE)

    if x_internal.size < 1:
        raise ValueError(
            f"Expected 'x' to have at least one element but got {x_internal.size} "
            f"elements."
        )

    return x_internal


def get_validated_order(n: Any) -> int:
    """
    Validates the input for the order of the Chebyshev polynomials or Hermite functions
    and returns the validated input.

    """

    # NumPy integers need to be converted to Python integers
    if isinstance(n, np.integer):
        n = int(n)

    if not isinstance(n, int):
        raise TypeError(f"Expected 'n' to be an integer but got type {type(n)}.")

    if n < 0:
        raise ValueError(f"Expected 'n' to be a non-negative integer but got {n}.")

    return n


def get_validated_alpha(alpha: Any) -> float:
    """
    Validates the input for the scaling factor for the independent variable of the
    Chebyshev polynomials or Hermite functions and returns the validated input.

    """

    # integers and NumPy scalars need to be converted to Python floats
    if isinstance(alpha, _real_scalar_types_no_pyfloat):
        alpha = float(alpha)

    if not isinstance(alpha, float):
        raise TypeError(
            f"Expected 'alpha' to be a float or integer but got type {type(alpha)}."
        )

    if alpha <= 0.0:
        raise ValueError(f"Expected 'alpha' to be a positive number but got {alpha}.")

    return alpha


def get_validated_offset_along_axis(
    offset: Any,
    which_axis: str,
) -> float:
    """
    Validates the input for the center (shift) along the an axis, e.g., the x-center
    along the x-axis, and returns the validated input.

    """

    if offset is not None:
        # integers and NumPy scalars need to be converted to Python floats
        if isinstance(offset, _real_scalar_types_no_pyfloat):
            offset = float(offset)

        if not isinstance(offset, float):
            raise TypeError(
                f"Expected the {which_axis}-'center' to be a float, integer, or None "
                f"but got type {type(offset)}."
            )

    else:
        offset = 0.0

    return offset


def get_validated_chebpoly_or_hermfunc_input(
    x: Any,
    n: Any,
    alpha: Any,
    x_center: Any,
) -> Tuple[np.ndarray, int, float, float]:
    """
    Validates the input for the Chebyshev polynomials or Hermite functions and returns
    the validated input.

    """

    # the input is validated according to the requirements of the higher level caller
    # functions
    return (
        get_validated_x_values(x=x),
        get_validated_order(n=n),
        get_validated_alpha(alpha=alpha),
        get_validated_offset_along_axis(
            offset=x_center,
            which_axis="x",
        ),
    )
