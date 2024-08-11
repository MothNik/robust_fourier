"""
This test suite implements the tests for the module :mod:`_utils._x_preprocessing`.

"""

# === Imports ===

from array import array as PythonArray
from typing import Literal, Union

import numpy as np
import pytest
from pandas import Series as PandasSeries

from robust_fourier._utils import get_validated_chebpoly_or_hermfunc_input
from robust_fourier._utils._x_preprocessing import _is_data_linked


@pytest.mark.parametrize(
    "original, expected",
    [
        (  # Test 0: a 1D-Array of float64
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            True,
        ),
        (  # Test 1: a scalar
            1.0,
            False,
        ),
        (  # Test 2: a list
            [1.0, 2.0, 3.0],
            "auto",
        ),
        (  # Test 3: a tuple
            (1.0, 2.0, 3.0),
            "auto",
        ),
        (  # Test 4: a 1D-Array of float32
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "auto",
        ),
        (  # Test 5: a 1D-Pandas Series of float64
            PandasSeries([1.0, 2.0, 3.0], dtype=np.float64),
            "auto",
        ),
        (  # Test 6: a 1D-Pandas Series of float32
            PandasSeries([1.0, 2.0, 3.0], dtype=np.float32),
            "auto",
        ),
        (  # Test 7: a 1D-Python Array of float64
            PythonArray("d", [1.0, 2.0, 3.0]),
            "auto",
        ),
        (  # Test 8: a 1D-Python Array of float32
            PythonArray("f", [1.0, 2.0, 3.0]),
            "auto",
        ),
    ],
)
def test_is_data_linked_identified_correctly_after_input_validation(
    original: np.ndarray,
    expected: Union[bool, Literal["auto"]],
) -> None:
    """
    This test checks whether the function :func:`_is_data_linked` correctly identifies
    whether the input data is copied after the same process as for the input validation
    of the Chebyshev polynomials and the Hermite functions.

    For indexable data types (lists, tuples, NumPy arrays, Pandas Series,
    Python arrays), the function dynamically checks whether the data is linked or copied
    by modifying one of the entries in the modified array and checking whether the
    original array is also modified.

    """

    # the input validation is called and the data copying is checked
    modified = get_validated_chebpoly_or_hermfunc_input(
        x=original,
        n=1,
        alpha=1.0,
        x_center=None,
    )[0]

    # the data copying is checked
    result = _is_data_linked(arr=modified, original=original)

    # if the expected is "auto", the modification is checked by setting the second entry
    # in the modified array to a different value and checking whether the original array
    # is also modified
    if expected == "auto":
        modify_value = -100.0
        modify_index = 1
        modified[modify_index] = modify_value
        expected = original[modify_index] == modify_value

    # the result is checked
    assert result == expected
