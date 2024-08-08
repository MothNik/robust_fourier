"""
This test suite implements the tests for the module :mod:`hermite_functions._validate`.

"""

# === Imports ===

from array import array as PythonArray
from typing import Any, Optional

import numpy as np
import pytest
from pandas import Series as PandasSeries

from .utils import (
    ALL_HERMITE_IMPLEMENTATIONS,
    HermiteFunctionBasisImplementations,
    setup_hermite_function_basis_implementations,
)

# === Tests ===


@pytest.mark.parametrize("implementation", ALL_HERMITE_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "x, n, alpha, x_center, expected",
    [
        (  # Test 0: x is a Python float
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 1: x is a Python integer
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 2: x is a NumPy float
            np.float32(1.0),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 3: x is a NumPy integer
            np.int32(1),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 4: x is complex
            1.0 + 1.0j,
            1,
            1.0,
            None,
            TypeError("Expected 'x' to be a real scalar or a real-value Array-like"),
        ),
        (  # Test 5: x is a list
            [1.0, 2.0, 3.0],
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 6: x is a tuple
            (1.0, 2.0, 3.0),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 7: x is a 1D-Array of float32
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 8: x is a 1D-Array of float64
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 9: x is a Pandas Series of float64
            PandasSeries([1.0, 2.0, 3.0], dtype=np.float64),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 10: x is a Pandas Series of float32
            PandasSeries([1.0, 2.0, 3.0], dtype=np.float32),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 11: x is a Python array of float64
            PythonArray("d", [1.0, 2.0, 3.0]),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 12: x is a Python array of float32
            PythonArray("f", [1.0, 2.0, 3.0]),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 13: x is a 2D array
            np.array([[1.0, 2.0, 3.0]]),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to be 1-dimensional"),
        ),
        (  # Test 14: x is a 2D list
            [[1.0, 2.0, 3.0]],
            1,
            1.0,
            None,
            ValueError("Expected 'x' to be 1-dimensional"),
        ),
        (  # Test 15: x is a 2D tuple
            ((1.0, 2.0, 3.0),),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to be 1-dimensional"),
        ),
        (  # Test 16: x is an empty list
            [],
            1,
            1.0,
            None,
            ValueError("Expected 'x' to have at least one element"),
        ),
        (  # Test 17: x is an empty tuple
            tuple(),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to have at least one element"),
        ),
        (  # Test 18: x is an empty NumPy array
            np.array([]),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to have at least one element"),
        ),
        (  # Test 19: x is an empty Pandas Series
            PandasSeries([]),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to have at least one element"),
        ),
        (  # Test 20: x is an empty Python array
            PythonArray("d", []),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to have at least one element"),
        ),
        (  # Test 21: n is a positive Python integer
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 22: n is positive NumPy integer
            1.0,
            np.int32(1),
            1.0,
            None,
            None,
        ),
        (  # Test 23: n is negative integer
            1.0,
            -1,
            1.0,
            None,
            ValueError("Expected 'n' to be a non-negative integer"),
        ),
        (  # Test 24: n is a float
            1.0,
            1.0,
            1.0,
            None,
            TypeError("Expected 'n' to be an integer"),
        ),
        (  # Test 25: alpha is a positive Python float
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 26: alpha is a positive NumPy float
            1.0,
            1,
            np.float32(1.0),
            None,
            None,
        ),
        (  # Test 27: alpha is a positive Python integer
            1.0,
            1,
            1,
            None,
            None,
        ),
        (  # Test 28: alpha is a positive NumPy integer
            1.0,
            1,
            np.int32(1),
            None,
            None,
        ),
        (  # Test 29: alpha is a zero float
            1.0,
            1,
            0.0,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 30: alpha is a zero integer
            1.0,
            1,
            0,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 31: alpha is a negative float
            1.0,
            1,
            -1.0,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 32: alpha is a negative integer
            1.0,
            1,
            -1,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 33: alpha is a complex number
            1.0,
            1,
            1.0 + 1.0j,
            None,
            TypeError("Expected 'alpha' to be a float or integer"),
        ),
        (  # Test 34: x_center is None
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 35: x_center is a Python float
            1.0,
            1,
            1.0,
            1.0,
            None,
        ),
        (  # Test 36: x_center is a NumPy float
            1.0,
            1,
            1.0,
            np.float32(1.0),
            None,
        ),
        (  # Test 37: x_center is a Python integer
            1.0,
            1,
            1.0,
            1,
            None,
        ),
        (  # Test 38: x_center is a Numpy integer
            1.0,
            1,
            1.0,
            np.int32(1),
            None,
        ),
        (  # Test 39: x_center is complex
            1.0,
            1,
            1.0,
            1.0 + 1.0j,
            TypeError("Expected the x-'center' to be a float, integer, or None"),
        ),
    ],
)
def test_dilated_hermite_functions_input_validation(
    x: Any,
    n: Any,
    alpha: Any,
    x_center: Any,
    expected: Optional[Exception],
    implementation: HermiteFunctionBasisImplementations,
) -> None:
    """
    This test checks whether the function input validation of the :func:`hermite_function_basis` implementations

    - passes if the input is correct and no exception is raised,
    - raises an exception if the input is incorrect.

    """  # noqa: E501

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            # the function is parametrized
            # NOTE: for the class interface the validation will happen in the
            #       constructor and a failure will thus happen here already
            func, kwargs = setup_hermite_function_basis_implementations(
                implementation=implementation,
                n=n,
                alpha=alpha,
                x_center=x_center,
            )

            # the function is called
            # NOTE: for the class interfaces the validation will happen in the actual
            #       function call and a failure will thus happen here
            func(
                x=x,  # type: ignore
                **kwargs,
            )

        return

    # if no exception should be raised, the function is called and if it finishes, the
    # test is passed
    # the function is parametrized
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation,
        n=n,
        alpha=alpha,
        x_center=x_center,
    )

    # the function is called
    func(
        x=x,  # type: ignore
        **kwargs,
    )

    return
