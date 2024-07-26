"""
This test suite implements the tests for the module :mod:`hermite_functions._validate`.

"""

# === Imports ===

from typing import Any, Optional

import numpy as np
import pytest

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
        (  # Test 0: x is a float
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 1: x is an integer
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 2: x is complex
            1.0 + 1.0j,
            1,
            1.0,
            None,
            TypeError("Expected 'x' to be a float, int, or an Array-like"),
        ),
        (  # Test 3: x is a list
            [1.0, 2.0, 3.0],
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 4: x is a tuple
            (1.0, 2.0, 3.0),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 5: x is a 1D-Array of float32
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 6: x is a 1D-Array of float64
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 7: x is a 2D array
            np.array([[1.0, 2.0, 3.0]]),
            1,
            1.0,
            None,
            ValueError("Expected 'x' to be 1-dimensional"),
        ),
        (  # Test 8: n is a positive integer
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 9: n is negative integer
            1.0,
            -1,
            1.0,
            None,
            ValueError("Expected 'n' to be a non-negative integer"),
        ),
        (  # Test 10: n is a float
            1.0,
            1.0,
            1.0,
            None,
            TypeError("Expected 'n' to be an integer"),
        ),
        (  # Test 11: alpha is a positive float
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 12: alpha is a positive integer
            1.0,
            1,
            1,
            None,
            None,
        ),
        (  # Test 13: alpha is a zero float
            1.0,
            1,
            0.0,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 14: alpha is a zero integer
            1.0,
            1,
            0,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 14: alpha is a negative float
            1.0,
            1,
            -1.0,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 15: alpha is a negative integer
            1.0,
            1,
            -1,
            None,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 16: alpha is a complex number
            1.0,
            1,
            1.0 + 1.0j,
            None,
            TypeError("Expected 'alpha' to be a float or integer"),
        ),
        (  # Test 17: x_center is None
            1.0,
            1,
            1.0,
            None,
            None,
        ),
        (  # Test 18: x_center is a float
            1.0,
            1,
            1.0,
            1.0,
            None,
        ),
        (  # Test 19: x_center is an integer
            1.0,
            1,
            1.0,
            1,
            None,
        ),
        (  # Test 20: x_center is complex
            1.0,
            1,
            1.0,
            1.0 + 1.0j,
            TypeError("Expected 'x_center' to be a float, integer, or None"),
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

    # the function is parametrized
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation
    )

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            func(
                x=x,  # type: ignore
                n=n,
                alpha=alpha,
                x_center=x_center,
                **kwargs,
            )

        return

    # if no exception should be raised, the function is called and if it finishes, the
    # test is passed
    func(
        x=x,  # type: ignore
        n=n,
        alpha=alpha,
        x_center=x_center,
        **kwargs,
    )

    return
