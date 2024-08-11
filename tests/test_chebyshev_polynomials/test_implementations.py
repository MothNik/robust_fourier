"""
This test suite implements the tests for the module :mod:`chebyshev_polynomials._func_interface`
and also the ``__call__`` method of the class :class:`ChebyshevPolyBasis` from the
module :mod:`chebyshev_polynomials._class_interface`.

"""  # noqa: E501

# === Imports ===

from array import array as PythonArray
from typing import Any, Literal, Optional, Type, Union

import numpy as np
import pytest
from pandas import Series as PandasSeries
from scipy.special import eval_chebyt as scipy_chebyshev_first_kind
from scipy.special import eval_chebyu as scipy_chebyshev_second_kind

from robust_fourier import chebyshev_poly_basis
from robust_fourier.chebyshev_polynomials._func_interface import (
    get_validated_chebyshev_kind,
)

from .utils import (
    ALL_CHEBYSHEV_IMPLEMENTATIONS,
    ChebyshevPolyBasisImplementations,
    setup_chebyshev_poly_basis_implementations,
)

# === Constants ===

# the number of x-values and the order of the Chebyshev polynomials for the test against
# the SciPy implementation
# NOTE: the number of x-values has to be an odd number to make sure that 0 is included
TEST_CHEBYSHEV_POLY_ORDER = 1_000
TEST_CHEBYSHEV_POLY_N_X_VALUES = 2_001

# the absolute and relative tolerances for the tests of the Chebyshev polynomials
# against the SciPy implementation
# NOTE: these have to be chosen relatively high because the Chebyshev polynomials are
#       quite sensitive to rounding errors during the centering and scaling applied in
#       the tests
TEST_CHEBYSHEV_POLY_FLOAT64_ATOL = 1e-9
TEST_CHEBYSHEV_POLY_FLOAT64_RTOL = 1e-9

assert TEST_CHEBYSHEV_POLY_N_X_VALUES % 2 == 1, "The number of x-values must be odd"

# === Tests ===


@pytest.mark.parametrize(
    "kind, expected",
    [
        (1, 1),  # Test 0: kind is 1
        ("first", 1),  # Test 1: kind is "first"
        (2, 2),  # Test 2: kind is 2
        ("second", 2),  # Test 3: kind is "second"
        (None, None),  # Test 4: kind is None
        ("both", None),  # Test 5: kind is "both"
        (  # Test 6: kind is of an invalid type
            complex(3, 5),
            TypeError("Expected 'kind' to be an integer, a string, or None"),
        ),
        (  # Test 7: kind is an invalid integer
            0,
            ValueError("Expected 'kind' to be one of"),
        ),
        (  # Test 8: kind is an invalid string
            "zero",
            ValueError("Expected 'kind' to be one of"),
        ),
    ],
)
def test_chebyshev_poly_kind_input_validation(
    kind: Any,
    expected: Union[Exception, Optional[Literal[1, 2]]],
) -> None:
    """
    This test checks whether the function input validation for the kind of the Chebyshev
    polynomials

    - passes if the input is correct and no exception is raised,
    - raises an exception if the input is incorrect.

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            kind_validated = get_validated_chebyshev_kind(kind=kind)

        return

    # if no exception should be raised, the output is checked to be as expected
    kind_validated = get_validated_chebyshev_kind(kind=kind)
    if expected is not None:
        assert kind_validated == expected

    else:
        assert kind_validated is None

    return


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("array_like_type", [np.ndarray, PandasSeries, PythonArray])
def test_centered_chebyshev_poly_do_not_modify_x_values(
    array_like_type: Type,
    dtype: Type,
) -> None:
    """
    This test checks whether the function :func:`chebyshev_poly_basis` does not modify
    the input x-values when the center is set.

    """

    # the x-values are set up ...
    x_values = np.array([19.0, 20.0, 21.0], dtype=dtype)
    x_values_original = x_values.copy()
    if array_like_type == PandasSeries:
        x_values = PandasSeries(x_values.tolist(), dtype=dtype)
    elif array_like_type == PythonArray:
        dtype_str = "f" if dtype == np.float32 else "d"
        x_values = PythonArray(dtype_str, x_values.tolist())  # type: ignore

    # the function is called with the center set
    chebyshev_poly_basis(
        x=x_values,
        n=1,
        alpha=1.0,
        x_center=20.0,
        kind=None,
    )

    # the x-values are checked to be unchanged
    assert np.array_equal(np.asarray(x_values), x_values_original)


@pytest.mark.parametrize("x_center", [-10.0, 0.0, None, 10.0])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("implementation", ALL_CHEBYSHEV_IMPLEMENTATIONS)
def test_dilated_chebyshev_poly_basis_against_scipy_reference(
    implementation: ChebyshevPolyBasisImplementations,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test checks the implementations of the Chebyshev polynomials against a symbolic
    reference implementation.

    The ``x_center`` is tested in a hacky manner by shifting the x-values and calling
    the function with the shifted values and the center. With this, the function - given
    that everything runs correctly - should be invariant to the centering and always
    return the reference values.

    """

    # first, the function call is set up
    # NOTE: for the cases when the first, the second, or both kinds are requested to
    #       cover all possible cases
    spec_kwargs = dict(
        implementation=implementation,
        n=TEST_CHEBYSHEV_POLY_ORDER,
        alpha=alpha,
        x_center=x_center,
    )
    func_first, kwargs_first = setup_chebyshev_poly_basis_implementations(
        kind="first",
        **spec_kwargs,  # type: ignore
    )
    func_second, kwargs_second = setup_chebyshev_poly_basis_implementations(
        kind="second",
        **spec_kwargs,  # type: ignore
    )
    func_both, kwargs_both = setup_chebyshev_poly_basis_implementations(
        kind="both",
        **spec_kwargs,  # type: ignore
    )

    # then, the x-values are set up and the Chebyshev polynomials are evaluated
    # NOTE: to be numerically precise, the x-values are set up in a way with as much
    #       accuracy as possible and Python's ``sum`` uses a compensated summation
    #       algorithm to reduce the rounding errors
    x_center_for_shift = x_center if x_center is not None else 0.0
    x_values = np.linspace(
        start=sum((x_center_for_shift, -alpha)),
        stop=sum((x_center_for_shift, alpha)),
        num=TEST_CHEBYSHEV_POLY_N_X_VALUES,
        dtype=np.float64,
    )
    chebyshev_basis_first_kind_alone = func_first(
        x=x_values,
        **kwargs_first,
    )
    chebyshev_basis_second_kind_alone = func_second(
        x=x_values,
        **kwargs_second,
    )
    (
        chebyshev_basis_first_kind_both,
        chebyshev_basis_second_kind_both,
    ) = func_both(
        x=x_values,
        **kwargs_both,
    )

    # the SciPy reference values are computed with the x-values between -1 and +1
    n_values = np.arange(
        start=0, stop=TEST_CHEBYSHEV_POLY_ORDER + 1, step=1, dtype=np.int64
    ).reshape((1, -1))
    x_values = np.linspace(
        start=-1.0,
        stop=+1.0,
        num=TEST_CHEBYSHEV_POLY_N_X_VALUES,
        dtype=np.float64,
    ).reshape((-1, 1))
    reference_chebyshev_basis_first_kind = scipy_chebyshev_first_kind(
        n_values,
        x_values,
    )
    reference_chebyshev_basis_second_kind = scipy_chebyshev_second_kind(
        n_values,
        x_values,
    )

    # the reference values are compared with the numerical results
    # NOTE: the numerical tolerances are relatively high because the Chebyshev
    #       polynomials become quite sensitive to rounding errors during the centering
    #       and scaling applied in the tests
    for basis, ref_basis in zip(
        (
            chebyshev_basis_first_kind_alone,
            chebyshev_basis_second_kind_alone,
            chebyshev_basis_first_kind_both,
            chebyshev_basis_second_kind_both,
        ),
        (
            reference_chebyshev_basis_first_kind,
            reference_chebyshev_basis_second_kind,
            reference_chebyshev_basis_first_kind,
            reference_chebyshev_basis_second_kind,
        ),
    ):

        assert np.allclose(
            basis,
            ref_basis,
            atol=TEST_CHEBYSHEV_POLY_FLOAT64_ATOL,
            rtol=TEST_CHEBYSHEV_POLY_FLOAT64_RTOL,
        )

    return
