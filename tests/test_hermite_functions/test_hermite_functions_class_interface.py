"""
This test suite implements the tests for the module :mod:`hermite_functions._class_interface`.

"""  # noqa: E501

# === Imports ===

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from robust_hermite_ft import HermiteFunctionBasis, approximate_hermite_funcs_fadeout_x
from robust_hermite_ft.fourier_transform import (
    TimeSpaceSignal,
    angular_frequency_grid,
    convert_discrete_to_continuous_ft,
    discrete_ft,
    grid_spacing,
)
from robust_hermite_ft.hermite_functions._class_interface import (
    _get_validated_time_space_symmetry,
)

# === Types ===

# the methods for the Hermite function computations of the class for the tests
HermiteFunctionMethod = Literal["__call__", "eval"]

# === Constants ===

# the common scaling factors alpha for the tests
TEST_SCALES_ALPHA = [0.5, 1.0, 2.0]
# the common time/space x-centers for the tests
TEST_TIME_SPACE_X_CENTERS = [-10.0, 0.0, None, 10.0]
# the common methods for the Hermite function computations by the class for the tests
TEST_HERMITE_FUNCTION_COMPUTATION_METHODS = [
    "__call__",  # calling the instance directly
    "eval",  # calling the static method ``eval``
]

# the orders n for the Hermite functions for the Fourier transform tests
# NOTE: it's important that at least one even and one odd order is included
FOURIER_TRANSFORM_TEST_ORDERS = [100, 101, 102]
# the number of x-values per Hermite function order for the Fourier transform tests
FOURIER_TRANSFORM_TEST_NUM_X_VALUES_PER_ORDER = 25
# the safety margin for the number of x-values to account for the outermost oscillations
# of the Hermite functions
FOURIER_TRANSFORM_TEST_X_VALUES_SAFETY_MARGIN = 0.1
# the absolute and relative tolerances for the Fourier transform tests
FOURIER_TRANSFORM_TEST_ATOL = 1e-13
FOURIER_TRANSFORM_TEST_RTOL = 1e-13

assert any(
    ft_n % 2 == 0 for ft_n in FOURIER_TRANSFORM_TEST_ORDERS
), "At least one even order must be included in the Fourier transform tests."
assert any(
    ft_n % 2 != 0 for ft_n in FOURIER_TRANSFORM_TEST_ORDERS
), "At least one odd order must be included in the Fourier transform tests."

# === Models ===


@dataclass
class HermiteOutputFormat:
    """
    Defines the format of the output of the ``__call__``  or ``eval`` method of the
    class :class:`HermiteFunctionBasis` for both the time/space and frequency domain.

    """

    # the expected number of columns of the output (independent of the domain)
    num_columns: int

    # the expected dtype of the output for the time/space domain ...
    time_space_dtype: Type
    # ... and the frequency domain
    frequency_dtype: Type


@dataclass
class HermiteFunctionOutputFormatTestSpecs:
    """
    Defines the test specifications for the output format of the ``__call__`` or
    ``eval`` method of the class :class:`HermiteFunctionBasis`.

    """

    # the order n of the Hermite functions
    n: int
    # the scaling factor alpha of the Hermite functions
    alpha: float
    # the time/space x-center of the Hermite functions
    time_space_x_center: Optional[float]
    # the time/space symmetry of the Hermite functions
    time_space_symmetry: Optional[Literal["none", "even", "odd"]]

    # the expected output format
    expected_output_format: HermiteOutputFormat


# === Auxiliary Functions ===


def setup_hermite_basis_evaluator(
    method: HermiteFunctionMethod,
    n: int,
    alpha: float,
    time_space_x_center: Optional[float],
    time_space_symmetry: Optional[Literal["none", "even", "odd"]],
) -> Tuple[Callable[..., NDArray[Union[np.float64, np.complex128]]], Dict[str, Any]]:
    """
    Sets up the Hermite function evaluator function and its keyword arguments based on
    the given method.

    """

    if method == "__call__":
        hermite_basis = HermiteFunctionBasis(
            n=n,
            alpha=alpha,
            time_space_x_center=time_space_x_center,
            time_space_symmetry=time_space_symmetry,
        )
        kwargs: Dict[str, Any] = dict()

        return hermite_basis.__call__, kwargs

    if method == "eval":
        kwargs = dict(
            n=n,
            alpha=alpha,
            time_space_x_center=time_space_x_center,
            time_space_symmetry=time_space_symmetry,
        )

        return HermiteFunctionBasis.eval, kwargs

    raise ValueError(f"Unknown method '{method}'.")


# === Tests ===


@pytest.mark.parametrize(
    "time_space_symmetry, expected",
    [
        (None, "none"),
        ("none", "none"),
        ("NoNe", "none"),
        ("even", "even"),
        ("EvEn", "even"),
        ("odd", "odd"),
        ("OdD", "odd"),
        (3, TypeError("The time/space symmetry must be None or a String")),
        ("this_is_wrong", ValueError("The time/space symmetry must be one of")),
    ],
)
def test_symmetry_input_validation(
    time_space_symmetry: Any,
    expected: Union[str, Exception],
) -> None:
    """
    This test checks whether the function input validation via the function
    :func:`_get_validated_time_space_symmetry`

    - passes if the input is correct and no exception is raised,
    - raises an exception if the input is incorrect.

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            _get_validated_time_space_symmetry(time_space_symmetry=time_space_symmetry)

        return

    # if no exception should be raised, the validated input is checked to be equal to
    # the expected output
    validated_time_space_symmetry = _get_validated_time_space_symmetry(
        time_space_symmetry=time_space_symmetry
    )
    assert validated_time_space_symmetry == expected

    return


def test_hermite_function_class_property_getters_setters_and_len() -> None:
    """
    This test checks whether the property getters and setters of the class
    :class:`HermiteFunctionBasis` work correctly.
    Besides, also the ``__len__`` method is tested, but during the test it is also
    considered in the same way as a property.

    """

    # the class instance is initialised with some values
    hermite_basis = HermiteFunctionBasis(
        n=1,
        alpha=1.0,
        time_space_x_center=None,
        time_space_symmetry=None,
    )

    # the properties are checked to be equal to be properly set during initialisation
    assert hermite_basis.n == 1
    assert len(hermite_basis) == 2
    assert hermite_basis.alpha == 1.0
    assert hermite_basis.time_space_x_center == 0.0
    assert hermite_basis.time_space_symmetry == "none"
    assert not hermite_basis.is_fully_validated  # evaluated lazily in ``__call__``

    # the properties are overwritten with new values
    hermite_basis.n = 2
    hermite_basis.alpha = 2.0
    hermite_basis.time_space_x_center = 10.0
    hermite_basis.time_space_symmetry = "odd"

    # the properties are checked to be equal to the new values
    assert hermite_basis.n == 2
    assert len(hermite_basis) == 1
    assert hermite_basis.alpha == 2.0
    assert hermite_basis.time_space_x_center == 10.0
    assert hermite_basis.time_space_symmetry == "odd"
    assert not hermite_basis.is_fully_validated  # evaluated lazily in ``__call__``

    # the Hermite functions are evaluated to trigger the internal validation, after the
    # symmetry was set to "even"
    hermite_basis.time_space_symmetry = "even"
    hermite_basis(x=1.0)

    # the property are checked once more to be equal to the new values
    assert hermite_basis.n == 2
    assert len(hermite_basis) == 2
    assert hermite_basis.alpha == 2.0
    assert hermite_basis.time_space_x_center == 10.0
    assert hermite_basis.time_space_symmetry == "even"
    assert hermite_basis.is_fully_validated  # evaluated in call

    # the property ``alpha`` is overwritten which should NOT invalidate the class
    hermite_basis.alpha = 0.5
    assert hermite_basis.n == 2
    assert len(hermite_basis) == 2
    assert hermite_basis.alpha == 0.5
    assert hermite_basis.time_space_x_center == 10.0
    assert hermite_basis.time_space_symmetry == "even"
    assert hermite_basis.is_fully_validated  # remains unchanged

    # now, the property ``time_space_x_center`` is overwritten which should also NOT
    # invalidate the class
    hermite_basis.time_space_x_center = -10.0
    assert hermite_basis.n == 2
    assert len(hermite_basis) == 2
    assert hermite_basis.alpha == 0.5
    assert hermite_basis.time_space_x_center == -10.0
    assert hermite_basis.time_space_symmetry == "even"
    assert hermite_basis.is_fully_validated  # remains unchanged

    # when the property ``n`` is overwritten, the class should be invalidated
    hermite_basis.n = 1
    assert hermite_basis.n == 1
    assert len(hermite_basis) == 1
    assert hermite_basis.alpha == 0.5
    assert hermite_basis.time_space_x_center == -10.0
    assert hermite_basis.time_space_symmetry == "even"
    assert not hermite_basis.is_fully_validated  # evaluated lazily in ``__call__``

    # after an evaluation, the class should be validated again
    hermite_basis(x=1.0)
    assert hermite_basis.is_fully_validated

    # finally, weh the property ``time_space_x_center`` is overwritten, the class should
    # be invalidated again
    hermite_basis.time_space_symmetry = "odd"
    assert hermite_basis.n == 1
    assert len(hermite_basis) == 1
    assert hermite_basis.alpha == 0.5
    assert hermite_basis.time_space_x_center == -10.0
    assert hermite_basis.time_space_symmetry == "odd"
    assert not hermite_basis.is_fully_validated  # evaluated lazily in ``__call__``

    # it should be validated again after one last evaluation
    hermite_basis(x=1.0)
    assert hermite_basis.is_fully_validated

    return


@pytest.mark.parametrize("time_space_x_center", TEST_TIME_SPACE_X_CENTERS)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
def test_hermite_function_class_raises_error_when_no_available_functions(
    alpha: float,
    time_space_x_center: Optional[float],
) -> None:
    """
    This test checks whether the class :class:`HermiteFunctionBasis` raises an error for
    the parameter setting ``n = 0`` and ``time_space_symmetry = "odd"`` when set like
    this both at the initialisation and via the property setter.

    Note that the validation is lazy, so the error is raised when the Hermite functions
    are computed.

    """

    reference_exception = ValueError(
        "There are no Hermite functions to compute with 'n = 0' and "
        "'time_space_symmetry = 'odd''."
    )

    # Test 0: wrong initialisation
    hermite_basis = HermiteFunctionBasis(
        n=0,
        alpha=alpha,
        time_space_x_center=time_space_x_center,
        time_space_symmetry="odd",
    )
    assert not hermite_basis.is_fully_validated
    with pytest.raises(type(reference_exception), match=str(reference_exception)):
        hermite_basis(x=1.0)

    # Test 1: wrong property setting for ``n``
    hermite_basis = HermiteFunctionBasis(
        n=1,
        alpha=alpha,
        time_space_x_center=time_space_x_center,
        time_space_symmetry="odd",
    )
    # one successful computation is carried out to trigger the internal validation
    assert not hermite_basis.is_fully_validated
    hermite_basis(x=1.0)
    assert hermite_basis.is_fully_validated
    # overwriting ``n`` should reset the internal validation ...
    hermite_basis.n = 0
    assert not hermite_basis.is_fully_validated
    # ... which then should cause an error at the next computation
    with pytest.raises(type(reference_exception), match=str(reference_exception)):
        hermite_basis(x=1.0)

    # Test 2: wrong property setting for ``time_space_symmetry``
    hermite_basis = HermiteFunctionBasis(
        n=0,
        alpha=alpha,
        time_space_x_center=time_space_x_center,
        time_space_symmetry="none",
    )
    # one successful computation is carried out to trigger the internal validation
    assert not hermite_basis.is_fully_validated
    hermite_basis(x=1.0)
    assert hermite_basis.is_fully_validated
    # overwriting ``time_space_symmetry`` should reset the internal validation ...
    hermite_basis.time_space_symmetry = "odd"
    assert not hermite_basis.is_fully_validated
    # ... which then should cause an error at the next computation
    with pytest.raises(type(reference_exception), match=str(reference_exception)):
        hermite_basis(x=1.0)

    return


@pytest.mark.parametrize("method", TEST_HERMITE_FUNCTION_COMPUTATION_METHODS)
def test_hermite_function_class_raises_error_for_invalid_x_omega_in_call(
    method: HermiteFunctionMethod,
) -> None:
    """
    This test checks whether the class :class:`HermiteFunctionBasis` raises an error
    when either both ``x`` and ``omega`` or neither of them are provided for the
    ``__call__`` or ``eval`` method.

    """

    reference_exception = ValueError(
        "Either 'x' or 'omega' must be given but not both or none of them."
    )
    # the Hermite function evaluator is set up
    hermite_basis_evaluator, kwargs = setup_hermite_basis_evaluator(
        method=method,
        n=1,
        alpha=1.0,
        time_space_x_center=None,
        time_space_symmetry=None,
    )

    # Test 0: neither ``x`` nor ``omega`` are provided
    with pytest.raises(type(reference_exception), match=str(reference_exception)):
        hermite_basis_evaluator(
            x=None,
            omega=None,
            **kwargs,
        )

    # Test 1: both ``x`` and ``omega`` are provided
    with pytest.raises(type(reference_exception), match=str(reference_exception)):
        hermite_basis_evaluator(
            x=1.0,
            omega=1.0,
            **kwargs,
        )

    return


@pytest.mark.parametrize("method", TEST_HERMITE_FUNCTION_COMPUTATION_METHODS)
@pytest.mark.parametrize(
    "test_specification",
    [
        HermiteFunctionOutputFormatTestSpecs(  # Test 0: even n, no symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 1: even n, no symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 2: even n, even symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 3: even n, odd symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=5,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 4: odd n, no symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 5: odd n, no symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 6: odd n, even symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 7: odd n, odd symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=None,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 8: even n, no symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 9: even n, no symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 10: even n, even symmetry, no shift  # noqa: E501
            n=10,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 11: even n, odd symmetry, no shift
            n=10,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=5,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 12: odd n, no symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 13: odd n, no symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 14: odd n, even symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 15: odd n, odd symmetry, no shift
            n=11,
            alpha=1.0,
            time_space_x_center=0.0,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.complex128,
                frequency_dtype=np.float64,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 16: even n, no symmetry, +shift
            n=10,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 17: even n, no symmetry, +shift
            n=10,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 18: even n, even symmetry, +shift
            n=10,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 19: even n, odd symmetry, +shift
            n=10,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=5,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 20: odd n, no symmetry, +shift
            n=11,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 21: odd n, no symmetry, +shift
            n=11,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 22: odd n, even symmetry, +shift
            n=11,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 23: odd n, odd symmetry, +shift
            n=11,
            alpha=1.0,
            time_space_x_center=10.0,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 24: even n, no symmetry, -shift
            n=10,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 25: even n, no symmetry, -shift
            n=10,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=11,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 26: even n, even symmetry, -shift
            n=10,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 27: even n, odd symmetry, -shift
            n=10,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=5,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 28: odd n, no symmetry, -shift
            n=11,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry=None,
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 29: odd n, no symmetry, -shift
            n=11,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry="none",
            expected_output_format=HermiteOutputFormat(
                num_columns=12,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 30: odd n, even symmetry, -shift
            n=11,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry="even",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.float64,
                frequency_dtype=np.complex128,
            ),
        ),
        HermiteFunctionOutputFormatTestSpecs(  # Test 31: odd n, odd symmetry, -shift
            n=11,
            alpha=1.0,
            time_space_x_center=-10.0,
            time_space_symmetry="odd",
            expected_output_format=HermiteOutputFormat(
                num_columns=6,
                time_space_dtype=np.complex128,
                frequency_dtype=np.complex128,
            ),
        ),
    ],
)
def test_hermite_function_class_output_format(
    test_specification: HermiteFunctionOutputFormatTestSpecs,
    method: HermiteFunctionMethod,
) -> None:
    """
    This test checks whether the output format of the ``__call__`` and ``eval`` method
    of the class :class:`HermiteFunctionBasis` is correct.

    """

    # the Hermite function evaluator is set up
    hermite_basis_evaluator, kwargs = setup_hermite_basis_evaluator(
        method=method,
        n=test_specification.n,
        alpha=test_specification.alpha,
        time_space_x_center=test_specification.time_space_x_center,
        time_space_symmetry=test_specification.time_space_symmetry,
    )

    # the output is computed (twice in a loop to test both scalar and Array inputs)
    x_or_omega_specs: List[Tuple[Union[float, ArrayLike], int]] = [
        (0.0, 1),  # value, expected number of rows
        (np.array([-1.0, 0.0, 1.0]), 3),
    ]
    for x_or_omega_val, expected_num_rows in x_or_omega_specs:
        # first, the output for the time/space domain is computed and checked
        time_space_output = hermite_basis_evaluator(
            x=x_or_omega_val,  # type: ignore
            **kwargs,  # type: ignore
        )

        assert time_space_output.shape == (
            expected_num_rows,
            test_specification.expected_output_format.num_columns,
        )
        assert (
            time_space_output.dtype
            == test_specification.expected_output_format.time_space_dtype
        )

        # then, the output for the frequency domain is computed and checked
        frequency_output = hermite_basis_evaluator(
            omega=x_or_omega_val,  # type: ignore
            **kwargs,  # type: ignore
        )

        assert frequency_output.shape == (
            expected_num_rows,
            test_specification.expected_output_format.num_columns,
        )
        assert (
            frequency_output.dtype
            == test_specification.expected_output_format.frequency_dtype
        )

    return


@pytest.mark.parametrize("method", TEST_HERMITE_FUNCTION_COMPUTATION_METHODS)
@pytest.mark.parametrize("time_space_x_center", TEST_TIME_SPACE_X_CENTERS)
@pytest.mark.parametrize("alpha", TEST_SCALES_ALPHA)
@pytest.mark.parametrize("n", FOURIER_TRANSFORM_TEST_ORDERS)
def test_hermite_function_continuous_fourier_transform(
    n: int,
    alpha: float,
    time_space_x_center: Optional[float],
    method: HermiteFunctionMethod,
) -> None:
    """
    This test checks whether the Continuous Fourier transform of the Hermite functions
    computed by the ``__call__`` and ``eval`` method of the class
    :class:`HermiteFunctionBasis` is correct.

    For this test, the analytical Continuous Fourier transform is computed and compared
    to the Continuous Fourier transform obtained from a numerical Discrete Fourier
    transform.

    """

    # first, the adequate x-values for the time/space domain are obtained
    # given that the Hermite function of order ``n`` has ``n`` roots and thus ``n + 1``
    # oscillations, the number of x-values is set to ``n + 1`` times the number of
    # values per oscillation (plus a safety margin for the outermost oscillations);
    # with this number of x-values the span from the leftmost to the rightmost numerical
    # fadeout point of the Hermite functions is covered
    x_fadeouts = approximate_hermite_funcs_fadeout_x(
        n=n,
        alpha=alpha,
        x_center=time_space_x_center,
    )
    num_x_values = int(
        (1.0 + FOURIER_TRANSFORM_TEST_X_VALUES_SAFETY_MARGIN)
        * (n + 1)
        * FOURIER_TRANSFORM_TEST_NUM_X_VALUES_PER_ORDER
    )
    x_values = np.linspace(
        start=x_fadeouts[0],
        stop=x_fadeouts[1],
        num=num_x_values,
    )

    # the angular frequencies corresponding to these x-values are computed
    omega_values = angular_frequency_grid(
        n=num_x_values,
        d=grid_spacing(x=x_values),
    )

    # the test is carried out for the four different time/space symmetries
    for time_space_symmetry in [None, "none", "even", "odd"]:
        # the Hermite function evaluator is set up
        hermite_basis_evaluator, kwargs = setup_hermite_basis_evaluator(
            method=method,
            n=n,
            alpha=alpha,
            time_space_x_center=time_space_x_center,
            time_space_symmetry=time_space_symmetry,  # type: ignore
        )

        # then, the Hermite functions for the time/space domain are computed ...
        time_space_hermite_funcs = hermite_basis_evaluator(x=x_values, **kwargs)
        # ... followed by their analytical Continuous Fourier transforms
        cft_analytical = hermite_basis_evaluator(omega=omega_values, **kwargs)

        # now, the numerical Continuous Fourier transform is computed for each order
        # (column) of the time/space Hermite functions and compared to the analytical
        # counterpart
        for iter_j in range(0, time_space_hermite_funcs.shape[1]):
            # the numerical Continuous Fourier transform is computed from the numerical
            # Discrete Fourier transform
            signal = TimeSpaceSignal(
                x=x_values,
                y=time_space_hermite_funcs[::, iter_j],
            )
            discrete_ft_numerical = discrete_ft(
                signal=signal,
                norm="ortho",
            )
            cft_numerical = convert_discrete_to_continuous_ft(
                dft=discrete_ft_numerical,
            )

            # the numerical and analytical Continuous Fourier transforms are compared
            assert np.allclose(
                cft_analytical[::, iter_j].real,
                cft_numerical.real,
                atol=FOURIER_TRANSFORM_TEST_ATOL,
                rtol=FOURIER_TRANSFORM_TEST_RTOL,
            )
            assert np.allclose(
                cft_analytical[::, iter_j].imag,
                cft_numerical.imag,
                atol=FOURIER_TRANSFORM_TEST_ATOL,
                rtol=FOURIER_TRANSFORM_TEST_RTOL,
            )
