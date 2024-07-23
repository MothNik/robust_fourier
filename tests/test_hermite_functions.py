"""
This test suite implements the tests for the module :mod:`hermite_functions`.

"""

# === Imports ===

import json
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, Tuple, Type, Union

import numpy as np
import pytest

from robust_hermite_ft import (
    hermite_function_basis,
    single_hermite_function,
    slow_hermite_function_basis,
)

from .reference_files.generate_hermfunc_references import (
    FILE_DIRECTORY,
    METADATA_FILENAME,
    HermiteFunctionsParameters,
    ReferenceHermiteFunctionsMetadata,
)

# === Constants ===

# the absolute and relative tolerances for the Hermite function tests for
# 1) single precision
HERMITE_FUNC_FLOAT32_ATOL = 1e-5
HERMITE_FUNC_FLOAT32_RTOL = 1e-5
# 2) double precision
HERMITE_FUNC_FLOAT64_ATOL = 1e-13
HERMITE_FUNC_FLOAT64_RTOL = 1e-13

# the factor for x-bound computations from alpha for the orthonormality test
ORTHONORMALITY_BOUND_FACTOR = 50.0
# the number of data points for the orthonormality test
ORTHONORMALITY_NUM_X = 50_001
# the absolute tolerance for the orthonormality test
ORTHONORMALITY_ATOL = 1e-13
# the pre-factor for the bound given by the Cramér's inequality
CRAMERS_INEQUALITY_FACTOR = np.pi ** (-0.25)
# the tolerance for the Cramér's inequality
CRAMERS_INEQUALITY_TOLERANCE = 1e-13

# === Models ===

# an Enum class for the different implementations of the Hermite functions


class HermiteFunctionImplementations(str, Enum):
    CYTHON_SINGLE = auto()
    CYTHON_PARALLEL = auto()
    NUMPY_SINGLE = auto()
    NUMBA_SINGLE = auto()


# a dataclass for the reference values of the dilated Hermite functions


@dataclass
class ReferenceHermiteFunctionBasis:
    """
    Contains the reference values for the dilated Hermite functions.

    """

    n: int
    alpha: float
    x_values: np.ndarray
    hermite_function_basis: np.ndarray
    ns_for_single_function: list[int]


# === Fixtures ===


@pytest.fixture
def reference_dilated_hermite_function_basis() -> (
    Generator[ReferenceHermiteFunctionBasis, None, None]
):
    """
    Loads the reference values for the dilated Hermite functions.

    Returns
    -------
    reference_dilated_hermite_function_basis : :class:`Generator`
        The reference values for the dilated Hermite functions.

    """  # noqa: E501

    # first, the metadata is loaded ...
    metadata_filepath = os.path.join(FILE_DIRECTORY, METADATA_FILENAME)
    with open(metadata_filepath, "r") as metadata_file:
        reference_metadata = json.load(metadata_file)

    # ... and cast into the corresponding dataclass for easier handling
    # NOTE: this is not really necessary but also serves as a sanity check for the
    #       metadata
    filename_parameters_mapping = {
        filename: HermiteFunctionsParameters(**parameters)
        for filename, parameters in reference_metadata["parameters_mapping"].items()
    }
    reference_metadata = ReferenceHermiteFunctionsMetadata(
        parameters_mapping=filename_parameters_mapping,
        computation_time_mapping=reference_metadata["computation_time_mapping"],
        num_digits=reference_metadata["num_digits"],
        x_values=np.array(reference_metadata["x_values"], dtype=np.float64),
    )

    # then, an iterator is created that yields the reference values from the files in
    # a lazy fashion
    def reference_iterator() -> Generator[ReferenceHermiteFunctionBasis, None, None]:
        for (
            filename,
            parameters,
        ) in reference_metadata.parameters_mapping.items():
            # the reference values are loaded
            filepath = os.path.join(FILE_DIRECTORY, filename)
            reference_hermite_function_basis = np.load(filepath)

            yield ReferenceHermiteFunctionBasis(
                n=parameters.n,
                alpha=parameters.alpha,
                x_values=np.array(reference_metadata.x_values, dtype=np.float64),
                hermite_function_basis=reference_hermite_function_basis,
                ns_for_single_function=parameters.ns_for_single_function,
            )

    # finally, the iterator is returned
    return reference_iterator()


# === Auxiliary Functions ===

# a function to set up the Hermite function implementations


def setup_hermite_function_implementations(
    implementation: HermiteFunctionImplementations,
) -> Tuple[
    Union[
        Callable[[np.ndarray, int, float, int], np.ndarray],
        Callable[[np.ndarray, int, float, bool], np.ndarray],
    ],
    Dict[str, Any],
]:
    """
    Sets up the Hermite function implementations by selecting the correct function and
    its keyword arguments.

    """

    if implementation == HermiteFunctionImplementations.CYTHON_SINGLE:
        return hermite_function_basis, dict(workers=1)

    elif implementation == HermiteFunctionImplementations.CYTHON_PARALLEL:
        return hermite_function_basis, dict(workers=-1)

    elif implementation == HermiteFunctionImplementations.NUMPY_SINGLE:
        return slow_hermite_function_basis, dict(jit=False)

    elif implementation == HermiteFunctionImplementations.NUMBA_SINGLE:
        return slow_hermite_function_basis, dict(jit=True)

    else:
        raise AssertionError(f"Unknown implementation: {implementation}")


# === Tests ===


@pytest.mark.parametrize("x_dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "implementation",
    [
        HermiteFunctionImplementations.CYTHON_SINGLE,
        HermiteFunctionImplementations.CYTHON_PARALLEL,
        HermiteFunctionImplementations.NUMPY_SINGLE,
        HermiteFunctionImplementations.NUMBA_SINGLE,
    ],
)
def test_dilated_hermite_function_basis(
    reference_dilated_hermite_function_basis: Generator[
        ReferenceHermiteFunctionBasis, None, None
    ],
    implementation: HermiteFunctionImplementations,
    x_dtype: Type,
) -> None:
    """
    This test checks the implementation of the function
    :func:`slow_hermite_function_basis` against the symbolic implementation of the
    Hermite functions.

    """

    # the reference values are loaded from the files and compared with the numerical
    # results
    for reference in reference_dilated_hermite_function_basis:
        # the numerical implementation is parametrized and called
        func, kwargs = setup_hermite_function_implementations(
            implementation=implementation
        )
        numerical_herm_func_basis = func(
            x=reference.x_values.astype(x_dtype),  # type: ignore
            n=reference.n,
            alpha=reference.alpha,
            **kwargs,
        )

        # the reference values are compared with the numerical results
        # NOTE: the numerical tolerance has to be based on the data type of the x-values
        #       because the build-up of rounding errors is quite pronounced due to the
        #       x-values being involved in the recursions
        if x_dtype == np.float32:
            atol = HERMITE_FUNC_FLOAT32_ATOL
            rtol = HERMITE_FUNC_FLOAT32_RTOL
        else:
            atol = HERMITE_FUNC_FLOAT64_ATOL
            rtol = HERMITE_FUNC_FLOAT64_RTOL

        assert np.allclose(
            numerical_herm_func_basis,
            reference.hermite_function_basis,
            atol=atol,
            rtol=rtol,
        ), f"For n = {reference.n} and alpha = {reference.alpha}"


@pytest.mark.parametrize(
    "implementation",
    [
        HermiteFunctionImplementations.CYTHON_SINGLE,
        HermiteFunctionImplementations.CYTHON_PARALLEL,
        HermiteFunctionImplementations.NUMPY_SINGLE,
        HermiteFunctionImplementations.NUMBA_SINGLE,
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_dilated_hermite_function_basis_orthonormal_and_bounded(
    alpha: float,
    implementation: HermiteFunctionImplementations,
) -> None:
    """
    This test checks whether the dilated Hermite functions generated by the function
    :func:`slow_hermite_function_basis` are orthonormal and bounded by the Cramér's
    inequality :math:`\\frac{\\sqrt{\\alpha}}{\\pi^{-\\frac{1}{4}}}`.

    To prove the first point, the dot product of the Hermite basis function matrix with
    itself is computed which - after scaling by the step size in ``x`` - should be the
    identity matrix.
    For the second point, the absolute values of the Hermite functions are checked to be
    within the bounds (plus some numerical tolerance).

    """

    # the number of orders is high (n = 1000) to ensure that the orthonormality is
    # thoroughly tested
    n = 1_000

    # the x-values need to be set up
    # they need to be sampled densely within a wide interval in which the Hermite
    # functions have already decayed to zero while the central bump (that could
    # potentially exceed the bounds) is still present for n <= 1000
    x_bound = ORTHONORMALITY_BOUND_FACTOR / alpha
    x_values = np.linspace(
        start=-x_bound,
        stop=x_bound,
        num=ORTHONORMALITY_NUM_X,
    )

    # the function is parametrized and called to evaluate the Hermite functions
    func, kwargs = setup_hermite_function_implementations(implementation=implementation)
    hermite_basis = func(
        x=x_values,  # type: ignore
        n=n,
        alpha=alpha,
        **kwargs,
    )

    # then, the are tested for being bounded by the given values (but reach it)
    # the bounds are calculated for the given alpha
    bound = CRAMERS_INEQUALITY_FACTOR * np.sqrt(alpha)
    assert np.all(np.abs(hermite_basis) <= bound + CRAMERS_INEQUALITY_TOLERANCE)
    assert np.any(np.abs(hermite_basis) >= bound - CRAMERS_INEQUALITY_TOLERANCE)

    # the orthonormality is tested by computing the dot product of the Hermite basis
    # function matrix with itself, i.e., ``X.T @ X`` which - after scaling by the step
    # size in ``x`` - should be the identity matrix
    delta_x = (x_values[-1] - x_values[0]) / (x_values.size - 1)
    dot_product = (hermite_basis.T @ hermite_basis) * delta_x

    # this product is now compared with the identity matrix
    assert np.allclose(dot_product, np.eye(n + 1), atol=ORTHONORMALITY_ATOL)


@pytest.mark.parametrize(
    "implementation",
    [
        HermiteFunctionImplementations.CYTHON_SINGLE,
        HermiteFunctionImplementations.CYTHON_PARALLEL,
        HermiteFunctionImplementations.NUMPY_SINGLE,
        HermiteFunctionImplementations.NUMBA_SINGLE,
    ],
)
@pytest.mark.parametrize(
    "x, n, alpha, exception",
    [
        (  # Test 0: x is complex
            1.0 + 1.0j,
            1,
            1.0,
            TypeError("Expected 'x' to be a float, int, or np.ndarray"),
        ),
        (  # Test 1: x is a list
            [1.0, 2.0, 3.0],
            1,
            1.0,
            TypeError("Expected 'x' to be a float, int, or np.ndarray"),
        ),
        (  # Test 2: x is a 2D array
            np.array([[1.0, 2.0, 3.0]]),
            1,
            1.0,
            ValueError("Expected 'x' to be 1-dimensional"),
        ),
        (  # Test 3: n is negative
            1.0,
            -1,
            1.0,
            ValueError("Expected 'n' to be a non-negative integer"),
        ),
        (  # Test 4: n is a float
            1.0,
            1.0,
            1.0,
            TypeError("Expected 'n' to be an integer"),
        ),
        (  # Test 5: alpha is zero
            1.0,
            1,
            0.0,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 6: alpha is negative
            1.0,
            1,
            -1.0,
            ValueError("Expected 'alpha' to be a positive number"),
        ),
        (  # Test 7: alpha is a complex number
            1.0,
            1,
            1.0 + 1.0j,
            TypeError("Expected 'alpha' to be a float or integer"),
        ),
    ],
)
def test_dilated_hermite_function_basis_invalid_input(
    x: np.ndarray,
    n: int,
    alpha: float,
    exception: Exception,
    implementation: HermiteFunctionImplementations,
) -> None:
    """
    This test checks whether the function :func:`hermite_function_basis` raises the
    correct exceptions for invalid input.

    """

    # the function is parametrized
    func, kwargs = setup_hermite_function_implementations(implementation=implementation)

    # the function is called and the exception is checked
    with pytest.raises(type(exception), match=str(exception)):
        func(
            x=x,  # type: ignore
            n=n,
            alpha=alpha,
            **kwargs,
        )

    return


def test_single_hermite_functions(
    reference_dilated_hermite_function_basis: Generator[
        ReferenceHermiteFunctionBasis, None, None
    ],
) -> None:
    """
    This test checks the implementation of the function
    :func:`single_hermite_function` against the symbolic implementation of the Hermite
    functions, but only for selected ones that aim to cover the most relevant cases
    given that the evaluation for a full basis would be way too expensive.

    """

    # the reference values are loaded from the files and compared with the numerical
    # results
    for reference in reference_dilated_hermite_function_basis:
        if len(reference.ns_for_single_function) < 1:
            continue

        # all the selected orders are tested
        for n in reference.ns_for_single_function:
            # the numerical implementation is called
            numerical_herm_func = single_hermite_function(
                x=reference.x_values,
                n=n,
                alpha=reference.alpha,
            )

            # the reference values are compared with the numerical results
            assert np.allclose(
                numerical_herm_func,
                reference.hermite_function_basis[::, n],
                atol=HERMITE_FUNC_FLOAT64_ATOL,
                rtol=HERMITE_FUNC_FLOAT64_RTOL,
            ), f"For n = {n} and alpha = {reference.alpha}"
