"""
This test suite implements the tests for the module :mod:`hermite_functions._func_interface`.

"""  # noqa: E501

# === Imports ===

import json
import os
from array import array as PythonArray
from dataclasses import dataclass
from typing import Generator, Literal, Optional, Type, Union

import numpy as np
import pytest
from pandas import Series as PandasSeries

from robust_hermite_ft import hermite_function_basis, single_hermite_function
from robust_hermite_ft.hermite_functions._func_interface import (
    _get_validated_hermite_function_input,
    _is_data_linked,
)

from ..reference_files.generate_hermfunc_references import (
    FILE_DIRECTORY,
    METADATA_FILENAME,
    HermiteFunctionsParameters,
    ReferenceHermiteFunctionsMetadata,
)
from .utils import (
    ALL_HERMITE_IMPLEMENTATIONS,
    HermiteFunctionBasisImplementations,
    setup_hermite_function_basis_implementations,
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


# === Tests ===


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
def test_is_data_linked_identified_correctly_after_hermite_functions_input_validation(
    original: np.ndarray,
    expected: Union[bool, Literal["auto"]],
) -> None:
    """
    This test checks whether the function :func:`_is_data_lined` correctly identifies
    whether the input data is copied after the same process as for the input validation
    of the Hermite functions implementations.

    For indexable data types (lists, tuples, NumPy arrays, Pandas Series,
    Python arrays), the function dynamically checks whether the data is linked or copied
    by modifying one of the entries in the modified array and checking whether the
    original array is also modified.

    """

    # the input validation is called and the data copying is checked
    modified = _get_validated_hermite_function_input(
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("array_like_type", [np.ndarray, PandasSeries, PythonArray])
def test_centered_hermite_functions_do_not_modify_x_values(
    array_like_type: Type,
    dtype: Type,
) -> None:
    """
    This test checks whether the function :func:`hermite_function_basis` does not modify
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
    hermite_function_basis(
        x=x_values,
        n=1,
        alpha=1.0,
        x_center=20.0,
        workers=1,
    )

    # the x-values are checked to be unchanged
    assert np.array_equal(np.asarray(x_values), x_values_original)


@pytest.mark.parametrize("x_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("x_center", [-10.0, 0.0, None, 10.0])
@pytest.mark.parametrize("implementation", ALL_HERMITE_IMPLEMENTATIONS)
def test_dilated_hermite_function_basis(
    reference_dilated_hermite_function_basis: Generator[
        ReferenceHermiteFunctionBasis, None, None
    ],
    implementation: HermiteFunctionBasisImplementations,
    x_center: Optional[float],
    x_dtype: Type,
) -> None:
    """
    This test checks the implementation of the function
    :func:`slow_hermite_function_basis` against the symbolic implementation of the
    Hermite functions.

    The ``x_center`` is tested in a hacky manner by shifting the x-values and calling
    the function with the shifted values and the center. With this, the function - given
    that everything runs correctly - should be invariant to the centering and always
    return the reference values.

    """

    # the reference values are loaded from the files and compared with the numerical
    # results
    for reference in reference_dilated_hermite_function_basis:
        # the numerical implementation is parametrized and called
        func, kwargs = setup_hermite_function_basis_implementations(
            implementation=implementation
        )
        x_center_for_shift = x_center if x_center is not None else 0.0
        numerical_herm_func_basis = func(
            x=(reference.x_values + x_center_for_shift).astype(x_dtype),  # type: ignore
            n=reference.n,
            alpha=reference.alpha,
            x_center=x_center,
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
        HermiteFunctionBasisImplementations.CYTHON_SINGLE,
        HermiteFunctionBasisImplementations.CYTHON_PARALLEL,
        HermiteFunctionBasisImplementations.NUMPY_SINGLE,
        HermiteFunctionBasisImplementations.NUMBA_SINGLE,
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_dilated_hermite_function_basis_orthonormal_and_bounded(
    alpha: float,
    implementation: HermiteFunctionBasisImplementations,
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
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation
    )
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


@pytest.mark.parametrize("x_center", [None, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("implementation", ALL_HERMITE_IMPLEMENTATIONS)
def test_hermite_functions_work_identically_for_all_x_types(
    implementation: HermiteFunctionBasisImplementations,
    alpha: float,
    x_center: Optional[float],
) -> None:
    """
    This test checks whether the Hermite functions can be evaluated for all possible
    types that are supported for ``x``, i.e.,

    - NumPy arrays
    - lists
    - tuples
    - Pandas Series
    - Python arrays
    - individual Python and NumPy floats

    """

    # the points for the check are set up as a NumPy-Array
    x_center_for_points = x_center if x_center is not None else 0.0
    x_points_to_check = np.linspace(start=-10.0, stop=10.0, num=1001)
    x_points_to_check /= alpha
    x_points_to_check += x_center_for_points

    # the Hermite function for the NumPy-Array x-points is computed as a reference
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation
    )
    hermite_basis_reference = func(
        x=x_points_to_check,  # type: ignore
        n=1,
        alpha=alpha,
        x_center=x_center,
        **kwargs,
    )

    # then, all the other types for ``x`` are checked
    # 1) a list
    hermite_basis_from_x_list = func(
        x=x_points_to_check.tolist(),  # type: ignore
        n=1,
        alpha=alpha,
        x_center=x_center,
        **kwargs,
    )
    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_list,
    )

    # 2) a tuple
    hermite_basis_from_x_tuple = func(
        x=tuple(x_points_to_check.tolist()),  # type: ignore
        n=1,
        alpha=alpha,
        x_center=x_center,
        **kwargs,
    )
    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_tuple,
    )

    # 3) a Pandas Series
    hermite_basis_from_x_pandas_series = func(
        x=PandasSeries(x_points_to_check.tolist()),  # type: ignore
        n=1,
        alpha=alpha,
        x_center=x_center,
        **kwargs,
    )
    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_pandas_series,
    )

    # 4) a Python Array
    hermite_basis_from_x_pyarray = func(
        x=PythonArray("d", x_points_to_check.tolist()),  # type: ignore
        n=1,
        alpha=alpha,
        x_center=x_center,
        **kwargs,
    )
    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_pyarray,
    )

    # 5) individual Python and NumPy floats
    for float_func in [float, np.float64]:
        hermite_basis_from_x_float = np.array(
            [
                func(
                    x=float_func(x_point),  # type: ignore
                    n=1,
                    alpha=alpha,
                    x_center=x_center,
                    **kwargs,
                )[0, ::]
                for x_point in x_points_to_check
            ]
        )
        assert np.array_equal(
            hermite_basis_reference,
            hermite_basis_from_x_float,
        )


@pytest.mark.parametrize("implementation", ALL_HERMITE_IMPLEMENTATIONS)
def test_hermite_functions_work_identically_for_all_n_alpha_x_center_types(
    implementation: HermiteFunctionBasisImplementations,
) -> None:
    """
    This test checks whether the Hermite functions can be evaluated for all possible
    types that are supported for ``n``, ``alpha``, and ``x_center``, i.e.,

    - Python and Numpy integers for ``n``
    - Python and  integers and floats for ``alpha`` and ``x_center``

    """

    # the reference values for ``n``, ``alpha``, and ``x_center`` are set up as Python
    # integers to avoid any float conversion issues
    n_ref_value = 11
    alpha_ref_value = 2
    x_center_ref_value = 10

    # the x-points for the check are set up as a NumPy-Array
    x_points_to_check = np.linspace(start=-10.0, stop=10.0, num=1001)
    x_points_to_check /= float(alpha_ref_value)
    x_points_to_check += float(x_center_ref_value)

    # the Hermite function for the x-points and Python integer parameters is computed as
    # a reference
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation
    )
    hermite_basis_reference = func(
        x=x_points_to_check,  # type: ignore
        n=n_ref_value,
        alpha=alpha_ref_value,
        x_center=x_center_ref_value,
        **kwargs,
    )

    # then, ``n`` is tested with a NumPy instead of a Python integer
    hermite_basis_from_n_numpy_int = func(
        x=x_points_to_check,  # type: ignore
        n=np.int64(n_ref_value),
        alpha=alpha_ref_value,
        x_center=x_center_ref_value,
        **kwargs,
    )
    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_n_numpy_int,
    )

    # afterwards, ``alpha`` is tested with a NumPy integer, a Python float, and a
    # NumPy float instead of a Python integer
    for alpha_conversion in [np.int64, float, np.float64]:
        hermite_basis_from_alpha_converted = func(
            x=x_points_to_check,  # type: ignore
            n=n_ref_value,
            alpha=alpha_conversion(alpha_ref_value),
            x_center=x_center_ref_value,
            **kwargs,
        )
        assert np.array_equal(
            hermite_basis_reference,
            hermite_basis_from_alpha_converted,
        )

    # finally, ``x_center`` is tested with a NumPy integer, a Python float, and a
    # NumPy float instead of a Python integer
    for x_center_conversion in [np.int64, float, np.float64]:
        hermite_basis_from_x_center_converted = func(
            x=x_points_to_check,  # type: ignore
            n=n_ref_value,
            alpha=alpha_ref_value,
            x_center=x_center_conversion(x_center_ref_value),
            **kwargs,
        )
        assert np.array_equal(
            hermite_basis_reference,
            hermite_basis_from_x_center_converted,
        )
