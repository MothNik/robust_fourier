"""
This test suite implements the tests for the module :mod:`hermite_functions._func_interface`
and also the ``__call__`` method of the class :class:`HermiteFunctionBasis` from the
module :mod:`hermite_functions._class_interface`.

"""  # noqa: E501

# === Imports ===

import json
import os
from array import array as PythonArray
from dataclasses import dataclass
from math import sqrt as pysqrt
from typing import Generator, Optional, Type

import numpy as np
import pytest
from pandas import Series as PandasSeries

from robust_fourier import (
    hermite_approx,
    hermite_function_vander,
    single_hermite_function,
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

# the absolute and relative tolerances for the symbolic Hermite function tests for
# 1) single precision
SYMBOLIC_TEST_HERMITE_FUNC_FLOAT32_ATOL = 1e-5
SYMBOLIC_TEST_HERMITE_FUNC_FLOAT32_RTOL = 1e-5
# 2) double precision
SYMBOLIC_TEST_HERMITE_FUNC_FLOAT64_ATOL = 1e-13
SYMBOLIC_TEST_HERMITE_FUNC_FLOAT64_RTOL = 1e-13

# the number of data points per Hermite function order for the orthonormality test
ORTHONORMALITY_TEST_NUM_X_PER_ORDER = 25
# the safety margin for the number of x-values to account for the outermost
# oscillations of the Hermite functions for the orthonormality test
ORTHONORMALITY_TEST_X_VALUES_SAFETY_MARGIN = 0.1
# the absolute tolerance for the orthonormality test
ORTHONORMALITY_TEST_ATOL = 1e-13
# the pre-factor for the bound given by the Cramér's inequality
CRAMERS_INEQUALITY_TEST_FACTOR = np.pi ** (-0.25)
# the relative tolerance for the Cramér's inequality
CRAMERS_INEQUALITY_TEST_RTOL = 1e-13

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
    hermite_func_vander: np.ndarray
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
                hermite_func_vander=reference_hermite_function_basis,
                ns_for_single_function=parameters.ns_for_single_function,
            )

    # finally, the iterator is returned
    return reference_iterator()


# === Tests ===


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("array_like_type", [np.ndarray, PandasSeries, PythonArray])
def test_centered_hermite_functions_do_not_modify_x_values(
    array_like_type: Type,
    dtype: Type,
) -> None:
    """
    This test checks whether the function :func:`hermite_function_vander` does not
    modify the input x-values when the center is set.

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
    hermite_function_vander(
        x=x_values,
        n=1,
        alpha=1.0,
        x_center=20.0,
    )

    # the x-values are checked to be unchanged
    assert np.array_equal(np.asarray(x_values), x_values_original)

    return


@pytest.mark.parametrize("x_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("x_center", [-10.0, 0.0, None, 10.0])
@pytest.mark.parametrize("implementation", ALL_HERMITE_IMPLEMENTATIONS)
def test_dilated_hermite_function_basis_against_symbolic_reference(
    reference_dilated_hermite_function_basis: Generator[
        ReferenceHermiteFunctionBasis, None, None
    ],
    implementation: HermiteFunctionBasisImplementations,
    x_center: Optional[float],
    x_dtype: Type,
) -> None:
    """
    This test checks the implementations of the Hermite functions against a symbolic
    reference implementation.

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
            implementation=implementation,
            n=reference.n,
            alpha=reference.alpha,
            x_center=x_center,
        )
        x_center_for_shift = x_center if x_center is not None else 0.0
        numerical_herm_func_basis = func(
            x=(reference.x_values + x_center_for_shift).astype(x_dtype),  # type: ignore
            **kwargs,
        )

        # the reference values are compared with the numerical results
        # NOTE: the numerical tolerance has to be based on the data type of the x-values
        #       because the build-up of rounding errors is quite pronounced due to the
        #       x-values being involved in the recursions
        if x_dtype == np.float32:
            atol = SYMBOLIC_TEST_HERMITE_FUNC_FLOAT32_ATOL
            rtol = SYMBOLIC_TEST_HERMITE_FUNC_FLOAT32_RTOL
        else:
            atol = SYMBOLIC_TEST_HERMITE_FUNC_FLOAT64_ATOL
            rtol = SYMBOLIC_TEST_HERMITE_FUNC_FLOAT64_RTOL

        assert np.allclose(
            numerical_herm_func_basis,
            reference.hermite_func_vander,
            atol=atol,
            rtol=rtol,
        ), f"Failed for or n = {reference.n} and alpha = {reference.alpha}"

        return


@pytest.mark.parametrize("implementation", ALL_HERMITE_IMPLEMENTATIONS)
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_dilated_hermite_function_basis_orthonormal_and_bounded(
    alpha: float,
    implementation: HermiteFunctionBasisImplementations,
) -> None:
    """
    This test checks whether the dilated Hermite functions generated by the function
    :func:`slow_hermite_function_basis` are orthonormal and bounded by the Cramér's
    inequality :math:`\\frac{1}{\\sqrt[4]{\\pi\\cdot\\alpha^{2}}}`.

    To prove the first point, the dot product of the Hermite basis function matrix with
    itself is computed which - after scaling by the step size in ``x`` - should be the
    identity matrix.
    For the second point, the absolute values of the Hermite functions are checked to be
    within the bounds (plus some numerical tolerance).

    """

    # the number of orders is high (n = 1000) to ensure that the orthonormality is
    # thoroughly tested
    n = 1_000

    # first, the adequate x-values for the time/space domain are obtained
    # given that the Hermite function of order ``n`` has ``n`` roots and thus ``n + 1``
    # oscillations, the number of x-values is set to ``n + 1`` times the number of
    # values per oscillation (plus a safety margin for the outermost oscillations);
    # with this number of x-values the span from the leftmost to the rightmost numerical
    # fadeout point of the Hermite functions is covered
    x_fadeouts = hermite_approx.x_fadeout(n=n, alpha=alpha)
    num_x_values = int(
        (1.0 + ORTHONORMALITY_TEST_X_VALUES_SAFETY_MARGIN)
        * (n + 1)
        * ORTHONORMALITY_TEST_NUM_X_PER_ORDER
    )
    x_values = np.linspace(
        start=x_fadeouts[0],
        stop=x_fadeouts[1],
        num=num_x_values,
    )

    # the function is parametrized and called to evaluate the Hermite functions
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation,
        n=n,
        alpha=alpha,
        x_center=None,
    )
    hermite_basis = func(
        x=x_values,  # type: ignore
        **kwargs,
    )

    # then, the are tested for being bounded by the given values (but reach it)
    # the bounds are calculated for the given alpha
    bound = CRAMERS_INEQUALITY_TEST_FACTOR / pysqrt(alpha)
    assert np.all(np.abs(hermite_basis) <= bound * (1.0 + CRAMERS_INEQUALITY_TEST_RTOL))
    assert np.any(np.abs(hermite_basis) >= bound * (1.0 - CRAMERS_INEQUALITY_TEST_RTOL))

    # the orthonormality is tested by computing the dot product of the Hermite basis
    # function matrix with itself, i.e., ``X.T @ X`` which - after scaling by the step
    # size in ``x`` - should be the identity matrix
    delta_x = (x_values[-1] - x_values[0]) / (x_values.size - 1)
    dot_product = (hermite_basis.T @ hermite_basis) * delta_x

    # this product is now compared with the identity matrix
    assert np.allclose(dot_product, np.eye(n + 1), atol=ORTHONORMALITY_TEST_ATOL)

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
                reference.hermite_func_vander[::, n],
                atol=SYMBOLIC_TEST_HERMITE_FUNC_FLOAT64_ATOL,
                rtol=SYMBOLIC_TEST_HERMITE_FUNC_FLOAT64_RTOL,
            ), f"For n = {n} and alpha = {reference.alpha}"

    return


@pytest.mark.parametrize("x_center", [None, 0.0, 1.0, -1.0])
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
    x_points_to_check = np.linspace(start=-10.0, stop=10.0, num=1_001)
    x_points_to_check *= alpha
    x_points_to_check += x_center_for_points

    # the Hermite functions for the NumPy-Array x-points are computed as a reference
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation,
        n=1,
        alpha=alpha,
        x_center=x_center,
    )
    hermite_basis_reference = func(
        x=x_points_to_check,  # type: ignore
        **kwargs,
    )

    # then, all the other types for ``x`` are checked
    # 1) a list
    hermite_basis_from_x_list = func(
        x=x_points_to_check.tolist(),  # type: ignore
        **kwargs,
    )

    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_list,
    )

    del hermite_basis_from_x_list

    # 2) a tuple
    hermite_basis_from_x_tuple = func(
        x=tuple(x_points_to_check.tolist()),  # type: ignore
        **kwargs,
    )

    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_tuple,
    )

    del hermite_basis_from_x_tuple

    # 3) a Pandas Series
    hermite_basis_from_x_pandas_series = func(
        x=PandasSeries(x_points_to_check.tolist()),  # type: ignore
        **kwargs,
    )

    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_pandas_series,
    )

    del hermite_basis_from_x_pandas_series

    # 4) a Python Array
    hermite_basis_from_x_pyarray = func(
        x=PythonArray("d", x_points_to_check.tolist()),  # type: ignore
        **kwargs,
    )

    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_x_pyarray,
    )

    del hermite_basis_from_x_pyarray

    # 5) individual Python and NumPy floats
    for float_func in [float, np.float64]:
        hermite_basis_from_x_float = np.array(
            [
                func(
                    x=float_func(x_point),  # type: ignore
                    **kwargs,
                )[0, ::]
                for x_point in x_points_to_check
            ]
        )

        assert np.array_equal(
            hermite_basis_reference,
            hermite_basis_from_x_float,
        )

        del hermite_basis_from_x_float

    return


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
    x_points_to_check = np.linspace(start=-10.0, stop=10.0, num=1_001)
    x_points_to_check *= float(alpha_ref_value)
    x_points_to_check += float(x_center_ref_value)

    # the Hermite functions for the x-points and Python integer parameters are computed
    # as a reference
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation,
        n=n_ref_value,
        alpha=alpha_ref_value,
        x_center=x_center_ref_value,
    )
    hermite_basis_reference = func(
        x=x_points_to_check,  # type: ignore
        **kwargs,
    )

    # then, ``n`` is tested with a NumPy instead of a Python integer
    func, kwargs = setup_hermite_function_basis_implementations(
        implementation=implementation,
        n=np.int64(n_ref_value),
        alpha=alpha_ref_value,
        x_center=x_center_ref_value,
    )
    hermite_basis_from_n_numpy_int = func(
        x=x_points_to_check,  # type: ignore
        **kwargs,
    )

    assert np.array_equal(
        hermite_basis_reference,
        hermite_basis_from_n_numpy_int,
    )

    del hermite_basis_from_n_numpy_int

    # afterwards, ``alpha`` is tested with a NumPy integer, a Python float, and a
    # NumPy float instead of a Python integer
    for alpha_conversion in [np.int64, float, np.float64]:
        func, kwargs = setup_hermite_function_basis_implementations(
            implementation=implementation,
            n=n_ref_value,
            alpha=alpha_conversion(alpha_ref_value),
            x_center=x_center_ref_value,
        )
        hermite_basis_from_alpha_converted = func(
            x=x_points_to_check,  # type: ignore
            **kwargs,
        )

        assert np.array_equal(
            hermite_basis_reference,
            hermite_basis_from_alpha_converted,
        )

        del hermite_basis_from_alpha_converted

    # finally, ``x_center`` is tested with a NumPy integer, a Python float, and a
    # NumPy float instead of a Python integer
    for x_center_conversion in [np.int64, float, np.float64]:
        func, kwargs = setup_hermite_function_basis_implementations(
            implementation=implementation,
            n=n_ref_value,
            alpha=alpha_ref_value,
            x_center=x_center_conversion(x_center_ref_value),
        )
        hermite_basis_from_x_center_converted = func(
            x=x_points_to_check,  # type: ignore
            **kwargs,
        )

        assert np.array_equal(
            hermite_basis_reference,
            hermite_basis_from_x_center_converted,
        )

        del hermite_basis_from_x_center_converted

    return
