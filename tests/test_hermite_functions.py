"""
This test suite implements the tests for the module :mod:`hermite_functions`.

"""

# === Imports ===

import json
import os
import warnings
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pytest

from src.hermite_functions import (
    _dilated_hermite_function_basis,
    _slogabs_dilated_hermite_polynomial_basis,
)

from .reference_files.generate_hermpoly_references import (
    FILE_DIRECTORY,
    METADATA_FILENAME,
    HermitePolynomialParameters,
    ReferenceHermitePolynomialsMetadata,
)

# === Models ===


@dataclass
class ReferenceHermitePolynomialBasis:
    """
    Contains the reference values for the dilated Hermite polynomials.

    """

    n: int
    alpha: float
    x_values: np.ndarray
    hermite_polynomial_basis: np.ndarray


# === Fixtures ===


@pytest.fixture
def reference_dilated_hermite_polynomial_basis() -> (
    Generator[ReferenceHermitePolynomialBasis, None, None]
):
    """
    Loads the reference values for the dilated Hermite polynomials.

    Returns
    -------
    reference_dilated_hermite_polynomial_basis : :class:`Generator`
        The reference values for the dilated Hermite polynomials.

    """  # noqa: E501

    # first, the metadata is loaded ...
    metadata_filepath = os.path.join(FILE_DIRECTORY, METADATA_FILENAME)
    with open(metadata_filepath, "r") as metadata_file:
        reference_metadata = json.load(metadata_file)

    # ... and cast into the corresponding dataclass for easier handling
    # NOTE: this is not really necessary but also serves as a sanity check for the
    #       metadata
    filename_parameters_mapping = {
        filename: HermitePolynomialParameters(**parameters)
        for filename, parameters in reference_metadata[
            "filename_parameters_mapping"
        ].items()
    }
    reference_metadata = ReferenceHermitePolynomialsMetadata(
        filename_parameters_mapping=filename_parameters_mapping,
        num_digits=reference_metadata["num_digits"],
        x_values=np.array(reference_metadata["x_values"], dtype=np.float64),
    )

    # then, an iterator is created that yields the reference values from the files in
    # a lazy fashion
    def reference_iterator() -> Generator[ReferenceHermitePolynomialBasis, None, None]:
        for (
            filename,
            parameters,
        ) in reference_metadata.filename_parameters_mapping.items():
            # the reference values are loaded
            filepath = os.path.join(FILE_DIRECTORY, filename)
            reference_hermite_polynomial_basis = np.load(filepath)

            yield ReferenceHermitePolynomialBasis(
                n=parameters.n,
                alpha=parameters.alpha,
                x_values=np.array(reference_metadata.x_values, dtype=np.float64),
                hermite_polynomial_basis=reference_hermite_polynomial_basis,
            )

    # finally, the iterator is returned
    return reference_iterator()


# === Tests ===


def test_slogabs_dilated_hermite_polynomial_basis(
    reference_dilated_hermite_polynomial_basis: Generator[
        ReferenceHermitePolynomialBasis, None, None
    ],
) -> None:
    """
    This test checks the implementation of the function
    :func:`_slogabs_dilated_hermite_polynomial_basis` against the symbolic
    implementation of the Hermite polynomials.

    It evaluates the exponential of the logarithm of the numerical results which is not
    a step that should be taken in practice. Here, this is solely done to compare the
    results with the symbolic reference.

    """

    # the reference values are loaded from the files
    for reference in reference_dilated_hermite_polynomial_basis:
        # the numerical evaluation to test computes the results in the log space and
        # also provides the signs of the Hermite polynomials at the points x
        logabs_hermpoly_basis, signs_hermpoly_basis = (
            _slogabs_dilated_hermite_polynomial_basis(
                x=reference.x_values,
                n=reference.n,
                alpha=reference.alpha,
            )
        )
        # the logarithms have to be exponentiated and multiplied with the signs to
        # obtain the numerical Hermite polynomials
        # NOTE: overflow might happen for some calculations since this is not the
        #       intended way of using the function; therefore warnings are suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hermpoly_basis = signs_hermpoly_basis * np.exp(logabs_hermpoly_basis)

        # the reference values are compared with the numerical results
        assert np.allclose(hermpoly_basis, reference.hermite_polynomial_basis)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_dilated_hermite_function_basis_orthonormal_and_bounded(alpha: float) -> None:
    """
    This test checks whether the generated dilated Hermite functions are orthonormal
    and bounded by :math:`\\frac{\\pm\\pi^{-\\frac{1}{4}}{\\sqrt{\\alpha}}`.

    To prove the first point, the dot product of the Hermite basis function matrix with
    itself is computed which - after scaling by the step size in ``x`` - should be the
    identity matrix.
    For the second point, the absolute values of the Hermite functions are checked to be
    within the bounds (plus some numerical tolerance).

    """

    # the number of order is high (n = 500) to ensure that the orthonormality is
    # thoroughly tested
    n = 500

    # the x-values need to be set up
    # they need to be sampled densely within the interval [-35 * alpha, 35 * alpha]
    # because there they have already decayed to zero while the central bump (that could
    # potentially exceed the bounds) is still present for n <= 500
    x_values = np.linspace(
        start=-35.0 * alpha,
        stop=35.0 * alpha,
        num=40_001,
    )

    # the dilated Hermite functions are evaluated
    hermite_basis = _dilated_hermite_function_basis(
        x=x_values,
        n=n,
        alpha=alpha,
    )

    # then, the are tested for being bounded by the given values
    # the bounds are calculated for the given alpha
    bound = np.pi ** (-0.25) / np.sqrt(alpha)
    assert np.all(np.abs(hermite_basis) <= bound + 1e-10)

    # the orthonormality is tested by computing the dot product of the Hermite basis
    # function matrix with itself, i.e., ``X.T @ X`` which - after scaling by the step
    # size in ``x`` - should be the identity matrix
    delta_x = (x_values[-1] - x_values[0]) / (x_values.size - 1)
    dot_product = (hermite_basis.T @ hermite_basis) * delta_x

    # this product is now compared with the identity matrix
    assert np.allclose(dot_product, np.eye(n + 1), atol=1e-10)
