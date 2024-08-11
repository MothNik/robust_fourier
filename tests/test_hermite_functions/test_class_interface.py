"""
This test suite implements the tests for the module :mod:`hermite_functions._class_interface`.

"""  # noqa: E501

# === Imports ===

import numpy as np
import pytest

from robust_fourier import HermiteFunctionBasis, hermite_function_basis

# === Tests ===


def test_hermite_function_basis_properties_and_len() -> None:
    """
    Tests the properties setters and getters of the class :class:`HermiteFunctionBasis`
    as well as its :meth:`__len__` method.

    """

    # an instance of the Hermite function basis is initialised
    hermite_basis = HermiteFunctionBasis(
        n=0,
        alpha=1.0,
        x_center=None,
        jit=True,
    )

    # the properties are tested
    assert hermite_basis.n == 0
    assert hermite_basis.alpha == 1.0
    assert hermite_basis.x_center == 0.0
    assert hermite_basis.jit is True
    assert len(hermite_basis) == 1

    # the results of calling the Hermite function basis are tested against the expected
    # results
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        hermite_function_basis(
            x=x_reference,
            n=0,
            alpha=1.0,
            x_center=None,
            jit=True,
        ),
        hermite_basis(x=x_reference),
    )

    # the ``n`` property is set
    hermite_basis.n = 1

    # the properties are tested
    assert hermite_basis.n == 1
    assert hermite_basis.alpha == 1.0
    assert hermite_basis.x_center == 0.0
    assert hermite_basis.jit is True
    assert len(hermite_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        hermite_function_basis(
            x=x_reference,
            n=1,
            alpha=1.0,
            x_center=None,
            jit=True,
        ),
        hermite_basis(x=x_reference),
    )

    # the ``alpha`` property is set
    hermite_basis.alpha = 2.0

    # the properties are tested
    assert hermite_basis.n == 1
    assert hermite_basis.alpha == 2.0
    assert hermite_basis.x_center == 0.0
    assert hermite_basis.jit is True
    assert len(hermite_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        hermite_function_basis(
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=None,
            jit=True,
        ),
        hermite_basis(x=x_reference),
    )

    # the ``x_center`` property is set to a numeric value
    hermite_basis.x_center = 0.0

    # the properties are tested
    assert hermite_basis.n == 1
    assert hermite_basis.alpha == 2.0
    assert hermite_basis.x_center == 0.0
    assert hermite_basis.jit is True
    assert len(hermite_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        hermite_function_basis(
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=0.0,
            jit=True,
        ),
        hermite_basis(x=x_reference),
    )

    # the ``x_center`` property is set to another numeric value
    hermite_basis.x_center = 10.0

    # the properties are tested
    assert hermite_basis.n == 1
    assert hermite_basis.alpha == 2.0
    assert hermite_basis.x_center == 10.0
    assert hermite_basis.jit is True
    assert len(hermite_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        hermite_function_basis(
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=10.0,
            jit=True,
        ),
        hermite_basis(x=x_reference),
    )

    # the ``jit`` property is set
    hermite_basis.jit = False

    # the properties are tested
    assert hermite_basis.n == 1
    assert hermite_basis.alpha == 2.0
    assert hermite_basis.x_center == 10.0
    assert hermite_basis.jit is False
    assert len(hermite_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        hermite_function_basis(
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=10.0,
            jit=False,
        ),
        hermite_basis(x=x_reference),
    )

    # finally, ``jit`` is set to a non-boolean value
    with pytest.raises(TypeError, match="Expected 'jit' to be a boolean"):
        hermite_basis.jit = "this is wrong!"  # type: ignore
