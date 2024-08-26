"""
This test suite implements the tests for the module :mod:`chebyshev_polynomials._class_interface`.

"""  # noqa: E501

# === Imports ===

from typing import Literal

import numpy as np
import pytest

from robust_fourier import ChebyshevPolynomialBasis, chebyshev_polyvander

# === Tests ===


@pytest.mark.parametrize("kind", [1, 2, "first", "second"])
def test_chebyshev_polynomials_basis_properties_and_len(
    kind: Literal[1, "first", 2, "second"]
) -> None:
    """
    Tests the properties setters and getters of the class :class:`ChebyshevPolynomialBasis`
    as well as its :meth:`__len__` method.

    """  # noqa: E501

    # the reference kind is set
    if kind in {1, "first"}:
        ref_kind = 1
    elif kind in {2, "second"}:
        ref_kind = 2
    else:
        raise AssertionError(f"Unexpected kind '{kind}'.")

    # an instance of the Chebyshev polynomial basis is initialised
    chebyshev_basis = ChebyshevPolynomialBasis(
        n=0,
        alpha=1.0,
        x_center=None,
        kind=kind,
        jit=True,
    )

    # the properties are tested
    assert chebyshev_basis.n == 0
    assert chebyshev_basis.alpha == 1.0
    assert chebyshev_basis.x_center == 0.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is True
    assert len(chebyshev_basis) == 1

    # the results of calling the Chebyshev polynomial basis are tested against the
    # expected results
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=0,
            alpha=1.0,
            x_center=None,
            kind=ref_kind,
            jit=True,
        ),
        chebyshev_basis(x=x_reference),
    )

    # the ``n`` property is set
    chebyshev_basis.n = 1

    # the properties are tested
    assert chebyshev_basis.n == 1
    assert chebyshev_basis.alpha == 1.0
    assert chebyshev_basis.x_center == 0.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is True
    assert len(chebyshev_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=1,
            alpha=1.0,
            x_center=None,
            kind=ref_kind,
            jit=True,
        ),
        chebyshev_basis(x=x_reference),
    )

    # the ``alpha`` property is set
    chebyshev_basis.alpha = 2.0

    # the properties are tested
    assert chebyshev_basis.n == 1
    assert chebyshev_basis.alpha == 2.0
    assert chebyshev_basis.x_center == 0.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is True
    assert len(chebyshev_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=None,
            kind=ref_kind,
            jit=True,
        ),
        chebyshev_basis(x=x_reference),
    )

    # the ``x_center`` property is set to a numeric value
    chebyshev_basis.x_center = 0.0

    # the properties are tested
    assert chebyshev_basis.n == 1
    assert chebyshev_basis.alpha == 2.0
    assert chebyshev_basis.x_center == 0.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is True
    assert len(chebyshev_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=0.0,
            kind=ref_kind,
            jit=True,
        ),
        chebyshev_basis(x=x_reference),
    )

    # the ``x_center`` property is set to another numeric value
    chebyshev_basis.x_center = 10.0

    # the properties are tested
    assert chebyshev_basis.n == 1
    assert chebyshev_basis.alpha == 2.0
    assert chebyshev_basis.x_center == 10.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is True
    assert len(chebyshev_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=10.0,
            kind=ref_kind,
            jit=True,
        ),
        chebyshev_basis(x=x_reference),
    )

    # the ``kind`` property is set
    if kind == 1:
        new_kind = 2
        ref_kind = 2
    elif kind == 2:
        new_kind = 1
        ref_kind = 1
    elif kind == "first":
        new_kind = "second"  # type: ignore
        ref_kind = 2
    elif kind == "second":
        new_kind = "first"  # type: ignore
        ref_kind = 1
    else:
        raise AssertionError(f"Unexpected kind '{kind}'.")

    chebyshev_basis.kind = new_kind  # type: ignore

    # the properties are tested
    assert chebyshev_basis.n == 1
    assert chebyshev_basis.alpha == 2.0
    assert chebyshev_basis.x_center == 10.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is True
    assert len(chebyshev_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=10.0,
            kind=ref_kind,
            jit=True,
        ),
        chebyshev_basis(x=x_reference),
    )

    # the ``jit`` property is set
    chebyshev_basis.jit = False

    # the properties are tested
    assert chebyshev_basis.n == 1
    assert chebyshev_basis.alpha == 2.0
    assert chebyshev_basis.x_center == 10.0
    assert chebyshev_basis.kind == ref_kind
    assert chebyshev_basis.jit is False
    assert len(chebyshev_basis) == 2

    # the direct call is tested
    x_reference = np.arange(start=-2.5, stop=3.0, step=0.5)
    assert np.array_equal(
        chebyshev_polyvander(  # type: ignore
            x=x_reference,
            n=1,
            alpha=2.0,
            x_center=10.0,
            kind=ref_kind,
            jit=False,
        ),
        chebyshev_basis(x=x_reference),
    )

    # finally, ``jit`` is set to a non-boolean value
    with pytest.raises(TypeError, match="Expected 'jit' to be a boolean"):
        chebyshev_basis.jit = "this is wrong!"  # type: ignore
