"""
Module :mod:`chebyshev_polynomials._func_interface`

This module implements the interface to the either NumPy-based or Numba-based
implementations of the Chebyshev polynomials of the first and second kind.

It augments them with an additional input validation which is better done in Python
and also handles the incorporation of the Numba-functions if Numba is available at
runtime.

"""

# === Imports ===

from typing import Any, Literal, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils import (
    IntScalar,
    RealScalar,
    get_validated_chebpoly_or_hermfunc_input,
    get_validated_x_values,
    normalise_x_values,
)
from ._numba_funcs import nb_chebyshev_poly_bases as _nb_chebyshev_poly_bases
from ._numpy_funcs import _chebyshev_poly_bases as _np_chebyshev_poly_bases

# === Constants ===

# the set of kinds for the first and second kind of Chebyshev polynomials
chebyshev_first_kind_set = {1, "first"}
chebyshev_second_kind_set = {2, "second"}
chebyshev_both_kinds_set = {"both", None}


# === Auxiliary Functions ===


@overload
def get_validated_chebyshev_kind(
    kind: Any,
    allow_both_kinds: Literal[False],
) -> Literal[1, 2]: ...


@overload
def get_validated_chebyshev_kind(
    kind: Any,
    allow_both_kinds: Literal[True] = True,
) -> Optional[Literal[1, 2]]: ...


@overload
def get_validated_chebyshev_kind(
    kind: Any,
    allow_both_kinds: bool = True,
) -> Optional[Literal[1, 2]]: ...


def get_validated_chebyshev_kind(
    kind: Any,
    allow_both_kinds: bool = True,
) -> Optional[Literal[1, 2]]:
    """
    Validates the input for the kind of Chebyshev polynomials and returns the validated
    input.

    """

    # if the kind is not of the expected type, an error is raised
    if not isinstance(kind, (int, str)) and kind is not None:
        raise TypeError(
            f"Expected 'kind' to be an integer, a string, or None, but got type "
            f"{type(kind)}."
        )

    # then, the kind is checked for being one of the allowed values
    kind_internal = kind
    if isinstance(kind, str):
        kind_internal = kind.lower()

    if kind_internal in chebyshev_first_kind_set:
        return 1

    if kind_internal in chebyshev_second_kind_set:
        return 2

    if kind_internal in chebyshev_both_kinds_set:
        if allow_both_kinds:
            return None

        raise ValueError(
            f"Expected 'kind' to be one of {chebyshev_first_kind_set} or "
            f"{chebyshev_second_kind_set}, but got '{kind}'."
        )

    raise ValueError(
        f"Expected 'kind' to be one of {chebyshev_first_kind_set}, "
        f"{chebyshev_second_kind_set}, or {chebyshev_both_kinds_set}, but got "
        f"'{kind}'."
    )


# === Main Functions ===


@overload
def chebyshev_poly_basis(
    x: Union[RealScalar, ArrayLike],
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
    kind: Literal[1, 2, "first", "second"] = "second",
    allow_both_kinds: bool = True,
    jit: bool = True,
    validate_parameters: bool = True,
) -> NDArray[np.float64]: ...


@overload
def chebyshev_poly_basis(
    x: Union[RealScalar, ArrayLike],
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
    kind: Optional[Literal["both"]] = "both",
    allow_both_kinds: bool = True,
    jit: bool = True,
    validate_parameters: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def chebyshev_poly_basis(
    x: Union[RealScalar, ArrayLike],
    n: IntScalar,
    alpha: RealScalar = 1.0,
    x_center: Optional[RealScalar] = None,
    kind: Optional[Literal[1, 2, "first", "second", "both"]] = "second",
    allow_both_kinds: bool = True,
    jit: bool = True,
    validate_parameters: bool = True,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Computes the basis of dilated Chebyshev polynomials up to order ``n`` for the given
    points ``x``. It makes use of a combined recursion formula that links the first and
    second kind of Chebyshev polynomials to evaluate them in a numerically stable way.

    Parameters
    ----------
    x : :class:`float` or :class:`int` or Array-like of shape (m,)
        The points at which the dilated Chebyshev polynomials are evaluated.
        Internally, it will be promoted to ``np.float64``.
        It has to contain at least one element.
    n : :class:`int`
        The order of the dilated Chebyshev polynomials.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Chebyshev polynomials.
        If ``None`` or ``0``, the functions are centered at the origin.
        Otherwise, the center is shifted to the given value.
    kind : {``1``, ``2``, ``"first"``, ``"second"``, ``"both"``} or None, default=``"second"``
        The kind of Chebyshev polynomials to compute, which can either

        - ``1`` or ``"first"`` for the first kind,
        - ``2`` or ``"second"`` for the second kind,
        - ``"both"`` or ``None`` for both kinds simultaneously (no significant
            performance impact due to the combined recursion formula; only available if
            ``allow_both_kinds`` is ``True``).

    allow_both_kinds : :class:`bool`, default=``True``
        Whether to allow the computation of both kinds of Chebyshev polynomials
        simultaneously (``True``) or not (``False``).
    jit : :class:`bool`, default=``True``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.
    validate_parameters : :class:`bool`, default=``True``
        Whether to validate all the input parameters (``True``) or only ``x``
        (``False``).
        Disabling the input checks is not recommended and was only implemented for
        internal use.

    Returns
    -------
    chebyshev_t1_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the Chebyshev polynomials of the first kind at the points ``x``.
        It will always be 2D even if ``x`` is a scalar.
        It is only returned if ``kind`` is ``1``, ``"first"``, ``"both"`` (for the
        latter, it is the first element of the tuple).

    chebyshev_u2_basis : :class:`numpy.ndarray` of shape (m, n + 1)
        The values of the Chebyshev polynomials of the second kind at the points ``x``.
        It will always be 2D even if ``x`` is a scalar.
        It is only returned if ``kind`` is ``2``, ``"second"``, ``"both"`` (for the
        latter, it is the second element of the tuple).

    Raises
    ------
    TypeError
        If any of the input arguments is not of the expected type.
    ValueError
        If ``x`` is not 1-dimensional after conversion to a NumPy array.
    ValueError
        If ``n`` is not a non-negative integer or ``alpha`` is not a positive number.
    ValueError
        If ``kind`` is not one of the allowed values.

    Notes
    -----
    The naive three-term recurrence relations for Chebyshev polynomials of the first and
    second kind have bad numerical properties because the error is increasing
    proportional to ``n * n``, i.e., quadratically with the order of the polynomial
    ``n``.
    However, a combined recurrence relation that links both kinds of Chebyshev
    polynomials has an error proportional to ``n``, i.e., a linear increase with the
    order of the polynomial ``n``. Already for ``n`` as little as ``4``, the combined
    recurrence relation outperforms the naive three-term recurrence relations. So, the
    roughly doubled computational cost of the combined recurrence relation is justified.
    On top of that, a combined evaluation offers the opportunity to easily compute the
    derivatives of the Chebyshev polynomials because the derivatives of the first kind
    are related to the polynomials of the second kind and vice versa.

    References
    ----------
    The implementation is taken from [1]_.

    .. [1] Hrycak T., Schmutzhard S., Accurate evaluation of Chebyshev polynomials in
       floating-point arithmetic, BIT Num

    """  # noqa: E501

    # --- Input validation ---

    if validate_parameters:
        (
            x_internal,
            n,
            alpha,
            x_center,
        ) = get_validated_chebpoly_or_hermfunc_input(
            x=x,
            n=n,
            alpha=alpha,
            x_center=x_center,
        )

    else:  # pragma: no cover
        x_internal = get_validated_x_values(x=x)

    kind_internal = get_validated_chebyshev_kind(
        kind=kind,
        allow_both_kinds=allow_both_kinds,
    )

    # --- Computation ---

    # if required, the x-values are centered
    x_internal = normalise_x_values(
        x_internal=x_internal,
        x=x,
        x_center=x_center,  # type: ignore
        alpha=alpha,  # type: ignore
    )

    # if requested, the Numba-accelerated implementation is used
    # NOTE: this does not have to necessarily involve Numba because it can also be
    #       the NumPy-based implementation under the hood
    func = _nb_chebyshev_poly_bases if jit else _np_chebyshev_poly_bases
    chebyshev_bases = func(  # type: ignore
        x=x_internal,  # type: ignore
        n=n,  # type: ignore
    )

    # --- Output post-processing ---

    # if only one kind is requested, the corresponding part of the output is returned
    if kind_internal in {1, 2}:
        return chebyshev_bases[kind_internal - 1]

    # if both kinds are requested, the full output is returned
    return chebyshev_bases
