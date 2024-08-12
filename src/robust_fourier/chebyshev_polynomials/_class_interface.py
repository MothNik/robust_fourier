"""
Module :mod:`chebyshev_polynomials._class_interface`

This module implements a class based interface to the Chebyshev polynomials via the
class :class:`ChebyshevPolynomialBasis`.

"""

# === Imports ===

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils import (
    IntScalar,
    RealScalar,
    get_validated_alpha,
    get_validated_offset_along_axis,
    get_validated_order,
)
from ._func_interface import chebyshev_poly_basis, get_validated_chebyshev_kind

# === Classes ===


class ChebyshevPolynomialBasis:
    """
    This class represents a basis of Chebyshev polynomials which are defined by their

    - order ``n``,
    - scaling factor ``alpha``,
    - center ``x_center`` (mu), and
    - kind (first or second kind).

    For their definition, please refer to the Notes section.

    Parameters
    ----------
    n : :class:`int`
        The order of the dilated Chebyshev polynomials.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Chebyshev polynomials.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.
    kind : {``1``, ``2``, ``"first"``, ``"second"``}, default=``"second"``
        The kind of Chebyshev polynomials to compute, which can either

        - ``1`` or ``"first"`` for the first kind,
        - ``2`` or ``"second"`` for the second kind.

    jit : :class:`bool`, default=``True``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.

    Attributes
    ----------
    n : :class:`int`
        The order of the Chebyshev polynomials.
    alpha : :class:`float`
        The scaling factor of the independent variables.
    x_center : :class:`float`
        The x-center mu of the Chebyshev polynomials.
    kind : {``1``, ``2``}
        The kind of Chebyshev polynomials to compute.

    Methods
    -------
    __len__()
        Returns the number of Chebyshev basis functions that will be computed with the
        given parameters.
    eval(x, n, alpha, x_center, validate_parameters=True)
        Evaluates the Chebyshev polynomials at the given points with the specified
        parameters.
    __call__(x)
        Evaluates the Chebyshev polynomials at the given points with the class
        parameters.

    Raises
    ------
    TypeError
        If any of the input parameters is not of the expected type.
    ValueError
        If ``n`` is not a non-negative integer, ``alpha`` is not a positive number, or
        ``kind`` is not one of the allowed values.

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

    # --- Constructor ---

    def __init__(
        self,
        n: IntScalar,
        alpha: RealScalar = 1.0,
        x_center: Optional[RealScalar] = None,
        kind: Literal[1, 2, "first", "second"] = "second",
        jit: bool = True,
    ):
        self._n: int = get_validated_order(n=n)
        self._alpha: float = get_validated_alpha(alpha=alpha)
        self._x_center: float = get_validated_offset_along_axis(
            offset=x_center,
            which_axis="x",
        )
        self._kind: Literal[1, 2] = get_validated_chebyshev_kind(
            kind=kind,
            allow_both_kinds=False,
        )
        self._jit: bool = jit

    # --- Properties ---

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: IntScalar) -> None:
        self._n = get_validated_order(n=value)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: RealScalar) -> None:
        self._alpha = get_validated_alpha(alpha=value)

    @property
    def x_center(self) -> float:
        return self._x_center

    @x_center.setter
    def x_center(self, value: Optional[RealScalar]) -> None:
        self._x_center = get_validated_offset_along_axis(
            offset=value,
            which_axis="x",
        )

    @property
    def kind(self) -> Literal[1, 2]:
        return self._kind

    @kind.setter
    def kind(self, value: Literal[1, 2]) -> None:
        self._kind = get_validated_chebyshev_kind(
            kind=value,
            allow_both_kinds=False,
        )

    @property
    def jit(self) -> bool:
        return self._jit

    @jit.setter
    def jit(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected 'jit' to be a boolean but it is of type "
                f"'{type(value).__name__}'."
            )

        self._jit = value

    # --- Public Methods ---

    @staticmethod
    def eval(
        x: Union[RealScalar, ArrayLike],
        n: IntScalar = 10,
        alpha: RealScalar = 1.0,
        x_center: Optional[RealScalar] = None,
        kind: Literal[1, 2, "first", "second"] = "second",
        jit: bool = True,
        validate_parameters: bool = True,
    ) -> NDArray[np.float64]:
        """
        Evaluates the Chebyshev polynomials at the given points with the specified
        parameters.

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
        kind : {``1``, ``2``, ``"first"``, ``"second"``}, default=``"second"``
            The kind of Chebyshev polynomials to compute, which can either

            - ``1`` or ``"first"`` for the first kind,
            - ``2`` or ``"second"`` for the second kind.

        jit : :class:`bool`, default=``True``
            Whether to use the Numba-accelerated implementation (``True``) or the
            NumPy-based implementation (``False``).
            If Numba is not available, the function silently falls back to the
            NumPy-based implementation.
        validate_parameters : :class:`bool`, default=``True``
            Whether to validate ``n``, ``alpha``, and ``x_center`` before evaluating the
            Chebyshev polynomials.
            Disabling the checks is highly discouraged and was only implemented for
            internal purposes.

        Returns
        -------
        chebyshev_basis : :class:`numpy.ndarray` of shape (m, n + 1) of dtype ``np.float64``
            The values of the dilated Chebyshev polynomials at the points ``x``.
            It will always be 2D even if ``x`` is a scalar.

        Notes
        -----
        For a detailed description of the Chebyshev polynomials, please refer to the
        docstring of the class :class:`ChebyshevPolynomialBasis` itself.

        """  # noqa: E501

        return chebyshev_poly_basis(
            x=x,
            n=n,
            alpha=alpha,
            x_center=x_center,
            kind=kind,
            allow_both_kinds=False,
            jit=jit,
            validate_parameters=validate_parameters,
        )

    # --- Magic Methods ---

    def __len__(self):
        """
        Returns the number of Chebyshev basis functions that will be computed, i.e.,
        ``n + 1``.

        """

        return self._n + 1

    def __call__(
        self,
        x: Union[RealScalar, ArrayLike],
    ) -> NDArray[np.float64]:
        """
        Evaluates the Chebyshev polynomials at the given points with the class
        parameters.
        Please refer to the docstrings of the class itself and the method :meth:`eval`
        for more information.

        """  # noqa: E501

        return self.eval(
            x=x,
            n=self._n,
            alpha=self._alpha,
            x_center=self._x_center,
            kind=self._kind,
            jit=self._jit,
            validate_parameters=False,
        )
