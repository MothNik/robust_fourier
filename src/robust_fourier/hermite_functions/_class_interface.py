"""
Module :mod:`hermite_functions._class_interface`

This module implements a class based interface to the Hermite functions via the class
:class:`HermiteFunctionBasis`.

"""

# === Imports ===

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils import (
    IntScalar,
    RealScalar,
    get_validated_alpha,
    get_validated_offset_along_axis,
    get_validated_order,
)
from ._func_interface import hermite_function_vander

# === Classes ===


class HermiteFunctionBasis:
    """
    This class represents a basis of Hermite functions which are defined by their

    - order ``n``,
    - scaling factor ``alpha``, and
    - center ``x_center`` (mu)

    For their definition, please refer to the Notes section.

    Parameters
    ----------
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variable ``x`` for
        ``x_scaled = x / alpha``.
        It must be a positive number ``> 0``.
    x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the dilated Hermite functions.
        If ``None`` or ``0``, the function is centered at the origin.
        Otherwise, the center is shifted to the given value.
    jit : :class:`bool`, default=``True``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.

    Attributes
    ----------
    n : :class:`int`
        The order of the Hermite functions.
    alpha : :class:`float`
        The scaling factor of the independent variables.
    x_center : :class:`float`
        The x-center mu of the Hermite functions.

    Methods
    -------
    __len__()
        Returns the number of Hermite basis functions that will be computed with the
        given parameters.
    eval(x, n, alpha, x_center, validate_parameters=True)
        Evaluates the Hermite functions at the given points with the specified
        parameters.
    __call__(x)
        Evaluates the Hermite functions at the given points with the class parameters.

    Raises
    ------
    TypeError
        If any of the input parameters is not of the expected type.
    ValueError
        If ``n`` is not a non-negative integer or ``alpha`` is not a positive number.

    Notes
    -----
    The dilated Hermite functions are defined as

    .. image:: docs/hermite_functions/equations/HF-01-Hermite_Functions_TimeSpace_Domain.svg

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/HF-02-Hermite_Polynomials_TimeSpace_Domain.svg

    Internally, they are computed in a numerically stable way that relies on a
    logarithmic scaling trick to avoid over- and underflow in the recursive calculation
    of the Hermite polynomials and the Gaussians. This allows for arbitrary large orders
    ``n`` to be evaluated.

    References
    ----------
    The implementation is an adaption of the Appendix in [1]_.

    .. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
       functions, BIT Numer Math (2009), 49, pp. 281–295, DOI: 10.1007/s10543-009-0216-1

    """  # noqa: E501

    # --- Constructor ---

    def __init__(
        self,
        n: IntScalar,
        alpha: RealScalar = 1.0,
        x_center: Optional[RealScalar] = None,
        jit: bool = True,
    ):
        self._n: int = get_validated_order(n=n)
        self._alpha: float = get_validated_alpha(alpha=alpha)
        self._x_center: float = get_validated_offset_along_axis(
            offset=x_center,
            which_axis="x",
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
        jit: bool = True,
        validate_parameters: bool = True,
    ) -> NDArray[np.float64]:
        """
        Evaluates the Hermite functions at the given points with the specified
        parameters.

        Parameters
        ----------
        x : :class:`float` or :class:`int` or Array-like of shape (m,)
            The points at which the dilated Hermite functions are evaluated.
            Internally, it will be promoted to ``np.float64``.
            It has to contain at least one element.
        n : :class:`int`
            The order of the dilated Hermite functions.
            It must be a non-negative integer ``>= 0``.
        alpha : :class:`float` or :class:`int`, default=``1.0``
            The scaling factor of the independent variable ``x`` for
            ``x_scaled = x / alpha``.
            It must be a positive number ``> 0``.
        x_center : :class:`float` or :class:`int` or ``None``, default=``None``
            The center of the dilated Hermite functions.
            If ``None`` or ``0``, the functions are centered at the origin.
            Otherwise, the center is shifted to the given value.
        jit : :class:`bool`, default=``True``
            Whether to use the Numba-accelerated implementation (``True``) or the
            NumPy-based implementation (``False``).
            If Numba is not available, the function silently falls back to the
            NumPy-based implementation.
        validate_parameters : :class:`bool`, default=``True``
            Whether to validate ``n``, ``alpha``, and ``x_center`` before evaluating the
            Hermite functions.
            Disabling the checks is highly discouraged and was only implemented for
            internal purposes.

        Returns
        -------
        hermite_basis : :class:`numpy.ndarray` of shape (m, n + 1) of dtype ``np.float64``
            The values of the dilated Hermite functions at the points ``x``.
            It will always be 2D even if ``x`` is a scalar.

        Notes
        -----
        For a detailed description of the Hermite functions, please refer to the
        docstring of the class :class:`HermiteFunctionBasis` itself.

        """  # noqa: E501

        return hermite_function_vander(
            x=x,
            n=n,
            alpha=alpha,
            x_center=x_center,
            jit=jit,
            validate_parameters=validate_parameters,
        )

    # --- Magic Methods ---

    def __len__(self):
        """
        Returns the number of Hermite basis functions that will be computed, i.e.,
        ``n + 1``.

        """

        return self._n + 1

    def __call__(
        self,
        x: Union[RealScalar, ArrayLike],
    ) -> NDArray[np.float64]:
        """
        Evaluates the Hermite functions at the given points with the class parameters.
        Please refer to the docstrings of the class itself and the method :meth:`eval`
        for more information.

        """  # noqa: E501

        return self.eval(
            x=x,
            n=self._n,
            alpha=self._alpha,
            x_center=self._x_center,
            jit=self._jit,
            validate_parameters=False,
        )
