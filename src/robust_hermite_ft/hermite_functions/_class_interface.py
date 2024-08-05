"""
Module :mod:`hermite_functions._class_interface`

This module implements a class based interface to the Hermite functions via the class
:class:`HermiteFunctionBasis`. The class allows to evaluate the Hermite functions at
arbitrary points in both the time/space and the frequency domain.

"""

# === Imports ===

from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._func_interface import hermite_function_basis
from ._validate import (
    IntScalar,
    RealScalar,
    get_validated_alpha,
    get_validated_offset_along_axis,
    get_validated_order,
)

# === Constants ====

# the aliases for the time/space symmetry
_time_space_symmetry_aliases: set[Literal["even", "odd", "none"]] = {
    "even",
    "odd",
    "none",
}

# === Auxiliary Functions ===


def _get_validated_time_space_symmetry(
    time_space_symmetry: Any,
) -> Literal["even", "odd", "none"]:
    """
    Validates the input for the time/space symmetry and returns the validated input.

    """

    if time_space_symmetry is None:
        return "none"

    if not isinstance(time_space_symmetry, str):
        raise TypeError(
            f"The time/space symmetry must be None or a String, but it is of type "
            f"'{type(time_space_symmetry).__name__}'."
        )

    time_space_symmetry_internal = time_space_symmetry.lower()
    if time_space_symmetry_internal in _time_space_symmetry_aliases:
        return time_space_symmetry_internal  # type: ignore

    raise ValueError(
        f"The time/space symmetry must be one of '{_time_space_symmetry_aliases}' but "
        f"it is '{time_space_symmetry_internal}'."
    )


def _validate_parameter_combination(
    n: int,
    alpha: float,
    time_space_x_center: float,
    time_space_symmetry: Literal["even", "odd", "none"],
) -> None:
    """
    Validates the parameter combination for the Hermite functions.

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite functions.
    alpha : :class:`float`
        The scaling factor of the independent variables.
    time_space_x_center : :class:`float`
        The center of the Hermite functions in the time/space domain.
    time_space_symmetry : ``"even"`` or ``"odd"`` or ``"none"``
        The symmetry to be assumed for the time space domain.

    Raises
    ------
    ValueError
        If there are no Hermite functions to compute (``n = 0`` and
        ``time_space_symmetry = "odd"``).

    """

    # if the order is 0 and the symmetry is odd, there are no Hermite functions to
    # compute, so an error is raised
    if n == 0 and time_space_symmetry == "odd":
        raise ValueError(
            "There are no Hermite functions to compute with 'n = 0' and "
            "'time_space_symmetry = 'odd''."
        )


def _get_num_effective_n(
    n: int,
    time_space_symmetry: Literal["even", "odd", "none"],
) -> int:
    """
    Computes the number of orders to consider based on the order and the symmetry.

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite functions.
    time_space_symmetry : ``"even"`` or ``"odd"`` or ``"none"``
        The symmetry to be assumed for the time space domain.

    Returns
    -------
    num_n : :class:`int`
        The number of orders to consider.

    """

    # Case 1: no symmetry
    # in this case, all orders are considered
    if time_space_symmetry == "none":
        return n + 1

    # Case 2: even symmetry
    # in this case, only the even orders are considered, i.e., the number of orders
    # is halved and rounded up
    if time_space_symmetry == "even":
        # NOTE: the following is a numerically safe integer ceiling division
        return -(-(n + 1) // 2)

    # Case 3: odd symmetry
    # in this case, only the odd orders are considered, i.e., the number of orders
    # is halved and rounded down
    return (n + 1) // 2


def _get_frequency_domain_hermite_complex_prefactors(
    num_effective_n: int,
    time_space_symmetry: Literal["even", "odd", "none"],
) -> NDArray[Union[np.float64, np.complex128]]:
    """
    Computes the prefactors for the Hermite functions in the frequency domain
    depending on the symmetry.

    The pre-factors are given by ``(0 - 1j) ** i`` for ``i in range(0, n + 1)`` where
    ``j``.
    However, his evaluation via a power is expensive and on top of that numerically
    inaccurate, so the pre-factors are directly specified from the repeating sequence
    ``[1, -j, -1, j]``.

    Parameters
    ----------
    num_effective_n : :class:`int`
        The number of orders to consider as computed by :func:`_get_num_effective_n`.
    time_space_symmetry : ``"even"`` or ``"odd"`` or ``"none"``
        The symmetry to be assumed for the time space domain.

    Returns
    -------
    pre_factors : :class:`numpy.ndarray` of shape (1, num_effective_n) of dtype ``np.float64`` or ``np.complex128``
        The pre-factors for the Hermite functions in the frequency domain.
        It is stored as a row-vector to leverage NumPy's broadcasting for a direct
        multiplication with the Hermite functions.

    """  # noqa: E501

    # Case 1: no symmetry
    # in this case, all pre-factors that repeat every 4th order are computed
    if time_space_symmetry == "none":
        base_pre_factors = [
            +1.0,
            complex(0.0, -1.0),
            -1.0,
            complex(0.0, +1.0),
        ]
        num_full_reps, rest = divmod(num_effective_n, 4)
        dtype = np.complex128

    # Case 2: even symmetry
    # in this case, only the even orders are considered and the pre-factors are
    # just [1, -1] that repeat every 2nd even order
    elif time_space_symmetry == "even":
        base_pre_factors = [+1.0, -1.0]
        num_full_reps, rest = divmod(num_effective_n, 2)
        dtype = np.float64  # type: ignore

    # Case 3: odd symmetry
    # in this case, only the odd orders are considered and the pre-factors are
    # [1j, -1j] that repeat every 2nd odd order
    else:
        base_pre_factors = [complex(0.0, -1.0), complex(0.0, +1.0)]
        num_full_reps, rest = divmod(num_effective_n, 2)
        dtype = np.complex128

    # the pre-factors are "computed" by repeating the base pre-factors
    # NOTE: the following is a row vector to leverage NumPy's broadcasting and
    #       multiply every row of the Hermite functions with the respective
    #       pre-factors
    return np.array(
        [base_pre_factors * num_full_reps + base_pre_factors[0:rest]],
        dtype=dtype,
    )


def _get_frequency_domain_shift_prefactors(
    omega: Union[RealScalar, ArrayLike],
    time_space_x_center: RealScalar,
) -> Optional[NDArray[np.complex128]]:
    """
    Computes the prefactors for the Hermite functions in the frequency domain
    depending on the shift.

    The pre-factors are given by ``exp(-1j * omega * time_space_x_center)`` where ``j``
    is the imaginary unit.

    Parameters
    ----------
    omega : :class:`float` or :class:`int` or Array-like of shape (m,)
        The angular frequency values at which the Hermite functions are evaluated.
    time_space_x_center : :class:`float` or :class:`int`
        The center of the Hermite functions in the time/space domain.

    Returns
    -------
    pre_factors : :class:`numpy.ndarray` of shape (m, 1) of dtype ``np.complex128`` or ``None``
        The pre-factors for the Hermite functions in the frequency domain.
        If the center is 0, the pre-factors are just 1 and ``None`` is returned to
        avoid unnecessary computations.
        Otherwise, the pre-factors are computed and returned. They are stored as a
        column-vector to leverage NumPy's broadcasting for a direct multiplication with
        the Hermite functions.

    """  # noqa: E501

    # if there is no shift, the pre-factors are just 1, so it is returned as None
    # to avoid unnecessary computations
    if time_space_x_center == 0.0:
        return None

    # otherwise, the shift characteristics of the Fourier transform are applied
    omega_internal = np.atleast_1d(omega)
    # NOTE: the following is a column vector to leverage NumPy's broadcasting and
    #       multiply every column of the Hermite functions with the respective
    #       pre-factors
    result = np.empty(
        shape=(omega_internal.size, 1),
        dtype=np.complex128,
    )

    return np.exp(
        complex(0.0, -1.0) * omega_internal[::, np.newaxis] * time_space_x_center,
        out=result,
    )


# === Classes ===


class HermiteFunctionBasis:
    """
    This class represents a basis of Hermite functions which are defined by their
    order ``n`` and their scaling factor ``alpha`` for the independent variables.
    It can be used to compute the Continuous Fourier transforms of the Hermite functions
    by making use of the fact that they are the eigenfunctions of the Fourier transform
    and thus - up to scaling factors and dilations - their own Fourier transforms.
    Please refer to the Notes for further details on the underlying definitions.

    Parameters
    ----------
    n : :class:`int`
        The order of the dilated Hermite functions.
        It must be a non-negative integer ``>= 0``.
    alpha : :class:`float` or :class:`int`, default=``1.0``
        The scaling factor of the independent variables.
        It must be a positive number ``> 0``.
        Please refer to the Notes for further details
    time_space_x_center : :class:`float` or :class:`int` or ``None``, default=``None``
        The center of the Hermite functions in the time/space domain.
        If ``None`` or ``0``, the functions are centered at the time/space domain's
        origin (``x = 0``).
        Otherwise, the center is shifted to the given value
        (``x = time_space_x_center``).
    time_space_symmetry : ``"even"`` or ``"odd"`` or ``"none"`` or ``None``, default=``None``
        The symmetry to be assumed for the time space domain with respect to
        ``time_space_x_center``.
        If ``"none"`` or ``None``, no symmetry is assumed.
        For ``"even"`` symmetry (axis-mirrored at a y-axis located at
        ``time_space_x_center``), only the even orders are considered while for
        ``"odd"`` symmetry (point-mirrored at ``time_space_x_center``; rotational
        symmetry), only the odd orders are considered.
        Please refer to the Notes for further details.

    Attributes
    ----------
    n : :class:`int`
        The order of the Hermite functions.
    alpha : :class:`float`
        The scaling factor of the independent variables.
    time_space_x_center : :class:`float`
        The x-center of the Hermite functions in the time/space domain.
    time_space_symmetry : ``"even"`` or ``"odd"`` or ``"none"``
        The symmetry to be assumed for the time space domain.

    Methods
    -------
    __len__()
        Returns the number of Hermite basis functions that will be computed with the
        given parameters.
    __call__(x, omega)
        Evaluates the Hermite functions at the given points in the time/space or the
        frequency domain.

    Raises
    ------
    TypeError
        If any of the input parameters is not of the expected type.
    ValueError
        If ``n`` is not a non-negative integer or ``alpha`` is not a positive number.
    ValueError
        If ``time_space_symmetry`` is not one of the allowed values.

    Notes
    -----
    Here, two definitions are made:

    - the time/space domain and
    - the frequency domain

    where

    - time/space is given by the independent variable ``x`` and
    - the angular frequency is given by the independent variable ``omega``.


    In the time/space domain, the Hermite functions are all real-valued by the
    definitions chosen here. They can be shifted left or right by specifying a center
    ``time_space_x_center`` which will only change the position of the Hermite functions
    but not their real-valued nature. Furthermore, The scaling factor ``alpha`` is
    applied to the independent variable ``x`` as ``x / alpha``, i.e., larger values of
    ``alpha`` will expand the Hermite functions further away from the origin.

    All in all, their definition is given by

    .. image:: docs/hermite_functions/equations/HF-01-Hermite_Functions_TimeSpace_Domain.svg

    with the Hermite polynomials

    .. image:: docs/hermite_functions/equations/HF-02-Hermite_Polynomials_TimeSpace_Domain.svg

    For the frequency domain on the other hand, the scaling factor ``alpha`` is applied
    to the independent variable ``omega`` as ``alpha * omega``. Thus, larger values of
    ``alpha`` will result in a contraction of the Hermite functions closer to the
    origin. In this domain, all Hermite functions of even order ``n`` (0, 2, 4, ...) are
    real-valued while all Hermite functions of odd order ``n`` (1, 3, 5, ...) are
    imaginary-valued. A nonzero center ``time_space_x_center`` in the time/space domain
    will result in a phase shift of the Hermite functions in the frequency domain.
    Consequently, the Hermite functions of all orders will be complex-valued due to
    shifts in the time/space domain.

    Based on the calculus rules of the Fourier transform, the Hermite functions in the
    frequency domain can be expressed as

    .. image:: docs/hermite_functions/equations/HF-07-Hermite_Functions_Frequency_Domain_pt_1.svg

    .. image:: docs/hermite_functions/equations/HF-08-Hermite_Functions_Frequency_Domain_pt_2.svg

    Moreover, the symmetry in the time/space domain can be taken into account by
    specifying the ``time_space_symmetry`` parameter. If it is ``"even"``/``"odd"`` only
    the even/odd orders of the Hermite functions have to considered independent of the
    domain.
    Even symmetry means that the Hermite functions are axis-mirrored at a y-axis located
    at the center ``time_space_x_center`` in the time/space domain, while odd symmetry
    means that they are point-mirrored at the ``time_space_x_center`` in the time/space
    domain (rotational symmetry).
    This fact will not reduce the computation time of the Hermite functions themselves
    at all. However, it will save a lot of irrelevant computations down the line, e.g.,
    for regression procedures. Furthermore, this can affect the complex-valued nature of
    the Hermite functions in the frequency domain.

    To sum is up, the time/space domain is defined to be always real-valued, thereby
    leading to the following patterns:

    - time/space domain → always real-valued

    On the other hand, the frequency domain is sensitive to both a shift and to the
    symmetry in the time/space domain, thereby leading to the following patterns:

    - frequency domain → time/space not shifted and not symmetric: complex-valued
    - frequency domain → time/space shifted and not symmetric: complex-valued
    - frequency domain → time/space not shifted and even symmetric: real-valued
    - frequency domain → time/space shifted and even symmetric: complex-valued
    - frequency domain → time/space not shifted and odd symmetric: imaginary-valued
    - frequency domain → time/space shifted and odd symmetric: complex-valued

    """  # noqa: E501

    # --- Constructor ---

    def __init__(
        self,
        n: IntScalar,
        alpha: RealScalar = 1.0,
        time_space_x_center: Optional[RealScalar] = None,
        time_space_symmetry: Optional[Literal["even", "odd", "none"]] = None,
    ):
        self._n: int = get_validated_order(n=n)
        self._alpha: float = get_validated_alpha(alpha=alpha)
        self._time_space_x_center: float = get_validated_offset_along_axis(
            center=time_space_x_center,
            which_axis="x",
        )
        self._time_space_symmetry: Literal["even", "odd", "none"] = (
            _get_validated_time_space_symmetry(time_space_symmetry=time_space_symmetry)
        )

        # it still needs to be checked if there are any Hermite functions to compute
        # but this will not take place in the constructor
        # instead, a flag is set for lazy evaluation
        self._is_fully_validated: bool = False

    # --- Properties ---

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: IntScalar) -> None:
        self._n = get_validated_order(n=value)
        self._is_fully_validated = False

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: RealScalar) -> None:
        self._alpha = get_validated_alpha(alpha=value)

    @property
    def time_space_x_center(self) -> float:
        return self._time_space_x_center

    @time_space_x_center.setter
    def time_space_x_center(self, value: Optional[RealScalar]) -> None:
        self._time_space_x_center = get_validated_offset_along_axis(
            center=value,
            which_axis="x",
        )

    @property
    def time_space_symmetry(self) -> Literal["even", "odd", "none"]:
        return self._time_space_symmetry

    @time_space_symmetry.setter
    def time_space_symmetry(
        self,
        value: Optional[Literal["even", "odd", "none"]],
    ) -> None:
        self._time_space_symmetry = _get_validated_time_space_symmetry(
            time_space_symmetry=value
        )
        self._is_fully_validated = False

    @property
    def is_fully_validated(self) -> bool:
        return self._is_fully_validated

    # --- Internal Methods ---

    def _validate_parameter_combination(self) -> None:
        """
        Validates the current parameter combination of the class.

        """

        _validate_parameter_combination(
            n=self._n,
            alpha=self._alpha,
            time_space_x_center=self._time_space_x_center,
            time_space_symmetry=self._time_space_symmetry,
        )

        # if the validation is successful, the flag is set to True
        self._is_fully_validated = True

    # --- Public Methods ---

    @staticmethod
    def eval(
        x: Union[RealScalar, ArrayLike, None] = None,
        omega: Union[RealScalar, ArrayLike, None] = None,
        n: IntScalar = 10,
        alpha: RealScalar = 1.0,
        time_space_x_center: Optional[RealScalar] = None,
        time_space_symmetry: Optional[Literal["even", "odd", "none"]] = None,
        validate_parameters: bool = True,
    ) -> NDArray[Union[np.float64, np.complex128]]:
        """
        Evaluates the Hermite functions at the given points in the specified domain.

        Parameters
        ----------
        x, omega : :class:`float` or :class:`int` or Array-like of shape (m,)
            The points at which the Hermite functions are evaluated in either the
            time/space domain or the frequency domain.
            If the time/space value(s) ``x`` is/are given, the Hermite functions are
            evaluated in the time/space domain.
            Conversely, if the angular frequency value(s) ``omega`` is/are given, the
            Hermite functions are evaluated in the frequency domain.
        n : :class:`int`, default=``10``
            The order of the dilated Hermite functions.
            It must be a non-negative integer ``>= 0``.
        alpha : :class:`float` or :class:`int`, default=``1.0``
            The scaling factor of the independent variables.
            It must be a positive number ``> 0``.
            Please refer to the Notes of the class docstring for further details.
        time_space_x_center : :class:`float` or :class:`int` or ``None``, default=``None``
            The center of the Hermite functions in the time/space domain.
            If ``None`` or ``0``, the functions are centered at the time/space domain's
            origin (``x = 0``).
            Otherwise, the center is shifted to the given value
            (``x = time_space_x_center``).
        time_space_symmetry : ``"even"`` or ``"odd"`` or ``"none"`` or ``None``, default=``None``
            The symmetry to be assumed for the time space domain with respect to
            ``time_space_x_center``.
            If ``"none"`` or ``None``, no symmetry is assumed.
            For ``"even"`` symmetry (axis-mirrored at a y-axis located at
            ``time_space_x_center``), only the even orders are considered while for
            ``"odd"`` symmetry (point-mirrored at ``time_space_x_center``; rotational
            symmetry), only the odd orders are considered.
            Please refer to the Notes of the class docstring for further details.
        validate_parameters : :class:`bool`, default=``True``
            Whether to validate ``n``, ``alpha``, ``time_space_x_center``, and
            ``time_space_symmetry`` before computing the Hermite functions.
            Disabling the checks is highly discouraged and was only implemented for
            internal purposes.

        Returns
        -------
        hermite_function_basis : :class:`numpy.ndarray` of shape (m, n_new) of dtype ``np.float64`` or ``np.complex128``
            The values of the dilated Hermite functions at the points ``x`` or
            ``omega``.
            Please refer to the Notes for its data type and shape.
            It will always be 2D even if ``x``/``omega`` is a scalar.

        Raises
        ------
        ValueError
            If both ``x`` and ``omega`` or neither of them is given.

        Notes
        -----
        The data type and the shape depend on the domain as well as the center and the
        assumed symmetry in the time/space domain. The following table is derived based
        on the Notes section of the docstring of the class itself.

        The shape and data types for the time/space domain are as follows:

        - no symmetry → (m, n + 1) of dtype ``np.float64``
        - even symmetry → (m, ceil((n + 1) / 2)) of dtype ``np.float64``
        - odd symmetry → (m, floor((n + 1) / 2)) of dtype ``np.float64``

        The shape and data types for the frequency domain are as follows:

        - time/space not shifted, no symmetry → (m, n + 1) of dtype ``np.complex128``
        - time/space shifted, no symmetry → (m, n + 1) of dtype ``np.complex128``
        - time/space not shifted, even symmetry → (m, ceil((n + 1) / 2)) of dtype ``np.float64``
        - time/space shifted, even symmetry → (m, ceil((n + 1) / 2)) of dtype ``np.complex128``
        - time/space not shifted, odd symmetry → (m, floor((n + 1) / 2)) of dtype ``np.complex128``
        - time/space shifted, odd symmetry → (m, floor((n + 1) / 2)) of dtype ``np.complex128``

        """  # noqa: E501

        # === Input Validation ===

        # if required, the input parameters are validated
        if validate_parameters:
            n = get_validated_order(n=n)
            alpha = get_validated_alpha(alpha=alpha)
            time_space_x_center = get_validated_offset_along_axis(
                center=time_space_x_center,
                which_axis="x",
            )
            time_space_symmetry = _get_validated_time_space_symmetry(
                time_space_symmetry=time_space_symmetry
            )
            _validate_parameter_combination(
                n=n,
                alpha=alpha,
                time_space_x_center=time_space_x_center,
                time_space_symmetry=time_space_symmetry,
            )

        # independent of whether the validation is disabled, an error is raised if both
        # x and omega are given or neither of them is given
        x_is_given = x is not None
        if x_is_given == (omega is not None):
            raise ValueError(
                "Either 'x' or 'omega' must be given but not both or none of them."
            )

        # === Computation of the Hermite Functions ===

        # the independent variable, the scaling factor, and the center (not necessarily
        # identical to the ``time_space_x_center``) are determined based on the domain
        if x_is_given:
            independent_variable = x
            center_internal = time_space_x_center
        else:
            independent_variable = omega
            alpha = 1.0 / alpha
            center_internal = 0.0

        # the Hermite function basis is computed with skipped parameter validation
        # because this was already done within the constructor
        hermite_basis = hermite_function_basis(
            x=independent_variable,  # type: ignore
            n=n,
            alpha=alpha,
            x_center=center_internal,
            validate_parameters=False,
        )

        # === Post-Processing ===

        # independent of the domain, the Hermite functions need to be sliced based on
        # the symmetry in the time/space domain
        # slicing is only necessary for even and odd symmetry and always has step size
        # 2; the only difference is the start index
        if time_space_symmetry == "even":
            hermite_basis = hermite_basis[::, 0:None:2]
        elif time_space_symmetry == "odd":
            hermite_basis = hermite_basis[::, 1:None:2]

        # for the time/space domain, there is nothing more to be done because the
        # definition of the Hermite functions was chosen to make them obey the most
        # basic definition of Hermite functions without any modifications
        if x_is_given:
            return hermite_basis

        # for the frequency domain, the respective pre-factors to account for the shift
        # are applied
        fourier_prefactors = _get_frequency_domain_shift_prefactors(
            omega=omega,  # type: ignore
            time_space_x_center=time_space_x_center,  # type: ignore
        )
        # NOTE: the factors are a column vector to leverage NumPy's broadcasting and
        #       multiply every column of the Hermite functions with the respective
        #       pre-factors
        if fourier_prefactors is not None:
            hermite_basis = hermite_basis * fourier_prefactors  # type: ignore

        # besides, the respective pre-factors to account for the Fourier transform and
        # the symmetry are applied
        fourier_prefactors = _get_frequency_domain_hermite_complex_prefactors(
            num_effective_n=_get_num_effective_n(
                n=n,  # type: ignore
                time_space_symmetry=time_space_symmetry,  # type: ignore
            ),
            time_space_symmetry=time_space_symmetry,  # type: ignore
        )
        # NOTE: the factors are a row vector to leverage NumPy's broadcasting and
        #       multiply every row of the Hermite functions with the respective
        #       pre-factors
        if fourier_prefactors.dtype == hermite_basis.dtype:  # type: ignore
            hermite_basis *= fourier_prefactors  # type: ignore
        else:
            hermite_basis = hermite_basis * fourier_prefactors  # type: ignore

        return hermite_basis

    # --- Magic Methods ---

    def __len__(self):
        """
        Returns the number of Hermite basis functions that will be computed with the
        given parameters, i.e., the order and the symmetry.

        """

        return _get_num_effective_n(
            n=self._n,
            time_space_symmetry=self._time_space_symmetry,
        )

    def __call__(
        self,
        x: Union[RealScalar, ArrayLike, None] = None,
        omega: Union[RealScalar, ArrayLike, None] = None,
    ) -> NDArray[Union[np.float64, np.complex128]]:
        """
        Evaluates the Hermite functions at the given points in the specified domain.
        Please refer to the docstring of the static method :func:`eval` for further
        details.

        """  # noqa: E501

        # first, the current parameter combination of the class is validated in a lazy
        # fashion, i.e., if it has not been done yet
        if not self._is_fully_validated:
            self._validate_parameter_combination()

        # the Hermite functions are computed with skipped parameter validation
        return self.eval(
            x=x,
            omega=omega,
            n=self._n,
            alpha=self._alpha,
            time_space_x_center=self._time_space_x_center,
            time_space_symmetry=self._time_space_symmetry,
            validate_parameters=False,
        )
