"""
Module :mod:`fourier_transform._fft_utils`

This module provides utility functions for the Fourier transform, such as

- the computation of the angular frequency grid for the discrete Fourier transform
- the conversion from a discrete Fourier transform to a continuous Fourier transform
    and vice versa

"""

# === Imports ===

from dataclasses import dataclass, field
from math import sqrt as pysqrt
from typing import Literal, Union

import numpy as np

# === Constants ===

NORM_BACKWARD = "backward"
NORM_ORTHOGONAL = "ortho"
NORM_FORWARD = "forward"

# === Types ===

_DftNorms = Literal["backward", "ortho", "forward"]
_FTConversionsTo = Literal["continuous", "discrete"]

# === Models ===


@dataclass
class TimeSpaceSignal:
    """
    Dataclass to represent the grid points and the signal in the time or space domain.

    Attributes
    ----------
    y : :class:`numpy.ndarray` of shape ``(n,)``
        The signal in the time or space domain.
    x : :class:`numpy.ndarray` of shape ``(n,)``
        The grid points in the time or space domain.
        It has to be evenly spaced and sorted in ascending order.
        If empty, an index-based grid is assumed.
    delta_x : :class:`float`
        The grid spacing.
        It is computed from the grid points.

    Notes
    -----
    The number of grid points can be obtained by ``len(time_space_domain)``.

    """

    y: np.ndarray
    x: np.ndarray = field(default_factory=lambda: np.array([]))

    delta_x: float = field(init=False)

    def __post_init__(self) -> None:
        # if there are no grid points, an index-based grid is assumed
        if self.x.size < 1:
            self.x = np.arange(
                start=0,
                stop=self.y.size,
                step=1,
                dtype=np.int64,
            ).astype(np.float64)

        # the grid spacing is computed from the grid points
        self.delta_x = grid_spacing(x=self.x)

    def __len__(self) -> int:
        return self.y.size


@dataclass
class _BaseFourierTransform:
    """
    Base class for the representation of a Fourier transform.

    Attributes
    ----------
    angular_frequencies : :class:`numpy.ndarray` of shape ``(n,)``
        The angular frequency grid points.
    time_space_signal : :class:`TimeSpaceDomainRepresentation`
        The grid points and the signal in the time or space domain.

    """

    angular_frequencies: np.ndarray
    time_space_signal: TimeSpaceSignal


@dataclass
class DiscreteFourierTransform(_BaseFourierTransform):
    """
    Dataclass to represent a discrete Fourier transform.

    Attributes
    ----------
    angular_frequencies : :class:`numpy.ndarray` of shape ``(n,)``
        The angular frequency grid points.
    dft : :class:`numpy.ndarray` of shape ``(n,)``
        The discrete Fourier transform.
    norm : {"backward", "ortho", "forward"}
        The normalization to use.
        Please refer to :func:`discrete_ft` for more details.
    time_space_signal : :class:`TimeSpaceDomainRepresentation`
        The grid points and the signal in the time or space domain.

    Properties
    ----------
    real : :class:`numpy.ndarray` of shape ``(n,)``
        The real part of the discrete Fourier transform.
    imag : :class:`numpy.ndarray` of shape ``(n,)``
        The imaginary part of the discrete Fourier transform.

    """

    dft: np.ndarray
    norm: _DftNorms

    @property
    def real(self) -> np.ndarray:
        return self.dft.real

    @property
    def imag(self) -> np.ndarray:
        return self.dft.imag


@dataclass
class ContinuousFourierTransform(_BaseFourierTransform):
    """
    Dataclass to represent a continuous Fourier transform.

    Attributes
    ----------
    angular_frequencies : :class:`numpy.ndarray` of shape ``(n,)``
        The angular frequency grid points.
    cft : :class:`numpy.ndarray` of shape ``(n,)``
        The continuous Fourier transform.
    time_space_signal : :class:`TimeSpaceDomainRepresentation`
        The grid points and the signal in the time or space domain.

    Properties
    ----------
    real : :class:`numpy.ndarray` of shape ``(n,)``
        The real part of the continuous Fourier transform.
    imag : :class:`numpy.ndarray` of shape ``(n,)``
        The imaginary part of the continuous Fourier transform.

    """

    cft: np.ndarray

    @property
    def real(self) -> np.ndarray:
        return self.cft.real

    @property
    def imag(self) -> np.ndarray:
        return self.cft.imag


# === Functions ===


def grid_spacing(x: np.ndarray) -> float:
    """
    Computes the grid spacing from a an evenly spaced grid.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape ``(n,)``
        The grid points.
        It has to be evenly spaced and sorted in strictly ascending order. Strictly
        ascending means that the difference between two consecutive grid points is
        strictly positive.

    Returns
    -------
    d : :class:`float`
        The grid spacing.

    Raises
    ------
    ValueError
        If the grid is not evenly spaced.

    """

    delta_x = (x[-1] - x[0]) / (x.size - 1)
    diff_x = np.diff(x)
    if (diff_x <= 0.0).any():
        raise ValueError("The grid points are not sorted in strictly ascending order.")

    if not np.allclose(
        diff_x,
        np.full(shape=(x.size - 1,), fill_value=delta_x),
    ):
        raise ValueError("The grid points are not evenly spaced.")

    return delta_x


def angular_frequency_grid(
    n: int,
    d: Union[float, int] = 1.0,
) -> np.ndarray:
    """
    Compute the angular frequency grid for the discrete Fourier transform.
    It is given by :math:`\\angular_frequencies = \\frac{2\\cdot\\pi\\cdot k}{n\\cdot d}`
    where

    * :math:`n` is the number of grid points
    * :math:`d` is the grid spacing

    Parameters
    ----------
    n : :class:`int`
        The number of grid points.
    d : :class:`float` or :class:`int`, default=1.0
        The grid spacing.

    Returns
    -------
    angular_frequencies : :class:`numpy.ndarray` of shape ``(n,)``
        The angular frequency grid.

    """  # noqa: E501

    return 2 * np.pi * np.fft.fftfreq(n=n, d=d)


def discrete_ft(
    signal: TimeSpaceSignal,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> DiscreteFourierTransform:
    """
    Object-oriented wrapper for the NumPy function :func:`numpy.fft.fft` to compute the
    discrete Fourier transform of a signal.

    Parameters
    ----------
    signal : :class:`TimeSpaceSignal`
        The signal in the time or space domain.
        For more details, please refer to :class:`TimeSpaceSignal`.
    norm : {"backward", "ortho", "forward"}, default="backward"
        The normalization to use.

        * ``"backward"``: no normalization is applied. Then, the inverse Fourier
            transform has to be multiplied by ``1/n``.
        * ``"ortho"``: the normalization factor is ``1/sqrt(n)`` for both the forward
            and backward Fourier transform.
        * ``"forward"``: the normalization factor is ``1/n`` for the Fourier transform
            and no normalization is needed for the inverse Fourier transform.

    Returns
    -------
    discrete_ft : :class:`DiscreteFourierTransform`
        The discrete Fourier transform.
        For more details, please refer to :class:`DiscreteFourierTransform`.

    """

    # the discrete Fourier transform is computed together with the angular frequency
    # grid ...
    angular_frequencies = angular_frequency_grid(n=len(signal), d=signal.delta_x)
    discrete_ft = np.fft.fft(a=signal.y, norm=norm)

    # ... and stored in a dataclass
    return DiscreteFourierTransform(
        angular_frequencies=angular_frequencies,
        dft=discrete_ft,
        norm=norm,
        time_space_signal=signal,
    )


def _ft_conversion_factors(
    x: np.ndarray,
    delta_x: float,
    angular_frequencies: np.ndarray,
    norm: _DftNorms,
    to_kind: _FTConversionsTo,
) -> np.ndarray:
    """
    Compute the factors for the conversion from a discrete Fourier transform to a
    continuous Fourier transform or vice versa.

    For the conversion from a discrete Fourier transform to a continuous Fourier
    transform, the correction is given by

    .. code-block:: python

        correction_factors = (phi * delta_x / np.sqrt(2 * np.pi)) *  * np.exp(-1j * angular_frequencies * x[0])
        continuous_ft = discrete_ft * correction_factors

    while for the conversion from a continuous Fourier transform to a discrete Fourier
    transform, the correction is given by

    .. code-block:: python

        correction_factors = (np.sqrt(2 * np.pi) / phi / delta_x) * np.exp(1j * angular_frequencies * x[0])
        discrete_ft = continuous_ft * correction_factors

    Depending on the normalization, the factor :math:`\\phi` is different:

    * ``"backward"``: :math:`\\phi = 1`
    * ``"ortho"``: :math:`\\phi = \\sqrt{n}`
    * ``"forward"``: :math:`\\phi = n`

    where :math:`n` is the number of grid points.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape ``(n,)``
        The grid points in the time or space domain.
    delta_x : :class:`float`
        The grid spacing.
    angular_frequencies : :class:`numpy.ndarray` of shape ``(n,)``
        The angular frequency grid.
    norm : {"backward", "ortho", "forward"}, default="backward"
        The normalization to use.
        Please refer to :func:`discrete_ft` for more details.
    from_kind : {"continuous", "discrete"}, default="discrete"
        Whether the conversion is from a discrete Fourier transform to a continuous
        Fourier transform (``"continuous"``) or vice versa (``"discrete"``).

    Returns
    -------
    correction_factors : :class:`numpy.ndarray` of shape ``(n,)``
        The correction factors.

    Raises
    ------
    ValueError
        If the normalization is unknown.

    """  # noqa: E501

    norm_inter = norm.lower()
    if norm_inter == NORM_BACKWARD:
        phi = 1.0
    elif norm_inter == NORM_ORTHOGONAL:
        phi = pysqrt(x.size)
    elif norm_inter == NORM_FORWARD:
        phi = x.size
    else:  # pragma: no cover
        raise ValueError(f"Unknown normalization '{norm}'.")

    if to_kind == "continuous":
        return (phi * delta_x / pysqrt(2.0 * np.pi)) * np.exp(
            -1j * angular_frequencies * x[0]
        )
    else:
        return (pysqrt(2.0 * np.pi) / phi / delta_x) * np.exp(
            1j * angular_frequencies * x[0]
        )


def convert_discrete_to_continuous_ft(
    dft: DiscreteFourierTransform,
) -> ContinuousFourierTransform:
    """
    Convert a discrete Fourier transform to a continuous Fourier transform.

    Parameters
    ----------
    dft : :class:`DiscreteFourierTransform`
        The discrete Fourier transform.
        For more details, please refer to :class:`DiscreteFourierTransform`.

    Returns
    -------
    cft : :class:`ContinuousFourierTransform`
        The continuous Fourier transform.
        For more details, please refer to :class:`ContinuousFourierTransform`.

    """

    # the conversion factors are computed ...
    conversion_factors = _ft_conversion_factors(
        x=dft.time_space_signal.x,
        delta_x=dft.time_space_signal.delta_x,
        angular_frequencies=dft.angular_frequencies,
        norm=dft.norm,
        to_kind="continuous",
    )

    # ... and the continuous Fourier transform is computed
    return ContinuousFourierTransform(
        angular_frequencies=dft.angular_frequencies,
        cft=dft.dft * conversion_factors,
        time_space_signal=dft.time_space_signal,
    )


def convert_continuous_to_discrete_ft(
    cft: ContinuousFourierTransform,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> DiscreteFourierTransform:
    """
    Convert a continuous Fourier transform to a discrete Fourier transform.

    Parameters
    ----------
    cft : :class:`ContinuousFourierTransform`
        The continuous Fourier transform.
        For more details, please refer to :class:`ContinuousFourierTransform`.
    norm : {"backward", "ortho", "forward"}, default="backward"
        The normalization to use.
        Please refer to :func:`discrete_ft` for more details.

    Returns
    -------
    dft : :class:`DiscreteFourierTransform`
        The discrete Fourier transform.
        For more details, please refer to :class:`DiscreteFourierTransform`.

    """

    # the conversion factors are computed ...
    conversion_factors = _ft_conversion_factors(
        x=cft.time_space_signal.x,
        delta_x=cft.time_space_signal.delta_x,
        angular_frequencies=cft.angular_frequencies,
        norm=norm,
        to_kind="discrete",
    )

    # ... and the discrete Fourier transform is computed
    return DiscreteFourierTransform(
        angular_frequencies=cft.angular_frequencies,
        dft=cft.cft * conversion_factors,
        norm=norm,
        time_space_signal=cft.time_space_signal,
    )
