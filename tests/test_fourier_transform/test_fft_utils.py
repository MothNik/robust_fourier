"""
This test suite implements the tests for the module :mod:`fourier_transform._fft_utils`.

"""

# === Imports ===

from math import isclose as pyisclose
from typing import Literal, Optional, Union

import numpy as np
import pytest

from robust_fourier.fourier_transform import (
    TimeSpaceSignal,
    convert_continuous_to_discrete_ft,
    convert_discrete_to_continuous_ft,
    discrete_ft,
)

# === Constants ===

# the cutoff factor for the Gaussian functions
GAUSSIAN_SIGMA_CUTOFF_FACTOR = 12.0
# the number of points for the Gaussian functions
GAUSSIAN_NUM_POINTS = 10_001
# the relative and absolute tolerances for the tests against the Gaussian functions
GAUSSIAN_RTOL = 1e-13
GAUSSIAN_ATOL = 1e-13

# === Auxiliary Functions ===


def gaussian(x: np.ndarray, sigma: Union[float, int]) -> np.ndarray:
    """
    Calculates an unscaled Gaussian function with standard deviation ``sigma``.

    """

    return np.exp(-(0.5 / sigma / sigma) * np.square(x))


def gaussian_cft(
    angular_frequencies: np.ndarray, sigma: Union[float, int]
) -> np.ndarray:
    """
    Calculates the continuous Fourier transform of an unscaled Gaussian function with
    standard deviation ``sigma``.

    """

    return sigma * np.exp(-0.5 * sigma * sigma * np.square(angular_frequencies))


# === Tests ===


@pytest.mark.parametrize(
    "x, y, x_attribute, delta_x",
    [
        (  # Test 0: explicit x
            np.array([2.0, 4.0, 6.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            2.0,
        ),
        (  # Test 1: semi-implicit x
            np.array([]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            1.0,
        ),
        (  # Test 2: implicit x
            None,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            1.0,
        ),
    ],
)
def test_time_space_signal_initialisation_normal_input(
    x: Optional[np.ndarray],
    y: np.ndarray,
    x_attribute: np.ndarray,
    delta_x: float,
) -> None:
    """
    Tests the initialisation of the dataclass :class:`TimeSpaceSignal` for normal
    cases.

    """

    # the signal is initialised
    if x is not None:
        signal = TimeSpaceSignal(x=x, y=y)
    else:
        signal = TimeSpaceSignal(y=y)

    # the x-values are checked ...
    assert np.array_equal(signal.x, x_attribute), "The x-values are incorrect."
    # ... followed by the y-values ...
    assert np.array_equal(signal.y, y), "The y-values are incorrect."
    # ... and the delta x-value
    assert pyisclose(
        a=signal.delta_x,
        b=delta_x,
        abs_tol=0.0,
        rel_tol=1e-15,
    ), "The delta x-value is incorrect."


@pytest.mark.parametrize(
    "x, error",
    [
        (  # Test 0: x is sorted in descending order
            np.array([2.0, 1.0, 0.0]),
            ValueError("The grid points are not sorted in strictly ascending order."),
        ),
        (  # Test 1: x contains a duplicate entry
            np.array([0.0, 1.0, 1.0]),
            ValueError("The grid points are not sorted in strictly ascending order."),
        ),
        (  # Test 2: all x-values are the same
            np.array([1.0, 1.0, 1.0]),
            ValueError("The grid points are not sorted in strictly ascending order."),
        ),
        (  # Test 3: the spacing between x-values is not constant
            np.array([0.0, 1.0, 3.0, 10.0, 20.0]),
            ValueError("The grid points are not evenly spaced."),
        ),
    ],
)
def test_time_space_signal_initialisation_invalid_input(
    x: np.ndarray,
    error: Exception,
) -> None:
    """
    Tests the initialisation of the dataclass :class:`TimeSpaceSignal` for invalid
    inputs by ensuring that the correct exceptions are raised.

    """

    with pytest.raises(type(error), match=str(error)):
        TimeSpaceSignal(
            x=x,
            y=np.random.rand(*x.shape),  # type: ignore
        )

    return


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_convert_continuous_to_discrete_ft_and_back_with_gaussian(
    norm: Literal["backward", "ortho", "forward"],
    sigma: Union[float, int],
) -> None:
    """
    Tests the functions

    - :func:`discrete_ft`
    - :func:`convert_continuous_to_discrete_ft`
    - :func:`convert_discrete_to_continuous_ft`

    from the module :mod:`fourier_transform._fft_utils` for a Gaussian function for
    which the continuous Fourier transform is known analytically.

    """

    # a densely sampled Gaussian is generated
    # NOTE: it is cut off at a high number of standard deviations to ensure that the
    #       discrete Fourier transform is calculated accurately
    x_stop = GAUSSIAN_SIGMA_CUTOFF_FACTOR * sigma
    x = np.linspace(
        start=-x_stop,
        stop=x_stop,
        num=GAUSSIAN_NUM_POINTS,
    )
    gaussian_values = gaussian(x=x, sigma=sigma)

    # the discrete Fourier transform is calculated
    signal = TimeSpaceSignal(x=x, y=gaussian_values)
    dft = discrete_ft(signal=signal, norm=norm)

    # the discrete Fourier transform is converted back to the continuous domain
    cft = convert_discrete_to_continuous_ft(dft=dft)

    # the continuous Fourier transform of the Gaussian is calculated
    expected_angular_frequencies = (
        2.0 * np.pi * np.fft.fftfreq(n=len(x), d=2 * x_stop / (x.size - 1))
    )
    expected_cft = gaussian_cft(
        angular_frequencies=expected_angular_frequencies,
        sigma=sigma,
    )

    # the comparison for the continuous Fourier transform is performed
    assert np.allclose(
        cft.angular_frequencies,
        expected_angular_frequencies,
    ), "The angular frequency grid is incorrect."
    # NOTE: the comparisons on the real and imaginary part are performed in a very
    #       strict manner to ensure that the implementation is correct
    assert np.allclose(
        cft.real,
        expected_cft.real,
        rtol=GAUSSIAN_RTOL,
        atol=GAUSSIAN_ATOL,
    ), "The real part of the continuous Fourier transform is incorrect."

    assert np.allclose(
        cft.imag,
        expected_cft.imag,
        rtol=GAUSSIAN_RTOL,
        atol=GAUSSIAN_ATOL,
    ), "The imaginary part of the continuous Fourier transform is incorrect."

    # then, the continuous Fourier transform is converted back to the discrete domain
    # and checked against the initial discrete Fourier transform
    dft_reconstructed = convert_continuous_to_discrete_ft(cft=cft, norm=norm)

    # the comparison for the discrete Fourier transform is performed
    assert np.array_equal(
        dft_reconstructed.angular_frequencies,
        dft.angular_frequencies,
    ), "The x-values of the discrete reconstruction are incorrect."

    assert (
        dft_reconstructed.norm == dft.norm
    ), "The normalization of the discrete reconstruction is incorrect."

    # NOTE: the comparisons on the real and imaginary part are performed in a very
    #       strict manner to ensure that the implementation is correct
    assert np.allclose(
        dft_reconstructed.real,
        dft.real,
        rtol=GAUSSIAN_RTOL,
        atol=GAUSSIAN_ATOL,
    ), "The real part of the discrete reconstruction is incorrect."

    assert np.allclose(
        dft_reconstructed.imag,
        dft.imag,
        rtol=GAUSSIAN_RTOL,
        atol=GAUSSIAN_ATOL,
    ), "The imaginary part of the discrete reconstruction is incorrect."
