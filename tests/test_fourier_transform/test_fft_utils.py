"""
This test suite implements the tests for the module :mod:`fourier_transform._fft_utils`.

"""

# === Imports ===

from typing import Literal, Union

import numpy as np
import pytest

from robust_hermite_ft.fourier_transform import (
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
