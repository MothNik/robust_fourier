"""
This script generates the reference values of the dilated Hermite polynomials by
symbolic calculation and stores them as NumPy binary files. Given that the symbolic
computations with 100 significant digits are very costly, they have to be precomputed
and stored for later use.

Despite the underlying multiprocessing, running ths script will take up to an hour to
complete.

"""  # noqa: E501

# === Imports ===

import json
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Dict, Tuple

import numpy as np
from sympy import Mul, diff, exp, symbols
from tqdm import tqdm

# === Constants ===

# the path to the directory where the NumPy binary files and metadata JSON file are
# stored
FILE_DIRECTORY = "./tests/reference_files/files"

# the name of the metadata JSON file
METADATA_FILENAME = "hermite_polynomials_metadata.json"

# the base filename for the NumPy binary files
_NUMPY_FILENAME_PREFIX = "ref_hermite_polynomials_"
_numpy_filename_base = _NUMPY_FILENAME_PREFIX + "order_{order:03d}_alpha_{alpha}.npy"
# the replacement for the decimal point in the scaling factor alpha
_DOT_REPLACEMENT = "-point-"


# === Models ===


@dataclass
class HermitePolynomialParameters:
    """
    Contains the parameters for the Hermite polynomials, namely

    - ``n``: the order of the Hermite polynomial, and
    - ``alpha``: the scaling factor of the independent variable ``x``.

    """

    n: int
    alpha: float


@dataclass
class ReferenceHermitePolynomialsMetadata:
    """
    Contains the metadata for the files storing the reference values of the dilated
    Hermite polynomials.

    """

    filename_parameters_mapping: Dict[str, HermitePolynomialParameters]
    num_digits: int
    x_values: np.ndarray


# === Functions ===


def _generate_dilated_hermite_polynomial_expression(n: int) -> Mul:
    """
    Generates a symbolic representation of the ``n``-th dilated Hermite polynomial that
    is defined as

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    Parameters
    ----------
    n : :class:`int`
        The order of the Hermite polynomial.

    Returns
    -------
    hermite_polynomial : :class:`sympy.Mul`
        The symbolic representation of the ``n``-th dilated Hermite polynomial.

    """

    # the hermite polynomials are defined as the product of an exponential and the
    # derivative of a Gaussian
    x, alpha = symbols("x, alpha")
    x_dilated_squared = x * x / (alpha * alpha)
    prefactor = 1 if n % 2 == 0 else -1
    exponential = exp(x_dilated_squared)
    gaussian_derivative = diff(exp(-x_dilated_squared), x, n)

    return prefactor * exponential * gaussian_derivative  # type: ignore


def _evaluate_single_hermite_polynomial(
    args: Tuple[int, np.ndarray, float, int],
) -> np.ndarray:
    """
    Evaluates the ``n``-th dilated Hermite polynomial at the given points ``x``.

    """

    iter_j, x, alpha, num_digits = args
    hermite_polynomial_func = _generate_dilated_hermite_polynomial_expression(n=iter_j)
    return np.array(
        [
            hermite_polynomial_func.evalf(
                subs={"x": x_value, "alpha": alpha}, n=num_digits
            )
            for x_value in x
        ]
    )


def _dilated_hermite_polynomial_basis(
    x: np.ndarray,
    n: int,
    alpha: float,
    num_digits: int = 16,
) -> np.ndarray:
    """
    Evaluates the first ``n + 1`` dilated Hermite polynomials at the given points ``x``.
    They are defined as

    .. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png

    Parameters
    ----------
    x : :class:`np.ndarray` of shape (m,)
        The points at which the Hermite polynomials are evaluated.
    n : :class:`int`
        The order of the Hermite polynomials.
    alpha : :class:`float`
        The scaling factor of the independent variable ``x``.
    num_digits : :class:`int`, default=16
        The number of digits used in the symbolic evaluation of the Hermite polynomials.
        For orders ``n >= 50`` and high ``x / alpha``-values, the symbolic evaluation
        might be inaccurate. In this case, going to quadruple precision
        (``n_digits~=32``) or higher might be necessary.

    Returns
    -------
    hermite_polynomial_basis : :class:`np.ndarray` of shape (m, n + 1)
        The values of the first ``n + 1`` dilated Hermite polynomials evaluated at the
        points ``x``.

    """

    # the Hermite polynomials are evaluated symbolically and the results are stored in a
    # NumPy array
    # NOTE: multiprocessing is used to speed up the computation
    hermite_polynomial_basis = np.empty(shape=(x.size, n + 1), dtype=np.float64)
    args_list = [(iter_j, x, alpha, num_digits) for iter_j in range(0, n + 1)]

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap(_evaluate_single_hermite_polynomial, args_list),
                total=n + 1,
                desc=f"Generating Hermite polynomials for alpha {alpha:.1f}",
                leave=False,
            )
        )

    for iter_j, res in enumerate(results):
        hermite_polynomial_basis[:, iter_j] = res

    return hermite_polynomial_basis


# === Main ===

# this part generates NumPy binary files for the first 200 dilated Hermite polynomials
# with different scaling factors evaluated at high precision for a series of 101 points
# in the range [-10, 10]
# NOTE: it is important that the number of points is odd to have a point at exactly 0

if __name__ == "__main__":

    # first, the parameters are defined
    # half the number of the points for the positive x-values
    HALF_NUM_X_POINTS = 51
    # the range to consider for the x-values
    HALF_X_RANGE = (0.0, 10.0)
    # the orders and scaling factors for the Hermite polynomials
    N_AND_ALPHA_COMBINATIONS = [
        HermitePolynomialParameters(n=0, alpha=0.5),  # special case of early return
        HermitePolynomialParameters(n=0, alpha=1.0),  # special case of early return
        HermitePolynomialParameters(n=0, alpha=2.0),  # special case of early return
        HermitePolynomialParameters(n=1, alpha=0.5),  # special case of early return
        HermitePolynomialParameters(n=1, alpha=1.0),  # special case of early return
        HermitePolynomialParameters(n=1, alpha=2.0),  # special case of early return
        HermitePolynomialParameters(n=200, alpha=0.5),
        HermitePolynomialParameters(n=200, alpha=1.0),
        HermitePolynomialParameters(n=200, alpha=2.0),
    ]
    # the number of significant digits used in the symbolic evaluation
    NUM_DIGITS = 100

    # the x-values are generated
    x_values = np.linspace(
        start=HALF_X_RANGE[0],
        stop=HALF_X_RANGE[1],
        num=HALF_NUM_X_POINTS,
        dtype=np.float64,
    )
    x_values = np.concatenate((-np.flip(x_values[1::]), x_values))
    assert np.isin(0.0, x_values), "The x-values do not contain 0.0."

    # first, all the existing files are removed from the directory
    progress_bar = tqdm(
        total=len(N_AND_ALPHA_COMBINATIONS) + 2,
        desc="Deleting existing files",
    )

    for filename in os.listdir(FILE_DIRECTORY):
        if filename.startswith(_NUMPY_FILENAME_PREFIX) or filename == METADATA_FILENAME:
            os.remove(os.path.join(FILE_DIRECTORY, filename))

    progress_bar.update(1)

    # the Hermite polynomials are evaluated for the different scaling factors and saved
    filename_parameters_mapping = dict()

    progress_bar.set_description("Generating Hermite polynomials")
    for parameters in N_AND_ALPHA_COMBINATIONS:
        # the basis of the Hermite polynomials is generated ...
        hermite_polynomial_basis = _dilated_hermite_polynomial_basis(
            x=x_values,
            n=parameters.n,
            alpha=parameters.alpha,
            num_digits=NUM_DIGITS,
        )

        # ... and stored in a NumPy binary file
        alpha_str = f"{parameters.alpha:.1f}".replace(".", _DOT_REPLACEMENT)
        filename = _numpy_filename_base.format(
            order=parameters.n,
            alpha=alpha_str,
        )
        filepath = os.path.join(FILE_DIRECTORY, filename)
        np.save(file=filepath, arr=hermite_polynomial_basis, allow_pickle=False)
        filename_parameters_mapping[filename] = asdict(parameters)

        progress_bar.update(1)

    # finally, the metadata for the generation is stored in a JSON file
    progress_bar.set_description("Storing metadata")
    reference_metadata = ReferenceHermitePolynomialsMetadata(
        filename_parameters_mapping=filename_parameters_mapping,
        num_digits=NUM_DIGITS,
        x_values=x_values.tolist(),
    )
    filepath = os.path.join(FILE_DIRECTORY, METADATA_FILENAME)

    with open(filepath, "w") as file:
        json.dump(asdict(reference_metadata), file, indent=4)

    progress_bar.update(1)
