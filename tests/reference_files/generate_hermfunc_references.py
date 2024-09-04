"""
This script generates the reference values of the dilated Hermite functions by
symbolic calculation and stores them as NumPy binary files. Given that the symbolic
computations with 200 significant digits are very costly, they have to be precomputed
and stored for later use.

Despite the underlying multiprocessing, running ths script will take up to three hours
to complete.

"""  # noqa: E501

# === Imports ===

import json
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from math import sqrt as pysqrt
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from sympy import Symbol as sp_Symbol
from sympy import exp as sp_exp
from sympy import pi as sp_pi
from sympy import sqrt as sp_sqrt
from sympy import symbols as sp_symbols
from tqdm import tqdm

# === Constants ===

# the path to the directory where the NumPy binary files and metadata JSON file are
# stored
FILE_DIRECTORY = "./tests/reference_files/files"

# the name of the metadata JSON file
METADATA_FILENAME = "hermite_functions_metadata.json"

# the base filename for the NumPy binary files
_NUMPY_FILENAME_PREFIX = "ref_hermite_functions_"
_numpy_filename_base = _NUMPY_FILENAME_PREFIX + "order_{order:03d}_alpha_{alpha}.npy"
# the replacement for the decimal point in the scaling factor alpha
_DOT_REPLACEMENT = "-point-"


# === Models ===


@dataclass
class HermiteFunctionsParameters:
    """
    Contains the parameters for the Hermite functions, namely

    - ``n``: the order of the Hermite functions,
    - ``alpha``: the scaling factor of the independent variable ``x``,
    - ``ns_for_single_test``: the orders against which the dedicated functions for the
        evaluation of single Hermite functions are tested.
        If empty, the dedicated functions are not tested.

    """

    n: int
    alpha: float
    ns_for_single_function: List[int] = field(default_factory=list)


@dataclass
class ReferenceHermiteFunctionsMetadata:
    """
    Contains the metadata for the files storing the reference values of the dilated
    Hermite functions, namely

    - ``parameters_mapping``: a mapping of the filenames to the parameters of
        the Hermite functions,
    - ``num_digits``: the number of significant digits used in the symbolic evaluation,
        and
    - ``x_values``: the points at which the Hermite functions are evaluated.

    """

    parameters_mapping: Dict[str, HermiteFunctionsParameters]
    num_digits: int
    x_values: np.ndarray


# === Functions ===


def _eval_sym_hermite_worker(
    row_index: int,
    x: np.ndarray,
    x_sym: sp_Symbol,
    n: int,
    alpha: float,
    expressions: np.ndarray,
    num_digits: int,
) -> Tuple[int, np.ndarray]:
    """
    Worker function to evaluate the Hermite functions at the given points ``x``.

    """

    # the Hermite functions are evaluated at the given points
    hermite_function_values = np.empty(shape=n + 1, dtype=np.float64)

    # the Hermite functions are evaluated using the recurrence relation
    for iter_j in range(0, n + 1):
        # the expression for the Hermite function is evaluated
        hermite_expression = expressions[iter_j]
        hermite_function_values[iter_j] = hermite_expression.subs(
            x_sym, x[row_index] / alpha
        ).evalf(n=num_digits)

    return row_index, hermite_function_values


def _eval_sym_dilated_hermite_function_basis(
    x: np.ndarray,
    n: int,
    alpha: float,
    num_digits: int = 16,
) -> np.ndarray:
    """
    Evaluates the first ``n + 1`` dilated Hermite functions at the given points ``x``.
    They are defined as

    .. image:: docs/hermite_functions/equations/HF-01-Hermite_Functions_TimeSpace_Domain.svg

    Parameters
    ----------
    x : :class:`np.ndarray` of shape (m,)
        The points at which the Hermite functions are evaluated.
    n : :class:`int`
        The order of the Hermite functions.
    alpha : :class:`float`
        The scaling factor of the independent variable ``x`` as ``x / alpha``.
    num_digits : :class:`int`, default=16
        The number of digits used in the symbolic evaluation of the Hermite
        functions.
        For orders ``n >= 50`` and high ``x / alpha``-values, the symbolic
        evaluation might be inaccurate. In this case, going to quadruple precision
        (``n_digits~=32``) or higher might be necessary.

    Returns
    -------
    hermite_func_vander : :class:`np.ndarray` of shape (m, n + 1)
        The values of the first ``n + 1`` dilated Hermite functions evaluated at the
        points ``x`` represented as a Vandermonde matrix.

    """  # noqa: E501

    # the Hermite functions are evaluated using their recurrence relation given by
    # h_{n+1}(x) = sqrt(2 / (n + 1)) * x * h_{n}(x) - sqrt(n / (n + 1)) * h_{n-1}(x)
    # with the initial conditions h_{-1}(x) = 0 and
    # h_{0}(x) = pi**(-1/4) * exp(-x**2 / 2)
    x_sym = sp_symbols("x")
    hermite_expressions = np.empty(shape=(n + 1), dtype=object)

    # the first two Hermite function expressions are defined with the involved
    # Gaussian function not multiplied in yet to avoid the build-up of large
    # expressions
    h_i_minus_1 = 0
    h_i = sp_exp(-(x_sym**2) / 2) / sp_sqrt(sp_sqrt(sp_pi))  # type: ignore
    hermite_expressions[0] = h_i

    # the Hermite functions are evaluated using the recurrence relation
    for iter_j in tqdm(
        range(0, n),
        desc="Generating Hermite expressions",
        leave=False,
    ):
        h_i_plus_1 = (
            sp_sqrt(2 / (iter_j + 1)) * x_sym * h_i
            - sp_sqrt(iter_j / (iter_j + 1)) * h_i_minus_1  # type: ignore
        )
        h_i_minus_1, h_i = h_i, h_i_plus_1
        hermite_expressions[iter_j + 1] = h_i

    # the Hermite functions are evaluated at the given points
    hermite_func_vander = np.empty(shape=(x.size, n + 1), dtype=np.float64)

    # the evaluation is done in parallel to speed up the process but a progress bar
    # is used to keep track of the progress
    with Pool() as pool:
        worker = partial(
            _eval_sym_hermite_worker,
            x=x,
            x_sym=x_sym,
            n=n,
            alpha=alpha,
            expressions=hermite_expressions,
            num_digits=num_digits,
        )
        results = list(
            tqdm(
                pool.imap(worker, range(0, x.size)),
                total=x.size,
                desc="Evaluating Hermite functions",
                leave=False,
            )
        )

    # the results are stored in the matrix
    for row_idx, row_values in results:
        hermite_func_vander[row_idx, ::] = row_values

    return hermite_func_vander / pysqrt(alpha)


# === Main Code ===

if __name__ == "__main__":

    # this part generates NumPy binary files for the first 250 dilated Hermite functions
    # with different scaling factors evaluated at high precision for a series of 501
    # points in the range [-45, 45]
    # NOTE: it is important that the number of points is odd to have a point at
    #       exactly 0

    # --- Setup ---

    # here, the parameters are defined
    # half the number of the points for the positive x-values
    HALF_NUM_X_POINTS = 251
    # the range to consider for the x-values
    HALF_X_RANGE = (0.0, 45.0)
    # the orders where to test the single Hermite function evaluation functions (wow,
    # what a name)
    NS_FOR_SINGLE_FUNCTION = (
        [
            0,  # special case of falling back to the Hermite basis function
            1,  # special odd case after falling back to the Hermite basis function
            2,  # special even case after falling back to the Hermite basis function
            10,  # little even number,
            11,  # little odd number,
        ]
        + np.arange(start=50, stop=251, step=50, dtype=int).tolist()  # large even
        + np.arange(start=51, stop=250, step=50, dtype=int).tolist()  # large odd
    )

    # the orders and scaling factors for the Hermite functions
    N_AND_ALPHA_COMBINATIONS = [
        HermiteFunctionsParameters(
            n=250,
            alpha=0.5,
            ns_for_single_function=NS_FOR_SINGLE_FUNCTION,
        ),
        HermiteFunctionsParameters(
            n=250,
            alpha=1.0,
            ns_for_single_function=NS_FOR_SINGLE_FUNCTION,
        ),
        HermiteFunctionsParameters(
            n=250,
            alpha=2.0,
            ns_for_single_function=NS_FOR_SINGLE_FUNCTION,
        ),
        HermiteFunctionsParameters(n=0, alpha=0.5),  # special case of early return
        HermiteFunctionsParameters(n=0, alpha=1.0),  # special case of early return
        HermiteFunctionsParameters(n=0, alpha=2.0),  # special case of early return
        HermiteFunctionsParameters(n=1, alpha=0.5),  # special case after early return
        HermiteFunctionsParameters(n=1, alpha=1.0),  # special case after early return
        HermiteFunctionsParameters(n=1, alpha=2.0),  # special case after early return
    ]
    # the number of significant digits used in the symbolic evaluation
    NUM_DIGITS = 200

    # --- Generation of the reference data ---

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

    # the Hermite functions are evaluated for the different scaling factors and saved
    filename_parameters_mapping: Dict[str, HermiteFunctionsParameters] = dict()

    progress_bar.set_description("Generating Hermite reference data")
    for parameters in N_AND_ALPHA_COMBINATIONS:
        # the basis of the Hermite functions is generated ...
        hermite_func_vander = _eval_sym_dilated_hermite_function_basis(
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
        np.save(file=filepath, arr=hermite_func_vander, allow_pickle=False)
        filename_parameters_mapping[filename] = parameters

        progress_bar.update(1)

    # finally, the metadata for the generation is stored in a JSON file
    progress_bar.set_description("Storing metadata")
    reference_metadata = ReferenceHermiteFunctionsMetadata(
        parameters_mapping=filename_parameters_mapping,
        num_digits=NUM_DIGITS,
        x_values=x_values.tolist(),
    )
    filepath = os.path.join(FILE_DIRECTORY, METADATA_FILENAME)

    with open(filepath, "w") as file:
        json.dump(asdict(reference_metadata), file, indent=4)

    progress_bar.update(1)
