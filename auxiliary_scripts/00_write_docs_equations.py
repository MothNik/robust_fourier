"""
This script will update the equations of the documentation of the package.

It relies on a Matplotlib-to-LaTeX converter function which was taken from
https://medium.com/@ealbanez/how-to-easily-convert-latex-to-images-with-python-9062184dc815

NOTE: THIS SCRIPT CAN ONLY BE RUN IF THE DEVELOPER MODE IS ENABLED BY SETTING THE
      ENVIRONMENT VARIABLE ``ROBFT_DEVELOPER`` TO ``true``.

"""  # noqa: E501

# === Imports ===

import json
import os
from dataclasses import asdict, dataclass
from hashlib import md5
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.rcParams["mathtext.fontset"] = "cm"  # font is Computer Modern

# === Models ===


@dataclass
class EquationSpecification:
    """
    Specifies the LaTeX expression and the path where the image of the equation will be
    stored.

    """

    image_path: str
    latex_expression: str

    @property
    def full_image_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), self.image_path)

    def __hash__(self) -> int:
        return int(
            md5(
                json.dumps(
                    asdict(self),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            16,
        )


# === Constants ===

# the file path where the hashes of the previous equation specifications are stored
# (relative to the current file)
PREVIOUS_EQUATION_SPECS_HASHES_FILE_PATH = "_equation_specs_hashes.json"

# the paths to there the images will be stored (relative to the current file) and their
# respective LaTeX expressions
EQUATION_SPECIFICATIONS = {
    "generic_fourier_basis_pair": EquationSpecification(
        image_path=("../docs/general/equations/GEN-01-Generic_Fourier_Basis_Pair.svg"),
        latex_expression=(
            r"$\lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
            r"\left( t\right)="
            r"\mathcal{F}^{-1}\left\{\Lambda_{n}^"
            r"{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
            r"\left(\omega\right)\right\}\left( t\right)$"
            + "\n\n"
            + r"$\Lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
            r"\left(\omega\right)="
            r"\mathcal{F}\left\{\lambda_{n}^{"
            r"\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
            r"\left( t\right)\right\}\left(\omega\right)$"
        ),
    ),
    "eulers_formula": EquationSpecification(
        image_path=("../docs/general/equations/GEN-02-Eulers_Formula.svg"),
        latex_expression=(r"$e^{i\cdot x}=cos\left(x\right)+i\cdot sin\left(x\right)$"),
    ),
    "eulers_formula_add_opposite_sign_arguments": EquationSpecification(
        image_path=(
            "../docs/general/equations"
            "/GEN-03-Eulers_Formula_Add_Opposite_Sign_Arguments.svg"
        ),
        latex_expression=(r"$e^{i\cdot x}+e^{-i\cdot x}=2\cdot cos\left(x\right)$"),
    ),
    "eulers_formula_subtract_opposite_sign_arguments": EquationSpecification(
        image_path=(
            "../docs/general/equations"
            "/GEN-04-Eulers_Formula_Subtract_Opposite_Sign_Arguments.svg"
        ),
        latex_expression=(
            r"$e^{i\cdot x}-e^{-i\cdot x}=2\cdot i\cdot sin\left(x\right)$"
        ),
    ),
    "chebyshev_polynomials_recurrence_relation_first_kind": EquationSpecification(
        image_path=(
            "../docs/chebyshev_polynomials/equations"
            "/CP-01-Chebyshev_Polynomials_Recurrence_Relation_First_Kind.svg"
        ),
        latex_expression=(
            r"$T_{n+1}^{\left(\alpha;\mu\right)}\left(x\right)=2\cdot "
            r"\frac{x-\mu}{\alpha}\cdot T_{n}^{\left(\alpha;\mu\right)}\left(x\right)-"
            r"T_{n-1}^{\left(\alpha;\mu\right)}\left(x\right)$"
            + "\n\n"
            + r"$T_{0}^{\left(\alpha;\mu\right)}\left(x\right)=1$"
            + "\n\n"
            + r"$T_{1}^{\left(\alpha;\mu\right)}\left(x\right)=\frac{x-\mu}{\alpha}$"
        ),
    ),
    "chebyshev_polynomials_recurrence_relation_second_kind": EquationSpecification(
        image_path=(
            "../docs/chebyshev_polynomials/equations"
            "/CP-02-Chebyshev_Polynomials_Recurrence_Relation_Second_Kind.svg"
        ),
        latex_expression=(
            r"$U_{n+1}^{\left(\alpha;\mu\right)}\left(x\right)=2\cdot "
            r"\frac{x-\mu}{\alpha}\cdot U_{n}^{\left(\alpha;\mu\right)}\left(x\right)-"
            r"U_{n-1}^{\left(\alpha;\mu\right)}\left(x\right)$"
            + "\n\n"
            + r"$U_{0}^{\left(\alpha;\mu\right)}\left(x\right)=1$"
            + "\n\n"
            + r"$U_{1}^{\left(\alpha;\mu\right)}\left(x\right)="
            r"2\cdot\frac{x-\mu}{\alpha}$"
        ),
    ),
    "hermite_functions_time_space_domain": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-01-Hermite_Functions_OfGenericX.svg"
        ),
        latex_expression=(
            r"$\psi_{n}^{\left(\alpha;\mu\right)}\left(x\right)=\frac{"
            r"exp\left(-\frac{1}{2}\cdot\left(\frac{x-\mu}{\alpha}\right)^{2}\right)}"
            r"{\sqrt[4]{\pi\cdot\alpha^{2}}\cdot\sqrt{n!\cdot"
            r"\left(\frac{2}{\alpha^{2}}\right)^{n}}}"
            r"\cdot H_{n}^{\left(\alpha;\mu\right)}\left(x\right)$"
        ),
    ),
    "hermite_polynomials_time_space_domain": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-02-Hermite_Polynomials_OfGenericX.svg"
        ),
        latex_expression=(
            r"$H_{n}^{\left(\alpha;\mu\right)}\left(x\right)=(-1)^{n}\cdot "
            r"exp\left(\left(\frac{x-\mu}{\alpha}\right)^{2}\right)\cdot"
            r"\left(\frac{d}{dx}\right)^n\cdot "
            r"exp\left(-\left(\frac{x-\mu}{\alpha}\right)^{2}\right)$"
        ),
    ),
    "hermite_functions_dilated_to_undilated": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-03-Hermite_Functions_Dilated_to_Undilated.svg"
        ),
        latex_expression=(
            r"$\psi_{n}^{\left(\alpha;\mu\right)}\left(x\right)="
            r"\frac{1}{\sqrt{\alpha}}"
            r"\cdot\psi_{n}\left(\frac{x-\mu}{\alpha}\right)$"
        ),
    ),
    "hermite_functions_recurrence_relation": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-04-Hermite_Functions_Recurrence_Relation.svg"
        ),
        latex_expression=(
            r"$\psi_{n+1}\left(x\right)=\sqrt{\frac{2}{n+1}}\cdot x\cdot\psi_{n}"
            r"\left(x\right)-\sqrt{\frac{n}{n+1}}\cdot\psi_{n-1}\left(x\right)$"
        ),
    ),
    "hermite_function_basic_definition": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-05-Hermite_Functions_Basic_Definition.svg"
        ),
        latex_expression=(
            r"$\psi_{n}\left(x\right)=\frac{exp\left(-\frac{1}{2}\cdot x^{2}\right)}"
            r"{\sqrt[4]{\pi}\cdot\sqrt{n!\cdot 2^{n}}}\cdot H_{n}\left(x\right)$"
        ),
    ),
    "hermite_polynomial_basic_definition": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-06-Hermite_Polynomials_Basic_Definition.svg"
        ),
        latex_expression=(
            r"$H_{n}\left(x\right)=(-1)^{n}\cdot exp\left(x^{2}\right)\cdot"
            r"\left(\frac{d}{dx}\right)^{n}\cdot exp\left(-x^{2}\right)$"
        ),
    ),
    (
        "hermite_functions_derived_basis_frequency_from_freq_at_origin_time_space_at_"
        "origin"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-07-Hermite_Functions_Derived_Basis_Frequency_from_Frequency_at_"
                "Origin_TimeSpace_at_Origin.svg"
            ),
            latex_expression=(
                r"$\Lambda_{n}^{\left(\beta;\gamma; t_{0}=0;\omega_{0}=0\right)}"
                r"\left(\omega\right)="
                r"\left(-i\right)^{n}\cdot\psi_{n}^{\left(\alpha=\gamma;\mu=0\right)}"
                r"\left(\omega\right)=\left(-i\right)^{n}"
                r"\cdot\frac{exp\left(-\frac{1}{2}"
                r"\cdot\left(\frac{\omega}{\gamma}\right)^{2}\right)}"
                r"{\sqrt[4]{\pi\cdot\gamma^{2}}\cdot\sqrt{n!\cdot"
                r"\left(\frac{2}{\gamma^{2}}\right)^{n}}}"
                r"\cdot H_{n}^{\left(\alpha=\gamma;\mu=0\right)}\left(\omega\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_time_space_from_freq_at_origin_time_space_"
        "at_origin"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-08-Hermite_Functions_Derived_Basis_TimeSpace_from_Frequency_at_"
                "Origin_TimeSpace_at_Origin.svg"
            ),
            latex_expression=(
                r"$\lambda_{n}^{\left(\beta;\gamma; t_{0}=0;\omega_{0}=0\right)}"
                r"\left( t\right)="
                r"\psi_{n}^{\left(\alpha=\beta;\mu=0\right)}\left( t\right)$"
                + "\n\n"
                + r"$\beta=\frac{1}{\gamma}$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_time_space_from_freq_at_origin_time_space_"
        "shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-09-Hermite_Functions_Derived_Basis_Time_Space_from_Frequency_at_"
                "Origin_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}=0\right)}"
                r"\left( t\right)="
                r"\psi_{n}^{\left(\alpha=\beta;\mu=t_{0}\right)}\left( t\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_frequency_from_freq_at_origin_time_"
        "space_shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-10-Hermite_Functions_Derived_Basis_Frequency_from_Frequency_at_"
                "Origin_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\Lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}=0\right)}"
                r"\left(\omega\right)="
                r"(-i)^{n}\cdot exp\left(-i\cdot t_{0}\cdot\omega\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\gamma;\mu=0\right)}\left(\omega\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_frequency_from_freq_shifted_time_space_"
        "at_origin"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-11-Hermite_Functions_Derived_Basis_Frequency_from_Frequency_"
                "Shifted_TimeSpace_at_Origin.svg"
            ),
            latex_expression=(
                r"$\Lambda_{n}^{\left(\beta;\gamma; t_{0}=0;\omega_{0}\right)}"
                r"\left(\omega\right)="
                r"(-i)^{n}\cdot\psi_{n}^{\left(\alpha=\gamma;\mu=\omega_{0}\right)}"
                r"\left(\omega\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_time_space_from_freq_shifted_time_space_"
        "at_origin"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-12-Hermite_Functions_Derived_Basis_TimeSpace_from_Frequency_"
                "Shifted_TimeSpace_at_Origin.svg"
            ),
            latex_expression=(
                r"$\lambda_{n}^{\left(\beta;\gamma; t_{0}=0;\omega_{0}\right)}"
                r"\left( t\right)="
                r"exp\left(i\cdot 2\cdot\pi\cdot\omega_{0}\cdot t\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\beta;\mu=0\right)}\left( t\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_frequency_from_freq_shifted_"
        "time_space_shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-13-Hermite_Functions_Derived_Basis_Frequency_from_Frequency_"
                "Shifted_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\Lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
                r"\left(\omega\right)="
                r"(-i)^{n}\cdot exp\left(-i\cdot t_{0}\cdot\omega\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\gamma;\mu=\omega_{0}\right)}"
                r"\left(\omega\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_frequency_complementary_from_freq_shifted_"
        "time_space_shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-14-Hermite_Functions_Derived_Basis_Frequency_Complementary_from_"
                "Frequency_Shifted_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\tilde{\Lambda}_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}="
                r"\left(\omega\right)="
                r"(-i)^{n}\cdot exp\left(-i\cdot t_{0}\cdot\omega\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\gamma;\mu=-\omega_{0}\right)}"
                r"\left(\omega\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_frequency_complementary_flipped_odd_orders_"
        "from_freq_shifted_time_space_shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-15-Hermite_Functions_Derived_Basis_Frequency_Complementary_"
                "Flipped_Odd_Orders_from_Frequency_Shifted_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\tilde{\Lambda}_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}="
                r"\left(\omega\right)="
                r"i^{n}\cdot exp\left(-i\cdot t_{0}\cdot\omega\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\gamma;\mu=-\omega_{0}\right)}"
                r"\left(\omega\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_frequency_symmetrized_from_freq_shifted_time_"
        "space_shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-16-Hermite_Functions_Derived_Basis_Frequency_Symmetrized_from_"
                "Frequency_Shifted_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\widehat{\Lambda}_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}"
                r"\right)}\left(\omega\right)="
                r"\Lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
                r"\left(\omega\right)+"
                r"\tilde{\Lambda}_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
                r"\left(\omega\right)=$" + "\n\n"
                r"$exp\left(-i\cdot t_{0}\cdot\omega\right)\cdot"
                r"\left(i^{n}\cdot\psi_{n}^{\left(\alpha=\gamma;\mu=-\omega_{0}\right)}"
                r"\left(\omega\right)+"
                r"(-i)^{n}\cdot\psi_{n}^{\left(\alpha=\gamma;\mu=\omega_{0}\right)}"
                r"\left(\omega\right)\right)$"
            ),
        )
    ),
    (
        "hermite_functions_derived_basis_time_space_single_bases_symmetrized_from_freq_"
        "shifted_time_space_shifted"
    ): (
        EquationSpecification(
            image_path=(
                "../docs/hermite_functions/equations"
                "/HF-17-Hermite_Functions_Derived_Basis_TimeSpace_Single_Bases_"
                "Symmetrized_from_Frequency_Shifted_TimeSpace_Shifted.svg"
            ),
            latex_expression=(
                r"$\lambda_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
                r"\left( t\right)="
                r"exp\left(i\cdot 2\cdot\pi\cdot\omega_{0}\cdot t\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\beta;\mu=t_{0}\right)}\left( t\right)$"
                + "\n\n"
                r"$\tilde{\lambda}_{n}^{\left(\beta;\gamma; t_{0};\omega_{0}\right)}"
                r"\left( t\right)="
                r"exp\left(-i\cdot 2\cdot\pi\cdot\omega_{0}\cdot t\right)\cdot"
                r"\psi_{n}^{\left(\alpha=\beta;\mu=t_{0}\right)}\left( t\right)$"
            ),
        )
    ),
}

# the fontsize of the equations
FONTSIZE: float = 22
# the resolution of the images in dots per inch
DPI: int = 100

# the list of plots where only the preview of the equation is shown
# if this is empty, all equations are saved as images
# otherwise, only a preview of the equations in this list is shown and nothing is saved
FIG_NAMES_TO_PREVIEW: List[str] = [
    # "generic_fourier_basis_pair",
    # "eulers_formula",
    # "eulers_formula_add_opposite_sign_arguments",
    # "eulers_formula_subtract_opposite_sign_arguments",
    # "chebyshev_polynomials_recurrence_relation_first_kind",
    # "chebyshev_polynomials_recurrence_relation_second_kind",
    # "hermite_functions_time_space_domain",
    # "hermite_polynomials_time_space_domain",
    # "hermite_functions_dilated_to_undilated",
    # "hermite_functions_recurrence_relation",
    # "hermite_function_basic_definition",
    # "hermite_polynomial_basic_definition",
    # (
    #     "hermite_functions_derived_basis_frequency_from_freq_at_origin_time_space_"
    #     "at_origin"
    # ),
    # (
    #     "hermite_functions_derived_basis_time_space_from_freq_at_origin_time_space_"
    #     "at_origin"
    # ),
    # (
    #     "hermite_functions_derived_basis_time_space_from_freq_at_origin_time_space_"
    #     "shifted"
    # ),
    # (
    #     "hermite_functions_derived_basis_frequency_from_freq_at_origin_time_"
    #     "space_shifted"
    # ),
    # (
    #     "hermite_functions_derived_basis_frequency_from_freq_shifted_time_space_"
    #     "at_origin"
    # ),
    # (
    #     "hermite_functions_derived_basis_time_space_from_freq_shifted_time_space_"
    #     "at_origin"
    # ),
    # (
    #     "hermite_functions_derived_basis_frequency_from_freq_shifted_"
    #     "time_space_shifted"
    # ),
    # (
    #     "hermite_functions_derived_basis_frequency_complementary_from_freq_shifted_"
    #     "time_space_shifted"
    # ),
    # (
    #     "hermite_functions_derived_basis_frequency_complementary_flipped_odd_orders_"
    #     "from_freq_shifted_time_space_shifted"
    # ),
    # (
    #     "hermite_functions_derived_basis_frequency_symmetrized_from_freq_shifted_"
    #     "time_space_shifted"
    # ),
    # (
    #     "hermite_functions_derived_basis_time_space_single_bases_symmetrized_from_"
    #     "freq_shifted_time_space_shifted"
    # ),
]

# === Functions ===


def load_previous_equation_specs_hashes(file_path: str) -> Dict[str, int]:
    """
    Loads the hashes of the previous equation specifications.

    """

    if not os.path.exists(file_path):
        return dict()

    with open(file_path, "r") as file:
        previous_hashes = json.load(file)

    return previous_hashes


def save_equation_specs_hashes(
    file_path: str,
    equation_specs_hashes: Dict[str, int],
) -> None:
    """
    Saves the hashes of the equation specifications.

    """

    with open(file_path, "w") as file:
        json.dump(
            equation_specs_hashes,
            file,
            indent=4,
        )

    return


def latex2image(
    equation_name: str,
    equation_specification: EquationSpecification,
    fontsize: float,
    dpi: int,
    preview_only: bool,
) -> None:
    """
    A simple function to generate an image from a LaTeX language string.

    Parameters
    ----------
    equation_name : :class:`str`
        The name of the equation.
    equation_specification : :class:`EquationSpecification`
        The specification of the equation.
    fontsize : :class:`float`
        The font size of the equation.
    dpi : :class:`int`
        The resolution of the image in dots per inch.
    preview_only : :class:`bool`
        Whether to show only the preview of the equation (``True``) or to save the image
        (``False``).

    """

    fig = plt.figure()
    _ = fig.text(
        x=0.5,
        y=0.5,
        s=equation_specification.latex_expression,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )
    fig.tight_layout()

    # if this is only a preview, the plot is shown and the function returns
    if preview_only:
        fig.suptitle(f"Equation {equation_name}")
        plt.show()

        return

    # otherwise, the plot is saved
    plt.savefig(
        equation_specification.full_image_path,
        bbox_inches="tight",
        dpi=dpi,
        transparent=False,
    )

    return


# === Main ===

if __name__ == "__main__" and os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":

    # the previews are handled
    # if there are previews, only these plots will be generated and no hashes are
    # required because no updates are performed
    if len(FIG_NAMES_TO_PREVIEW) > 0:
        fig_names_to_generate = list(set(FIG_NAMES_TO_PREVIEW))
        previous_hashes_file_path = ""
        previous_hashes = dict()

    # if there are no previews, all plots will be generated and the hashes of the
    # previous equation specifications are required to check whether the equations have
    # changed and need to be updated
    else:
        fig_names_to_generate = list(EQUATION_SPECIFICATIONS.keys())
        previous_hashes_file_path = os.path.join(
            os.path.dirname(__file__),
            PREVIOUS_EQUATION_SPECS_HASHES_FILE_PATH,
        )
        previous_hashes = load_previous_equation_specs_hashes(
            file_path=previous_hashes_file_path,
        )

    # all the equations are generated
    progress_bar = tqdm(
        total=len(fig_names_to_generate),
        desc="Generating equation images",
    )
    index = 0
    for name, specification in EQUATION_SPECIFICATIONS.items():
        # if the equation is not in the list of equations to be generated, it is skipped
        if name not in fig_names_to_generate:
            continue

        # it is checked whether the equation should only be previewed
        preview_only = name in FIG_NAMES_TO_PREVIEW

        # if this is not a preview, it has to be checked whether the equation has
        # changed and an update of the saved image is necessary
        if not preview_only:
            # the hash of the current equation specification is computed and
            # compared to the hash of the previous equation specification (if available)
            current_hash = hash(specification)
            previous_hash = previous_hashes.get(name, None)

            # if both hashes coincide AND the file of the image exists, the equation
            # has not changed and will not be updated
            if current_hash == previous_hash and os.path.exists(
                specification.full_image_path
            ):
                tqdm.write(
                    f"Equation '{name}' has not changed and will not be updated."
                )
                index += 1
                progress_bar.update(1)
                continue

        # the image of the equation is generated
        latex2image(
            equation_name=name,
            equation_specification=specification,
            fontsize=FONTSIZE,
            dpi=DPI,
            preview_only=preview_only,
        )

        index += 1
        progress_bar.update(1)

    # if there were no previews, the hashes of the current equation specifications are
    # saved
    if len(FIG_NAMES_TO_PREVIEW) < 1:
        current_equation_specs_hashes = {
            name: hash(specification)
            for name, specification in EQUATION_SPECIFICATIONS.items()
        }
        save_equation_specs_hashes(
            file_path=previous_hashes_file_path,
            equation_specs_hashes=current_equation_specs_hashes,
        )

elif __name__ == "__main__":
    print(
        "This script can only be run if the developer mode is enabled by setting the "
        "environment variable 'ROBFT_DEVELOPER' to 'true'."
    )
