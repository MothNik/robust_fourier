"""
This script will update the equations of the documentation of the package.

It relies on a Matplotlib-to-LaTeX converter function which was taken from
https://medium.com/@ealbanez/how-to-easily-convert-latex-to-images-with-python-9062184dc815

NOTE: THIS SCRIPT CAN ONLY BE RUN IF THE DEVELOPER MODE IS ENABLED BY SETTING THE
      ENVIRONMENT VARIABLE ``ROBFT_DEVELOPER`` TO ``true``.

"""  # noqa: E501

# === Imports ===

import os
from dataclasses import dataclass
from typing import List, Optional, Union

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

    def __iter__(self):
        return iter((self.latex_expression, self.full_image_path))


# === Constants ===

# the paths to there the images will be stored (relative to the current file) and their
# respective LaTeX expressions
EQUATION_SPECIFICATIONS = {
    "chebyshev_polynomials_recurrence_relation_first_kind": EquationSpecification(
        image_path=(
            "../docs/chebyshev_polynomials/equations"
            "/CP-01-Chebyshev_Polynomials_Recurrence_Relation_First_Kind.png"
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
            "/CP-02-Chebyshev_Polynomials_Recurrence_Relation_Second_Kind.png"
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
            "/HF-01-Hermite_Functions_TimeSpace_Domain.png"
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
            "/HF-02-Hermite_Polynomials_TimeSpace_Domain.png"
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
            "/HF-03-Hermite_Functions_Dilated_to_Undilated.png"
        ),
        latex_expression=(
            r"$\psi_{n}^{\left(\alpha;\mu\right)}\left(x\right)=\frac{1}{\sqrt{\alpha}}"
            r"\cdot\psi_{n}\left(\frac{x-\mu}{\alpha}\right)$"
        ),
    ),
    "hermite_functions_recurrence_relation": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-04-Hermite_Functions_Recurrence_Relation.png"
        ),
        latex_expression=(
            r"$\psi_{n+1}\left(x\right)=\sqrt{\frac{2}{n+1}}\cdot x\cdot\psi_{n}"
            r"\left(x\right)-\sqrt{\frac{n}{n+1}}\cdot\psi_{n-1}\left(x\right)$"
        ),
    ),
    "hermite_function_basic_definition": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-05-Hermite_Functions_Basic_Definition.png"
        ),
        latex_expression=(
            r"$\psi_{n}\left(x\right)=\frac{exp\left(-\frac{1}{2}\cdot x^{2}\right)}"
            r"{\sqrt[4]{\pi}\cdot\sqrt{n!\cdot 2^{n}}}\cdot H_{n}\left(x\right)$"
        ),
    ),
    "hermite_polynomial_basic_definition": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-06-Hermite_Polynomials_Basic_Definition.png"
        ),
        latex_expression=(
            r"$H_{n}\left(x\right)=(-1)^{n}\cdot exp\left(x^{2}\right)\cdot"
            r"\left(\frac{d}{dx}\right)^{n}\cdot exp\left(-x^{2}\right)$"
        ),
    ),
    "hermite_functions_frequency_domain_part_one": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-07-Hermite_Functions_Frequency_Domain_pt_1.png"
        ),
        latex_expression=(
            r"$\Psi_{n}^{\left(\alpha;\mu\right)}\left(\omega\right)="
            r"\left(-j\right)^{n}\cdot exp\left(-j\cdot\mu\cdot\omega\right)\cdot"
            r"\psi_{n}^{\left(\frac{1}{\alpha};\mu=0\right)}\left(\omega\right)$"
        ),
    ),
    "hermite_functions_frequency_domain_part_two": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-08-Hermite_Functions_Frequency_Domain_pt_2.png"
        ),
        latex_expression=(
            r"$\Psi_{n}^{\left(\alpha;\mu\right)}\left(\omega\right)="
            r"\left(-j\right)^{n}\cdot\frac{exp\left(-\frac{1}{2}\cdot"
            r"\left(\alpha\cdot\omega\right)^{2}-j\cdot\mu\cdot\omega\right)}"
            r"{\sqrt[4]{\frac{\pi}{\alpha^{2}}}\cdot\sqrt{n!\cdot\left("
            r"2\cdot\alpha^{2}\right)^{n}}}"
            r"\cdot H_{n}^{\left(\frac{1}{\alpha};\mu=0\right)}\left(\omega\right)$"
        ),
    ),
}

# the fontsize of the equations
FONTSIZE: float = 10
# the resolution of the images in dots per inch
DPI: int = 1200

# whether to show only the preview of the specified equations
# if None, all equations will be generated and saved
# otherwise, only the equations with the specified indices will be shown as previews
PREVIEW_ONLY_NAMES: Union[str, List[str], None] = None

# === Functions ===


def latex2image(
    latex_expression: str,
    image_path: str,
    fontsize: float,
    dpi: int,
    preview_name: Optional[str] = None,
) -> None:
    """
    A simple function to generate an image from a LaTeX language string.

    Parameters
    ----------
    latex_expression : :class:`str`
        The equation in LaTeX markup language.
    image_path : str or path-like
        The full path to the image file including the file name and extension.
    fontsize : :class:`float`
        The font size of the equation.
    dpi : :class:`int`
        The resolution of the image in dots per inch.
    preview_name : :class:`str` or ``None``, default=``None``
        The name of the equation. If specified, only this equation will be shown with
        this name as a title. Further execution will be blocked until the plot is
        closed.

    """

    fig = plt.figure()
    _ = fig.text(
        x=0.5,
        y=0.5,
        s=latex_expression,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )
    fig.tight_layout()

    # if this is only a preview, the plot is shown and the function returns
    if preview_name is not None:
        fig.suptitle(f"Equation {preview_name}")
        plt.show()

        return

    # otherwise, the plot is saved
    plt.savefig(
        image_path,
        bbox_inches="tight",
        dpi=dpi,
        transparent=False,
    )

    return


# === Main ===

if __name__ == "__main__" and os.getenv("ROBFT_DEVELOPER", "false").lower() == "true":

    # the previews are enabled for the specified indices or disabled if None
    preview_names = [None] * len(EQUATION_SPECIFICATIONS)
    make_fig_names = list(EQUATION_SPECIFICATIONS.keys())
    if PREVIEW_ONLY_NAMES is not None:
        if isinstance(PREVIEW_ONLY_NAMES, str):
            preview_names = [PREVIEW_ONLY_NAMES]  # type: ignore
        else:
            preview_names = PREVIEW_ONLY_NAMES  # type: ignore

        make_fig_names = preview_names  # type: ignore

    progress_bar = tqdm(total=len(make_fig_names), desc="Generating equation images")
    index = 0
    for name, (latex_expression, image_path) in EQUATION_SPECIFICATIONS.items():
        if name not in make_fig_names:
            continue

        latex2image(
            latex_expression=latex_expression,
            image_path=image_path,
            fontsize=FONTSIZE,
            dpi=DPI,
            preview_name=preview_names[index],
        )

        index += 1
        progress_bar.update(1)


elif __name__ == "__main__":
    print(
        "This script can only be run if the developer mode is enabled by setting the "
        "environment variable 'ROBFT_DEVELOPER' to 'true'."
    )
