"""
This script will update the equations of the documentation of the package.

It relies on a Matplotlib-to-LaTeX converter function which was taken from
https://medium.com/@ealbanez/how-to-easily-convert-latex-to-images-with-python-9062184dc815

NOTE: THIS SCRIPT CAN ONLY BE RUN IF THE DEVELOPER MODE IS ENABLED BY SETTING THE
      ENVIRONMENT VARIABLE ``ROBHERMFT_DEVELOPER`` TO ``true``.

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
    "hermite_functions_time_space_domain": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-01-Hermite_Functions_TimeSpace_Domain.svg"
        ),
        latex_expression=(
            r"$\psi_{n}^{\left(\alpha;\mu\right)}\left(x\right)=\frac{exp\left("
            r"-\frac{1}{2}\cdot\left(\frac{x-\mu}{\alpha}\right)^{2}\right)}"
            r"{\sqrt[4]{\pi\cdot\alpha^{2}}\cdot\sqrt{n!\cdot\left("
            r"\frac{2}{\alpha^{2}}\right)^{n}}}"
            r"\cdot H_{n}^{\left(\alpha;\mu\right)}\left(x\right)$"
        ),
    ),
    "hermite_polynomials_time_space_domain": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-02-Hermite_Polynomials_TimeSpace_Domain.svg"
        ),
        latex_expression=(
            r"$H_{n}^{\left(\alpha;\mu\right)}\left(x\right)=(-1)^{n}\cdot exp"
            r"\left(\left(\frac{x-\mu}{\alpha}\right)^{2}\right)\cdot\left("
            r"\frac{d}{dx}\right)^n\cdot exp\left(-\left("
            r"\frac{x-\mu}{\alpha}\right)^{2}\right)$"
        ),
    ),
    "hermite_functions_dilated_to_undilated": EquationSpecification(
        image_path=(
            "../docs/hermite_functions/equations"
            "/HF-03-Hermite_Functions_Dilated_to_Undilated.svg"
        ),
        latex_expression=(
            r"$\psi_{n}^{\left(\alpha;\mu\right)}\left(x\right)=\frac{1}{\sqrt{\alpha}}"
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
            r"$H_{n}\left(x\right)=(-1)^{n}\cdot exp\left(x^{2}\right)\cdot\left("
            r"\frac{d}{dx}\right)^{n}\cdot exp\left(-x^{2}\right)$"
        ),
    ),
}

# the fontsize of the equations
FONTSIZE: float = 15
# the resolution of the images in dots per inch
DPI: int = 100

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

if (
    __name__ == "__main__"
    and os.getenv("ROBHERMFT_DEVELOPER", "false").lower() == "true"
):

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
        "environment variable 'ROBHERMFT_DEVELOPER' to 'true'."
    )
