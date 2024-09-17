"""
This script implements utility classes and functions that are used in the ``07``
examples. It is not intended to be executed.

"""

# === Imports ===

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from matplotlib.axes import Axes
from matplotlib.patches import ArrowStyle

# === Types ===

FourierDomains = Literal["time_space", "frequency"]
ComplexAxis = Literal["real", "imaginary"]

# === Models ===


@dataclass
class AxisLimits:
    """
    Holds the limits of an axis.

    """

    lower: float
    upper: float

    # --- Properties ---

    @property
    def width(self) -> float:
        return self.upper - self.lower

    # --- Methods ---

    def get_scaled_width(self, scale: float) -> float:
        return self.width * scale


@dataclass
class ArrowProperties:
    """
    Holds the properties of an arrow.

    """

    arrowstyle: ArrowStyle
    connectionstyle: str
    shrinkA: float
    shrinkB: float
    facecolor: Optional[str]
    edgecolor: Optional[str] = None
    linewidth: float = 2.0

    def __post_init__(self):
        if self.edgecolor is None:
            self.edgecolor = self.facecolor

    # --- Methods ---

    def as_arrow_properties(
        self,
        rad: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns the arrow properties as a dictionary.

        """

        arrow_properties = asdict(self)
        if rad is not None:
            arrow_properties["connectionstyle"] = arrow_properties[
                "connectionstyle"
            ].format(rad=rad)

        return arrow_properties


# === Constants ===

# the fontsize for the labels
LABEL_FONTSIZE = 18

# the label text for the center arrow
CENTER_ARROW_LABEL = "Center"
# the label text for the fadeout arrows
FADEOUT_TIME_SPACE_LABEL = "Fade-\nout"
FADEOUT_FREQUENCY_LABEL = "Band-\nlimit"

# the arrow properties for the center arrow ...
CENTER_ARROWPROPS = ArrowProperties(
    arrowstyle=ArrowStyle(
        "Fancy",
        head_length=0.6,
        head_width=0.6,
        tail_width=0.4,
    ),
    shrinkA=5.0,
    shrinkB=0.0,
    connectionstyle="arc3,rad={rad:.1f}",
    facecolor="black",
    linewidth=2.0,
)
# ... and the fadeout arrows
FADEOUT_ARROWPROPS = ArrowProperties(
    arrowstyle=ArrowStyle(
        "Fancy",
        head_length=0.6,
        head_width=0.6,
        tail_width=0.4,
    ),
    shrinkA=5.0,
    shrinkB=0.0,
    connectionstyle="arc3,rad=0.0",
    facecolor="white",
    edgecolor="black",
    linewidth=2.0,
)

# the bbox property dictionary for normal text annotations
NORMAL_TEXT_ANNOTATION_BBOX_PROPS = dict(
    boxstyle="round",
    facecolor="white",
    edgecolor="none",
)

# a bbox property dictionary for the text annotations with reduced padding
REDUCED_PADDING_TEXT_ANNOTATION_BBOX_PROPS = dict(
    boxstyle="round",
    facecolor="white",
    edgecolor="none",
    pad=0.05,
)

# an empty textbox bbox property dictionary
EMPTY_TEXTBOX_BBOX_PROPS = dict(
    boxstyle="round",
    facecolor="none",
    edgecolor="none",
)

# a textbox bbox property assignment dictionary for the normal labelling textboxes
NORMAL_TEXTBOX_BBOX_PROPS_ASSIGNMENT = {
    CENTER_ARROW_LABEL: NORMAL_TEXT_ANNOTATION_BBOX_PROPS,
    FADEOUT_FREQUENCY_LABEL: NORMAL_TEXT_ANNOTATION_BBOX_PROPS,
    FADEOUT_TIME_SPACE_LABEL: NORMAL_TEXT_ANNOTATION_BBOX_PROPS,
    "\n": EMPTY_TEXTBOX_BBOX_PROPS,
    "": EMPTY_TEXTBOX_BBOX_PROPS,
}

# a textbox bbox property assignment dictionary for the labelling textboxes with
# reduced padding
REDUCED_PADDING_TEXTBOX_BBOX_PROPS_ASSIGNMENT = {
    CENTER_ARROW_LABEL: REDUCED_PADDING_TEXT_ANNOTATION_BBOX_PROPS,
    FADEOUT_FREQUENCY_LABEL: REDUCED_PADDING_TEXT_ANNOTATION_BBOX_PROPS,
    FADEOUT_TIME_SPACE_LABEL: REDUCED_PADDING_TEXT_ANNOTATION_BBOX_PROPS,
    "\n": EMPTY_TEXTBOX_BBOX_PROPS,
    "": EMPTY_TEXTBOX_BBOX_PROPS,
}

# === Functions ===


def is_even(
    value: int,
) -> bool:
    """
    Returns whether the given value is even.

    """

    return value % 2 == 0


def is_desired_axis(
    iter_i: int,
    domain: Union[FourierDomains, List[FourierDomains]],
    complex_axis: Union[ComplexAxis, List[ComplexAxis]],
) -> bool:
    """
    Returns whether the given axis is in the desired domain and has the desired
    complex axis.

    """

    # the following axes layout is assumed:
    # - first column => time/space domain
    # - second column => frequency domain
    # - first row => real axis
    # - second row => imaginary axis
    row_i, col_j = divmod(iter_i, 2)

    domain_assignment = dict(
        time_space=0,
        frequency=1,
    )
    complex_axis_assignment = dict(
        real=0,
        imaginary=1,
    )

    if isinstance(domain, str):
        domain = [domain]
    if isinstance(complex_axis, str):
        complex_axis = [complex_axis]

    if len(domain) == 0:
        domain = ["time_space", "frequency"]
    if len(complex_axis) == 0:
        complex_axis = ["real", "imaginary"]

    for dom in domain:
        for c_axis in complex_axis:
            if (
                domain_assignment[dom] == col_j
                and complex_axis_assignment[c_axis] == row_i
            ):
                return True

    return False


def is_time_space_axis(
    iter_i: int,
) -> bool:
    """
    Returns whether the given axis is in the time/space domain.

    """

    return is_desired_axis(
        iter_i=iter_i,
        domain="time_space",
        complex_axis=list(),
    )


def is_real_axis(
    iter_i: int,
) -> bool:
    """
    Returns whether the given axis is in the real axis.

    """

    return is_desired_axis(
        iter_i=iter_i,
        domain=list(),
        complex_axis="real",
    )


def get_axis_limits(
    ax: Axes,
    which: Literal["x", "y"],
) -> AxisLimits:
    """
    Returns the limits of the given axis.

    """

    axis_lim_getter = getattr(ax, f"get_{which}lim")

    axis_lim_left, axis_lim_right = axis_lim_getter()
    axis_limits = AxisLimits(
        lower=axis_lim_left,
        upper=axis_lim_right,
    )

    return axis_limits


def get_both_axis_limits(
    ax: Axes,
) -> tuple[AxisLimits, AxisLimits]:
    """
    Returns the limits of both axes.

    """

    x_limits = get_axis_limits(ax=ax, which="x")
    y_limits = get_axis_limits(ax=ax, which="y")

    return x_limits, y_limits


def get_and_freeze_axis_limits(
    ax: Axes,
    which: Literal["x", "y"],
) -> AxisLimits:
    """
    Freezes the limits of the given axis and returns them.

    """

    axis_lim_setter = getattr(ax, f"set_{which}lim")

    axis_limits = get_axis_limits(ax=ax, which=which)
    axis_lim_setter(axis_limits.lower, axis_limits.upper)

    return axis_limits


def get_and_freeze_both_axis_limits(
    ax: Axes,
) -> tuple[AxisLimits, AxisLimits]:
    """
    Freezes the limits of both axes and returns them.

    """

    x_limits = get_and_freeze_axis_limits(ax=ax, which="x")
    y_limits = get_and_freeze_axis_limits(ax=ax, which="y")

    return x_limits, y_limits


# === Main ===

if __name__ == "__main__":
    raise Exception("This script is not intended to be executed.")
