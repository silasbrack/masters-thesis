from enum import Enum


class PlotWidth(Enum):
    TEXT = 4.2125984252  # 107 mm
    FULL = 6.33464566929  # 107 mm + 6.2 mm + 47.7 mm
    MARGIN = 1.87795275591  # 47.7 mm


class AspectRatio(Enum):
    GOLDEN = 1.61803398875
    SQUARE = 1.0
    FOUR_BY_THREE = 4.0 / 3.0


def compute_figsize(width: PlotWidth, aspect_ratio: AspectRatio, ncol: int = 1, nrow: int = 1):
    if ncol > 1 and nrow > 1:
        raise ValueError("Cannot have more than one row and column.")

    return width.value, nrow * width.value / ncol / aspect_ratio.value
