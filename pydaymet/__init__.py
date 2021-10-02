"""Top-level package for PyDaymet."""
from pkg_resources import DistributionNotFound, get_distribution

from .core import Daymet
from .exceptions import (
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingCRS,
    MissingItems,
)
from .pet import potential_et
from .print_versions import show_versions
from .pydaymet import get_bycoords, get_bygeom

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "999"

__all__ = [
    "Daymet",
    "get_bycoords",
    "get_bygeom",
    "potential_et",
    "show_versions",
    "InvalidInputRange",
    "InvalidInputType",
    "InvalidInputValue",
    "MissingItems",
    "MissingCRS",
]
