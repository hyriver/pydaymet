"""Top-level package for PyDaymet."""
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
    import importlib.metadata
except ImportError:
    import importlib_metadata

    __version__ = importlib_metadata.version("pydaymet")
else:
    __version__ = importlib.metadata.version("pydaymet")

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
