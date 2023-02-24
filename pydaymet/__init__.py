"""Top-level package for PyDaymet."""
from importlib.metadata import PackageNotFoundError, version

from pydaymet.core import Daymet
from pydaymet.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingCRSError,
    MissingItemError,
)
from pydaymet.pet import potential_et
from pydaymet.print_versions import show_versions
from pydaymet.pydaymet import get_bycoords, get_bygeom

try:
    __version__ = version("pydaymet")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "Daymet",
    "get_bycoords",
    "get_bygeom",
    "potential_et",
    "show_versions",
    "InputRangeError",
    "InputTypeError",
    "InputValueError",
    "MissingItemError",
    "MissingCRSError",
]
