"""Top-level package for PyDaymet."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pydaymet import exceptions
from pydaymet.core import Daymet, separate_snow
from pydaymet.pet import potential_et
from pydaymet.print_versions import show_versions
from pydaymet.pydaymet import get_bycoords, get_bygeom, get_bystac
from pydaymet._utils import daymet_tiles

try:
    __version__ = version("pydaymet")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "Daymet",
    "__version__",
    "daymet_tiles",
    "exceptions",
    "get_bycoords",
    "get_bygeom",
    "get_bystac",
    "potential_et",
    "separate_snow",
    "show_versions",
]
