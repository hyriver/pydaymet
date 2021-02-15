"""Top-level package for PyDaymet."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue, MissingItems
from .print_versions import show_versions
from .pydaymet import get_bycoords, get_bygeom, get_byloc

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
