"""Top-level package for PyNHD."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputType, InvalidInputValue, InvalidInputRange, MissingItems, MissingInput
from .pydaymet import get_bygeom, get_byloc

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
