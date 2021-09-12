"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pydaymet namespace for doctest."""
    import pydaymet as daymet

    doctest_namespace["daymet"] = daymet
