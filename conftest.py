"""Configuration for pytest."""

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Return a CliRunner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pydaymet namespace for doctest."""
    import pydaymet as daymet

    doctest_namespace["daymet"] = daymet
