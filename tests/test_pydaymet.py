"""Tests for PyDaymet package."""
import io

import pytest
from shapely.geometry import Polygon

import pydaymet as daymet


@pytest.fixture
def geometry():
    return Polygon(
        [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    )


@pytest.fixture
def dates():
    return ("2000-01-01", "2000-01-12")


@pytest.fixture
def variables():
    return ["tmin"]


def test_byloc(dates, variables):
    coords = (-1431147.7928, 318483.4618)
    crs = "epsg:3542"

    daymet.get_byloc(coords, dates, crs=crs)
    st_p = daymet.get_byloc(coords, dates, crs=crs, variables=variables, pet=True)
    yr_p = daymet.get_byloc(coords, 2010, crs=crs, variables=variables)
    assert (
        abs(st_p.iloc[10]["pet (mm/day)"] - 2.393) < 1e-3
        and abs(yr_p.iloc[10]["tmin (deg c)"] - 11.5) < 1e-1
    )


def test_bygeom(geometry, dates, variables):
    daymet.get_bygeom(geometry, dates, fill_holes=True)
    daymet.get_bygeom(geometry.bounds, dates)
    st_g = daymet.get_bygeom(geometry, dates, variables=variables, pet=True)
    yr_g = daymet.get_bygeom(geometry, 2010, variables=variables)
    assert (
        abs(st_g.isel(time=10, x=5, y=10).pet.values.item() - 0.596) < 1e-3
        and abs(yr_g.isel(time=10, x=5, y=10).tmin.values.item() - (-18.0)) < 1e-1
    )


def test_show_versions():
    f = io.StringIO()
    daymet.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
