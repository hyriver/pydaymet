"""Tests for PyDaymet package."""
import pytest
from shapely.geometry import Polygon

import pydaymet as daymet


@pytest.fixture
def geometry_nat():
    return Polygon(
        [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    )


def test_daymet(geometry_nat):
    coords = (-118.47, 34.16)
    dates = ("2000-01-01", "2000-01-12")
    variables = ["tmin"]

    st_p = daymet.get_byloc(coords, dates=dates)
    st_p = daymet.get_byloc(coords, dates=dates, variables=variables, pet=True)
    yr_p = daymet.get_byloc(coords, years=2010, variables=variables)

    st_g = daymet.get_bygeom(geometry_nat, dates=dates, fill_holes=True)
    st_g = daymet.get_bygeom(geometry_nat.bounds, dates=dates)
    st_g = daymet.get_bygeom(geometry_nat, dates=dates, variables=variables, pet=True)
    yr_g = daymet.get_bygeom(geometry_nat, years=2010, variables=variables)
    assert (
        abs(st_g.isel(time=10, x=5, y=10).pet.values.item() - 0.596) < 1e-3
        and abs(yr_g.isel(time=10, x=5, y=10).tmin.values.item() - (-18.0)) < 1e-1
        and abs(st_p.iloc[10]["pet (mm/day)"] - 2.393) < 1e-3
        and abs(yr_p.iloc[10]["tmin (deg c)"] - 11.5) < 1e-1
    )
