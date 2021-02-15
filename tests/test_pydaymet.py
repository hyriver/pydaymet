"""Tests for PyDaymet package."""
import io

import pytest
from shapely.geometry import Polygon

import pydaymet as daymet

GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
DAY = ("2000-01-01", "2000-01-12")
YEAR = 2010
VAR = "tmin"


def test_byloc():
    coords = (-1431147.7928, 318483.4618)
    crs = "epsg:3542"

    pet = daymet.get_byloc(coords, DAY, crs=crs, pet=True)
    st_p = daymet.get_byloc(coords, DAY, crs=crs, variables=VAR)
    yr_p = daymet.get_byloc(coords, YEAR, crs=crs, variables=VAR)

    daily = daymet.get_bycoords(coords, DAY, variables=VAR, loc_crs=crs)
    monthly = daymet.get_bycoords(coords, YEAR, variables=VAR, loc_crs=crs, time_scale="monthly")
    annual = daymet.get_bycoords(coords, YEAR, variables=VAR, loc_crs=crs, time_scale="annual")

    assert (
        abs(pet["pet (mm/day)"].mean() - 2.286) < 1e-3
        and abs(st_p["tmin (deg c)"].mean() - 6.917) < 1e-3
        and abs(yr_p["tmin (deg c)"].mean() - 11.458) < 1e-3
        and abs(daily["tmin (degrees C)"].mean() - 6.917) < 1e-3
        and abs(monthly["tmin (degrees C)"].mean() - 11.435) < 1e-3
        and abs(annual["tmin (degrees C)"].mean() - 11.458) < 1e-3
    )


def test_bygeom():
    pet = daymet.get_bygeom(GEOM, DAY, pet=True)
    bounds = daymet.get_bygeom(GEOM.bounds, DAY)
    daily = daymet.get_bygeom(GEOM, DAY, variables=VAR)
    monthly = daymet.get_bygeom(GEOM, YEAR, variables=VAR, time_scale="monthly")
    annual = daymet.get_bygeom(GEOM, YEAR, variables=VAR, time_scale="annual")

    assert (
        abs(pet.pet.mean().values - 0.629) < 1e-3
        and abs(bounds.tmin.mean().values - (-9.433)) < 1e-3
        and abs(daily.tmin.mean().values - (-9.421)) < 1e-3
        and abs(monthly.tmin.mean().values - 1.311) < 1e-3
        and abs(annual.tmin.mean().values - 1.361) < 1e-3
    )


def test_show_versions():
    f = io.StringIO()
    daymet.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
