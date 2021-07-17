"""Tests for PyDaymet package."""
import io
import shutil
from pathlib import Path
from typing import Tuple

import cytoolz as tlz
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

import pydaymet as daymet

GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
DAY = ("2000-01-01", "2000-01-12")
YEAR = 2010
VAR = ["prcp", "tmin"]
DEF_CRS = "epsg:4326"
ALT_CRS = "epsg:3542"
COORDS = (-1431147.7928, 318483.4618)
DATES = ("2000-01-01", "2000-12-31")
SMALL = 1e-3


def test_byloc():
    pet = daymet.get_byloc(COORDS, DATES, crs=ALT_CRS, pet=True)
    st_p = daymet.get_byloc(COORDS, DATES, crs=ALT_CRS, variables=VAR)
    yr_p = daymet.get_byloc(COORDS, YEAR, crs=ALT_CRS, variables=VAR)

    assert (
        abs(pet["pet (mm/day)"].mean() - 3.497) < SMALL
        and abs(st_p["tmin (deg c)"].mean() - 12.056) < SMALL
        and abs(yr_p["tmin (deg c)"].mean() - 11.458) < SMALL
    )


class TestByCoords:
    def test_daily(self):
        clm = daymet.get_bycoords(COORDS, DATES, variables=VAR, crs=ALT_CRS)
        assert abs(clm["prcp (mm/day)"].mean() - 1.005) < SMALL

    def test_monthly(self):
        clm = daymet.get_bycoords(COORDS, YEAR, variables=VAR, crs=ALT_CRS, time_scale="monthly")
        assert abs(clm["tmin (degrees C)"].mean() - 11.435) < SMALL

    def test_annual(self):
        clm = daymet.get_bycoords(COORDS, YEAR, variables=VAR, crs=ALT_CRS, time_scale="annual")
        assert abs(clm["tmin (degrees C)"].mean() - 11.458) < SMALL


class TestByGeom:
    def test_pet(self):
        pet = daymet.get_bygeom(GEOM, DAY, pet=True)
        assert abs(pet.pet.mean().values - 0.6269) < SMALL

    def test_bounds(self):
        prcp = daymet.get_bygeom(GEOM.bounds, DAY)
        assert abs(prcp.prcp.mean().values - 3.4999) < SMALL

    def test_daily(self):
        daily = daymet.get_bygeom(GEOM, DAY, variables=VAR)
        assert abs(daily.tmin.mean().values - (-9.421)) < SMALL

    def test_monthly(self):
        monthly = daymet.get_bygeom(GEOM, YEAR, variables=VAR, time_scale="monthly")
        assert abs(monthly.tmin.mean().values - 1.311) < SMALL

    def test_annual(self):
        annual = daymet.get_bygeom(GEOM, YEAR, variables=VAR, time_scale="annual")
        assert abs(annual.tmin.mean().values - 1.361) < SMALL

    def test_region(self):
        hi_ext = (-160.3055, 17.9539, -154.7715, 23.5186)
        pr_ext = (-67.9927, 16.8443, -64.1195, 19.9381)
        hi = daymet.get_bygeom(hi_ext, YEAR, variables=VAR, region="hi", time_scale="annual")
        pr = daymet.get_bygeom(pr_ext, YEAR, variables=VAR, region="pr", time_scale="annual")

        assert (
            abs(hi.prcp.mean().values - 1035.233) < SMALL
            and abs(pr.tmin.mean().values - 21.441) < SMALL
        )


def test_cli_grid(script_runner):
    params = {
        "id": "geo_test",
        "start": "2000-01-01",
        "end": "2000-05-31",
        "region": "na",
    }
    geo_gpkg = "nat_geo.gpkg"
    gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0])
    gdf.to_file(geo_gpkg)
    ret = script_runner.run(
        "pydaymet",
        geo_gpkg,
        "geometry",
        DEF_CRS,
        *list(tlz.concat([["-v", v] for v in VAR])),
        "-t",
        "monthly",
        "-s",
        "geo_map",
    )
    shutil.rmtree(geo_gpkg)
    shutil.rmtree("geo_map")
    assert ret.success
    assert "Retrieved climate data for 1 item(s)." in ret.stdout
    assert ret.stderr == ""


def test_cli_coords(script_runner):
    params = {
        "id": "coords_test",
        "x": -1431147.7928,
        "y": 318483.4618,
        "start": DAY[0],
        "end": DAY[1],
        "region": "na",
    }
    coord_csv = "coords.csv"
    df = pd.DataFrame(params, index=[0])
    df.to_csv(coord_csv)
    ret = script_runner.run(
        "pydaymet",
        coord_csv,
        "coords",
        ALT_CRS,
        *list(tlz.concat([["-v", v] for v in VAR])),
        "-p",
        "-s",
        "geo_coords",
    )
    script_runner.run(
        "pydaymet",
        coord_csv,
        "coords",
        ALT_CRS,
        *list(tlz.concat([["-v", v] for v in VAR])),
        "-p",
        "-s",
        "geo_coords",
    )
    Path(coord_csv).unlink()
    shutil.rmtree("geo_coords")
    assert ret.success
    assert "Retrieved climate data for 1 item(s)." in ret.stdout
    assert ret.stderr == ""


def test_show_versions():
    f = io.StringIO()
    daymet.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
