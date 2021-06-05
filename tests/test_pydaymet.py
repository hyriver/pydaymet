"""Tests for PyDaymet package."""
import io
import shutil
from pathlib import Path

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


def test_byloc():
    coords = (-1431147.7928, 318483.4618)
    dates = ("2000-01-01", "2000-12-31")

    pet = daymet.get_byloc(coords, dates, crs=ALT_CRS, pet=True)
    st_p = daymet.get_byloc(coords, dates, crs=ALT_CRS, variables=VAR)
    yr_p = daymet.get_byloc(coords, YEAR, crs=ALT_CRS, variables=VAR)

    daily = daymet.get_bycoords(coords, dates, variables=VAR, crs=ALT_CRS)
    monthly = daymet.get_bycoords(coords, YEAR, variables=VAR, crs=ALT_CRS, time_scale="monthly")
    annual = daymet.get_bycoords(coords, YEAR, variables=VAR, crs=ALT_CRS, time_scale="annual")

    assert (
        abs(pet["pet (mm/day)"].mean() - 4.076) < 1e-3
        and abs(st_p["tmin (deg c)"].mean() - 12.056) < 1e-3
        and abs(yr_p["tmin (deg c)"].mean() - 11.458) < 1e-3
        and abs(daily["prcp (mm/day)"].mean() - 1.005) < 1e-3
        and abs(monthly["tmin (degrees C)"].mean() - 11.435) < 1e-3
        and abs(annual["tmin (degrees C)"].mean() - 11.458) < 1e-3
    )


def test_bygeom():
    pet = daymet.get_bygeom(GEOM, DAY, pet=True)
    prcp = daymet.get_bygeom(GEOM.bounds, DAY)
    daily = daymet.get_bygeom(GEOM, DAY, variables=VAR)
    monthly = daymet.get_bygeom(GEOM, YEAR, variables=VAR, time_scale="monthly")
    annual = daymet.get_bygeom(GEOM, YEAR, variables=VAR, time_scale="annual")

    assert (
        abs(pet.pet.mean().values - 0.629) < 1e-3
        and abs(prcp.prcp.mean().values - 3.513) < 1e-3
        and abs(daily.tmin.mean().values - (-9.421)) < 1e-3
        and abs(monthly.tmin.mean().values - 1.311) < 1e-3
        and abs(annual.tmin.mean().values - 1.361) < 1e-3
    )


def test_region():
    hi_ext = (-160.3055, 17.9539, -154.7715, 23.5186)
    pr_ext = (-67.9927, 16.8443, -64.1195, 19.9381)
    hi = daymet.get_bygeom(hi_ext, YEAR, variables=VAR, region="hi", time_scale="annual")
    pr = daymet.get_bygeom(pr_ext, YEAR, variables=VAR, region="pr", time_scale="annual")

    assert (
        abs(hi.prcp.mean().values - 1035.233) < 1e-3 and abs(pr.tmin.mean().values - 21.441) < 1e-3
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
    Path(coord_csv).unlink()
    shutil.rmtree("geo_coords")
    assert ret.success
    assert "Retrieved climate data for 1 item(s)." in ret.stdout
    assert ret.stderr == ""


def test_show_versions():
    f = io.StringIO()
    daymet.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
