from pathlib import Path
import shutil

import pandas as pd
import pytest
from pydantic import ValidationError
import geopandas as gpd

import pydaymet as daymet
from pydaymet import InvalidInputRange, InvalidInputType, InvalidInputValue, MissingItems
from pydaymet.cli import cli
from shapely.geometry import Polygon

GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
COORDS = (-69.77, 45.07)
DATES = ("2000-01-01", "2000-12-31")


def test_invalid_variable():
    with pytest.raises(InvalidInputValue) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, variables="tt")
    assert "Given variables" in str(ex.value)


def test_invalid_pet_timescale():
    with pytest.raises(ValidationError) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, pet="hargreaves_samani", time_scale="monthly")
    assert "PET can only" in str(ex.value)


def test_invalid_timescale():
    with pytest.raises(InvalidInputValue) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, time_scale="subdaily")
    assert "time_scale" in str(ex.value)


def test_invalid_region():
    with pytest.raises(InvalidInputValue) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, region="nn")
    assert "region" in str(ex.value)


def test_invalid_coords():
    with pytest.raises(InvalidInputRange) as ex:
        _ = daymet.get_bycoords((0, 0), DATES)
    assert "Valid bounding box" in str(ex.value)


def test_invalid_date():
    with pytest.raises(InvalidInputRange) as ex:
        _ = daymet.get_bycoords(COORDS, ("1950-01-01", "2010-01-01"))
    assert "1980" in str(ex.value)


def test_invalid_year():
    with pytest.raises(InvalidInputRange) as ex:
        _ = daymet.get_bycoords(COORDS, 1950)
    assert "1980" in str(ex.value)


def test_invalid_year_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = daymet.get_bycoords(COORDS, "1950")
    assert "or int" in str(ex.value)


def test_invalid_date_tuple():
    with pytest.raises(InvalidInputType) as ex:
        _ = daymet.get_bycoords(COORDS, ("2010-01-01"))
    assert "(start, end)" in str(ex.value)


class TestCLIFails:
    """Test the command-line interface exceptions."""
    def test_cli_missing_col(self, runner):
        params = {
            "id": "coords_test",
            "lon": -100,
            "start": "2000-01-01",
            "end": "2000-01-12",
        }
        coord_csv = "coords.csv"
        df = pd.DataFrame(params, index=[0])
        df.to_csv(coord_csv)
        ret = runner.invoke(
            cli,
            [
                "coords",
                coord_csv,
                "-p",
                "hargreaves_samani",
                "-s",
                "geo_coords",
            ],
        )
        assert ret.exit_code == 1
        assert isinstance(ret.exception, MissingItems)
        assert "lat" in str(ret.exception)
        Path(coord_csv).unlink()

    def test_wrong_geo_format(self, runner):
        params = {
            "id": "geo_test",
            "start": "2000-01-01",
            "end": "2000-05-31",
        }
        geo_feather = "nat_geo.feather"
        gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0], crs="epsg:4326")
        gdf.to_feather(geo_feather)
        ret = runner.invoke(
            cli,
            [
                "geometry",
                geo_feather,
                "-s",
                "geo_map",
            ],
        )
        Path(geo_feather).unlink()
        shutil.rmtree("geo_map")
        assert ret.exit_code == 1
        assert isinstance(ret.exception, InvalidInputType)
        assert "gpkg" in str(ret.exception)

    def test_wrong_geo_crs(self, runner):
        params = {
            "id": "geo_test",
            "start": "2000-01-01",
            "end": "2000-05-31",
        }
        geo_gpkg = "nat_geo.gpkg"
        gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0], crs="epsg:4326")
        gdf.to_crs("epsg:3542").to_file(geo_gpkg)
        ret = runner.invoke(
            cli,
            [
                "geometry",
                geo_gpkg,
                "-s",
                "geo_map",
            ],
        )
        shutil.rmtree(geo_gpkg)
        shutil.rmtree("geo_map")
        assert ret.exit_code == 1
        assert isinstance(ret.exception, InvalidInputValue)
        assert "4326" in str(ret.exception)

    def test_wrong_coords_format(self, runner):
        params = {
            "id": "coords_test",
            "lon": -69.77,
            "lat": 45.07,
            "start": "2000-01-01",
            "end": "2000-12-31",
        }
        coord_paquet = "coords.paquet"
        df = pd.DataFrame(params, index=[0])
        df.to_parquet(coord_paquet)
        ret = runner.invoke(
            cli,
            [
                "coords",
                coord_paquet,
                "-s",
                "geo_coords",
            ],
        )
        Path(coord_paquet).unlink()
        shutil.rmtree("geo_coords")
        assert ret.exit_code == 1
        assert isinstance(ret.exception, InvalidInputType)
        assert "csv" in str(ret.exception)
