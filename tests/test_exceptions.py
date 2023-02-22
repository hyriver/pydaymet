import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

import pydaymet as daymet
from pydaymet import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingCRSError,
    MissingItemError,
)
from pydaymet.cli import cli

GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
COORDS = (-69.77, 45.07)
DATES = ("2000-01-01", "2000-12-31")


def test_invalid_variable():
    with pytest.raises(InputValueError) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, variables="tt")
    assert "variables" in str(ex.value)


def test_invalid_pet_timescale():
    with pytest.raises(InputValueError) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, pet="hargreaves_samani", time_scale="monthly")
    assert "pet" in str(ex.value)


def test_invalid_timescale():
    with pytest.raises(InputValueError) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, time_scale="subdaily")
    assert "time_scale" in str(ex.value)


def test_invalid_region():
    with pytest.raises(InputValueError) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, region="nn")
    assert "region" in str(ex.value)


def test_invalid_coords():
    with pytest.raises(InputRangeError) as ex:
        _ = daymet.get_bycoords((0, 0), DATES)
    assert "-136.8989" in str(ex.value)


def test_invalid_date():
    with pytest.raises(InputRangeError) as ex:
        _ = daymet.get_bycoords(COORDS, ("1950-01-01", "2010-01-01"))
    assert "1980" in str(ex.value)


def test_invalid_year():
    with pytest.raises(InputRangeError) as ex:
        _ = daymet.get_bycoords(COORDS, 1950)
    assert "1980" in str(ex.value)


def test_invalid_year_type():
    with pytest.raises(InputTypeError) as ex:
        _ = daymet.get_bycoords(COORDS, "1950")
    assert "or int" in str(ex.value)


def test_invalid_date_tuple():
    with pytest.raises(InputTypeError) as ex:
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
            "pet": "hargreaves_samani",
        }
        coord_csv = "coords_missing_co.csv"
        save_dir = "test_cli_missing_col"
        df = pd.DataFrame(params, index=[0])
        df.to_csv(coord_csv)
        ret = runner.invoke(
            cli,
            [
                "coords",
                coord_csv,
                "-s",
                save_dir,
            ],
        )
        Path(coord_csv).unlink()
        shutil.rmtree(save_dir, ignore_errors=True)
        assert ret.exit_code == 1
        assert isinstance(ret.exception, MissingItemError)
        assert "lat" in str(ret.exception)

    def test_wrong_geo_format(self, runner):
        params = {
            "id": "geo_test",
            "start": "2000-01-01",
            "end": "2000-05-31",
        }
        geo_feather = "geo_wrong_format.feather"
        save_dir = "test_wrong_geo_format"
        gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0], crs=4326)
        gdf.to_feather(geo_feather)
        ret = runner.invoke(
            cli,
            [
                "geometry",
                geo_feather,
                "-s",
                save_dir,
            ],
        )
        Path(geo_feather).unlink()
        shutil.rmtree(save_dir, ignore_errors=True)
        assert ret.exit_code == 1
        assert isinstance(ret.exception, InputTypeError)
        assert "gpkg" in str(ret.exception)

    def test_wrong_geo_crs(self, runner):
        params = {
            "id": "geo_test",
            "start": "2000-01-01",
            "end": "2000-05-31",
        }
        geo_gpkg = Path("wrong_geo_crs.gpkg")
        save_dir = "test_wrong_geo_crs"
        gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0], crs=None)
        gdf.to_file(geo_gpkg)
        ret = runner.invoke(
            cli,
            [
                "geometry",
                str(geo_gpkg),
                "-s",
                save_dir,
            ],
        )
        if geo_gpkg.is_dir():
            shutil.rmtree(geo_gpkg)
        else:
            geo_gpkg.unlink()
        shutil.rmtree(save_dir, ignore_errors=True)
        assert ret.exit_code == 1
        assert isinstance(ret.exception, MissingCRSError)
        assert "CRS" in str(ret.exception)

    def test_wrong_coords_format(self, runner):
        params = {
            "id": "coords_test",
            "lon": -69.77,
            "lat": 45.07,
            "start": "2000-01-01",
            "end": "2000-12-31",
        }
        coord_paquet = "wrong_coords_format.paquet"
        save_dir = "test_wrong_coords_format"
        df = pd.DataFrame(params, index=[0])
        df.to_parquet(coord_paquet)
        ret = runner.invoke(
            cli,
            [
                "coords",
                coord_paquet,
                "-s",
                save_dir,
            ],
        )
        Path(coord_paquet).unlink()
        shutil.rmtree(save_dir, ignore_errors=True)
        assert ret.exit_code == 1
        assert isinstance(ret.exception, InputTypeError)
        assert "csv" in str(ret.exception)
