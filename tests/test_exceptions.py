import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from pydantic import ValidationError
from shapely.geometry import Polygon

import pydaymet as daymet
from pydaymet import (
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingCRS,
    MissingItems,
)
from pydaymet.cli import cli

try:
    import typeguard  # noqa: F401
except ImportError:
    has_typeguard = False
else:
    has_typeguard = True

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


@pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
def test_invalid_year_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = daymet.get_bycoords(COORDS, "1950")
    assert "or int" in str(ex.value)


@pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
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
        assert isinstance(ret.exception, MissingItems)
        assert "lat" in str(ret.exception)

    def test_wrong_geo_format(self, runner):
        params = {
            "id": "geo_test",
            "start": "2000-01-01",
            "end": "2000-05-31",
        }
        geo_feather = "geo_wrong_format.feather"
        save_dir = "test_wrong_geo_format"
        gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0], crs="epsg:4326")
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
        assert isinstance(ret.exception, InvalidInputType)
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
        assert isinstance(ret.exception, MissingCRS)
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
        assert isinstance(ret.exception, InvalidInputType)
        assert "csv" in str(ret.exception)
