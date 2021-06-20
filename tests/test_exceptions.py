import pytest
from pydantic import ValidationError

import pydaymet as daymet
from pydaymet import InvalidInputRange, InvalidInputType, InvalidInputValue, MissingItems

COORDS = (-69.77, 45.07)
DATES = ("2000-01-01", "2000-12-31")


def test_invalid_variable():
    with pytest.raises(InvalidInputValue) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, variables="tt")
    assert "Given variables" in str(ex.value)


def test_invalid_pet_timescale():
    with pytest.raises(ValidationError) as ex:
        _ = daymet.get_bycoords(COORDS, DATES, pet=True, time_scale="monthly")
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
