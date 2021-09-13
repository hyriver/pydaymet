"""Core class for the Daymet functions."""
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import shapely.geometry as sgeom
from pydantic import BaseModel, validator

from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue

DEF_CRS = "epsg:4326"
DATE_FMT = "%Y-%m-%d"


__all__ = ["Daymet"]


class DaymetBase(BaseModel):
    """Base class for validating Daymet requests.

    Parameters
    ----------
    pet : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, ``hargreaves_samani``, and
        None (don't compute PET). The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``None``.
    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly summaries),
        or annual (annual summaries). Defaults to daily.
    variables : list, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
        Defaults to None i.e., all the variables are downloaded.
    region : str, optional
        Region in the US, defaults to na. Acceptable values are:

        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico

    References
    ----------
    .. footbibliography::
    """

    pet: Optional[str] = None
    time_scale: str = "daily"
    variables: List[str] = ["all"]
    region: str = "na"

    @validator("pet")
    def _valid_pet(cls, v):
        valid_methods = ["penman_monteith", "hargreaves_samani", "priestley_taylor", None]
        if v not in valid_methods:
            raise InvalidInputValue("pet", valid_methods)
        return v

    @validator("variables")
    def _valid_variables(cls, v, values) -> List[str]:
        valid_variables = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
        if "all" in v:
            return valid_variables

        variables = [v] if isinstance(v, str) else v

        if not set(variables).issubset(set(valid_variables)):
            raise InvalidInputValue("variables", valid_variables)

        if values["pet"] is not None:
            variables = list({"tmin", "tmax", "srad", "dayl"} | set(variables))
        return variables

    @validator("time_scale")
    def _valid_timescales(cls, v, values):
        valid_timescales = ["daily", "monthly", "annual"]
        if v not in valid_timescales:
            raise InvalidInputValue("time_scale", valid_timescales)

        if values["pet"] is not None and v != "daily":
            msg = "PET can only be computed at daily scale i.e., time_scale must be daily."
            raise InvalidInputRange(msg)
        return v

    @validator("region")
    def _valid_regions(cls, v):
        valid_regions = ["na", "hi", "pr"]
        if v not in valid_regions:
            raise InvalidInputValue("region", valid_regions)
        return v


class Daymet:
    """Base class for Daymet requests.

    Parameters
    ----------
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
        Defaults to None i.e., all the variables are downloaded.
    pet : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, ``hargreaves_samani``, and
        None (don't compute PET). The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``None``.
    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly summaries),
        or annual (annual summaries). Defaults to daily.
    region : str, optional
        Region in the US, defaults to na. Acceptable values are:

        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        variables: Optional[Union[Iterable[str], str]] = None,
        pet: Optional[str] = None,
        time_scale: str = "daily",
        region: str = "na",
    ) -> None:

        _variables = ["all"] if variables is None else variables
        _variables = [_variables] if isinstance(_variables, str) else _variables
        validated = DaymetBase(variables=_variables, pet=pet, time_scale=time_scale, region=region)
        self.variables = validated.variables
        self.pet = validated.pet
        self.time_scale = validated.time_scale
        self.region = validated.region

        self.region_bbox = {
            "na": sgeom.box(-136.8989, 6.0761, -6.1376, 69.077),
            "hi": sgeom.box(-160.3055, 17.9539, -154.7715, 23.5186),
            "pr": sgeom.box(-67.9927, 16.8443, -64.1195, 19.9381),
        }
        self.invalid_bbox_msg = "\n".join(
            [
                f"Input coordinates are outside the Daymet range for region ``{region}``.",
                f"Valid bounding box is: {self.region_bbox[region].bounds}",
            ]
        )
        if self.region == "pr":
            self.valid_start = pd.to_datetime("1950-01-01")
        else:
            self.valid_start = pd.to_datetime("1980-01-01")
        self.valid_end = pd.to_datetime("2020-12-31")
        self._invalid_yr = (
            "Daymet database ranges from " + f"{self.valid_start.year} to {self.valid_end.year}."
        )
        self.time_codes = {"daily": 1840, "monthly": 1855, "annual": 1852}

        self.daymet_table = pd.DataFrame(
            {
                "Parameter": [
                    "Day length",
                    "Precipitation",
                    "Shortwave radiation",
                    "Snow water equivalent",
                    "Maximum air temperature",
                    "Minimum air temperature",
                    "Water vapor pressure",
                ],
                "Abbr": ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"],
                "Units": ["s/day", "mm/day", "W/m2", "kg/m2", "degrees C", "degrees C", "Pa"],
                "Description": [
                    "Duration of the daylight period in seconds per day. "
                    + "This calculation is based on the period of the day during which the "
                    + "sun is above a hypothetical flat horizon",
                    "Daily total precipitation in millimeters per day, sum of"
                    + " all forms converted to water-equivalent. Precipitation occurrence on "
                    + "any given day may be ascertained.",
                    "Incident shortwave radiation flux density in watts per square meter, "
                    + "taken as an average over the daylight period of the day. "
                    + "NOTE: Daily total radiation (MJ/m2/day) can be calculated as follows: "
                    + "((srad (W/m2) * dayl (s/day)) / l,000,000)",
                    "Snow water equivalent in kilograms per square meter."
                    + " The amount of water contained within the snowpack.",
                    "Daily maximum 2-meter air temperature in degrees Celsius.",
                    "Daily minimum 2-meter air temperature in degrees Celsius.",
                    "Water vapor pressure in pascals. Daily average partial pressure of water vapor.",
                ],
            }
        )

        self.units = dict(zip(self.daymet_table["Abbr"], self.daymet_table["Units"]))

    @staticmethod
    def check_dates(dates: Union[Tuple[str, str], Union[int, List[int]]]) -> None:
        """Check if input dates are in correct format and valid."""
        if not isinstance(dates, (tuple, list, int)):
            raise InvalidInputType(
                "dates", "tuple, list, or int", "(start, end), year, or [years, ...]"
            )

        if isinstance(dates, tuple) and len(dates) != 2:
            raise InvalidInputType(
                "dates", "Start and end should be passed as a tuple of length 2."
            )

    def dates_todict(self, dates: Tuple[str, str]) -> Dict[str, str]:
        """Set dates by start and end dates as a tuple, (start, end)."""
        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InvalidInputType("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])

        if start < self.valid_start or end > self.valid_end:
            raise InvalidInputRange(self._invalid_yr)

        return {
            "start": start.strftime(DATE_FMT),
            "end": end.strftime(DATE_FMT),
        }

    def years_todict(self, years: Union[List[int], int]) -> Dict[str, str]:
        """Set date by list of year(s)."""
        years = [years] if isinstance(years, int) else years

        if min(years) < self.valid_start.year or max(years) > self.valid_end.year:
            raise InvalidInputRange(self._invalid_yr)

        return {"years": ",".join(str(y) for y in years)}

    def dates_tolist(
        self, dates: Tuple[str, str]
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Correct dates for Daymet accounting for leap years.

        Daymet doesn't account for leap years and removes Dec 31 when
        it's leap year.

        Parameters
        ----------
        dates : tuple
            Target start and end dates.

        Returns
        -------
        list
            All the dates in the Daymet database within the provided date range.
        """
        date_dict = self.dates_todict(dates)
        start = pd.to_datetime(date_dict["start"]) + pd.DateOffset(hour=12)
        end = pd.to_datetime(date_dict["end"]) + pd.DateOffset(hour=12)

        period = pd.date_range(start, end)
        nl = period[~period.is_leap_year]
        lp = period[(period.is_leap_year) & (~period.strftime(DATE_FMT).str.endswith("12-31"))]
        _period = period[(period.isin(nl)) | (period.isin(lp))]
        years = [_period[_period.year == y] for y in _period.year.unique()]
        return [(y[0], y[-1]) for y in years]

    def years_tolist(
        self, years: Union[List[int], int]
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Correct dates for Daymet accounting for leap years.

        Daymet doesn't account for leap years and removes Dec 31 when
        it's leap year.

        Parameters
        ----------
        years: list
            A list of target years.

        Returns
        -------
        list
            All the dates in the Daymet database within the provided date range.
        """
        date_dict = self.years_todict(years)
        start_list, end_list = [], []
        for year in date_dict["years"].split(","):
            s = pd.to_datetime(f"{year}0101")
            start_list.append(s + pd.DateOffset(hour=12))
            e = pd.to_datetime(f"{year}1230") if s.is_leap_year else pd.to_datetime(f"{year}1231")
            end_list.append(e + pd.DateOffset(hour=12))
        return list(zip(start_list, end_list))
