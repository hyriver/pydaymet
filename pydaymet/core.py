"""Core class for the Daymet functions."""
import functools
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import xarray as xr
from pydantic import BaseModel, validator

from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue

try:
    from numba import njit, prange

    ngjit = functools.partial(njit, cache=True, nogil=True, parallel=True)
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range

    def ngjit(ntypes):  # type: ignore
        def decorator_njit(func):  # type: ignore
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


DF = TypeVar("DF", pd.DataFrame, xr.Dataset)
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
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
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
    snow: bool = False
    time_scale: str = "daily"
    variables: List[str] = ["all"]
    region: str = "na"

    @validator("pet")
    def _pet(cls, v: Optional[str]) -> Optional[str]:
        valid_methods = ["penman_monteith", "hargreaves_samani", "priestley_taylor", None]
        if v not in valid_methods:
            raise InvalidInputValue("pet", valid_methods)
        return v

    @validator("variables")
    def _variables(cls, v: List[str], values: Dict[str, str]) -> List[str]:
        valid_variables = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
        if "all" in v:
            return valid_variables

        if not set(v).issubset(set(valid_variables)):
            raise InvalidInputValue("variables", valid_variables)

        if values["pet"] is not None:
            v = list(set(v).union({"tmin", "tmax", "srad", "dayl"}))

        if values["snow"]:
            v = list(set(v).union({"tmin"}))
        return v

    @validator("time_scale")
    def _timescales(cls, v: str, values: Dict[str, str]) -> str:
        valid_timescales = ["daily", "monthly", "annual"]
        if v not in valid_timescales:
            raise InvalidInputValue("time_scale", valid_timescales)

        if values["pet"] is not None and v != "daily":
            msg = "PET can only be computed at daily scale i.e., time_scale must be daily."
            raise InvalidInputRange(msg)
        return v

    @validator("region")
    def _regions(cls, v: str) -> str:
        valid_regions = ["na", "hi", "pr"]
        if v not in valid_regions:
            raise InvalidInputValue("region", valid_regions)
        return v


@ngjit("f8[::1](f8[::1], f8[::1], f8, f8)")  # type: ignore
def _separate_snow(
    prcp: np.ndarray, tmin: np.ndarray, t_rain: float = 2.5, t_snow: float = 0.0  # type: ignore
) -> np.ndarray:  # type: ignore
    """Separate snow in precipitation."""
    t_rng = t_rain - t_snow
    snow = np.zeros_like(prcp)

    for t in prange(prcp.shape[0]):
        if tmin[t] > t_rain:
            snow[t] = 0.0
        elif tmin[t] < t_snow:
            snow[t] = prcp[t]
        else:
            snow[t] = prcp[t] * (t_rain - tmin[t]) / t_rng
    return snow


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
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
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
        snow: bool = False,
        time_scale: str = "daily",
        region: str = "na",
    ) -> None:

        _variables = ["all"] if variables is None else variables
        _variables = [_variables] if isinstance(_variables, str) else _variables
        validated = DaymetBase(
            variables=_variables, pet=pet, snow=snow, time_scale=time_scale, region=region
        )
        self.variables = validated.variables
        self.pet = validated.pet
        self.time_scale = validated.time_scale
        self.region = validated.region
        self.snow = validated.snow

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
        if self.time_scale == "monthly":
            start = start.replace(day=14)
            end = end.replace(day=17)

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

    def dates_tolist(self, dates: Tuple[str, str]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
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

    def years_tolist(self, years: Union[List[int], int]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
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

    @staticmethod
    def _snow_point(climate: pd.DataFrame, t_rain: float, t_snow: float) -> pd.DataFrame:
        """Separate snow from precipitation."""
        clm = climate.copy()
        clm["snow (mm/day)"] = _separate_snow(
            clm["prcp (mm/day)"].to_numpy("f8"),
            clm["tmin (degrees C)"].to_numpy("f8"),
            np.float64(t_rain),
            np.float64(t_snow),
        )
        return clm

    @staticmethod
    def _snow_gridded(climate: xr.Dataset, t_rain: float, t_snow: float) -> xr.Dataset:
        """Separate snow from precipitation."""
        clm = climate.copy().chunk({"time": -1})

        def snow_func(
            prcp: xr.DataArray, tmin: xr.DataArray, t_rain: float, t_snow: float
        ) -> xr.DataArray:
            """Separate snow based on Martinez and Gupta (2010)."""
            return _separate_snow(  # type: ignore
                np.array(prcp, dtype="f8"),
                np.array(tmin, dtype="f8"),
                np.float64(t_rain),
                np.float64(t_snow),
            )

        clm["snow"] = xr.apply_ufunc(
            snow_func,
            clm.prcp,
            clm.tmin,
            t_rain,
            t_snow,
            input_core_dims=[["time"], ["time"], [], []],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[clm.prcp.dtype],
        ).transpose("time", "y", "x")
        clm["snow"].attrs["units"] = "mm/day"
        clm["snow"].attrs["long_name"] = "daily snowfall"
        return clm

    def separate_snow(self, clm: DF, t_rain: float = 2.5, t_snow: float = 0.0) -> DF:
        """Separate snow based on :footcite:t:`Martinez_2010`.

        Parameters
        ----------
        clm : pandas.DataFrame or xarray.Dataset
            Climate data that should include ``prcp`` and ``tmin``.
        t_rain : float, optional
            Threshold for temperature for considering rain, defaults to 2.5 degrees C.
        t_snow : float, optional
            Threshold for temperature for considering snow, defaults to 0.0 degrees C.

        Returns
        -------
        pandas.DataFrame or xarray.Dataset
            Input data with ``snow (mm/day)`` column if input is a ``pandas.DataFrame``,
            or ``snow`` variable if input is an ``xarray.Dataset``.

        References
        ----------
        .. footbibliography::
        """
        if not HAS_NUMBA:
            warnings.warn("Numba not installed. Using slow pure python version.", UserWarning)

        if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
            raise InvalidInputType("clm", "pandas.DataFrame or xarray.Dataset")

        if isinstance(clm, xr.Dataset):
            return self._snow_gridded(clm, t_rain, t_snow)
        return self._snow_point(clm, t_rain, t_snow)
