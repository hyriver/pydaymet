"""Core class for the Daymet functions."""
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as sgeom
import xarray as xr

from pydaymet.exceptions import InputRangeError, InputTypeError, InputValueError
from pydaymet.pet import PET_VARS

try:
    from numba import config as numba_config
    from numba import njit, prange

    ngjit = functools.partial(njit, cache=True, nogil=True)
    numba_config.THREADING_LAYER = "workqueue"
    has_numba = True
except ImportError:
    has_numba = False
    prange = range
    numba_config = None
    njit = None

    def ngjit(ntypes, parallel=None):  # type: ignore
        def decorator_njit(func):  # type: ignore
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


if TYPE_CHECKING:
    DF = TypeVar("DF", pd.DataFrame, xr.Dataset)

DATE_FMT = "%Y-%m-%d"
# Default snow params from https://doi.org/10.5194/gmd-11-1077-2018
T_RAIN = 2.5  # degC
T_SNOW = 0.6  # degC

__all__ = ["Daymet"]


@dataclass
class DaymetBase:
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

    pet: str | None
    snow: bool
    time_scale: str
    variables: Iterable[str]
    region: str

    def __post_init__(self) -> None:
        valid_variables = ("dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp")
        if "all" in self.variables:
            self.variables = valid_variables

        if not set(self.variables).issubset(set(valid_variables)):
            raise InputValueError("variables", valid_variables)

        if self.pet:
            valid_methods = ("penman_monteith", "hargreaves_samani", "priestley_taylor")
            if self.pet not in valid_methods:
                raise InputValueError("pet", valid_methods)

            self.variables = list(set(self.variables).union(PET_VARS[self.pet]))

        if self.snow:
            self.variables = list(set(self.variables).union({"tmin"}))

        valid_timescales = ["daily", "monthly", "annual"]
        if self.time_scale not in valid_timescales:
            raise InputValueError("time_scale", valid_timescales)

        if self.pet and self.time_scale != "daily":
            msg = "time_scale when pet is True"
            raise InputValueError(msg, ["daily"])

        valid_regions = ["na", "hi", "pr"]
        if self.region not in valid_regions:
            raise InputValueError("region", valid_regions)


@ngjit("f8[::1](f8[::1], f8[::1], f8, f8)")
def _separate_snow(
    prcp: npt.NDArray[np.float64],
    tmin: npt.NDArray[np.float64],
    t_rain: np.float64,
    t_snow: np.float64,
) -> npt.NDArray[np.float64]:
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
        variables: Iterable[str] | str | None = None,
        pet: str | None = None,
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
        if self.region == "pr":
            self.valid_start = pd.to_datetime("1950-01-01")
        else:
            self.valid_start = pd.to_datetime("1980-01-01")
        self.valid_end = pd.to_datetime(f"{datetime.now().year - 1}-12-31")
        self.time_codes = {"daily": 2129, "monthly": 2131, "annual": 2130}

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
                    " ".join(
                        [
                            "Duration of the daylight period in seconds per day.",
                            "This calculation is based on the period of the day during which the",
                            "sun is above a hypothetical flat horizon",
                        ]
                    ),
                    " ".join(
                        [
                            "Daily total precipitation in millimeters per day, sum of",
                            "all forms converted to water-equivalent. Precipitation occurrence on",
                            "any given day may be ascertained.",
                        ]
                    ),
                    " ".join(
                        [
                            "Incident shortwave radiation flux density in watts per square meter,",
                            "taken as an average over the daylight period of the day.",
                            "NOTE: Daily total radiation (MJ/m2/day) can be calculated as",
                            "follows: ((srad (W/m2) * dayl (s/day)) / l,000,000)",
                        ]
                    ),
                    " ".join(
                        [
                            "Snow water equivalent in kilograms per square meter."
                            "The amount of water contained within the snowpack."
                        ]
                    ),
                    "Daily maximum 2-meter air temperature in degrees Celsius.",
                    "Daily minimum 2-meter air temperature in degrees Celsius.",
                    " ".join(
                        [
                            "Water vapor pressure in pascals. Daily average partial",
                            "pressure of water vapor.",
                        ]
                    ),
                ],
            }
        )

        self.units = dict(zip(self.daymet_table["Abbr"], self.daymet_table["Units"]))
        self.units["snow"] = "mm/day"
        self.units["pet"] = "mm/day"

        self.long_names = dict(zip(self.daymet_table["Abbr"], self.daymet_table["Parameter"]))
        self.long_names["snow"] = "Snow"
        self.long_names["pet"] = "Potential Evapotranspiration"

        self.descriptions = dict(zip(self.daymet_table["Abbr"], self.daymet_table["Description"]))
        self.descriptions["snow"] = " ".join(
            [
                "Daily total snow in millimeters per day,",
                "computed by partitioning the total precipitation into snow and rain.",
            ]
        )
        if self.pet:
            self.descriptions["pet"] = " ".join(
                [
                    "Daily potential evapotranspiration in millimeters per day,",
                    f"computed using the {self.pet.replace('_', '-').title()} method.",
                ]
            )

    @staticmethod
    def check_dates(dates: tuple[str, str] | int | list[int]) -> None:
        """Check if input dates are in correct format and valid."""
        if not isinstance(dates, (tuple, list, int, range)):
            raise InputTypeError(
                "dates",
                "tuple, list, range, or int",
                "(start, end), range(start, end), or [years, ...]",
            )

        if isinstance(dates, tuple) and len(dates) != 2:
            raise InputTypeError("dates", "Start and end should be passed as a tuple of length 2.")

    def dates_todict(self, dates: tuple[str, str]) -> dict[str, str]:
        """Set dates by start and end dates as a tuple, (start, end)."""
        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InputTypeError("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])
        if self.time_scale == "monthly":
            start = start.replace(day=14)
            end = end.replace(day=17)

        if self.time_scale == "annual":
            start = start.replace(day=6)
            end = end.replace(day=8)

        if start < self.valid_start or end > self.valid_end:
            raise InputRangeError("start/end", f"from {self.valid_start} to {self.valid_end}")

        return {
            "start": start.strftime(DATE_FMT),
            "end": end.strftime(DATE_FMT),
        }

    def years_todict(self, years: list[int] | int | range) -> dict[str, str]:
        """Set date by list of year(s)."""
        years = [years] if isinstance(years, int) else list(years)

        if min(years) < self.valid_start.year or max(years) > self.valid_end.year:
            raise InputRangeError(
                "start/end", f"from {self.valid_start.year} to {self.valid_end.year}"
            )

        return {"years": ",".join(str(y) for y in years)}

    def dates_tolist(self, dates: tuple[str, str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
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

    def years_tolist(self, years: list[int] | int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
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
        clm = climate.copy()

        def snow_func(
            prcp: npt.NDArray[np.float64],
            tmin: npt.NDArray[np.float64],
            t_rain: float,
            t_snow: float,
        ) -> npt.NDArray[np.float64]:
            """Separate snow based on Martinez and Gupta (2010)."""
            return _separate_snow(
                prcp.astype("f8"),
                tmin.astype("f8"),
                np.float64(t_rain),
                np.float64(t_snow),
            )

        clm["snow"] = xr.apply_ufunc(
            snow_func,
            clm["prcp"],
            clm["tmin"],
            t_rain,
            t_snow,
            input_core_dims=[["time"], ["time"], [], []],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[clm["prcp"].dtype],
        ).transpose("time", "y", "x")
        clm["snow"].attrs["units"] = "mm/day"
        clm["snow"].attrs["long_name"] = "daily snowfall"
        return clm

    def separate_snow(self, clm: DF, t_rain: float = T_RAIN, t_snow: float = T_SNOW) -> DF:
        """Separate snow based on :footcite:t:`Martinez_2010`.

        Parameters
        ----------
        clm : pandas.DataFrame or xarray.Dataset
            Climate data that should include ``prcp`` and ``tmin``.
        t_rain : float, optional
            Threshold for temperature for considering rain, defaults to 2.5 degrees C.
        t_snow : float, optional
            Threshold for temperature for considering snow, defaults to 0.6 degrees C.

        Returns
        -------
        pandas.DataFrame or xarray.Dataset
            Input data with ``snow (mm/day)`` column if input is a ``pandas.DataFrame``,
            or ``snow`` variable if input is an ``xarray.Dataset``.

        References
        ----------
        .. footbibliography::
        """
        if not has_numba:
            warnings.warn(
                "Numba not installed. Using slow pure python version.", UserWarning, stacklevel=2
            )

        if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
            raise InputTypeError("clm", "pandas.DataFrame or xarray.Dataset")

        if isinstance(clm, xr.Dataset):
            return self._snow_gridded(clm, t_rain, t_snow)  # type: ignore
        return self._snow_point(clm, t_rain, t_snow)
