"""Access the Daymet database for both single single pixel and gridded queries."""
import io
from itertools import product
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import py3dep
import pygeoogc as ogc
import pygeoutils as geoutils
import rasterio.transform as rio_transform
import xarray as xr
from pygeoogc import MatchCRS, RetrySession, ServiceURL
from shapely.geometry import MultiPolygon, Polygon

from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue, MissingItems

DEF_CRS = "epsg:4326"
DATE_FMT = "%Y-%m-%d"
DATE_REQ = "%Y-%m-%dT%H:%M:%SZ"


class Daymet:
    """Base class for Daymet requests.

    Parameters
    ----------
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
        Defaults to None i.e., all the variables are downloaded.
    pet : bool, optional
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False
    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly summaries),
        or annual (annual summaries). Defaults to daily.
    """

    def __init__(
        self,
        variables: Optional[Union[List[str], str]] = None,
        pet: bool = False,
        time_scale: str = "daily",
    ) -> None:
        self.session = RetrySession()

        vars_table = pd.read_html("https://daymet.ornl.gov/overview")[1]

        self.units = dict(zip(vars_table["Abbr"], vars_table["Units"]))

        valid_times = ["daily", "monthly", "annual"]
        if time_scale not in valid_times:
            raise InvalidInputValue("time_scale", valid_times)

        self.time_scale = time_scale
        self.code = {"daily": 1840, "monthly": 1855, "annual": 1852}

        valid_variables = vars_table.Abbr.to_list()
        if variables is None:
            self.variables = valid_variables
        else:
            self.variables = variables if isinstance(variables, list) else [variables]

            if not set(self.variables).issubset(set(valid_variables)):
                raise InvalidInputValue("variables", valid_variables)

            if pet:
                reqs = ("tmin", "tmax", "vp", "srad", "dayl")
                self.variables = list(set(reqs) | set(self.variables))

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

    @staticmethod
    def dates_todict(dates: Tuple[str, str]) -> Dict[str, str]:
        """Set dates by start and end dates as a tuple, (start, end)."""
        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InvalidInputType("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])

        if start < pd.to_datetime("1980-01-01"):
            raise InvalidInputRange("Daymet database ranges from 1980 to 2019.")

        return {
            "start": start.strftime(DATE_FMT),
            "end": end.strftime(DATE_FMT),
        }

    @staticmethod
    def years_todict(years: Union[List[int], int]) -> Dict[str, str]:
        """Set date by list of year(s)."""
        years = years if isinstance(years, list) else [years]
        return {"years": ",".join(str(y) for y in years)}

    def dates_tolist(
        self, dates: Tuple[str, str]
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Correct dates for Daymet accounting for leap years.

        Daymet doesn't account for leap years and removes Dec 31 when
        it's leap year. This function returns all the dates in the
        Daymet database within the provided date range.
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
        it's leap year. This function returns all the dates in the
        Daymet database for the provided years.
        """
        date_dict = self.years_todict(years)
        start_list, end_list = [], []
        for year in date_dict["years"].split(","):
            s = pd.to_datetime(f"{year}0101")
            start_list.append(s + pd.DateOffset(hour=12))
            if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0):
                end_list.append(pd.to_datetime(f"{year}1230") + pd.DateOffset(hour=12))
            else:
                end_list.append(pd.to_datetime(f"{year}1231") + pd.DateOffset(hour=12))
        return list(zip(start_list, end_list))

    @staticmethod
    def pet_byloc(
        clm_df: pd.DataFrame,
        coords: Tuple[float, float],
        crs: str = DEF_CRS,
        alt_unit: bool = False,
    ) -> pd.DataFrame:
        """Compute Potential EvapoTranspiration using Daymet dataset for a single location.

        The method is based on `FAO-56 <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The following variables are required:
        tmin (deg c), tmax (deg c), lat, lon, vp (Pa), srad (W/m2), dayl (s/day)
        The computed PET's unit is mm/day.

        Parameters
        ----------
        clm_df : DataFrame
            A dataframe with columns named as follows:
            ``tmin (deg c)``, ``tmax (deg c)``, ``vp (Pa)``, ``srad (W/m^2)``, ``dayl (s)``
        coords : tuple of floats
            Coordinates of the daymet data location as a tuple, (x, y).
        crs : str, optional
            The spatial reference of the input coordinate, defaults to epsg:4326
        alt_unit : str, optional
            Whether to use alternative units rather than the official ones, defaults to False.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with an additional column named ``pet (mm/day)``
        """
        units = {
            "dayl": ("s/day", "s"),
            "srad": ("W/m2", "W/m^2"),
            "tmax": ("degrees C", "deg c"),
            "tmin": ("degrees C", "deg c"),
        }

        va_pa = "vp (Pa)"
        tmin_c = f"tmin ({units['tmin'][alt_unit]})"
        tmax_c = f"tmax ({units['tmax'][alt_unit]})"
        srad_wm2 = f"srad ({units['srad'][alt_unit]})"
        dayl_s = f"dayl ({units['dayl'][alt_unit]})"
        tmean_c = "tmean (deg c)"

        reqs = [tmin_c, tmax_c, va_pa, srad_wm2, dayl_s]

        _check_requirements(reqs, clm_df.columns)

        clm_df[tmean_c] = 0.5 * (clm_df[tmax_c] + clm_df[tmin_c])
        delta_v = (
            4098
            * (
                0.6108
                * np.exp(
                    17.27 * clm_df[tmean_c] / (clm_df[tmean_c] + 237.3),
                )
            )
            / ((clm_df[tmean_c] + 237.3) ** 2)
        )
        elevation = py3dep.elevation_bycoords([coords], crs)[0]

        pa = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
        gamma = pa * 0.665e-3

        rho_s = 0.0  # recommended for daily data
        clm_df[va_pa] = clm_df[va_pa] * 1e-3

        e_max = 0.6108 * np.exp(17.27 * clm_df[tmax_c] / (clm_df[tmax_c] + 237.3))
        e_min = 0.6108 * np.exp(17.27 * clm_df[tmin_c] / (clm_df[tmin_c] + 237.3))
        e_s = (e_max + e_min) * 0.5
        e_def = e_s - clm_df[va_pa]

        u_2m = 2.0  # recommended when no data is available

        jday = clm_df.index.dayofyear
        r_surf = clm_df[srad_wm2] * clm_df[dayl_s] * 1e-6

        alb = 0.23

        jp = 2.0 * np.pi * jday / 365.0
        d_r = 1.0 + 0.033 * np.cos(jp)
        delta_r = 0.409 * np.sin(jp - 1.39)
        phi = coords[1] * np.pi / 180.0
        w_s = np.arccos(-np.tan(phi) * np.tan(delta_r))
        r_aero = (
            24.0
            * 60.0
            / np.pi
            * 0.082
            * d_r
            * (w_s * np.sin(phi) * np.sin(delta_r) + np.cos(phi) * np.cos(delta_r) * np.sin(w_s))
        )
        rad_s = (0.75 + 2e-5 * elevation) * r_aero
        rad_ns = (1.0 - alb) * r_surf
        rad_nl = (
            4.903e-9
            * (((clm_df[tmax_c] + 273.16) ** 4 + (clm_df[tmin_c] + 273.16) ** 4) * 0.5)
            * (0.34 - 0.14 * np.sqrt(clm_df[va_pa]))
            * ((1.35 * r_surf / rad_s) - 0.35)
        )
        rad_n = rad_ns - rad_nl

        clm_df["pet (mm/day)"] = (
            0.408 * delta_v * (rad_n - rho_s)
            + gamma * 900.0 / (clm_df[tmean_c] + 273.0) * u_2m * e_def
        ) / (delta_v + gamma * (1 + 0.34 * u_2m))
        clm_df[va_pa] = clm_df[va_pa] * 1.0e3

        return clm_df.drop(columns=tmean_c)

    @staticmethod
    def pet_bygrid(clm_ds: xr.Dataset) -> xr.Dataset:
        """Compute Potential EvapoTranspiration using Daymet dataset.

        The method is based on `FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The following variables are required:
        tmin (deg c), tmax (deg c), lat, lon, vp (Pa), srad (W/m2), dayl (s/day)
        The computed PET's unit is mm/day.

        Parameters
        ----------
        clm_ds : xarray.DataArray
            The dataset should include the following variables:
            ``tmin``, ``tmax``, ``lat``, ``lon``, ``vp``, ``srad``, ``dayl``

        Returns
        -------
        xarray.DataArray
            The input dataset with an additional variable called ``pet``.
        """
        keys = list(clm_ds.keys())
        reqs = ["tmin", "tmax", "lat", "lon", "vp", "srad", "dayl"]

        _check_requirements(reqs, keys)

        dtype = clm_ds.tmin.dtype
        dates = clm_ds["time"]
        clm_ds["tmean"] = 0.5 * (clm_ds["tmax"] + clm_ds["tmin"])
        clm_ds["delta_r"] = (
            4098
            * (0.6108 * np.exp(17.27 * clm_ds["tmean"] / (clm_ds["tmean"] + 237.3)))
            / ((clm_ds["tmean"] + 237.3) ** 2)
        )

        res = clm_ds.res[0] * 1.0e3
        elev = py3dep.elevation_bygrid(clm_ds.x.values, clm_ds.y.values, clm_ds.crs, res)
        attrs = clm_ds.attrs
        clm_ds = xr.merge([clm_ds, elev])
        clm_ds.attrs = attrs
        clm_ds["elevation"] = clm_ds.elevation.where(
            ~np.isnan(clm_ds.isel(time=0)[keys[0]]), drop=True
        )

        pa = 101.3 * ((293.0 - 0.0065 * clm_ds["elevation"]) / 293.0) ** 5.26
        clm_ds["gamma"] = pa * 0.665e-3

        rho_s = 0.0  # recommended for daily data
        clm_ds["vp"] *= 1e-3

        e_max = 0.6108 * np.exp(17.27 * clm_ds["tmax"] / (clm_ds["tmax"] + 237.3))
        e_min = 0.6108 * np.exp(17.27 * clm_ds["tmin"] / (clm_ds["tmin"] + 237.3))
        e_s = (e_max + e_min) * 0.5
        clm_ds["e_def"] = e_s - clm_ds["vp"]

        u_2m = 2.0  # recommended when no wind data is available

        lat = clm_ds.isel(time=0).lat
        clm_ds["time"] = pd.to_datetime(clm_ds.time.values).dayofyear.astype(dtype)
        r_surf = clm_ds["srad"] * clm_ds["dayl"] * 1e-6

        alb = 0.23

        jp = 2.0 * np.pi * clm_ds["time"] / 365.0
        d_r = 1.0 + 0.033 * np.cos(jp)
        delta_r = 0.409 * np.sin(jp - 1.39)
        phi = lat * np.pi / 180.0
        w_s = np.arccos(-np.tan(phi) * np.tan(delta_r))
        r_aero = (
            24.0
            * 60.0
            / np.pi
            * 0.082
            * d_r
            * (w_s * np.sin(phi) * np.sin(delta_r) + np.cos(phi) * np.cos(delta_r) * np.sin(w_s))
        )
        rad_s = (0.75 + 2e-5 * clm_ds["elevation"]) * r_aero
        rad_ns = (1.0 - alb) * r_surf
        rad_nl = (
            4.903e-9
            * (((clm_ds["tmax"] + 273.16) ** 4 + (clm_ds["tmin"] + 273.16) ** 4) * 0.5)
            * (0.34 - 0.14 * np.sqrt(clm_ds["vp"]))
            * ((1.35 * r_surf / rad_s) - 0.35)
        )
        clm_ds["rad_n"] = rad_ns - rad_nl

        clm_ds["pet"] = (
            0.408 * clm_ds["delta_r"] * (clm_ds["rad_n"] - rho_s)
            + clm_ds["gamma"] * 900.0 / (clm_ds["tmean"] + 273.0) * u_2m * clm_ds["e_def"]
        ) / (clm_ds["delta_r"] + clm_ds["gamma"] * (1 + 0.34 * u_2m))
        clm_ds["pet"].attrs["units"] = "mm/day"

        clm_ds["time"] = dates
        clm_ds["vp"] *= 1.0e3

        clm_ds = clm_ds.drop_vars(["delta_r", "gamma", "e_def", "rad_n", "tmean"])

        return clm_ds


def get_byloc(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
    variables: Optional[Union[List[str], str]] = None,
    pet: bool = False,
) -> pd.DataFrame:
    """Get daily climate data from Daymet for a single point.

    This function uses Daymet's RESTful service to get the daily
    climate data and does not support monthly and annual summaries.
    If you want to get the summaries directly use get_bycoords function.

    Parameters
    ----------
    coords : tuple
        Longitude and latitude of the location of interest as a tuple (lon, lat)
    dates : tuple or list
        Either a tuple (start, end) or a list of years [YYYY, ...].
    crs :  str, optional
        The spatial reference of the input coordinates, defaults to epsg:4326
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
        Defaults to None i.e., all the variables are downloaded.
    pet : bool, optional
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False

    Returns
    -------
    pandas.DataFrame
        Daily climate data for a location
    """
    daymet = Daymet(variables, pet)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_dict = daymet.dates_todict(dates)
    else:
        dates_dict = daymet.years_todict(dates)

    if not (isinstance(coords, tuple) and len(coords) == 2):
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    _coords = MatchCRS.coords(((coords[0],), (coords[1],)), crs, DEF_CRS)
    lon, lat = (_coords[0][0], _coords[1][0])

    if not ((14.5 < lat < 52.0) or (-131.0 < lon < -53.0)):
        raise InvalidInputRange(
            "The location is outside the Daymet dataset. "
            + "The acceptable range is: "
            + "14.5 < lat < 52.0 and -131.0 < lon < -53.0"
        )

    payload = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "vars": ",".join(daymet.variables),
        "format": "json",
        **dates_dict,
    }

    r = daymet.session.get(ServiceURL().restful.daymet_point, payload)

    clm = pd.DataFrame(r.json()["data"])
    clm.index = pd.to_datetime(clm.year * 1000.0 + clm.yday, format="%Y%j")
    clm = clm.drop(["year", "yday"], axis=1)

    if pet:
        clm = daymet.pet_byloc(clm, (lon, lat), alt_unit=True)
    return clm


def get_bycoords(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    loc_crs: str = DEF_CRS,
    variables: Optional[List[str]] = None,
    pet: bool = False,
    region: str = "na",
    time_scale: str = "daily",
) -> xr.Dataset:
    """Get point-data from the Daymet database at 1-km resolution.

    This function uses THREDDS data service to get the coordinates
    and supports getting monthly and annual summaries of the climate
    data directly from the server.

    Parameters
    ----------
    coords : tuple
        Coordinates of the location of interest as a tuple (lon, lat)
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    loc_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    pet : bool
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False
    region : str, optional
        Region in the US, defaults to na. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico
    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly summaries),
        or annual (annual summaries). Defaults to daily.

    Returns
    -------
    xarray.Dataset
        Daily climate data within a geometry
    """
    daymet = Daymet(variables, pet, time_scale)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    _coords = MatchCRS.coords(((coords[0],), (coords[1],)), loc_crs, DEF_CRS)
    coords = (_coords[0][0], _coords[1][0])
    urls = coord_urls(daymet.code[time_scale], coords, region, daymet.variables, dates_itr)

    clm = pd.concat(
        (
            pd.concat(
                pd.read_csv(io.BytesIO(r), parse_dates=[0], usecols=[0, 3], index_col=[0])
                for r in ogc.async_requests(u, "binary", max_workers=8)
            )
            for u in urls
        ),
        axis=1,
    )
    clm.columns = [c.replace('[unit="', " (").replace('"]', ")") for c in clm.columns]

    if "prcp (mm)" in clm:
        clm = clm.rename(columns={"prcp (mm)": "prcp (mm/day)"})

    clm.index = pd.to_datetime(clm.index.strftime("%Y-%m-%d"))

    if pet:
        clm = daymet.pet_byloc(clm, coords, alt_unit=False)
    return clm


def get_bygeom(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    geo_crs: str = DEF_CRS,
    variables: Optional[List[str]] = None,
    pet: bool = False,
    region: str = "na",
    time_scale: str = "daily",
) -> xr.Dataset:
    """Get gridded data from the Daymet database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or bbox
        The geometry of the region of interest.
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    pet : bool
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False
    region : str, optional
        Region in the US, defaults to na. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico
    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly average),
        or annual (annual average). Defaults to daily.

    Returns
    -------
    xarray.Dataset
        Daily climate data within a geometry
    """
    daymet = Daymet(variables, pet, time_scale)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    _geometry = geoutils.geo2polygon(geometry, geo_crs, DEF_CRS)
    urls = gridded_urls(
        daymet.code[time_scale], _geometry.bounds, region, daymet.variables, dates_itr
    )

    clm = xr.open_mfdataset(ogc.async_requests(urls, "binary", max_workers=8))

    for k, v in daymet.units.items():
        if k in clm.variables:
            clm[k].attrs["units"] = v

    clm = clm.drop_vars(["lambert_conformal_conic"])

    crs = " ".join(
        [
            "+proj=lcc",
            "+lat_1=25",
            "+lat_2=60",
            "+lat_0=42.5",
            "+lon_0=-100",
            "+x_0=0",
            "+y_0=0",
            "+ellps=WGS84",
            "+units=km",
            "+no_defs",
        ]
    )
    clm.attrs["crs"] = crs
    clm.attrs["nodatavals"] = (0.0,)

    xdim, ydim = "x", "y"
    height, width = clm.sizes[ydim], clm.sizes[xdim]

    left, right = clm[xdim].min().item(), clm[xdim].max().item()
    bottom, top = clm[ydim].min().item(), clm[ydim].max().item()

    x_res = abs(left - right) / (width - 1)
    y_res = abs(top - bottom) / (height - 1)

    left -= x_res * 0.5
    right += x_res * 0.5
    top += y_res * 0.5
    bottom -= y_res * 0.5

    clm.attrs["transform"] = rio_transform.from_bounds(left, bottom, right, top, width, height)
    clm.attrs["res"] = (x_res, y_res)
    clm.attrs["bounds"] = (left, bottom, right, top)

    if pet:
        clm = daymet.pet_bygrid(clm)

    if isinstance(clm, xr.Dataset):
        for v in clm:
            clm[v].attrs["crs"] = crs
            clm[v].attrs["nodatavals"] = (0.0,)

    return geoutils.xarray_geomask(clm, geometry, geo_crs)


def get_filename(
    code: int,
    region: str,
) -> Dict[int, Callable[[str], str]]:
    """Generate an iterable URL list for downloading Daymet data.

    Parameters
    ----------
    code : int
        Endpoint code which should be one of the following:
        * 1840: Daily
        * 1855: Monthly average
        * 1852: Annual average
    region : str
        Region in the US. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    valid_regions = ["na", "hi", "pr"]
    if region not in valid_regions:
        raise InvalidInputValue("region", valid_regions)

    valid_codes = [1840, 1855, 1852]
    if code not in valid_codes:
        raise InvalidInputValue("code", valid_codes)

    return {
        1840: lambda v: f"daily_{region}_{v}",
        1855: lambda v: f"{v}_monttl_{region}" if v == "prcp" else f"{v}_monavg_{region}",
        1852: lambda v: f"{v}_annttl_{region}" if v == "prcp" else f"{v}_annavg_{region}",
    }


def coord_urls(
    code: int,
    coord: Tuple[float, float],
    region: str,
    variables: List[str],
    dates: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> Iterator[List[Tuple[str, Dict[str, str]]]]:
    """Generate an iterable URL list for downloading Daymet data.

    Parameters
    ----------
    code : int
        Endpoint code which should be one of the following:
        * 1840: Daily
        * 1855: Monthly average
        * 1852: Annual average
    coord : tuple of length 2
        Coordinates in EPSG:4326 CRS (lon, lat)
    region : str
        Region in the US. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico
    variables : list
        A list of Daymet variables
    date : list
        A list of dates

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    time_scale = get_filename(code, region)

    lon, lat = coord
    base_url = f"{ServiceURL().restful.daymet}/{code}"
    return (
        [
            (
                f"{base_url}/daymet_v4_{time_scale[code](v)}_{s.year}.nc",
                {
                    "var": v,
                    "longitude": f"{lon}",
                    "latitude": f"{lat}",
                    "time_start": s.strftime(DATE_REQ),
                    "time_end": e.strftime(DATE_REQ),
                    "accept": "csv",
                },
            )
            for s, e in dates
        ]
        for v in variables
    )


def gridded_urls(
    code: int,
    bounds: Tuple[float, float, float, float],
    region: str,
    variables: List[str],
    dates: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Generate an iterable URL list for downloading Daymet data.

    Parameters
    ----------
    code : int
        Endpoint code which should be one of the following:
        * 1840: Daily
        * 1855: Monthly average
        * 1852: Annual average
    bounds : tuple of length 4
        Bounding box (west, south, east, north)
    region : str
        Region in the US. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico
    variables : list
        A list of Daymet variables
    date : list
        A list of dates

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    time_scale = get_filename(code, region)

    west, south, east, north = bounds
    base_url = f"{ServiceURL().restful.daymet}/{code}"
    return (
        (
            f"{base_url}/daymet_v4_{time_scale[code](v)}_{s.year}.nc",
            {
                "var": v,
                "north": f"{north}",
                "west": f"{west}",
                "east": f"{east}",
                "south": f"{south}",
                "disableProjSubset": "on",
                "horizStride": "1",
                "time_start": s.strftime(DATE_REQ),
                "time_end": e.strftime(DATE_REQ),
                "timeStride": "1",
                "addLatLon": "true",
                "accept": "netcdf",
            },
        )
        for v, (s, e) in product(variables, dates)
    )


def _check_requirements(reqs: Iterable, cols: List[str]) -> None:
    """Check for all the required data.

    Parameters
    ----------
    reqs : iterable
        A list of required data names (str)
    cols : list
        A list of variable names (str)
    """
    if not isinstance(reqs, Iterable):
        raise InvalidInputType("reqs", "iterable")

    missing = [r for r in reqs if r not in cols]
    if missing:
        raise MissingItems(missing)
