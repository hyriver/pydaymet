"""Access the Daymet database for both single single pixel and gridded queries."""
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import py3dep
import pygeoogc as ogc
import pygeoutils as geoutils
import xarray as xr
from pygeoogc import MatchCRS, RetrySession, ServiceURL
from shapely.geometry import Polygon

from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue, MissingItems

DEF_CRS = "epsg:4326"
DATE_FMT = "%Y-%m-%d"


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
    """

    def __init__(
        self, variables: Optional[Union[List[str], str]] = None, pet: bool = False,
    ) -> None:
        self.session = RetrySession()

        vars_table = pd.read_html("https://daymet.ornl.gov/overview")[1]

        self.units = dict(zip(vars_table["Abbr"], vars_table["Units"]))

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
        clm_df: pd.DataFrame, coords: Tuple[float, float], crs: str = DEF_CRS
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

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with an additional column named ``pet (mm/day)``
        """
        va_pa = "vp (Pa)"
        tmin_c = "tmin (deg c)"
        tmax_c = "tmax (deg c)"
        srad_wm2 = "srad (W/m^2)"
        dayl_s = "dayl (s)"
        tmean_c = "tmean (deg c)"

        reqs = [tmin_c, tmax_c, va_pa, srad_wm2, dayl_s]

        _check_requirements(reqs, clm_df)

        clm_df[tmean_c] = 0.5 * (clm_df[tmax_c] + clm_df[tmin_c])
        delta_v = (
            4098
            * (0.6108 * np.exp(17.27 * clm_df[tmean_c] / (clm_df[tmean_c] + 237.3),))
            / ((clm_df[tmean_c] + 237.3) ** 2)
        )
        elevation = py3dep.elevation_byloc(coords, crs)

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

        return clm_df

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
        clm_ds["tmean"].attrs["units"] = "degree C"
        clm_ds["delta_r"] = (
            4098
            * (0.6108 * np.exp(17.27 * clm_ds["tmean"] / (clm_ds["tmean"] + 237.3)))
            / ((clm_ds["tmean"] + 237.3) ** 2)
        )

        gridxy = (clm_ds.x.values, clm_ds.y.values)
        res = clm_ds.res[0] * 1000
        elev = py3dep.elevation_bygrid(gridxy, clm_ds.crs, res)
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

        clm_ds = clm_ds.drop_vars(["delta_r", "gamma", "e_def", "rad_n"])

        return clm_ds


def get_byloc(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
    variables: Optional[Union[List[str], str]] = None,
    pet: bool = False,
) -> pd.DataFrame:
    """Get daily climate data from Daymet for a single point.

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

    if isinstance(dates, tuple) and len(dates) == 2:
        dates_dict = daymet.dates_todict(dates)
    elif isinstance(dates, (list, int)):
        dates_dict = daymet.years_todict(dates)
    else:
        raise InvalidInputType("dates", "tuple or list", "(start, end) or [2001, 2010, ...]")

    if isinstance(coords, tuple) and len(coords) == 2:
        _coords = MatchCRS.coords(((coords[0],), (coords[1],)), crs, DEF_CRS)
        lon, lat = (_coords[0][0], _coords[1][0])
    else:
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

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
        clm = daymet.pet_byloc(clm, (lon, lat))
    return clm


def get_bygeom(
    geometry: Union[Polygon, Tuple[float, float, float, float]],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    geo_crs: str = DEF_CRS,
    variables: Optional[List[str]] = None,
    pet: bool = False,
    fill_holes: bool = False,
) -> xr.Dataset:
    """Gridded data from the Daymet database at 1-km resolution.

    The data is clipped using NetCDF Subset Service.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon or bbox
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
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior, defaults to False.

    Returns
    -------
    xarray.Dataset
        Daily climate data within a geometry
    """
    daymet = Daymet(variables, pet)

    if isinstance(dates, tuple) and len(dates) == 2:
        dates_itr = daymet.dates_tolist(dates)
    elif isinstance(dates, (list, int)):
        dates_itr = daymet.years_tolist(dates)
    else:
        raise InvalidInputType("dates", "tuple or list", "(start, end) or [2001, 2010, ...]")

    _geometry = geoutils.geo2polygon(geometry, geo_crs, DEF_CRS)
    _geometry = Polygon(_geometry.exterior) if fill_holes else _geometry

    west, south, east, north = _geometry.bounds
    base_url = ServiceURL().restful.daymet_grid
    urls = []

    for s, e in dates_itr:
        for v in daymet.variables:
            urls.append(
                base_url
                + "&".join(
                    [
                        f"{s.year}/daymet_v3_{v}_{s.year}_na.nc4?var=lat",
                        "var=lon",
                        f"var={v}",
                        f"north={north}",
                        f"west={west}",
                        f"east={east}",
                        f"south={south}",
                        "disableProjSubset=on",
                        "horizStride=1",
                        f'time_start={s.strftime("%Y-%m-%dT%H:%M:%SZ")}',
                        f'time_end={e.strftime("%Y-%m-%dT%H:%M:%SZ")}',
                        "timeStride=1",
                        "accept=netcdf",
                    ]
                )
            )

    def getter(url):
        return xr.load_dataset(daymet.session.get(url).content)

    data = xr.merge(ogc.utils.threading(getter, urls, max_workers=8))

    for k, v in daymet.units.items():
        if k in data.variables:
            data[k].attrs["units"] = v

    data = data.drop_vars(["lambert_conformal_conic"])

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
    data.attrs["crs"] = crs

    x_res, y_res = data.x.diff("x").min().item(), data.y.diff("y").min().item()
    # PixelAsArea Convention
    x_origin = data.x.values[0] - x_res / 2.0
    y_origin = data.y.values[0] - y_res / 2.0

    transform = (x_res, 0, x_origin, 0, y_res, y_origin)

    x_end = x_origin + data.dims["x"] * x_res
    y_end = y_origin + data.dims["y"] * y_res
    x_options = np.array([x_origin, x_end])
    y_options = np.array([y_origin, y_end])

    data.attrs["transform"] = transform
    data.attrs["res"] = (x_res, y_res)
    data.attrs["bounds"] = (
        x_options.min(),
        y_options.min(),
        x_options.max(),
        y_options.max(),
    )

    if pet:
        data = daymet.pet_bygrid(data)

    return geoutils.xarray_geomask(data, _geometry, DEF_CRS)


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
