"""Access the Daymet database for both single single pixel and gridded queries."""
import functools
import io
import itertools
from ssl import SSLContext
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import async_retriever as ar
import pandas as pd
import pygeoutils as geoutils
import pyproj
import rioxarray  # noqa: F401
import xarray as xr
from pygeoogc import ServiceError, ServiceURL
from pygeoogc import utils as ogcutils
from shapely.geometry import MultiPolygon, Point, Polygon

from .core import Daymet
from .exceptions import InvalidInputRange, InvalidInputType
from .pet import potential_et

DEF_CRS = "epsg:4326"
DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
EXPIRE = -1
MAX_CONN = 10


def _get_filename(
    region: str,
) -> Dict[int, Callable[[str], str]]:
    """Get correct filenames based on region and variable of interest."""
    return {
        1840: lambda v: f"daily_{region}_{v}",
        1855: lambda v: f"{v}_monttl_{region}" if v == "prcp" else f"{v}_monavg_{region}",
        1852: lambda v: f"{v}_annttl_{region}" if v == "prcp" else f"{v}_annavg_{region}",
    }


def _coord_urls(
    code: int,
    coord: Tuple[float, float],
    region: str,
    variables: Iterable[str],
    dates: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[List[Tuple[str, Dict[str, Dict[str, str]]]]]:
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
    dates : list
        A list of dates

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    time_scale = _get_filename(region)

    lon, lat = coord
    base_url = f"{ServiceURL().restful.daymet}/{code}"
    return [
        [
            (
                f"{base_url}/daymet_v4_{time_scale[code](v)}_{s.year}.nc",
                {
                    "params": {
                        "var": v,
                        "longitude": f"{lon}",
                        "latitude": f"{lat}",
                        "time_start": s.strftime(DATE_FMT),
                        "time_end": e.strftime(DATE_FMT),
                        "accept": "csv",
                    }
                },
            )
            for s, e in dates
        ]
        for v in variables
    ]


def get_bycoords(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: Union[str, pyproj.CRS] = DEF_CRS,
    variables: Optional[Union[Iterable[str], str]] = None,
    region: str = "na",
    time_scale: str = "daily",
    pet: Optional[str] = None,
    pet_params: Optional[Dict[str, float]] = None,
    snow: bool = False,
    snow_params: Optional[Dict[str, float]] = None,
    ssl: Union[SSLContext, bool, None] = None,
) -> pd.DataFrame:
    """Get point-data from the Daymet database at 1-km resolution.

    This function uses THREDDS data service to get the coordinates
    and supports getting monthly and annual summaries of the climate
    data directly from the server.

    Parameters
    ----------
    coords : tuple
        Coordinates of the location of interest as a tuple (lon, lat)
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years ``[2001, 2010, ...]``.
    crs : str, optional
        The CRS of the input geometry, defaults to ``"epsg:4326"``.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    region : str, optional
        Target region in the US, defaults to ``na``. Acceptable values are:

        * ``na``: Continental North America
        * ``hi``: Hawaii
        * ``pr``: Puerto Rico

    time_scale : str, optional
        Data time scale which can be ``daily``, ``monthly`` (monthly summaries),
        or ``annual`` (annual summaries). Defaults to ``daily``.
    pet : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, ``hargreaves_samani``, and
        None (don't compute PET). The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``None``.
    pet_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the PET function.
        Defaults to ``None``.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0}``.
    ssl : bool or SSLContext, optional
        SSLContext to use for the connection, defaults to None. Set to False to disable
        SSL certification verification.

    Returns
    -------
    pandas.DataFrame
        Daily climate data for a location.

    Examples
    --------
    >>> import pydaymet as daymet
    >>> coords = (-1431147.7928, 318483.4618)
    >>> dates = ("2000-01-01", "2000-12-31")
    >>> clm = daymet.get_bycoords(
    ...     coords,
    ...     dates,
    ...     crs="epsg:3542",
    ...     pet="hargreaves_samani",
    ...     ssl=False
    ... )
    >>> clm["pet (mm/day)"].mean()
    3.713

    References
    ----------
    .. footbibliography::
    """
    daymet = Daymet(variables, pet, snow, time_scale, region)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    if not (isinstance(coords, tuple) and len(coords) == 2):
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    coords = ogcutils.match_crs([coords], crs, DEF_CRS)[0]

    if not Point(*coords).within(daymet.region_bbox[region]):
        raise InvalidInputRange(daymet.invalid_bbox_msg)

    url_kwds = _coord_urls(
        daymet.time_codes[time_scale], coords, daymet.region, daymet.variables, dates_itr
    )
    url_kwd_list = [tuple(zip(*u)) for u in url_kwds]

    retrieve = functools.partial(ar.retrieve_binary, max_workers=MAX_CONN, ssl=ssl)
    clm = pd.concat(
        (
            pd.concat(
                pd.read_csv(io.BytesIO(r), parse_dates=[0], usecols=[0, 3], index_col=[0])
                for r in retrieve(u, k)
            )
            for u, k in url_kwd_list
        ),
        axis=1,
    )
    clm.columns = [c.replace('[unit="', " (").replace('"]', ")") for c in clm.columns]

    if "prcp (mm)" in clm:
        clm = clm.rename(columns={"prcp (mm)": "prcp (mm/day)"})

    clm = clm.set_index(pd.to_datetime(clm.index.strftime("%Y-%m-%d")))

    if pet is not None:
        clm = potential_et(clm, coords, method=pet, params=pet_params)

    if snow:
        params = {"t_rain": 2.5, "t_snow": 0} if snow_params is None else snow_params
        clm = daymet.separate_snow(clm, **params)
    return clm


def _gridded_urls(
    code: int,
    bounds: Tuple[float, float, float, float],
    region: str,
    variables: Iterable[str],
    dates: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[str, Dict[str, Dict[str, str]]]]:
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
    dates : list
        A list of dates

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    time_scale = _get_filename(region)

    west, south, east, north = bounds
    base_url = f"{ServiceURL().restful.daymet}/{code}"
    return [
        (
            f"{base_url}/daymet_v4_{time_scale[code](v)}_{s.year}.nc",
            {
                "params": {
                    "var": v,
                    "north": f"{north}",
                    "west": f"{west}",
                    "east": f"{east}",
                    "south": f"{south}",
                    "disableProjSubset": "on",
                    "horizStride": "1",
                    "time_start": s.strftime(DATE_FMT),
                    "time_end": e.strftime(DATE_FMT),
                    "timeStride": "1",
                    "addLatLon": "true",
                    "accept": "netcdf",
                }
            },
        )
        for v, (s, e) in itertools.product(variables, dates)
    ]


def get_bygeom(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: Union[str, pyproj.CRS] = DEF_CRS,
    variables: Optional[Union[Iterable[str], str]] = None,
    region: str = "na",
    time_scale: str = "daily",
    pet: Optional[str] = None,
    pet_params: Optional[Dict[str, float]] = None,
    snow: bool = False,
    snow_params: Optional[Dict[str, float]] = None,
    ssl: Union[SSLContext, bool, None] = None,
) -> xr.Dataset:
    """Get gridded data from the Daymet database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or bbox
        The geometry of the region of interest.
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    region : str, optional
        Region in the US, defaults to na. Acceptable values are:

        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico

    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly average),
        or annual (annual average). Defaults to daily.
    pet : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, ``hargreaves_samani``, and
        None (don't compute PET). The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``None``.
    pet_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the PET function.
        Defaults to ``None``.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0}``.
    ssl : bool or SSLContext, optional
        SSLContext to use for the connection, defaults to None. Set to False to disable
        SSL certification verification.

    Returns
    -------
    xarray.Dataset
        Daily climate data within the target geometry.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> import pydaymet as daymet
    >>> geometry = Polygon(
    ...     [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    ... )
    >>> clm = daymet.get_bygeom(geometry, 2010, variables="tmin", time_scale="annual")
    >>> clm["tmin"].mean().compute().item()
    1.361

    References
    ----------
    .. footbibliography::
    """
    daymet = Daymet(variables, pet, snow, time_scale, region)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    crs = ogcutils.validate_crs(crs)
    _geometry = geoutils.geo2polygon(geometry, crs, DEF_CRS)

    if not _geometry.intersects(daymet.region_bbox[region]):
        raise InvalidInputRange(daymet.invalid_bbox_msg)

    urls, kwds = zip(
        *_gridded_urls(
            daymet.time_codes[time_scale],
            _geometry.bounds,
            daymet.region,
            daymet.variables,
            dates_itr,
        )
    )

    try:
        clm: xr.Dataset = xr.open_mfdataset(  # type: ignore
            (
                io.BytesIO(r)
                for r in ar.retrieve_binary(urls, request_kwds=kwds, max_workers=MAX_CONN, ssl=ssl)
            ),
            engine="scipy",
            coords="minimal",
        )
    except ValueError as ex:
        msg = (
            "The service did NOT process your request successfully. "
            + "Check your inputs and try again."
        )
        raise ServiceError(msg) from ex

    if len(clm.lat.dims) > 2:
        clm["lat"] = clm.lat.isel(time=0, drop=True)
        clm["lon"] = clm.lon.isel(time=0, drop=True)

    clm.attrs["crs"] = " ".join(
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
    clm = clm.rio.write_crs(clm.attrs["crs"], grid_mapping_name="lambert_conformal_conic")
    clm = geoutils.xarray_geomask(clm, _geometry, DEF_CRS)
    clm = clm.drop_vars(["spatial_ref"])
    for v in clm:
        if "grid_mapping" in clm[v].attrs:
            _ = clm[v].attrs.pop("grid_mapping")

    if pet is not None:
        clm = potential_et(clm, method=pet, params=pet_params)

    if snow:
        params = {"t_rain": 2.5, "t_snow": 0} if snow_params is None else snow_params
        clm = daymet.separate_snow(clm, **params)
    return clm
