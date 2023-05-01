"""Access the Daymet database for both single single pixel and gridded queries."""
from __future__ import annotations

import functools
import io
import itertools
import re
from typing import TYPE_CHECKING, Callable, Generator, Iterable, Sequence, Union, cast

import async_retriever as ar
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import pyproj
import xarray as xr
from pygeoogc import ServiceError, ServiceURL
from pygeoutils import Coordinates

from pydaymet.core import T_RAIN, T_SNOW, Daymet
from pydaymet.exceptions import InputRangeError, InputTypeError
from pydaymet.pet import potential_et

if TYPE_CHECKING:
    from pathlib import Path

    from shapely.geometry import MultiPolygon, Polygon

    CRSTYPE = Union[int, str, pyproj.CRS]

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
MAX_CONN = 10

__all__ = ["get_bycoords", "get_bygeom"]


def _get_filename(
    region: str,
) -> dict[int, Callable[[str], str]]:
    """Get correct filenames based on region and variable of interest."""
    return {
        2129: lambda v: f"daily_{region}_{v}",
        2131: lambda v: f"{v}_monttl_{region}" if v == "prcp" else f"{v}_monavg_{region}",
        2130: lambda v: f"{v}_annttl_{region}" if v == "prcp" else f"{v}_annavg_{region}",
    }


def _coord_urls(
    code: int,
    coord: tuple[float, float],
    region: str,
    variables: Iterable[str],
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> Generator[list[tuple[str, dict[str, dict[str, str]]]], None, None]:
    """Generate an iterable URL list for downloading Daymet data.

    Parameters
    ----------
    code : int
        Endpoint code which should be one of the following:

        * 2129: Daily
        * 2131: Monthly average
        * 2130: Annual average

    coord : tuple of length 2
        Coordinates in EPSG:4326 CRS (lon, lat)
    region : str
        Region in the US. Acceptable values are:

        * ``na``: Continental North America
        * ``hi``: Hawaii
        * ``pr``: Puerto Rico

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
    return (
        [
            (
                f"{base_url}/daymet_v4_{time_scale[code](v)}_{s.year}.nc",
                {
                    "params": {
                        "var": v,
                        "longitude": f"{lon:0.6f}",
                        "latitude": f"{lat:0.6f}",
                        "time_start": s.strftime(DATE_FMT),
                        "time_end": e.strftime(DATE_FMT),
                        "accept": "csv",
                    }
                },
            )
            for s, e in dates
        ]
        for v in variables
    )


def _get_lon_lat(
    coords: list[tuple[float, float]] | tuple[float, float],
    coords_id: Sequence[str | int] | None = None,
    crs: CRSTYPE = 4326,
    to_xarray: bool = False,
) -> tuple[list[float], list[float]]:
    """Get longitude and latitude from a list of coordinates."""
    coords_list = geoutils.coords_list(coords)

    if to_xarray and coords_id is not None and len(coords_id) != len(coords_list):
        raise InputTypeError("coords_id", "list with the same length as of coords")

    coords_list = ogc.match_crs(coords_list, crs, 4326)
    lon, lat = zip(*coords_list)
    return list(lon), list(lat)


def _by_coord(
    lon: float,
    lat: float,
    daymet: Daymet,
    time_scale: str,
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    pet: str | None,
    pet_params: dict[str, float] | None,
    snow: bool,
    snow_params: dict[str, float] | None,
    ssl: bool,
) -> pd.DataFrame:
    """Get climate data for a coordinate and return as a DataFrame."""
    coords = (lon, lat)
    url_kwds = _coord_urls(
        daymet.time_codes[time_scale], coords, daymet.region, daymet.variables, dates
    )
    retrieve = functools.partial(ar.retrieve_text, max_workers=MAX_CONN, ssl=ssl)
    clm = pd.concat(
        (
            pd.concat(
                pd.read_csv(io.StringIO(r), parse_dates=[0], usecols=[0, 3], index_col=[0])
                for r in retrieve(u, k)  # type: ignore
            )
            for u, k in (zip(*u) for u in url_kwds)
        ),
        axis=1,
    )
    clm.columns = [c.replace('[unit="', " (").replace('"]', ")") for c in clm.columns]

    if "prcp (mm)" in clm:
        clm = clm.rename(columns={"prcp (mm)": "prcp (mm/day)"})

    clm = clm.set_index(pd.to_datetime(clm.index.strftime("%Y-%m-%d")))
    clm = clm.where(clm > -9999)

    if pet is not None:
        clm = potential_et(clm, coords, method=pet, params=pet_params)  # type: ignore

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = daymet.separate_snow(clm, **params)
    clm.index.name = "time"
    return clm


def get_bycoords(
    coords: list[tuple[float, float]] | tuple[float, float],
    dates: tuple[str, str] | int | list[int],
    coords_id: Sequence[str | int] | None = None,
    crs: CRSTYPE = 4326,
    variables: Iterable[str] | str | None = None,
    region: str = "na",
    time_scale: str = "daily",
    pet: str | None = None,
    pet_params: dict[str, float] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    ssl: bool = True,
    to_xarray: bool = False,
) -> pd.DataFrame | xr.Dataset:
    """Get point-data from the Daymet database at 1-km resolution.

    This function uses THREDDS data service to get the coordinates
    and supports getting monthly and annual summaries of the climate
    data directly from the server.

    Parameters
    ----------
    coords : tuple or list of tuples
        Coordinates of the location(s) of interest as a tuple (x, y)
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years ``[2001, 2010, ...]``.
    coords_id : list of int or str, optional
        A list of identifiers for the coordinates. This option only applies when ``to_xarray``
        is set to ``True``. If not provided, the coordinates will be enumerated.
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input coordinates, defaults to ``EPSG:4326``.
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
        Model-specific parameters as a dictionary, defaults to ``None``. An important
        parameter for ``priestley_taylor`` and ``penman_monteith`` methods is
        ``arid_correction`` which is used to correct the actual vapor pressure
        for arid regions. Since relative humidity is not provided by Daymet, the actual
        vapor pressure is computed assuming that the dewpoint temperature is equal to
        the minimum temperature. However, for arid regions, FAO 56 suggests to subtract
        minimum temperature by 2-3 °C to account for the fact that in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2°C from the
        minimum temperature for computing the actual vapor pressure.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    ssl : bool, optional
        Whether to verify SSL certification, defaults to ``True``.
    to_xarray : bool, optional
        Return the data as an ``xarray.Dataset``. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Daily climate data for a single or list of locations.

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

    lon, lat = _get_lon_lat(coords, coords_id, crs, to_xarray)
    points = Coordinates(lon, lat, daymet.region_bbox[region].bounds).points
    n_pts = len(points)
    if n_pts == 0 or n_pts != len(lon):
        raise InputRangeError("coords", f"within {daymet.region_bbox[region].bounds}")

    by_coord = functools.partial(
        _by_coord,
        daymet=daymet,
        time_scale=time_scale,
        dates=dates_itr,
        pet=pet,
        pet_params=pet_params,
        snow=snow,
        snow_params=snow_params,
        ssl=ssl,
    )
    clm_list = itertools.starmap(by_coord, zip(points.x, points.y))

    idx = list(coords_id) if coords_id is not None else [f"P{i}" for i in range(n_pts)]
    if to_xarray:
        clm_ds = xr.concat(
            (xr.Dataset.from_dataframe(clm) for clm in clm_list), dim=pd.Index(idx, name="id")
        )
        clm_ds = clm_ds.rename(
            {n: re.sub(r"\([^\)]*\)", "", str(n)).strip() for n in clm_ds.data_vars}
        )
        clm_ds["time"] = pd.DatetimeIndex(pd.to_datetime(clm_ds["time"]).date)
        for v in clm_ds.data_vars:
            clm_ds[v].attrs["units"] = daymet.units[v]
            clm_ds[v].attrs["long_name"] = daymet.long_names[v]
            clm_ds[v].attrs["description"] = daymet.descriptions[v]
        return clm_ds

    if n_pts == 1:
        clm = next(iter(clm_list), pd.DataFrame())
    else:
        clm = cast("pd.DataFrame", pd.concat(clm_list, keys=idx, axis=1))
        clm = clm.columns.set_names(["id", "variable"])
    clm = clm.set_index(pd.DatetimeIndex(pd.to_datetime(clm.index).date))
    return clm


def _gridded_urls(
    code: int,
    bounds: tuple[float, float, float, float],
    region: str,
    variables: Iterable[str],
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> Generator[tuple[str, dict[str, dict[str, str]]], None, None]:
    """Generate an iterable URL list for downloading Daymet data.

    Parameters
    ----------
    code : int
        Endpoint code which should be one of the following:

        * 2129: Daily
        * 2131: Monthly average
        * 2130: Annual average

    bounds : tuple of length 4
        Bounding box (west, south, east, north)
    region : str
        Region in the US. Acceptable values are:

        * ``na``: Continental North America
        * ``hi``: Hawaii
        * ``pr``: Puerto Rico

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
    return (
        (
            f"{base_url}/daymet_v4_{time_scale[code](v)}_{s.year}.nc",
            {
                "params": {
                    "var": v,
                    "north": f"{north:0.6f}",
                    "west": f"{west:0.6f}",
                    "east": f"{east:0.6f}",
                    "south": f"{south:0.6f}",
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
    )


def _open_dataset(f: Path) -> xr.Dataset:
    """Open a dataset using ``xarray``."""
    with xr.open_dataset(f, engine="scipy") as ds:
        return ds.load()


def get_bygeom(
    geometry: Polygon | MultiPolygon | tuple[float, float, float, float],
    dates: tuple[str, str] | int | list[int],
    crs: CRSTYPE = 4326,
    variables: Iterable[str] | str | None = None,
    region: str = "na",
    time_scale: str = "daily",
    pet: str | None = None,
    pet_params: dict[str, float] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    ssl: bool = True,
) -> xr.Dataset:
    """Get gridded data from the Daymet database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or bbox
        The geometry of the region of interest.
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs : str, int, or pyproj.CRS, optional
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
        Model-specific parameters as a dictionary, defaults to ``None``. Valid
        parameters are:

        * ``penman_monteith``: ``soil_heat_flux``, ``albedo``, ``alpha``,
          and ``arid_correction``.
        * ``priestley_taylor``: ``soil_heat_flux``, ``albedo``, and ``arid_correction``.
        * ``hargreaves_samani``: None.

        Default values for the parameters are: ``soil_heat_flux`` = 0, ``albedo`` = 0.23,
        ``alpha`` = 1.26, and ``arid_correction`` = False.
        An important parameter for ``priestley_taylor`` and ``penman_monteith`` methods
        is ``arid_correction`` which is used to correct the actual vapor pressure
        for arid regions. Since relative humidity is not provided by Daymet, the actual
        vapor pressure is computed assuming that the dewpoint temperature is equal to
        the minimum temperature. However, for arid regions, FAO 56 suggests to subtract
        minimum temperature by 2-3 °C to account for the fact that in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2 °C from the
        minimum temperature for computing the actual vapor pressure.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    ssl : bool, optional
        Whether to verify SSL certification, defaults to ``True``.

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

    crs = ogc.validate_crs(crs)
    _geometry = geoutils.geo2polygon(geometry, crs, 4326)

    if not _geometry.intersects(daymet.region_bbox[region]):
        raise InputRangeError("geometry", f"within {daymet.region_bbox[region].bounds}")

    urls, kwds = zip(
        *_gridded_urls(
            daymet.time_codes[time_scale],
            _geometry.bounds,
            daymet.region,
            daymet.variables,
            dates_itr,
        )
    )
    urls = cast("list[str]", list(urls))
    kwds = cast("list[dict[str, dict[str, str]]]", list(kwds))

    clm_files = ogc.streaming_download(
        urls,
        kwds,
        file_extention="nc",
        ssl=ssl,
        n_jobs=MAX_CONN,
    )
    try:
        # open_mfdataset can run into too many open files error so we use merge
        # https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets
        clm = xr.merge(_open_dataset(f) for f in clm_files)
    except ValueError as ex:
        msg = (
            "Daymet did NOT process your request successfully. "
            + "Check your inputs and try again."
        )
        raise ServiceError(msg) from ex

    if len(clm.lat.dims) > 2:
        clm["lat"] = clm.lat.isel(time=0, drop=True)
        clm["lon"] = clm.lon.isel(time=0, drop=True)

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
    clm = xr.where(clm > -9999, clm, np.nan, keep_attrs=True)
    for v in clm:
        clm[v].rio.write_nodata(np.nan, inplace=True)
    clm = geoutils.xd_write_crs(clm, crs, "lambert_conformal_conic")
    clm = cast("xr.Dataset", clm)
    clm = geoutils.xarray_geomask(clm, _geometry, 4326)

    if pet:
        clm = potential_et(clm, method=pet, params=pet_params)

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = daymet.separate_snow(clm, **params)

    clm["time"] = pd.DatetimeIndex(pd.to_datetime(clm["time"]).date)
    return clm
