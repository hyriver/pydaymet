"""Access the Daymet database for both single single pixel and gridded queries."""

# pyright: reportArgumentType=false,reportCallIssue=false,reportReturnType=false
from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Callable, Literal
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import xarray as xr

import pydaymet._utils as utils
from pydaymet.core import T_RAIN, T_SNOW, Daymet, separate_snow
from pydaymet.exceptions import (
    InputRangeError,
    InputTypeError,
    MissingDependencyError,
    ServiceError,
)
from pydaymet.pet import potential_et

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pyproj
    from shapely import Polygon

    CRSType = int | str | pyproj.CRS
    PETMethods = Literal["penman_monteith", "priestley_taylor", "hargreaves_samani"]

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"

__all__ = ["get_bycoords", "get_bygeom", "get_bystac"]

URL = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac"


def _get_filename(
    region: str,
) -> dict[int, Callable[[str], str]]:
    """Get correct filenames based on region and variable of interest."""
    return {
        2129: lambda v: f"daily_{region}_{v}",
        2131: lambda v: f"{v}_monttl_{region}" if v == "prcp" else f"{v}_monavg_{region}",
        2130: lambda v: f"{v}_annttl_{region}" if v == "prcp" else f"{v}_annavg_{region}",
    }


def _get_lon_lat(
    coords: list[tuple[float, float]] | tuple[float, float],
    bounds: tuple[float, float, float, float],
    coords_id: Sequence[str | int] | None,
    crs: CRSType,
    to_xarray: bool,
) -> tuple[list[float], list[float]]:
    """Get longitude and latitude from a list of coordinates."""
    coords_list = utils.transform_coords(coords, crs, 4326)

    if to_xarray and coords_id is not None and len(coords_id) != len(coords_list):
        raise InputTypeError("coords_id", "list with the same length as of coords")

    lon, lat = utils.validate_coords(coords_list, bounds).T
    return lon.tolist(), lat.tolist()


def _by_coord(
    coords: tuple[float, float],
    csv_file: Path,
    pet: PETMethods | None,
    pet_params: dict[str, float] | None,
    snow: bool,
    snow_params: dict[str, float] | None,
) -> pd.DataFrame:
    """Get climate data for a coordinate and return as a DataFrame."""
    clm = pd.read_csv(csv_file, skiprows=6)
    clm["time"] = pd.to_datetime(
        clm["year"].astype(str) + "-" + clm["yday"].astype(str), format="%Y-%j"
    )
    clm = clm.drop(columns=["year", "yday"]).set_index("time")
    clm = clm.where(clm > -9999)
    clm.columns = clm.columns.str.replace("deg c", "degrees C").str.replace("^", "")

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)

    if pet is not None:
        clm = potential_et(clm, coords, method=pet, params=pet_params)
    return clm


def get_bycoords(
    coords: list[tuple[float, float]] | tuple[float, float],
    dates: tuple[str, str] | int | list[int],
    coords_id: Sequence[str | int] | None = None,
    crs: CRSType = 4326,
    variables: Iterable[Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]]
    | Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]
    | None = None,
    region: Literal["na", "hi", "pr"] = "na",
    time_scale: Literal["daily", "monthly", "annual"] = "daily",
    pet: PETMethods | None = None,
    pet_params: dict[str, float] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    to_xarray: bool = False,
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> pd.DataFrame | xr.Dataset:
    """Get point-data from the Daymet database at 1-km resolution.

    This function uses THREDDS data service to get the coordinates
    and supports getting monthly and annual summaries of the climate
    data directly from the server.

    Parameters
    ----------
    coords : tuple or list of tuples
        Coordinates of the location(s) of interest as a tuple (x, y)
    dates : tuple or list
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
        the minimum temperature. However, for arid regions, FAO 56 suggests subtracting
        the minimum temperature by 2-3 °C to account for aridity, since in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2 °C from the
        minimum temperature before computing the actual vapor pressure.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    to_xarray : bool, optional
        Return the data as an ``xarray.Dataset``. Defaults to ``False``.
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

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
    ...     crs=3542,
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
        # dates_itr = daymet.dates_tolist(dates)
        req_start = pd.to_datetime(dates[0]).strftime(DATE_FMT)
        req_end = pd.to_datetime(dates[1]).strftime(DATE_FMT)
        req_years = None
    else:
        # dates_itr = daymet.years_tolist(dates)
        dates = [dates] if isinstance(dates, int) else dates
        req_years = ",".join(map(str, dates))
        req_start, req_end = None, None

    lons, lats = _get_lon_lat(coords, daymet.region_bbox[region].bounds, coords_id, crs, to_xarray)
    n_pts = len(lons)

    req_vars = ",".join(daymet.variables)
    if req_years is None:
        base_url = (
            "https://daymet.ornl.gov/single-pixel/api/data?lat={}&lon={}&vars={}&start={}&end={}"
        )
        urls = [
            base_url.format(lat, lon, req_vars, req_start, req_end) for lat, lon in zip(lats, lons)
        ]
    else:
        base_url = "https://daymet.ornl.gov/single-pixel/api/data?lat={}&lon={}&vars={}&years={}"
        urls = [base_url.format(lat, lon, req_vars, req_years) for lat, lon in zip(lats, lons)]

    idx = list(coords_id) if coords_id is not None else list(range(n_pts))
    idx = dict(zip(zip(lons, lats), idx))
    csv_files = utils.download_files(urls, "csv", validate_filesize, conn_timeout)
    clm_list = {
        idx[c]: _by_coord(c, f, pet, pet_params, snow, snow_params)
        for c, f in zip(zip(lons, lats), csv_files)
    }
    if to_xarray:
        clm_ds = xr.concat(
            (xr.Dataset.from_dataframe(clm) for clm in clm_list.values()),
            dim=pd.Index(list(clm_list), name="id"),
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
        return next(iter(clm_list.values()), pd.DataFrame())
    return pd.concat(clm_list.values(), keys=list(clm_list), axis=1, names=["id", "variable"])


def _gridded_urls(
    code: int,
    bounds: tuple[float, float, float, float],
    region: str,
    variables: Iterable[str],
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[str]:
    """Generate a list of URLs for downloading Daymet data.

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
    list
        A list of generated URLs.
    """
    time_scale = _get_filename(region)
    west, south, east, north = bounds
    return [
        f"{URL}/{code}/daymet_v4_{time_scale[code](v)}_{s.year}.nc?"
        + urlencode(
            {
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
        )
        for v, (s, e) in itertools.product(variables, dates)
    ]


def _open_dataset(f: Path) -> xr.Dataset:
    """Open a dataset using ``xarray``."""
    with xr.open_dataset(f) as ds:
        return ds.load()


def get_bygeom(
    geometry: Polygon | tuple[float, float, float, float],
    dates: tuple[str, str] | int | list[int],
    crs: CRSType = 4326,
    variables: Iterable[Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]]
    | Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]
    | None = None,
    region: Literal["na", "hi", "pr"] = "na",
    time_scale: Literal["daily", "monthly", "annual"] = "daily",
    pet: PETMethods | None = None,
    pet_params: dict[str, float] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> xr.Dataset:
    """Get gridded data from the Daymet database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon or tuple
        The geometry of the region of interest. It can be a shapely Polygon or a tuple
        of length 4 representing the bounding box (minx, miny, maxx, maxy).
    dates : tuple or list
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
        the minimum temperature. However, for arid regions, FAO 56 suggests subtracting
        the minimum temperature by 2-3 °C to account for aridity, since in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2 °C from the
        minimum temperature before computing the actual vapor pressure.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

    Returns
    -------
    xarray.Dataset
        Daily climate data within the target geometry.

    Examples
    --------
    >>> from shapely import Polygon
    >>> import pydaymet as daymet
    >>> geometry = Polygon(
    ...     [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    ... )
    >>> clm = daymet.get_bygeom(geometry, 2010, variables="tmin", time_scale="annual")
    >>> clm["tmin"].mean().item()
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

    crs = utils.validate_crs(crs)
    _geometry = utils.to_geometry(geometry, crs, 4326)

    if not _geometry.intersects(daymet.region_bbox[region]):
        raise InputRangeError("geometry", f"within {daymet.region_bbox[region].bounds}")

    urls = _gridded_urls(
        daymet.time_codes[time_scale],
        _geometry.bounds,
        daymet.region,
        daymet.variables,
        dates_itr,
    )

    clm_files = utils.download_files(urls, "nc", validate_filesize, conn_timeout)
    try:
        # open_mfdataset can run into too many open files error so we use merge
        # https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets
        clm = xr.merge(_open_dataset(f) for f in clm_files)
    except ValueError as ex:
        msg = " ".join(
            (
                "Daymet did NOT process your request successfully.",
                "Check your inputs and try again.",
            )
        )
        raise ServiceError(msg) from ex

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
    if "spatial_ref" in clm:
        clm = clm.drop_vars("spatial_ref")
    clm = utils.write_crs(clm, crs)
    clm = utils.clip_dataset(clm, _geometry, 4326)
    clm = clm.rio.reproject(crs.replace("units=km", "units=m"), resolution=1000)

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)

    if pet:
        clm = potential_et(clm, method=pet, params=pet_params)

    clm["time"] = pd.DatetimeIndex(pd.to_datetime(clm["time"]).date)
    return clm


def get_bystac(
    geometry: Polygon | tuple[float, float, float, float],
    dates: tuple[str, str],
    crs: CRSType = 4326,
    variables: Iterable[Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]]
    | Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]
    | None = None,
    region: Literal["na", "hi", "pr"] = "na",
    time_scale: Literal["daily", "monthly", "annual"] = "daily",
    res_km: int = 1,
    pet: PETMethods | None = None,
    pet_params: dict[str, float] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
) -> xr.Dataset:
    """Get gridded Daymet from STAC.

    .. versionadded:: 0.16.1
    .. note::
        This function provides access to the Daymet data from Microsoft's
        the Planetary Computer:
        https://planetarycomputer.microsoft.com/dataset/group/daymet.
        Although this function can be much faster than :func:`get_bygeom`,
        currently, it gives access to Daymet v4.2 from 1980 to 2020. For
        accessing the latest version of Daymet (v4.5) you need to use
        :func:`get_bygeom`.

        Also, this function requires ``fsspec``, ``dask``, ``zarr``, and
        ``pystac-client`` packages. They can be installed using
        ``pip install fsspec dask zarr pystac-client`` or
        ``conda install fsspec dask-core zarr pystac-client``.

    Parameters
    ----------
    geometry : Polygon or tuple
        The geometry of the region of interest. It can be a shapely Polygon or a tuple
        of length 4 representing the bounding box (minx, miny, maxx, maxy).
    dates : tuple
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
    res_km : int, optional
        Spatial resolution in kilometers, defaults to 1. For values
        greater than 1, the data will be aggregated (coarsend) using mean.
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
        the minimum temperature. However, for arid regions, FAO 56 suggests subtracting
        the minimum temperature by 2-3 °C to account for aridity, since in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2 °C from the
        minimum temperature before computing the actual vapor pressure.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.

    Returns
    -------
    xarray.Dataset
        Daily climate data within the target geometry.

    Examples
    --------
    >>> from shapely import Polygon
    >>> geometry = Polygon(
    ...     [[-69.77, 45.07], [-69.70, 45.07], [-69.70, 45.15], [-69.77, 45.15], [-69.77, 45.07]]
    ... )
    >>> clm = daymet.get_bystac(
    ...     geometry,
    ...     ("2010-01-01", "2010-01-02"),
    ...     variables="prcp",
    ...     res_km=4,
    ...     snow=True,
    ...     pet="hargreaves_samani",
    ... )
    >>> clm["pet"].mean().item()
    0.3

    References
    ----------
    .. footbibliography::
    """
    try:
        import dask.config
        import fsspec
        import pystac
    except ImportError as ex:
        raise MissingDependencyError from ex

    daymet = Daymet(variables, pet, snow, time_scale, region)

    crs = utils.validate_crs(crs)
    if not utils.to_geometry(geometry, crs, 4326).intersects(daymet.region_bbox[region]):
        raise InputRangeError("geometry", f"within {daymet.region_bbox[region].bounds}")

    if not isinstance(res_km, int) or res_km < 1:
        raise InputTypeError("res_km", "positive integer", "1")

    if (
        not isinstance(dates, tuple)
        or len(dates) != 2
        or not all(isinstance(d, str) for d in dates)
    ):
        raise InputTypeError("dates", "tuple of (start, end) dates", "('2000-01-01', '2000-12-31')")
    start = pd.to_datetime(dates[0]).strftime("%Y-%m-%d")
    end = pd.to_datetime(dates[1]).strftime("%Y-%m-%d")
    time_slice = slice(start, end)

    url = (
        "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
        f"daymet-{time_scale}-{region}"
    )
    collection = pystac.read_file(url)
    asset = collection.assets["zarr-https"]

    store = fsspec.get_mapper(asset.href)
    ds = xr.open_zarr(store, decode_coords="all", **asset.extra_fields["xarray:open_kwargs"])

    _geometry = utils.to_geometry(geometry, crs, ds.rio.crs)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        if res_km > 1:
            clm = (
                ds[daymet.variables]
                .sel(time=time_slice)
                .rio.clip_box(*_geometry.bounds)
                .coarsen(dim={"x": res_km, "y": res_km}, boundary="trim")
                .mean()
                .load()
            )
        else:
            clm = ds[daymet.variables].sel(time=time_slice).rio.clip_box(*_geometry.bounds).load()
    ds.close()
    clm = utils.clip_dataset(clm, _geometry, ds.rio.crs)

    lat = clm["lat"].values
    lon = clm["lon"].values
    clm = clm.drop_vars(["lat", "lon"])
    clm["lat"] = (("y", "x"), lat)
    clm["lon"] = (("y", "x"), lon)
    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)

    if pet:
        clm = potential_et(clm, method=pet, params=pet_params)
    return clm
