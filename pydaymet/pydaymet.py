"""Access the Daymet database for both single single pixel and gridded queries."""
import io
import itertools
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import async_retriever as ar
import pandas as pd
import pygeoutils as geoutils
import rasterio.features as rio_features
import xarray as xr
from pygeoogc import MatchCRS, ServiceURL
from shapely.geometry import MultiPolygon, Point, Polygon

from .core import Daymet
from .exceptions import InvalidInputRange, InvalidInputType

DEF_CRS = "epsg:4326"
DATE_REQ = "%Y-%m-%dT%H:%M:%SZ"


def get_byloc(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
    variables: Optional[Union[Iterable[str], str]] = None,
    pet: bool = False,
) -> pd.DataFrame:
    """Get daily climate data from Daymet for a single point.

    .. deprecated:: 0.9.0
        Please use ``get_bycoords`` instead. This function will be removed in the future.

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
        `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__.
        The default is False

    Returns
    -------
    pandas.DataFrame
        Daily climate data for a location.
    """
    msg = "Please use get_bycoords instead. This function will be removed in the future."
    warnings.warn(msg, DeprecationWarning)

    daymet = Daymet(variables, pet)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_dict = daymet.dates_todict(dates)
    else:
        dates_dict = daymet.years_todict(dates)

    if not (isinstance(coords, tuple) and len(coords) == 2):
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    lon, lat = MatchCRS(crs, DEF_CRS).coords([coords])[0]

    if not ((14.5 < lat < 52.0) or (-131.0 < lon < -53.0)):
        raise InvalidInputRange(
            "The location is outside the Daymet dataset. "
            + "The acceptable range is: "
            + "14.5 < lat < 52.0 and -131.0 < lon < -53.0"
        )

    params = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "vars": ",".join(daymet.variables),
        "format": "json",
        **dates_dict,
    }

    r = ar.retrieve([ServiceURL().restful.daymet_point], "json", [{"params": params}])

    clm = pd.DataFrame(r[0]["data"])
    clm.index = pd.to_datetime(clm.year * 1000.0 + clm.yday, format="%Y%j")
    clm = clm.drop(["year", "yday"], axis=1)

    if pet:
        clm = daymet.pet_bycoords(clm, (lon, lat), alt_unit=True)
    return clm


def get_bycoords(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
    variables: Optional[Union[Iterable[str], str]] = None,
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
        Start and end dates as a tuple (start, end) or a list of years ``[2001, 2010, ...]``.
    crs : str, optional
        The CRS of the input geometry, defaults to ``"epsg:4326"``.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    pet : bool
        Whether to compute evapotranspiration based on
        `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__.
        The default is False
    region : str, optional
        Target region in the US, defaults to ``na``. Acceptable values are:

        * ``na``: Continental North America
        * ``hi``: Hawaii
        * ``pr``: Puerto Rico

    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly summaries),
        or annual (annual summaries). Defaults to daily.

    Returns
    -------
    pandas.DataFrame
        Daily climate data for a location.

    Examples
    --------
    >>> import pydaymet as daymet
    >>> coords = (-1431147.7928, 318483.4618)
    >>> dates = ("2000-01-01", "2000-12-31")
    >>> clm = daymet.get_bycoords(coords, dates, crs="epsg:3542", pet=True)
    >>> clm["pet (mm/day)"].mean()
    3.472
    """
    daymet = Daymet(variables, pet, time_scale, region)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    if not (isinstance(coords, tuple) and len(coords) == 2):
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    coords = MatchCRS(crs, DEF_CRS).coords([coords])[0]

    if not Point(*coords).within(daymet.region_bbox[region]):
        raise InvalidInputRange(daymet.invalid_bbox_msg)

    url_kwds = _coord_urls(
        daymet.time_codes[time_scale], coords, daymet.region, daymet.variables, dates_itr
    )
    url_kwd_list = [tuple(zip(*u)) for u in url_kwds]

    clm = pd.concat(
        (
            pd.concat(
                pd.read_csv(io.BytesIO(r), parse_dates=[0], usecols=[0, 3], index_col=[0])
                for r in ar.retrieve(u, "binary", request_kwds=k, max_workers=8)
            )
            for u, k in url_kwd_list
        ),
        axis=1,
    )
    clm.columns = [c.replace('[unit="', " (").replace('"]', ")") for c in clm.columns]

    if "prcp (mm)" in clm:
        clm = clm.rename(columns={"prcp (mm)": "prcp (mm/day)"})

    clm = clm.set_index(pd.to_datetime(clm.index.strftime("%Y-%m-%d")))

    if pet:
        clm = daymet.pet_bycoords(clm, coords, alt_unit=False)
    return clm


def get_bygeom(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
    variables: Optional[Union[Iterable[str], str]] = None,
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
    crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    pet : bool
        Whether to compute evapotranspiration based on
        `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__.
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
    """
    daymet = Daymet(variables, pet, time_scale, region)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    _geometry = geoutils.pygeoutils._geo2polygon(geometry, crs, DEF_CRS)

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
        clm = xr.open_mfdataset(
            (io.BytesIO(r) for r in ar.retrieve(urls, "binary", request_kwds=kwds, max_workers=8)),
            engine="scipy",
            coords="minimal",
        )
    except ValueError:
        msg = (
            "The server did NOT process your request successfully. "
            + "Check your inputs and try again."
        )
        raise ValueError(msg)

    for k, v in daymet.units.items():
        if k in clm.variables:
            clm[k].attrs["units"] = v

    clm = clm.drop_vars(["lambert_conformal_conic"])

    daymet_crs = " ".join(
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
    clm.attrs["crs"] = daymet_crs
    clm.attrs["nodatavals"] = (0.0,)
    transform, _, _ = geoutils.pygeoutils._get_transform(clm, ("y", "x"))
    clm.attrs["transform"] = transform
    clm.attrs["res"] = (transform.a, transform.e)

    if pet:
        clm = daymet.pet_bygrid(clm)

    if isinstance(clm, xr.Dataset):
        for v in clm:
            clm[v].attrs["crs"] = crs
            clm[v].attrs["nodatavals"] = (0.0,)

    return _xarray_geomask(clm, geometry, crs)


def _xarray_geomask(
    ds: Union[xr.Dataset, xr.DataArray],
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    geo_crs: str,
) -> Union[xr.Dataset, xr.DataArray]:
    """Mask a ``xarray.Dataset`` based on a geometry.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    geometry : Polygon, MultiPolygon, or tuple of length 4
        The geometry or bounding box to mask the data
    geo_crs : str
        The spatial reference of the input geometry

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input dataset with a mask applied (np.nan)
    """
    ds_dims = ("y", "x")
    transform, width, height = geoutils.pygeoutils._get_transform(ds, ds_dims)
    _geometry = geoutils.pygeoutils._geo2polygon(geometry, geo_crs, ds.crs)

    _mask = rio_features.geometry_mask([_geometry], (height, width), transform, invert=True)

    coords = {ds_dims[0]: ds.coords[ds_dims[0]], ds_dims[1]: ds.coords[ds_dims[1]]}
    mask = xr.DataArray(_mask, coords, dims=ds_dims)

    ds_masked = ds.where(mask, drop=True)
    ds_masked.attrs["transform"] = transform
    ds_masked.attrs["bounds"] = _geometry.bounds

    return ds_masked


def _get_filename(
    region: str,
) -> Dict[int, Callable[[str], str]]:
    """Generate an iterable URL list for downloading Daymet data.

    Parameters
    ----------
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
    dates: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
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
                        "time_start": s.strftime(DATE_REQ),
                        "time_end": e.strftime(DATE_REQ),
                        "accept": "csv",
                    }
                },
            )
            for s, e in dates
        ]
        for v in variables
    ]


def _gridded_urls(
    code: int,
    bounds: Tuple[float, float, float, float],
    region: str,
    variables: Iterable[str],
    dates: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
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
                    "time_start": s.strftime(DATE_REQ),
                    "time_end": e.strftime(DATE_REQ),
                    "timeStride": "1",
                    "addLatLon": "true",
                    "accept": "netcdf",
                }
            },
        )
        for v, (s, e) in itertools.product(variables, dates)
    ]
