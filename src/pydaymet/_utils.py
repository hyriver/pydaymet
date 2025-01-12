"""Some utilities for PyDaymet."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Generator, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import numpy as np
import pyproj
import shapely
import urllib3
from pyproj import Transformer
from pyproj.exceptions import CRSError as ProjCRSError
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window
from rioxarray.exceptions import OneDimensionalRaster
from shapely import MultiPolygon, Polygon, STRtree, ops
from shapely.geometry import shape
from urllib3.exceptions import HTTPError

from pydaymet.exceptions import DownloadError, InputRangeError, InputTypeError

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray
    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry

    CRSTYPE = int | str | pyproj.CRS
    POLYTYPE = Polygon | MultiPolygon | tuple[float, float, float, float]
    NUMBER = int | float | np.number

__all__ = [
    "clip_dataset",
    "daymet_tiles",
    "download_files",
    "sample_window",
    "to_geometry",
    "transform_coords",
    "validate_coords",
    "validate_crs",
    "write_crs",
]
CHUNK_SIZE = 1048576  # 1 MB
TransformerFromCRS = lru_cache(Transformer.from_crs)


def validate_crs(crs: CRSTYPE) -> str:
    """Validate a CRS.

    Parameters
    ----------
    crs : str, int, or pyproj.CRS
        Input CRS.

    Returns
    -------
    str
        Validated CRS as a string.
    """
    try:
        return pyproj.CRS(crs).to_string()
    except ProjCRSError as ex:
        raise InputTypeError("crs", "a valid CRS") from ex


def transform_coords(
    coords: Sequence[tuple[float, float]], in_crs: CRSTYPE, out_crs: CRSTYPE
) -> list[tuple[float, float]]:
    """Transform coordinates from one CRS to another."""
    try:
        pts = shapely.points(coords)
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    x, y = shapely.get_coordinates(pts).T
    x_proj, y_proj = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform(x, y)
    return list(zip(x_proj, y_proj))


def _geo_transform(geom: BaseGeometry, in_crs: CRSTYPE, out_crs: CRSTYPE) -> BaseGeometry:
    """Transform a geometry from one CRS to another."""
    project = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform
    return ops.transform(project, geom)


def validate_coords(
    coords: Iterable[tuple[float, float]], bounds: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    """Validate coordinates within a bounding box."""
    try:
        pts = shapely.points(list(coords))
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    if shapely.contains(shapely.box(*bounds), pts).all():
        return shapely.get_coordinates(pts)
    raise InputRangeError("coords", f"within {bounds}")


def to_geometry(
    geometry: BaseGeometry | tuple[float, float, float, float],
    geo_crs: CRSTYPE | None = None,
    crs: CRSTYPE | None = None,
) -> BaseGeometry:
    """Return a Shapely geometry and optionally transformed to a new CRS.

    Parameters
    ----------
    geometry : shaple.Geometry or tuple of length 4
        Any shapely geometry object or a bounding box (minx, miny, maxx, maxy).
    geo_crs : int, str, or pyproj.CRS, optional
        Spatial reference of the input geometry, defaults to ``None``.
    crs : int, str, or pyproj.CRS
        Target spatial reference, defaults to ``None``.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        A shapely geometry object.
    """
    is_geom = np.atleast_1d(shapely.is_geometry(geometry))
    if is_geom.all() and len(is_geom) == 1:
        geom = geometry
    elif isinstance(geometry, Iterable) and len(geometry) == 4 and np.isfinite(geometry).all():
        geom = shapely.box(*geometry)
    else:
        raise InputTypeError("geometry", "a shapley geometry or tuple of length 4")

    if geo_crs is not None and crs is not None:
        return _geo_transform(geom, geo_crs, crs)
    elif geo_crs is None and crs is not None:
        return geom
    raise InputTypeError("geo_crs/crs", "either both None or both valid CRS")


def _transform_xy(
    dataset: DatasetReader, xy: Iterable[tuple[float, float]]
) -> Generator[tuple[int, int], None, None]:
    # Transform x, y coordinates to row, col
    # Chunked to reduce calls, thus unnecessary overhead, to rowcol()
    dt = dataset.transform
    _xy = iter(xy)
    while True:
        buf = tuple(islice(_xy, 0, 256))
        if not buf:
            break
        x, y = rowcol(dt, *zip(*buf))
        yield from zip(x, y)


def sample_window(
    dataset: DatasetReader,
    xy: Iterable[tuple[float, float]],
    window: int = 5,
    indexes: int | list[int] | None = None,
    masked: bool = False,
    resampling: int = 1,
) -> Generator[NDArray[np.floating], None, None]:
    """Interpolate pixel values at given coordinates by interpolation.

    Notes
    -----
    This function is adapted from the ``rasterio.sample.sample_gen`` function of
    `RasterIO <https://rasterio.readthedocs.io/en/latest/api/rasterio.sample.html#rasterio.sample.sample_gen>`__.

    Parameters
    ----------
    dataset : rasterio.DatasetReader
        Opened in ``"r"`` mode.
    xy : iterable
        Pairs of x, y coordinates in the dataset's reference system.
    window : int, optional
        Size of the window to read around each point. Must be odd.
        Default is 5.
    indexes : int or list of int, optional
        Indexes of dataset bands to sample, defaults to all bands.
    masked : bool, optional
        Whether to mask samples that fall outside the extent of the dataset.
        Default is ``False``.
    resampling : int, optional
        Resampling method to use. See rasterio.enums.Resampling for options.
        Default is 1, i.e., ``Resampling.bilinear``.

    Yields
    ------
    numpy.array
        An array of length equal to the number of specified indexes
        containing the interpolated values for the bands corresponding to those indexes.
    """
    height = dataset.height
    width = dataset.width
    if indexes is None:
        indexes = dataset.indexes
    elif isinstance(indexes, int):
        indexes = [indexes]
    indexes = cast("list[int]", indexes)
    nodata = np.full(len(indexes), (dataset.nodata or 0), dtype=dataset.dtypes[0])
    if masked:
        mask_flags = [set(dataset.mask_flag_enums[i - 1]) for i in indexes]
        dataset_is_masked = any(
            {MaskFlags.alpha, MaskFlags.per_dataset, MaskFlags.nodata} & enums
            for enums in mask_flags
        )
        mask = [not (dataset_is_masked and enums == {MaskFlags.all_valid}) for enums in mask_flags]
        nodata = np.ma.array(nodata, mask=mask)

    if window % 2 == 0:
        raise InputTypeError("window", "odd integer")

    half_window = window // 2

    for row, col in _transform_xy(dataset, xy):
        if 0 <= row < height and 0 <= col < width:
            col_start = max(0, col - half_window)
            row_start = max(0, row - half_window)
            data = dataset.read(
                indexes,
                window=Window(col_start, row_start, window, window),  # pyright: ignore[reportCallIssue]
                out_shape=(len(indexes), 1, 1),
                resampling=Resampling(resampling),
                masked=masked,
            )

            yield data[:, 0, 0]
        else:
            yield nodata


def _download(url: str, fname: Path, http: urllib3.HTTPSConnectionPool) -> None:
    """Download a file from a URL."""
    parsed_url = urlparse(url)
    path = f"{parsed_url.path}?{parsed_url.query}"
    head = http.request("HEAD", path)
    fsize = int(head.headers.get("Content-Length", -1))
    if fname.exists() and fname.stat().st_size == fsize:
        return
    fname.unlink(missing_ok=True)
    fname.write_bytes(http.request("GET", path).data)


def download_files(url_list: list[str], file_extension: str, rewrite: bool = False) -> list[Path]:
    """Download multiple files concurrently."""
    hr_cache = os.getenv("HYRIVER_CACHE_NAME")
    cache_dir = Path(hr_cache).parent if hr_cache else Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)

    http = urllib3.HTTPSConnectionPool(
        urlparse(url_list[0]).netloc,
        maxsize=10,
        block=True,
        retries=urllib3.Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 504],
            allowed_methods=["HEAD", "GET"],
        ),
    )

    file_list = [
        Path(cache_dir, f"{hashlib.sha256(url.encode()).hexdigest()}.{file_extension}")
        for url in url_list
    ]
    if rewrite:
        _ = [f.unlink(missing_ok=True) for f in file_list]
    max_workers = min(4, os.cpu_count() or 1, len(url_list))
    if max_workers == 1:
        _ = [_download(url, path, http) for url, path in zip(url_list, file_list)]
        return file_list

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(_download, url, path, http): url
            for url, path in zip(url_list, file_list)
        }
        for future in as_completed(future_to_url):
            try:
                future.result()
            except Exception as e:  # noqa: PERF203
                raise DownloadError(future_to_url[future], e) from e
    return file_list


def write_crs(ds: xr.Dataset, crs: CRSTYPE) -> xr.Dataset:
    """Write geo reference info into a dataset or dataarray."""
    ds = ds.rio.write_transform()
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    if "grid_mapping" in ds.coords:
        ds = ds.drop_vars("grid_mapping")
    ds = ds.rio.write_crs(crs, grid_mapping_name="lambert_conformal_conic")
    ds = ds.rio.write_coordinate_system()
    return ds


def clip_dataset(
    ds: xr.Dataset,
    geometry: Polygon | MultiPolygon | tuple[float, float, float, float],
    crs: CRSTYPE,
) -> xr.Dataset:
    """Mask a ``xarray.Dataset`` based on a geometry."""
    attrs = {v: ds[v].attrs for v in ds}

    geom = to_geometry(geometry, crs, ds.rio.crs)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            ds = ds.rio.clip([geom])
    except OneDimensionalRaster:
        ds = ds.rio.clip([geom], all_touched=True)

    _ = [ds[v].rio.update_attrs(attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds


@lru_cache(maxsize=10)
def _fetch_geojson(url: str) -> dict[str, Any]:
    """Fetch Daymet tiles from a GeoJSON file."""
    try:
        resp = urllib3.request(
            "GET",
            url,
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 504],
                allowed_methods=["GET"],
            ),
        )
    except HTTPError as e:
        raise DownloadError(url, e) from e
    return json.loads(resp.data)["features"]


def daymet_tiles(geometry: POLYTYPE, geo_crs: CRSTYPE = 4326) -> NDArray[np.str_]:
    """Retrieve Daymet tiles from a GeoJSON file."""
    url = "/".join(
        (
            "https://raw.githubusercontent.com/ornldaac/daymet-TDStiles-batch",
            "refs/heads/master/Python/Daymet_v4_Tiles.geojson",
        )
    )
    tile_ids, tile_geoms = zip(
        *(
            (feature["properties"]["TileID"], shape(feature["geometry"]))
            for feature in _fetch_geojson(url)
        )
    )
    tile_ids = np.array(tile_ids)
    geom = to_geometry(geometry, geo_crs, 4326)
    return tile_ids[STRtree(tile_geoms).query(geom, predicate="intersects")]
