"""Some utilities for PyDaymet."""

# pyright: reportMissingTypeArgument=false
from __future__ import annotations

import json
import os
from collections.abc import Generator, Iterable, Sequence
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import numpy as np
import pyproj
import shapely
import tiny_retriever as terry
from pyproj import Transformer
from pyproj.exceptions import CRSError as ProjCRSError
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window
from rioxarray.exceptions import OneDimensionalRaster
from shapely import Polygon, STRtree, ops
from shapely.geometry import shape

from pydaymet.exceptions import DownloadError, InputRangeError, InputTypeError

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray
    from rasterio.io import DatasetReader

    CRSType = int | str | pyproj.CRS
    PolyType = Polygon | tuple[float, float, float, float]
    Number = int | float | np.number

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

TransformerFromCRS = lru_cache(Transformer.from_crs)


def validate_crs(crs: CRSType) -> str:
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
    coords: Sequence[tuple[float, float]], in_crs: CRSType, out_crs: CRSType
) -> list[tuple[float, float]]:
    """Transform coordinates from one CRS to another."""
    try:
        pts = shapely.points(np.atleast_2d(coords))
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    x, y = shapely.get_coordinates(pts).T
    x_proj, y_proj = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform(x, y)
    return list(zip(x_proj, y_proj))


def _geo_transform(geom: Polygon, in_crs: CRSType, out_crs: CRSType) -> Polygon:
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
        return shapely.get_coordinates(pts).round(6)
    raise InputRangeError("coords", f"within {bounds}")


def to_geometry(
    geometry: Polygon | tuple[float, float, float, float],
    geo_crs: CRSType | None = None,
    crs: CRSType | None = None,
) -> Polygon:
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
    shapely.Polygon
        A shapely geometry object.
    """
    is_geom = np.atleast_1d(shapely.is_geometry(geometry))
    if is_geom.all() and len(is_geom) == 1:
        geom = geometry
    elif isinstance(geometry, Iterable) and len(geometry) == 4 and np.isfinite(geometry).all():
        geom = shapely.box(*geometry)
    else:
        raise InputTypeError("geometry", "a shapley geometry or tuple of length 4")
    geom = cast("Polygon", geom)
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


def _get_prefix(url: str, with_var: bool) -> str:
    """Get the file prefix for creating a unique filename from a URL."""
    query = urlparse(url).query
    lat = parse_qs(query).get("latitude", ["grid"])[0]
    lon = parse_qs(query).get("longitude", ["grid"])[0]
    if with_var:
        var = parse_qs(query).get("var", ["var"])[0]
        return f"{lon}_{lat}_{var}"
    return f"{lon}_{lat}"


def download_files(
    url_list: list[str], f_ext: Literal["csv", "nc"], rewrite: bool = False, timeout: int = 1000
) -> list[Path]:
    """Download multiple files concurrently."""
    hr_cache = os.getenv("HYRIVER_CACHE_NAME")
    cache_dir = Path(hr_cache).parent if hr_cache else Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)

    with_var = f_ext == "nc"
    file_list = [
        Path(
            cache_dir,
            terry.unique_filename(url, prefix=_get_prefix(url, with_var), file_extension=f_ext),
        )
        for url in url_list
    ]
    if rewrite:
        _ = [f.unlink(missing_ok=True) for f in file_list]
    terry.download(url_list, file_list, timeout=timeout)
    return file_list


def write_crs(ds: xr.Dataset, crs: CRSType) -> xr.Dataset:
    """Write geo reference info into a dataset or dataarray."""
    ds = ds.rio.write_transform()
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    for v in ds.data_vars:
        _ = ds[v].attrs.pop("grid_mapping", None)
    ds = ds.rio.write_crs(crs, grid_mapping_name="lambert_conformal_conic")
    ds = ds.rio.write_coordinate_system()
    return ds


def clip_dataset(
    ds: xr.Dataset,
    geometry: Polygon | tuple[float, float, float, float],
    crs: CRSType,
) -> xr.Dataset:
    """Mask a ``xarray.Dataset`` based on a geometry."""
    attrs = {v: ds[v].attrs for v in ds}

    geom = to_geometry(geometry, crs, ds.rio.crs)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, Polygon):
            ds = ds.rio.clip([geom])
    except OneDimensionalRaster:
        ds = ds.rio.clip([geom], all_touched=True)

    _ = [ds[v].rio.update_attrs(attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds


@lru_cache(maxsize=10)
def _fetch_geojson(url: str) -> list[dict[str, Any]]:
    """Fetch Daymet tiles from a GeoJSON file."""
    try:
        with urlopen(url) as response:
            return json.loads(response.read())["features"]
    except HTTPError as e:
        raise DownloadError(url, e) from e


def daymet_tiles(geometry: PolyType, geo_crs: CRSType = 4326) -> NDArray[np.str_]:
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
