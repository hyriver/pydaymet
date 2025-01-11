"""Some utilities for PyDaymet."""
from __future__ import annotations

from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, cast, Generator, Iterable
from functools import lru_cache
from pyproj import Transformer
import ssl
from rioxarray.exceptions import OneDimensionalRaster
import hashlib
import os
from itertools import islice
import xarray as xr
import pyproj
from pyproj.exceptions import CRSError as ProjCRSError
from pydaymet.exceptions import InputTypeError, InputRangeError, DownloadError
from shapely import Polygon, MultiPolygon, ops
import shapely
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from rasterio.io import DatasetReader

    CRSTYPE = int | str | pyproj.CRS
    POLYTYPE = Polygon | MultiPolygon | tuple[float, float, float, float]
    NUMBER = int | float | np.number

__all__ = [
    "validate_crs",
    "transform_coords",
    "validate_coords",
    "geo2poly",
    "sample_window",
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


def _geo_transform(
    geom: POLYTYPE, in_crs: CRSTYPE, out_crs: CRSTYPE
) -> Polygon | MultiPolygon:
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
    if shapely.contains(shapely.box(*bounds), shapely.points(coords)).all():
        return shapely.get_coordinates(pts)
    raise InputRangeError("coords", f"within {bounds}")


def geo2poly(
    geometry: POLYTYPE,
    geo_crs: CRSTYPE | None = None,
    crs: CRSTYPE | None = None,
) -> Polygon | MultiPolygon:
    """Convert a geometry to a Shapely's Polygon and transform to any CRS.

    Parameters
    ----------
    geometry : Polygon or tuple of length 4
        Polygon or bounding box (west, south, east, north).
    geo_crs : int, str, or pyproj.CRS, optional
        Spatial reference of the input geometry, defaults to ``None``.
    crs : int, str, or pyproj.CRS
        Target spatial reference, defaults to ``None``.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        A (Multi)Polygon in the target CRS, if different from the input CRS.
    """
    if isinstance(geometry, (Polygon, MultiPolygon)):
        geom = geometry
    elif (
        not isinstance(geometry, Sequence)
        or len(geometry) != 4
        or not all(isinstance(x, (int, float)) for x in geometry)
    ):
        geom = shapely.box(*geometry)
    else:
        raise InputTypeError("geometry", "(Multi)Polygon or tuple of length 4")

    if geo_crs is not None and crs is not None:
        geom = _geo_transform(geom, geo_crs, crs)
    if not geom.is_valid:
        geom = geom.buffer(0.0)
        geom = cast("Polygon | MultiPolygon", geom)
    return geom


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

    .. note::

        This function is adapted from
        the ``rasterio.sample.sample_gen`` function of
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


def _download(url: str, file_path: Path, context: ssl.SSLContext | None) -> None:
    """Download a file from a URL with streaming."""
    with urlopen(url, context=context) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        if file_path.exists() and file_path.stat().st_size == total_size:
            return

        downloaded = 0
        with file_path.open("wb") as out_file:
            while chunk := response.read(CHUNK_SIZE):
                out_file.write(chunk)
                downloaded += len(chunk)

        if downloaded != total_size:
            file_path.unlink(missing_ok=True)
            raise DownloadError(url, "Downloaded file size mismatch")


def streaming_download(url_list: list[str], file_extension: str, disable_ssl: bool) -> list[Path]:
    """Download multiple files concurrently."""
    context = None
    if disable_ssl:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    max_workers = min(4, os.cpu_count() or 1, len(url_list))
    file_list = [Path(f"{hashlib.sha256(url.encode()).hexdigest()}.{file_extension}") for url in url_list]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(_download, url, path, context): url
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

    geom = geo2poly(geometry, crs, ds.rio.crs)
    ds = write_crs(ds)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            ds = ds.rio.clip([geom])
    except OneDimensionalRaster:
        ds = ds.rio.clip([geom], all_touched=True)

    _ = [ds[v].rio.update_attrs(attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds
