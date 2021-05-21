"""Command-line interface for PyDaymet."""
import os
from pathlib import Path
from typing import List, Optional, Union

import click
import geopandas as gpd
import pandas as pd

from . import pydaymet as daymet
from .exceptions import MissingItems


def get_target_df(
    tdf: Union[pd.DataFrame, gpd.GeoDataFrame], req_cols: List[str]
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Check if all required columns exists in the dataframe.

    It also re-orders the columns based on req_cols order.
    """
    missing = [c for c in req_cols if c not in tdf]
    if len(missing) > 0:
        raise MissingItems(missing)
    return tdf[req_cols]


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("target", type=click.Path(exists=True))
@click.argument("target_type", type=click.Choice(["geometry", "coords"], case_sensitive=False))
@click.argument("crs", type=str)
@click.option(
    "--variables",
    "-v",
    multiple=True,
    default=["prcp"],
    help="Target variables. You can pass this flag multiple times for multiple variables.",
)
@click.option(
    "-t",
    "--time_scale",
    type=click.Choice(["daily", "monthly", "annual"], case_sensitive=False),
    default="daily",
    help="Target time scale.",
)
@click.option(
    "-p",
    "--pet",
    is_flag=True,
    help="Compute PET.",
)
@click.option(
    "-s",
    "--save_dir",
    type=click.Path(exists=False),
    default="clm_daymet",
    help="Path to a directory to save the requested files. "
    + "Extension for the outputs is .nc for geometry and .csv for coords.",
)
def main(
    target: Path,
    target_type: str,
    crs: str,
    variables: Optional[Union[List[str], str]] = None,
    time_scale: str = "daily",
    pet: bool = False,
    save_dir: Union[str, Path] = "clm_daymet",
):
    r"""Retrieve cliamte data within geometries or elevations for a list of coordinates.

    TARGET: Path to a geospatial file (any file that geopandas.read_file can open) or a csv file.

    The input files should have three columns:

        - id: Feature identifiers that daymet uses as the output netcdf filenames.

        - start: Starting time.

        - end: Ending time.

        - region: Target region (na for CONUS, hi for Hawaii, and pr for Puerto Rico.

    If target_type is geometry, an additional geometry column is required.
    If it is coords, two additional columns are needed: x and y.

    TARGET_TYPE: Type of input file: "coords" for csv and "geometry" for geospatial.

    CRS: CRS of the input data.

    Examples:

        $ pydaymet ny_coords.csv coords epsg:4326 -v prcp -v tmin -p -t monthly

        $ pydaymet ny_geom.gpkg geometry epsg:3857 -v prcp
    """  # noqa: D412
    save_dir = Path(save_dir)
    target = Path(target)
    if not save_dir.exists():
        os.makedirs(save_dir, exist_ok=True)

    get_func = {"coords": daymet.get_bycoords, "geometry": daymet.get_bygeom}
    cols = ["dates", "region"]
    extra_args = {"crs": crs, "variables": variables, "time_scale": time_scale, "pet": pet}

    if target_type == "geometry":
        target_df = gpd.read_file(target, crs=crs)
        target_df["dates"] = list(target_df[["start", "end"]].itertuples(index=False, name=None))
        req_args = ["id", "geometry"] + cols
        target_df = get_target_df(target_df, req_args)
        save_func = "to_netcdf"
        save_ext = "nc"
    else:
        target_df = pd.read_csv(target)
        if not ("x" in target_df and "y" in target_df):
            raise MissingItems(["x", "y"])
        target_df["coords"] = list(target_df[["x", "y"]].itertuples(index=False, name=None))
        target_df["dates"] = list(target_df[["start", "end"]].itertuples(index=False, name=None))
        req_args = ["id", "coords"] + cols
        target_df = get_target_df(target_df, req_args)
        save_func = "to_csv"
        save_ext = "csv"

    click.echo(f"Found {len(target_df)} items in {target}. Retrieving ...")
    with click.progressbar(
        target_df.itertuples(index=False, name=None),
        label="Getting climate data",
        length=len(target_df),
    ) as bar:
        for i, *args in bar:
            fname = Path(save_dir, f"{i}.{save_ext}")
            if fname.exists():
                continue
            clm = get_func[target_type](**dict(zip(req_args[1:], args)), **extra_args)  # type: ignore
            getattr(clm, save_func)(fname)

    click.echo(f"Retrieved climate data for {len(target_df)} item(s).")
