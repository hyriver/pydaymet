"""Core class for the Daymet functions."""

# pyright: reportArgumentType=false
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS

import pydaymet._utils as utils
from pydaymet.exceptions import InputTypeError, InputValueError, MissingItemError

if TYPE_CHECKING:
    from collections.abc import Hashable, KeysView

    CRSType = int | str | CRS
    DataArray = TypeVar("DataArray", pd.Series, xr.DataArray)
    PETMethods = Literal["penman_monteith", "priestley_taylor", "hargreaves_samani"]

__all__ = ["potential_et"]

PET_VARS = {
    "penman_monteith": ("tmin", "tmax", "srad", "dayl"),
    "priestley_taylor": ("tmin", "tmax", "srad", "dayl"),
    "hargreaves_samani": ("tmin", "tmax"),
}
NAME_MAP = {
    "prcp": "prcp (mm/day)",
    "tmin": "tmin (degrees C)",
    "tmax": "tmax (degrees C)",
    "srad": "srad (W/m2)",
    "dayl": "dayl (s)",
    "vp": "vp (Pa)",
    "swe": "swe (kg/m2)",
}


def saturation_vapor(temperature: DataArray) -> DataArray:
    """Compute saturation vapor pressure :footcite:t:`Allen_1998` Eq. 11 [kPa].

    Parameters
    ----------
    temperature : xarray.DataArray or pandas.Series
        Temperature in °C.

    Returns
    -------
    xarray.DataArray or pandas.Series
        Saturation vapor pressure in kPa.

    References
    ----------
    .. footbibliography::
    """
    return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))  # pyright: ignore[reportReturnType]


def actual_vapor_pressure(tmin_c: DataArray, arid_correction: bool) -> DataArray:
    """Compute actual vapor pressure :footcite:t:`Allen_1998` Eq. 12 [kPa].

    Notes
    -----
    Since relative humidity is not provided by Daymet, the actual vapor pressure is
    computed assuming that the dewpoint temperature is equal to the minimum temperature.
    However, for arid regions, FAO 56 suggests to subtract minimum temperature by 2-3 °C
    to account for the fact that in arid regions, the air might not be saturated when its
    temperature is at its minimum.

    Parameters
    ----------
    tmin_c : pandas.Series or xarray.DataArray
        Minimum temperature in degrees Celsius.
    arid_correction : bool
        Whether to apply the arid correction.

    Returns
    -------
    pandas.Series or xarray.DataArray
        Actual vapor pressure in kPa.

    References
    ----------
    .. footbibliography::
    """
    if arid_correction:
        return saturation_vapor(tmin_c - 2.0)
    return saturation_vapor(tmin_c)


def vapor_pressure(tmax_c: DataArray, tmin_c: DataArray) -> DataArray:
    """Compute saturation and actual vapor pressure :footcite:t:`Allen_1998` Eq. 12 [kPa].

    Parameters
    ----------
    tmax_c : pandas.Series or xarray.DataArray
        Maximum temperature in degrees Celsius.
    tmin_c : pandas.Series or xarray.DataArray
        Minimum temperature in degrees Celsius.

    Returns
    -------
    pandas.Series or xarray.DataArray
        Saturation vapor pressure in kPa.

    References
    ----------
    .. footbibliography::
    """
    e_s = (saturation_vapor(tmax_c) + saturation_vapor(tmin_c)) * 0.5
    return cast("DataArray", e_s)


@overload
def extraterrestrial_radiation(dayofyear: xr.DataArray, lat: xr.DataArray) -> xr.DataArray: ...


@overload
def extraterrestrial_radiation(dayofyear: pd.Series, lat: float) -> pd.Series: ...


def extraterrestrial_radiation(
    dayofyear: pd.Series | xr.DataArray, lat: float | xr.DataArray
) -> pd.Series | xr.DataArray:
    """Compute Extraterrestrial Radiation using :footcite:t:`Allen_1998` Eq. 28 [MJ m^-2 h^-1].

    Parameters
    ----------
    dayofyear : pandas.Series or xarray.DataArray
        Time as day of year.
    lat : float or xarray.DataArray
        Latitude.

    Returns
    -------
    pandas.Series or xarray.DataArray
        Extraterrestrial Radiation in MJ m^-2 h^-1.

    References
    ----------
    .. footbibliography::
    """
    jp = 2.0 * np.pi * dayofyear / 365.0
    d_r = 1.0 + 0.033 * np.cos(jp)
    delta_r = 0.409 * np.sin(jp - 1.39)
    phi = lat * np.pi / 180.0
    w_s = np.arccos(-np.tan(phi) * np.tan(delta_r))
    return (  # pyright: ignore[reportReturnType]
        24.0
        * 60.0
        / np.pi
        * 0.082
        * d_r
        * (w_s * np.sin(phi) * np.sin(delta_r) + np.cos(phi) * np.cos(delta_r) * np.sin(w_s))
    )


def net_radiation(
    srad: DataArray,
    dayl: DataArray,
    elevation: float | xr.DataArray,
    tmax: DataArray,
    tmin: DataArray,
    e_a: DataArray,
    rad_a: pd.Series | xr.DataArray,
    albedo: float,
) -> DataArray:
    """Compute net radiation using :footcite:t:`Allen_1998` Eq. 40 [MJ m^-2 day^-1].

    Parameters
    ----------
    srad : pandas.Series or xarray.DataArray
        Solar radiation [MJ m^-2 day^-1].
    dayl : pandas.Series or xarray.DataArray
        Daylength [h].
    elevation : float or xarray.DataArray
        Elevation [m].
    tmax : pandas.Series or xarray.DataArray
        Maximum temperature [°C].
    tmin : pandas.Series or xarray.DataArray
        Minimum temperature [°C].
    e_a : pandas.Series or xarray.DataArray
        Actual vapor pressure [kPa].
    rad_a : pandas.Series or xarray.Dataset
        Extraterrestrial radiation [MJ m^-2 day^-1].
    albedo : float
        Albedo.

    Returns
    -------
    pandas.Series or xarray.DataArray
        Net radiation in MJ m^-2 day^-1.

    References
    ----------
    .. footbibliography::
    """
    r_surf = srad * dayl * 1e-6
    rad_s = (0.75 + 2e-5 * elevation) * rad_a
    rad_s = cast("DataArray", rad_s)
    rad_ns = (1.0 - albedo) * r_surf
    rad_nl = (
        4.903e-9
        * ((np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4)) * 0.5)
        * (0.34 - 0.14 * np.sqrt(e_a))
        * (1.35 * r_surf / rad_s - 0.35)
    )
    rad_net = rad_ns - rad_nl
    rad_net = cast("DataArray", rad_net)
    return rad_net


def psychrometric_constant(elevation: float | xr.DataArray, lmbda: DataArray) -> DataArray:
    """Compute the psychrometric constant :footcite:t:`Allen_1998` Eq. 8 [kPa °C^-1]..

    Parameters
    ----------
    elevation : float or xarray.DataArray
        Elevation of the location in meters.
    lmbda : pandas.Series or xarray.DataArray
        Latent heat of vaporization in J/kg, defaults to 0.0065.

    Returns
    -------
    pandas.Series or xarray.DataArray
        The psychrometric constant in kPa °C^-1.

    References
    ----------
    .. footbibliography::
    """
    # Atmospheric pressure [kPa]
    pa = 101.3 * np.power((293.15 - 0.0065 * elevation) / 293.15, 9.80665 / (0.0065 * 286.9))
    pa = cast("float | xr.DataArray", pa)
    gamma = 1.013e-3 * pa / (0.622 * lmbda)
    gamma = cast("DataArray", gamma)
    return gamma


def vapor_slope(tmean_c: DataArray) -> DataArray:
    """Compute slope of saturation vapor pressure :footcite:t:`Allen_1998` Eq. 1 [kPa/°C].

    Parameters
    ----------
    tmean_c : pandas.Series or xarray.DataArray
        The mean temperature [°C].

    Returns
    -------
    pandas.Series or xarray.DataArray
        The slope of the saturation vapor pressure curve in kPa.

    References
    ----------
    .. footbibliography::
    """
    return 4098 * saturation_vapor(tmean_c) / np.square(tmean_c + 237.3)  # pyright: ignore[reportReturnType]


def check_requirements(reqs: Iterable[str], cols: KeysView[Hashable] | pd.Index) -> None:
    """Check for all the required data.

    Parameters
    ----------
    reqs : iterable
        A list of required data names (str)
    cols : list of str
        A list of variable names (str)
    """
    if not isinstance(reqs, Iterable):
        raise InputTypeError("reqs", "iterable")

    missing = [r for r in reqs if r not in cols]
    if missing:
        raise MissingItemError(missing)


@lru_cache
def _get_location_elevation(coords: tuple[float, float], crs: int) -> float:
    """Get the elevation of the location."""
    url = "/".join(
        (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation",
            "13/TIFF/USGS_Seamless_DEM_13.vrt",
        )
    )
    with rasterio.open(url) as src:
        xy = utils.transform_coords([coords], crs, src.crs.to_epsg())
        elev = np.array(list(utils.sample_window(src, xy, 33))).ravel()
    return elev[0]


@dataclass(frozen=True)
class PetParams:
    soil_heat_flux: float = 0.0
    albedo: float = 0.23
    alpha: float = 1.26
    arid_correction: bool = False

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        """Return the field names of the dataclass."""
        return tuple(field.name for field in fields(cls))


class PETCoords:
    """Compute Potential EvapoTranspiration for a single location.

    Parameters
    ----------
    clm : DataFrame
        For ``penman_monteith`` method, the dataset must include at least
        the following variables: ``tmin (degrees C)``, ``tmax (degrees C)``,
        ``srad (W/m2)``, and ``dayl (s)``. Also, if ``u2m (m/s)``
        (wind at 2 m level) is available, it will be used. Otherwise, 2-m wind
        speed is considered to be 2 m/s. For the ``hargreaves_samani``
        method, the dataset must include ``tmin (degrees C)`` and ``tmax (degrees C)``.
    coords : tuple of floats
        Coordinates of the daymet data location as a tuple, (x, y).
    method : str
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, and ``hargreaves_samani``.
        The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
    crs : str, int, or pyproj.CRS, optional
        The spatial reference of the input coordinate, defaults to epsg:4326.
    params : dict, optional
        Model-specific parameters as a dictionary, defaults to ``None``.
    """

    def __init__(
        self,
        clm: pd.DataFrame,
        coords: tuple[float, float],
        method: PETMethods,
        crs: CRSType = 4326,
        params: dict[str, float] | None = None,
    ) -> None:
        self.clm = clm
        self.crs = utils.validate_crs(crs)
        self.coords = utils.transform_coords([coords], self.crs, 4326)[0]
        self.method = method
        valid_methods = ("penman_monteith", "hargreaves_samani", "priestley_taylor")
        if self.method not in valid_methods:
            raise InputValueError("method", valid_methods)

        if params is None:
            self.params = PetParams()
        else:
            if any(k not in PetParams.fields() for k in params):
                raise InputValueError("params", PetParams.fields())
            self.params = PetParams(**params)

        self.tmin = "tmin (degrees C)"
        self.tmax = "tmax (degrees C)"
        self.srad = "srad (W/m2)"
        self.dayl = "dayl (s)"
        self.u2m = "u2m (m/s)"

        self.tmean = 0.5 * (self.clm[self.tmax] + self.clm[self.tmin])
        dayofyear = pd.to_datetime(self.clm.index).dayofyear.to_numpy("uint16")
        self.dayofyear = pd.Series(dayofyear, index=self.clm.index)
        self.clm_vars = self.clm.columns
        self.req_vars = {k: tuple(NAME_MAP[v] for v in var) for k, var in PET_VARS.items()}

    def compute(self) -> pd.DataFrame:
        """Compute Potential EvapoTranspiration."""
        if self.method == "penman_monteith":
            return self.penman_monteith()
        if self.method == "hargreaves_samani":
            return self.hargreaves_samani()
        return self.priestley_taylor()

    def penman_monteith(self) -> pd.DataFrame:
        """Compute Potential EvapoTranspiration using :footcite:t:`Allen_1998` Eq. 6.

        Notes
        -----
        The method is based on :footcite:t:`Allen_1998`
        assuming that soil heat flux density is zero.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with an additional column named ``pet (mm/day)``

        References
        ----------
        .. footbibliography::
        """
        check_requirements(self.req_vars["penman_monteith"], self.clm_vars)

        vp_slope = vapor_slope(self.tmean)
        elevation = _get_location_elevation(self.coords, self.crs)

        # Latent Heat of Vaporization [MJ/kg]
        lmbda = 2.501 - 0.002361 * self.tmean
        gamma = psychrometric_constant(elevation, lmbda)

        # Saturation Vapor Pressure [kPa]
        e_s = vapor_pressure(self.clm[self.tmax], self.clm[self.tmin])
        e_a = actual_vapor_pressure(self.clm[self.tmin], self.params.arid_correction)

        rad_a = extraterrestrial_radiation(self.dayofyear, self.coords[1])
        rad_n = net_radiation(
            self.clm[self.srad],
            self.clm[self.dayl],
            elevation,
            self.clm[self.tmax],
            self.clm[self.tmin],
            e_a,
            rad_a,
            self.params.albedo,
        )

        # recommended when no data is not available to estimate wind speed
        u_2m = self.clm.get(self.u2m, 2.0)
        self.clm["pet (mm/day)"] = (
            0.408 * vp_slope * (rad_n - self.params.soil_heat_flux)
            + gamma * 900.0 / (self.tmean + 273.0) * u_2m * (e_s - e_a)
        ) / (vp_slope + gamma * (1 + 0.34 * u_2m))  # pyright: ignore[reportOperatorIssue]

        return self.clm

    def priestley_taylor(self) -> pd.DataFrame:
        """Compute Potential EvapoTranspiration using :footcite:t:`Priestley_1972`.

        Notes
        -----
        The method is based on :footcite:t:`Priestley_1972`
        assuming that soil heat flux density is zero.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with an additional column named ``pet (mm/day)``.

        References
        ----------
        .. footbibliography::
        """
        check_requirements(self.req_vars["priestley_taylor"], self.clm_vars)

        self.tmean = 0.5 * (self.clm[self.tmax] + self.clm[self.tmin])
        vp_slope = vapor_slope(self.tmean)
        elevation = _get_location_elevation(self.coords, self.crs)

        # Latent Heat of Vaporization [MJ/kg]
        lmbda = 2.501 - 0.002361 * self.tmean
        gamma = psychrometric_constant(elevation, lmbda)

        e_a = actual_vapor_pressure(self.clm[self.tmin], self.params.arid_correction)
        rad_a = extraterrestrial_radiation(self.dayofyear, self.coords[1])
        rad_n = net_radiation(
            self.clm[self.srad],
            self.clm[self.dayl],
            elevation,
            self.clm[self.tmax],
            self.clm[self.tmin],
            e_a,
            rad_a,
            self.params.albedo,
        )

        self.clm["pet (mm/day)"] = (
            self.params.alpha
            * vp_slope
            * (rad_n - self.params.soil_heat_flux)
            / ((vp_slope + gamma) * lmbda)
        )

        return self.clm

    def hargreaves_samani(self) -> pd.DataFrame:
        """Compute Potential EvapoTranspiration using :footcite:t:`Hargreaves_1982`.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with an additional column named ``pet (mm/day)``.

        References
        ----------
        .. footbibliography::
        """
        check_requirements(self.req_vars["hargreaves_samani"], self.clm_vars)

        self.tmean = 0.5 * (self.clm[self.tmax] + self.clm[self.tmin])
        rad_a = extraterrestrial_radiation(self.dayofyear, self.coords[1]) / 2.43
        self.clm["pet (mm/day)"] = (
            0.0023
            * (self.tmean + 17.8)
            * np.sqrt(self.clm[self.tmax] - self.clm[self.tmin])
            * rad_a
        )

        return self.clm


class PETGridded:
    """Compute Potential EvapoTranspiration using gridded climate data.

    Parameters
    ----------
    clm : xarray.DataArray
        For ``penman_monteith`` method, the dataset must include at least
        the following variables: ``tmin``, ``tmax``, ``lat``, ``lon``,
        ``srad``, ``dayl``. Also, if ``u2m`` (wind at 2 m level) is available,
        it will be used. Otherwise, 2-m wind speed is considered to be 2 m/s.
        For the ``hargreaves_samani`` method, the dataset must include ``tmin``,
        ``tmax``, and ``lat``.
    method : str
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, and ``hargreaves_samani``.
        The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
    params : dict, optional
        Model-specific parameters as a dictionary, defaults to ``None``.
    """

    def _add_elevation(self):
        """Add elevation to the dataset."""
        import rasterio

        url = "/".join(
            (
                "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation",
                "1/TIFF/USGS_Seamless_DEM_1.vrt",
            )
        )
        lon = self.clm["lon"].to_numpy().ravel()
        lat = self.clm["lat"].to_numpy().ravel()
        mask = ~np.isnan(lon) & ~np.isnan(lat)
        grid = zip(lon[mask], lat[mask])
        elev = np.full_like(lon, np.nan)
        with rasterio.open(url) as src:
            elev[mask] = np.array(list(utils.sample_window(src, grid, 33))).ravel()
        self.clm["elevation"] = self.clm["lon"].copy(data=elev.reshape(self.clm["lon"].shape))
        dtype = self.clm["lon"].dtype
        self.clm["elevation"] = self.clm["elevation"].astype(dtype)
        self.clm["elevation"].attrs.update({"units": "m", "long_name": "elevation"})
        self.clm["elevation"] = self.clm["elevation"].rio.write_crs(
            self.clm.rio.crs, grid_mapping_name=self.clm.rio.grid_mapping
        )

    def __init__(
        self,
        clm: xr.Dataset,
        method: PETMethods,
        params: dict[str, float] | None = None,
    ) -> None:
        self.clm = clm
        self.method = method
        valid_methods = ("penman_monteith", "hargreaves_samani", "priestley_taylor")
        if self.method not in valid_methods:
            raise InputValueError("method", valid_methods)

        if params is None:
            self.params = PetParams()
        else:
            try:
                self.params = PetParams(**params)
            except TypeError as e:
                raise InputValueError("params", PetParams.fields()) from e

        self.res_m = abs(self.clm.rio.resolution()[0])
        self.crs = CRS(self.clm.rio.crs)
        if self.crs.axis_info[0].unit_name == "kilometre":
            self.res_m *= 1000.0

        if "elevation" not in self.clm:
            chunksizes = None
            if all(d in self.clm.chunksizes for d in ("time", "y", "x")):
                chunksizes = self.clm.chunksizes
            self._add_elevation()
            if chunksizes is not None:
                self.clm = self.clm.chunk(chunksizes)

        self.clm["tmean"] = 0.5 * (self.clm["tmax"] + self.clm["tmin"])
        self.dayofyear = self.clm["time"].dt.dayofyear
        self.lat = self.clm["lat"]
        self.clm_vars = self.clm.keys()
        self.req_vars = PET_VARS

    def compute(self) -> xr.Dataset:
        """Compute Potential EvapoTranspiration."""
        if self.method == "penman_monteith":
            return self.penman_monteith()
        if self.method == "hargreaves_samani":
            return self.hargreaves_samani()
        return self.priestley_taylor()

    @staticmethod
    def set_pet_attrs(clm: xr.Dataset) -> xr.Dataset:
        """Set new attributes to the input dataset.

        Parameters
        ----------
        clm : xarray.DataArray
            The dataset to which the new attributes are added.
        """
        clm["pet"].attrs.update(
            {"units": "mm/day", "long_name": "daily potential evapotranspiration"}
        )
        clm["pet"] = clm["pet"].astype(clm["tmin"].dtype)
        clm["pet"] = clm["pet"].rio.write_crs(clm.rio.crs, grid_mapping_name=clm.rio.grid_mapping)
        return clm

    def penman_monteith(self) -> xr.Dataset:
        """Compute Potential EvapoTranspiration using :footcite:t:`Allen_1998` Eq. 6.

        Notes
        -----
        The method is based on :footcite:t:`Allen_1998`
        assuming that soil heat flux density is zero.

        Returns
        -------
        xarray.Dataset
            The input dataset with an additional variable called ``pet`` in mm/day.

        References
        ----------
        .. footbibliography::
        """
        check_requirements(self.req_vars["penman_monteith"], self.clm_vars)

        # Slope of saturation vapor pressure [kPa/°C]
        self.clm["vp_slope"] = vapor_slope(self.clm["tmean"])

        # Latent Heat of Vaporization [MJ/kg]
        self.clm["lambda"] = 2.501 - 0.002361 * self.clm["tmean"]
        self.clm["gamma"] = psychrometric_constant(self.clm["elevation"], self.clm["lambda"])

        # Saturation vapor pressure [kPa]
        self.clm["e_s"] = vapor_pressure(self.clm["tmax"], self.clm["tmin"])
        self.clm["e_a"] = actual_vapor_pressure(self.clm["tmin"], self.params.arid_correction)

        rad_a = extraterrestrial_radiation(self.dayofyear, self.lat)
        self.clm["rad_n"] = net_radiation(
            self.clm["srad"],
            self.clm["dayl"],
            self.clm["elevation"],
            self.clm["tmax"],
            self.clm["tmin"],
            self.clm["e_a"],
            rad_a,
            self.params.albedo,
        )

        # recommended when no data is not available to estimate wind speed
        u_2m = self.clm.get("u2m", 2.0)
        self.clm["pet"] = (
            0.408 * self.clm["vp_slope"] * (self.clm["rad_n"] - self.params.soil_heat_flux)
            + self.clm["gamma"]
            * 900.0
            / (self.clm["tmean"] + 273.0)
            * u_2m
            * (self.clm["e_s"] - self.clm["e_a"])
        ) / (self.clm["vp_slope"] + self.clm["gamma"] * (1.0 + 0.34 * u_2m))

        self.clm = self.clm.drop_vars(
            ["vp_slope", "gamma", "rad_n", "tmean", "lambda", "e_s", "e_a"]
        )

        return self.set_pet_attrs(self.clm)

    def priestley_taylor(self) -> xr.Dataset:
        """Compute Potential EvapoTranspiration using :footcite:t:`Priestley_1972`.

        Notes
        -----
        The method is based on :footcite:t:`Priestley_1972`
        assuming that soil heat flux density is zero.

        Returns
        -------
        xarray.Dataset
            The input dataset with an additional variable called ``pet`` in mm/day.

        References
        ----------
        .. footbibliography::
        """
        check_requirements(self.req_vars["priestley_taylor"], self.clm_vars)

        # Slope of saturation vapor pressure [kPa/°C]
        self.clm["vp_slope"] = vapor_slope(self.clm["tmean"])

        # Latent Heat of Vaporization [MJ/kg]
        self.clm["lambda"] = 2.501 - 0.002361 * self.clm["tmean"]
        self.clm["gamma"] = psychrometric_constant(self.clm["elevation"], self.clm["lambda"])

        self.clm["e_a"] = actual_vapor_pressure(self.clm["tmin"], self.params.arid_correction)
        rad_a = extraterrestrial_radiation(self.dayofyear, self.lat)
        self.clm["rad_n"] = net_radiation(
            self.clm["srad"],
            self.clm["dayl"],
            self.clm["elevation"],
            self.clm["tmax"],
            self.clm["tmin"],
            self.clm["e_a"],
            rad_a,
            self.params.albedo,
        )

        self.clm["pet"] = (
            self.params.alpha
            * self.clm["vp_slope"]
            * (self.clm["rad_n"] - self.params.soil_heat_flux)
            / ((self.clm["vp_slope"] + self.clm["gamma"]) * self.clm["lambda"])
        )

        self.clm = self.clm.drop_vars(["vp_slope", "gamma", "lambda", "rad_n", "tmean", "e_a"])

        return self.set_pet_attrs(self.clm)

    def hargreaves_samani(self) -> xr.Dataset:
        """Compute Potential EvapoTranspiration using :footcite:t:`Hargreaves_1982`.

        Returns
        -------
        xarray.Dataset
            The input dataset with an additional variable called ``pet`` in mm/day.

        References
        ----------
        .. footbibliography::
        """
        check_requirements(self.req_vars["hargreaves_samani"], self.clm_vars)

        rad_a = extraterrestrial_radiation(self.dayofyear, self.lat) / 2.43
        self.clm["pet"] = (
            0.0023
            * (self.clm["tmean"] + 17.8)
            * np.sqrt(self.clm["tmax"] - self.clm["tmin"])
            * rad_a
        )

        self.clm = self.clm.drop_vars("tmean")

        return self.set_pet_attrs(self.clm)


@overload
def potential_et(
    clm: pd.DataFrame,
    coords: tuple[float, float],
    crs: CRSType,
    method: Literal["penman_monteith", "priestley_taylor", "hargreaves_samani"] = ...,
    params: dict[str, float] | None = ...,
) -> pd.DataFrame: ...


@overload
def potential_et(
    clm: xr.Dataset,
    coords: None = None,
    crs: None = None,
    method: Literal["penman_monteith", "priestley_taylor", "hargreaves_samani"] = ...,
    params: dict[str, float] | None = ...,
) -> xr.Dataset: ...


def potential_et(
    clm: pd.DataFrame | xr.Dataset,
    coords: tuple[float, float] | None = None,
    crs: CRSType | None = 4326,
    method: Literal[
        "penman_monteith", "priestley_taylor", "hargreaves_samani"
    ] = "hargreaves_samani",
    params: dict[str, float] | None = None,
) -> pd.DataFrame | xr.Dataset:
    """Compute Potential EvapoTranspiration for both gridded and a single location.

    Parameters
    ----------
    clm : pandas.DataFrame or xarray.Dataset
        The dataset must include at least the following variables:

        * Minimum temperature in degree celsius
        * Maximum temperature in degree celsius
        * Solar radiation in in W/m2
        * Daylight duration in seconds

        Optionally, for ``penman_monteith``, wind speed at 2-m level
        will be used if available, otherwise, default value of 2 m/s
        will be assumed. Table below shows the variable names
        that the function looks for in the input data.

        ==================== ==================
        ``pandas.DataFrame`` ``xarray.Dataset``
        ==================== ==================
        ``tmin (degrees C)`` ``tmin``
        ``tmax (degrees C)`` ``tmax``
        ``srad (W/m2)``      ``srad``
        ``dayl (s)``         ``dayl``
        ``u2m (m/s)``        ``u2m``
        ==================== ==================

    coords : tuple of floats, optional
        Coordinates of the daymet data location as a tuple, (x, y). This is required when ``clm``
        is a ``DataFrame``.
    crs : str, int, or pyproj.CRS, optional
        The spatial reference of the input coordinate, defaults to ``EPSG:4326``. This is only used
        when ``clm`` is a ``DataFrame``.
    method : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, and ``hargreaves_samani``.
        The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``hargreaves_samani``.
    params : dict, optional
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
        minimum temperature by 2-3 °C to account for the fact that in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2 °C from the
        minimum temperature for computing the actual vapor pressure.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        The input DataFrame/Dataset with an additional variable named ``pet (mm/day)`` for
        ``pandas.DataFrame`` and ``pet`` for ``xarray.Dataset``.

    References
    ----------
    .. footbibliography::
    """
    if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
        raise InputTypeError("clm", "pd.DataFrame or xr.Dataset")

    if isinstance(clm, pd.DataFrame):
        if coords is None or crs is None:
            raise MissingItemError(["coords", "crs"])
        return PETCoords(clm, coords, method, crs, params).compute()

    with xr.set_options(keep_attrs=True):
        return PETGridded(clm, method, params).compute()
