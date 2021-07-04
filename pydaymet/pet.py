"""Core class for the Daymet functions."""
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import py3dep
import xarray as xr

from .exceptions import InvalidInputType, InvalidInputValue, MissingItems
DEF_CRS = "epsg:4326"


__all__ = ["potential_et"]


def potential_et(
    clm: Union[pd.DataFrame, xr.Dataset],
    coords: Optional[Tuple[float, float]] = None,
    crs: str = "epsg:4326",
    alt_unit: bool = False,
    method: str = "fao56",
) -> Union[pd.DataFrame, xr.Dataset]:
    """Compute Potential EvapoTranspiration for both gridded and a single location.

    Notes
    -----
    The method is based on
    `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__
    assuming that soil heat flux density is zero.

    Parameters
    ----------
    clm : pandas.DataFrame or xarray.Dataset
        The dataset must include at least the following variables:

        * Minimum temperature in degree celsius
        * Maximum temperature in degree celsius
        * Solar radiation in in W/m^2
        * Daylight duration in seconds

        Optionally, relative humidity and wind speed at 2-m level will be used if available.

        Table below shows the variable names that the function looks for in the input data.

        ================ ========
        DataFrame        Dataset
        ================ ========
        ``tmin (deg c)`` ``tmin``
        ``tmax (deg c)`` ``tmax``
        ``srad (W/m^2)`` ``srad``
        ``dayl (s)``     ``dayl``
        ``rh (-)``       ``rh``
        ``u2 (m/s)``     ``u2``
        ================ ========

        If relative humidity and wind speed at 2-m level are not available,
        actual vapour pressure is assumed to be saturation vapour pressure at daily minimum
        temperature and 2-m wind speed is considered to be 2 m/s.
    coords : tuple of floats, optional
        Coordinates of the daymet data location as a tuple, (x, y). This is required when ``clm``
        is a ``DataFrame``.
    crs : str, optional
        The spatial reference of the input coordinate, defaults to ``epsg:4326``. This is only used
        when ``clm`` is a ``DataFrame``.
    alt_unit : str, optional
        Whether to use alternative units rather than the official ones, defaults to False.
    method : str, optional
        Method for computing PET. At the moment only ``fao56`` is supported which is based on
        `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__ assuming that
        soil heat flux density is zero.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        The input DataFrame/Dataset with an additional variable named ``pet (mm/day)`` for
        DataFrame and ``pet`` for Dataset.
    """
    valid_methods = ["fao56"]
    if method not in valid_methods:
        raise InvalidInputValue("method", valid_methods)

    if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
        raise InvalidInputType("clm", "pd.DataFrame or xr.Dataset")

    pet: Union[PETCoords, PETGridded]
    if isinstance(clm, pd.DataFrame):
        if coords is None:
            raise MissingItems(["coords"])

        pet = PETCoords(clm, coords, crs, alt_unit)
    else:
        pet = PETGridded(clm)

    return getattr(pet, method)()


@dataclass
class PETCoords:
    """Compute Potential EvapoTranspiration for a single location.

    Notes
    -----
    The method is based on
    `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__
    assuming that soil heat flux density is zero.

    Parameters
    ----------
    clm : DataFrame
        The dataset must include at least the following variables:
        ``tmin (deg c)``, ``tmax (deg c)``, ``srad (W/m^2)``, and ``dayl (s)``.
        Also, if ``rh (-)`` (relative humidity) and ``u2 (m/s)`` (wind at 2 m level)
        are available, they are used. Otherwise, actual vapour pressure is assumed
        to be saturation vapour pressure at daily minimum temperature and 2-m wind
        speed is considered to be 2 m/s.
    coords : tuple of floats
        Coordinates of the daymet data location as a tuple, (x, y).
    crs : str, optional
        The spatial reference of the input coordinate, defaults to epsg:4326.
    alt_unit : str, optional
        Whether to use alternative units rather than the official ones, defaults to False.
    """

    clm: pd.DataFrame
    coords: Tuple[float, float]
    crs: str = DEF_CRS
    alt_unit: bool = False

    def fao56(self) -> pd.DataFrame:
        """Compute Potential EvapoTranspiration using FAO56 for a single location.

        Notes
        -----
        The method is based on
        `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__
        assuming that soil heat flux density is zero.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with an additional column named ``pet (mm/day)``
        """
        units = {
            "srad": ("W/m2", "W/m^2"),
            "temp": ("degrees C", "deg c"),
        }

        tmin_c = f"tmin ({units['temp'][self.alt_unit]})"
        tmax_c = f"tmax ({units['temp'][self.alt_unit]})"
        srad_wm2 = f"srad ({units['srad'][self.alt_unit]})"
        dayl_s = "dayl (s)"
        tmean_c = "tmean (deg c)"
        rh = "rh (-)"
        u2 = "u2 (m/s)"

        reqs = [tmin_c, tmax_c, srad_wm2, dayl_s]

        _check_requirements(reqs, self.clm.columns)

        self.clm[tmean_c] = 0.5 * (self.clm[tmax_c] + self.clm[tmin_c])
        delta_v = (
            4098
            * (
                0.6108
                * np.exp(
                    17.27 * self.clm[tmean_c] / (self.clm[tmean_c] + 237.3),
                )
            )
            / ((self.clm[tmean_c] + 237.3) ** 2)
        )
        elevation = py3dep.elevation_bycoords([self.coords], self.crs)[0]

        # Atmospheric pressure [kPa]
        pa = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
        # Latent Heat of Vaporization [MJ/kg]
        lmbda = 2.501 - 0.002361 * self.clm[tmean_c]
        # Psychrometric constant [kPa/°C]
        gamma = 1.013e-3 * pa / (0.622 * lmbda)

        e_max = 0.6108 * np.exp(17.27 * self.clm[tmax_c] / (self.clm[tmax_c] + 237.3))
        e_min = 0.6108 * np.exp(17.27 * self.clm[tmin_c] / (self.clm[tmin_c] + 237.3))
        e_s = (e_max + e_min) * 0.5
        e_a = self.clm[rh] * e_s * 1e-2 if rh in self.clm else e_min
        e_def = e_s - e_a

        jday = self.clm.index.dayofyear
        r_surf = self.clm[srad_wm2] * self.clm[dayl_s] * 1e-6

        alb = 0.23

        jp = 2.0 * np.pi * jday / 365.0
        d_r = 1.0 + 0.033 * np.cos(jp)
        delta_r = 0.409 * np.sin(jp - 1.39)
        phi = self.coords[1] * np.pi / 180.0
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
            * (((self.clm[tmax_c] + 273.16) ** 4 + (self.clm[tmin_c] + 273.16) ** 4) * 0.5)
            * (0.34 - 0.14 * np.sqrt(e_a))
            * ((1.35 * r_surf / rad_s) - 0.35)
        )
        rad_n = rad_ns - rad_nl

        # recommended for daily data
        rho_s = 0.0
        # recommended when no data is available
        u_2m = self.clm[u2] if u2 in self.clm else 2.0
        self.clm["pet (mm/day)"] = (
            0.408 * delta_v * (rad_n - rho_s)
            + gamma * 900.0 / (self.clm[tmean_c] + 273.0) * u_2m * e_def
        ) / (delta_v + gamma * (1 + 0.34 * u_2m))

        self.clm = self.clm.drop(columns=tmean_c)
        return self.clm


@dataclass
class PETGridded:
    """Compute Potential EvapoTranspiration using gridded climate data.

    Parameters
    ----------
    clm : xarray.DataArray
        The dataset must include at least the following variables:
        ``tmin``, ``tmax``, ``lat``, ``lon``, ``srad``, ``dayl``. Also, if
        ``rh`` (relative humidity) and ``u2`` (wind at 2 m level)
        are available, they are used. Otherwise, actual vapour pressure is assumed
        to be saturation vapour pressure at daily minimum temperature and 2-m wind
        speed is considered to be 2 m/s.
    """

    clm: xr.Dataset

    def fao56(self) -> xr.Dataset:
        """Compute Potential EvapoTranspiration using FAO56 for a gridded data.

        Notes
        -----
        The method is based on
        `FAO Penman-Monteith equation <http://www.fao.org/3/X0490E/x0490e06.htm>`__
        assuming that soil heat flux density is zero.

        Returns
        -------
        xarray.Dataset
            The input dataset with an additional variable called ``pet`` in mm/day.
        """
        keys = list(self.clm.keys())
        reqs = ["tmin", "tmax", "lat", "lon", "srad", "dayl"]

        _check_requirements(reqs, keys)

        dtype = self.clm.tmin.dtype
        dates = self.clm["time"]
        self.clm["tmean"] = 0.5 * (self.clm["tmax"] + self.clm["tmin"])

        # Slope of saturation vapour pressure [kPa/°C]
        self.clm["delta_r"] = (
            4098
            * (0.6108 * np.exp(17.27 * self.clm["tmean"] / (self.clm["tmean"] + 237.3)))
            / ((self.clm["tmean"] + 237.3) ** 2)
        )

        res = self.clm.res[0] * 1.0e3
        elev = py3dep.elevation_bygrid(self.clm.x.values, self.clm.y.values, self.clm.crs, res)
        self.clm = xr.merge([self.clm, elev], combine_attrs="override")
        self.clm["elevation"] = self.clm.elevation.where(
            ~np.isnan(self.clm.isel(time=0)[keys[0]]), drop=True
        ).T

        # Atmospheric pressure [kPa]
        pa = 101.3 * ((293.0 - 0.0065 * self.clm["elevation"]) / 293.0) ** 5.26
        # Latent Heat of Vaporization [MJ/kg]
        lmbda = 2.501 - 0.002361 * self.clm["tmean"]
        # Psychrometric constant [kPa/°C]
        self.clm["gamma"] = 1.013e-3 * pa / (0.622 * lmbda)

        # Saturation vapor pressure [kPa]
        e_max = 0.6108 * np.exp(17.27 * self.clm["tmax"] / (self.clm["tmax"] + 237.3))
        e_min = 0.6108 * np.exp(17.27 * self.clm["tmin"] / (self.clm["tmin"] + 237.3))
        e_s = (e_max + e_min) * 0.5

        self.clm["e_a"] = self.clm["rh"] * e_s * 1e-2 if "rh" in keys else e_min
        self.clm["e_def"] = e_s - self.clm["e_a"]

        lat = self.clm.isel(time=0).lat
        self.clm["time"] = pd.to_datetime(self.clm.time.values).dayofyear.astype(dtype)
        r_surf = self.clm["srad"] * self.clm["dayl"] * 1e-6

        alb = 0.23

        jp = 2.0 * np.pi * self.clm["time"] / 365.0
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
        rad_s = (0.75 + 2e-5 * self.clm["elevation"]) * r_aero
        rad_ns = (1.0 - alb) * r_surf
        rad_nl = (
            4.903e-9
            * (((self.clm["tmax"] + 273.16) ** 4 + (self.clm["tmin"] + 273.16) ** 4) * 0.5)
            * (0.34 - 0.14 * np.sqrt(self.clm["e_a"]))
            * ((1.35 * r_surf / rad_s) - 0.35)
        )
        self.clm["rad_n"] = rad_ns - rad_nl

        # recommended for daily data
        rho_s = 0.0
        # recommended when no data is available
        u_2m = self.clm["u2"] if "u2" in keys else 2.0
        self.clm["pet"] = (
            0.408 * self.clm["delta_r"] * (self.clm["rad_n"] - rho_s)
            + self.clm["gamma"] * 900.0 / (self.clm["tmean"] + 273.0) * u_2m * self.clm["e_def"]
        ) / (self.clm["delta_r"] + self.clm["gamma"] * (1 + 0.34 * u_2m))
        self.clm["pet"].attrs["units"] = "mm/day"

        self.clm["time"] = dates

        self.clm = self.clm.drop_vars(["delta_r", "gamma", "e_def", "rad_n", "tmean", "e_a"])

        return self.clm


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