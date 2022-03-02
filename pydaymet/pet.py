"""Core class for the Daymet functions."""
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import py3dep
import pygeoogc as ogc
import pyproj
import xarray as xr

from .exceptions import InvalidInputType, InvalidInputValue, MissingItems

DEF_CRS = "epsg:4326"
DF = TypeVar("DF", pd.DataFrame, xr.Dataset)
DS = TypeVar("DS", pd.Series, xr.DataArray)

__all__ = ["potential_et"]


def saturation_vapour(temperature: DS) -> DS:
    """Compute saturation vapour pressure :footcite:t:`Allen_1998` Eq. 11 [kPa].

    Parameters
    ----------
    temperature : xarray.DataArray or pandas.Series
        Temperature in °C.

    Returns
    -------
    xarray.DataArray or pandas.Series
        Saturation vapour pressure in kPa.

    References
    ----------
    .. footbibliography::
    """  # noqa: DAR203
    return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))  # type: ignore


def vapour_pressure(
    tmax_c: DS,
    tmin_c: DS,
    rh: Optional[DS] = None,
) -> Union[Tuple[DS, DS], Tuple[DS, DS]]:
    """Compute saturation and actual vapour pressure :footcite:t:`Allen_1998` Eq. 12 [kPa].

    Parameters
    ----------
    tmax_c : pandas.Series or xarray.DataArray
        Maximum temperature in degrees Celsius.
    tmin_c : pandas.Series or xarray.DataArray
        Minimum temperature in degrees Celsius.
    rh : pandas.Series or xarray.DataArray, optional
        Relative humidity in %.

    Returns
    -------
    tuple of pandas.Series or tuple of xarray.DataArray
        Saturation vapour pressure in kPa and actual vapour pressure in kPa.

    References
    ----------
    .. footbibliography::
    """  # noqa: DAR203
    e_max = saturation_vapour(tmax_c)
    e_min = saturation_vapour(tmin_c)
    e_s = (e_max + e_min) * 0.5
    if rh is not None:
        e_a = rh * e_s * 1e-2
    else:
        e_a = e_min
    return e_s, e_a


def extraterrestrial_radiation(
    dayofyear: Union[pd.Index, xr.DataArray], lat: Union[float, xr.DataArray]
) -> Union[pd.Index, xr.DataArray]:
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
    """  # noqa: DAR203
    jp = 2.0 * np.pi * dayofyear / 365.0
    d_r = 1.0 + 0.033 * np.cos(jp)
    delta_r = 0.409 * np.sin(jp - 1.39)
    phi = lat * np.pi / 180.0
    w_s = np.arccos(-np.tan(phi) * np.tan(delta_r))
    return (
        24.0
        * 60.0
        / np.pi
        * 0.082
        * d_r
        * (w_s * np.sin(phi) * np.sin(delta_r) + np.cos(phi) * np.cos(delta_r) * np.sin(w_s))
    )


def net_radiation(
    srad: DS,
    dayl: DS,
    elevation: Union[float, xr.DataArray],
    tmax: DS,
    tmin: DS,
    e_a: DS,
    rad_a: Union[pd.Index, xr.DataArray],
) -> DS:
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
        Actual vapour pressure [kPa].
    rad_a : pandas.Series or xarray.Dataset
        Extraterrestrial radiation [MJ m^-2 day^-1].

    Returns
    -------
    pandas.Series or xarray.DataArray
        Net radiation in MJ m^-2 day^-1.

    References
    ----------
    .. footbibliography::
    """  # noqa: DAR203
    r_surf = srad * dayl * 1e-6

    alb = 0.23
    rad_s = (0.75 + 2e-5 * elevation) * rad_a
    rad_ns = (1.0 - alb) * r_surf
    rad_nl = (
        4.903e-9
        * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * 0.5)
        * (0.34 - 0.14 * np.sqrt(e_a))
        * ((1.35 * r_surf / rad_s) - 0.35)
    )
    return rad_ns - rad_nl  # type: ignore


def psychrometric_constant(elevation: Union[float, xr.DataArray], lmbda: DS) -> DS:
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
    """  # noqa: DAR203
    # Atmospheric pressure [kPa]
    pa = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
    return 1.013e-3 * pa / (0.622 * lmbda)


def vapour_slope(tmean_c: DS) -> DS:
    """Compute the slope of the saturation vapour pressure curve :footcite:t:`Allen_1998` Eq. 1 [kPa].

    Parameters
    ----------
    tmean_c : pandas.Series or xarray.DataArray
        The mean temperature [°C].

    Returns
    -------
    pandas.Series or xarray.DataArray
        The slope of the saturation vapour pressure curve in kPa.

    References
    ----------
    .. footbibliography::
    """  # noqa: DAR203
    return (  # type: ignore
        4098
        * (
            0.6108
            * np.exp(
                17.27 * tmean_c / (tmean_c + 237.3),
            )
        )
        / ((tmean_c + 237.3) ** 2)
    )


def check_requirements(reqs: Iterable[str], cols: List[str]) -> None:
    """Check for all the required data.

    Parameters
    ----------
    reqs : iterable
        A list of required data names (str)
    cols : list of str
        A list of variable names (str)
    """
    if not isinstance(reqs, Iterable):
        raise InvalidInputType("reqs", "iterable")

    missing = [r for r in reqs if r not in cols]
    if missing:
        raise MissingItems(missing)


class PETCoords:
    """Compute Potential EvapoTranspiration for a single location.

    Parameters
    ----------
    clm : DataFrame
        For ``penman_monteith`` method, the dataset must include at least the following variables:
        ``tmin (degrees C)``, ``tmax (degrees C)``, ``srad (W/m2)``, and ``dayl (s)``.
        Also, if ``rh (-)`` (relative humidity) and ``u2 (m/s)`` (wind at 2 m level)
        are available, they are used. Otherwise, actual vapour pressure is assumed
        to be saturation vapour pressure at daily minimum temperature and 2-m wind
        speed is considered to be 2 m/s. For the ``hargreaves_samani`` method, the dataset
        must include ``tmin (degrees C)`` and ``tmax (degrees C)``.
    coords : tuple of floats
        Coordinates of the daymet data location as a tuple, (x, y).
    crs : str, optional
        The spatial reference of the input coordinate, defaults to epsg:4326.
    params : dict, optional
        Model-specific parameters as a dictionary, defaults to ``None``.
    """

    def __init__(
        self,
        clm: pd.DataFrame,
        coords: Tuple[float, float],
        crs: Union[str, pyproj.CRS] = DEF_CRS,
        params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.clm = clm
        self.coords = ogc.utils.match_crs([coords], crs, DEF_CRS)[0]
        self.params = params if isinstance(params, dict) else {"soil_heat": 0.0}

        # recommended when no data is not available to estimate soil heat flux
        if "soil_heat" not in self.params:
            self.params["soil_heat"] = 0.0

        self.tmin = "tmin (degrees C)"
        self.tmax = "tmax (degrees C)"
        self.srad = "srad (W/m2)"
        self.dayl = "dayl (s)"
        self.rh = "rh (-)"
        self.u2 = "u2 (m/s)"

        self.tmean = 0.5 * (self.clm[self.tmax] + self.clm[self.tmin])
        self.clm_vars = list(self.clm.columns)
        self.req_vars = {
            "penman_monteith": [self.tmin, self.tmax, self.srad, self.dayl],
            "priestley_taylor": [self.tmin, self.tmax, self.srad, self.dayl],
            "hargreaves_samani": [self.tmin, self.tmax],
        }

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

        vp_slope = vapour_slope(self.tmean)
        elevation = py3dep.elevation_bycoords([self.coords], source="tnm")[0]

        # Latent Heat of Vaporization [MJ/kg]
        lmbda = 2.501 - 0.002361 * self.tmean
        gamma = psychrometric_constant(elevation, lmbda)

        # Saturation Vapor Pressure [kPa]
        rh = self.clm[self.rh] if self.rh in self.clm else None
        e_s, e_a = vapour_pressure(self.clm[self.tmax], self.clm[self.tmin], rh)

        rad_a = extraterrestrial_radiation(self.clm.index.dayofyear, self.coords[1])
        rad_n = net_radiation(
            self.clm[self.srad],
            self.clm[self.dayl],
            elevation,
            self.clm[self.tmax],
            self.clm[self.tmin],
            e_a,
            rad_a,
        )

        # recommended when no data is not available to estimate wind speed
        u_2m = self.clm[self.u2] if self.u2 in self.clm else 2.0
        self.clm["pet (mm/day)"] = (
            0.408 * vp_slope * (rad_n - self.params["soil_heat"])
            + gamma * 900.0 / (self.tmean + 273.0) * u_2m * (e_s - e_a)
        ) / (vp_slope + gamma * (1 + 0.34 * u_2m))

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
        vp_slope = vapour_slope(self.tmean)
        elevation = py3dep.elevation_bycoords([self.coords], source="tnm")[0]

        # Latent Heat of Vaporization [MJ/kg]
        lmbda = 2.501 - 0.002361 * self.tmean
        gamma = psychrometric_constant(elevation, lmbda)

        # Saturation Vapor Pressure [kPa]
        rh = self.clm[self.rh] if self.rh in self.clm else None
        _, e_a = vapour_pressure(self.clm[self.tmax], self.clm[self.tmin], rh)

        rad_a = extraterrestrial_radiation(self.clm.index.dayofyear, self.coords[1])
        rad_n = net_radiation(
            self.clm[self.srad],
            self.clm[self.dayl],
            elevation,
            self.clm[self.tmax],
            self.clm[self.tmin],
            e_a,
            rad_a,
        )

        # value for humid conditions
        if "alpha" not in self.params:
            self.params["alpha"] = 1.26

        self.clm["pet (mm/day)"] = (
            self.params["alpha"]
            * vp_slope
            * (rad_n - self.params["soil_heat"])
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
        rad_a = extraterrestrial_radiation(self.clm.index.dayofyear, self.coords[1]) / 2.43
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
        For ``penman_monteith`` method, the dataset must include at least the following variables:
        ``tmin``, ``tmax``, ``lat``, ``lon``, ``srad``, ``dayl``. Also, if
        ``rh`` (relative humidity) and ``u2`` (wind at 2 m level)
        are available, they are used. Otherwise, actual vapour pressure is assumed
        to be saturation vapour pressure at daily minimum temperature and 2-m wind
        speed is considered to be 2 m/s. For the ``hargreaves_samani`` method, the dataset
        must include ``tmin``, ``tmax``, and ``lat``.
    params : dict, optional
        Model-specific parameters as a dictionary, defaults to ``None``.
    """

    def __init__(
        self,
        clm: xr.Dataset,
        params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.clm = clm
        self.params = params if isinstance(params, dict) else {"soil_heat": 0.0}
        self.res = 1.0e3
        self.crs = " ".join(
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

        self.clm["tmean"] = 0.5 * (self.clm["tmax"] + self.clm["tmin"])
        self.clm["elevation"] = py3dep.elevation_bygrid(
            self.clm.x.values, self.clm.y.values, self.crs, self.res
        ).chunk({"x": clm.chunksizes["x"], "y": clm.chunksizes["y"]})

        # recommended when no data is not available to estimate soil heat flux
        if "soil_heat" not in self.params:
            self.params["soil_heat"] = 0.0

        self.clm_vars = list(self.clm.keys())
        self.req_vars = {
            "penman_monteith": ["tmin", "tmax", "lat", "srad", "dayl"],
            "priestley_taylor": ["tmin", "tmax", "lat", "srad", "dayl"],
            "hargreaves_samani": ["tmin", "tmax", "lat"],
        }

    @staticmethod
    def set_new_attrs(clm: xr.Dataset) -> xr.Dataset:
        """Set new attributes to the input dataset.

        Parameters
        ----------
        clm : xarray.DataArray
            The dataset to which the new attributes are added.
        """
        dtype = clm[next(iter(clm.keys()))].dtype
        clm["elevation"].attrs = {"units": "m", "long_name": "elevation"}
        clm["elevation"] = clm["elevation"].astype(dtype)
        clm["pet"].attrs = {"units": "mm/day", "long_name": "daily potential evapotranspiration"}
        clm["pet"] = clm["pet"].astype(dtype)
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

        # Slope of saturation vapour pressure [kPa/°C]
        self.clm["vp_slope"] = vapour_slope(self.clm["tmean"])

        # Latent Heat of Vaporization [MJ/kg]
        self.clm["lambda"] = 2.501 - 0.002361 * self.clm["tmean"]
        self.clm["gamma"] = psychrometric_constant(self.clm["elevation"], self.clm["lambda"])

        # Saturation vapor pressure [kPa]
        rh = self.clm["rh"] if "rh" in self.clm_vars else None
        self.clm["e_s"], self.clm["e_a"] = vapour_pressure(self.clm["tmax"], self.clm["tmin"], rh)

        rad_a = extraterrestrial_radiation(self.clm["time"].dt.dayofyear, self.clm.lat)
        self.clm["rad_n"] = net_radiation(
            self.clm["srad"],
            self.clm["dayl"],
            self.clm["elevation"],
            self.clm["tmax"],
            self.clm["tmin"],
            self.clm["e_a"],
            rad_a,
        )

        # recommended when no data is not available to estimate wind speed
        u_2m = self.clm["u2"] if "u2" in self.clm_vars else 2.0
        self.clm["pet"] = (
            0.408 * self.clm["vp_slope"] * (self.clm["rad_n"] - self.params["soil_heat"])
            + self.clm["gamma"]
            * 900.0
            / (self.clm["tmean"] + 273.0)
            * u_2m
            * (self.clm["e_s"] - self.clm["e_a"])
        ) / (
            self.clm["vp_slope"] + self.clm["gamma"] * (1.0 + 0.34 * u_2m)  # type: ignore
        )

        self.clm = self.clm.drop_vars(
            ["vp_slope", "gamma", "rad_n", "tmean", "e_a", "lambda", "e_s"]
        )

        return self.set_new_attrs(self.clm)

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

        # Slope of saturation vapour pressure [kPa/°C]
        self.clm["vp_slope"] = vapour_slope(self.clm["tmean"])

        # Latent Heat of Vaporization [MJ/kg]
        self.clm["lambda"] = 2.501 - 0.002361 * self.clm["tmean"]
        self.clm["gamma"] = psychrometric_constant(self.clm["elevation"], self.clm["lambda"])

        # Saturation vapor pressure [kPa]
        rh = self.clm["rh"] if "rh" in self.clm_vars else None
        _, self.clm["e_a"] = vapour_pressure(self.clm["tmax"], self.clm["tmin"], rh)

        rad_a = extraterrestrial_radiation(self.clm["time"].dt.dayofyear, self.clm.lat)
        self.clm["rad_n"] = net_radiation(
            self.clm["srad"],
            self.clm["dayl"],
            self.clm["elevation"],
            self.clm["tmax"],
            self.clm["tmin"],
            self.clm["e_a"],
            rad_a,
        )

        # value for humid conditions
        if "alpha" not in self.params:
            self.params["alpha"] = 1.26

        self.clm["pet"] = (
            self.params["alpha"]
            * self.clm["vp_slope"]
            * (self.clm["rad_n"] - self.params["soil_heat"])
            / ((self.clm["vp_slope"] + self.clm["gamma"]) * self.clm["lambda"])
        )

        self.clm = self.clm.drop_vars(["vp_slope", "gamma", "lambda", "rad_n", "tmean", "e_a"])

        return self.set_new_attrs(self.clm)

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

        lat = self.clm.lat
        rad_a = extraterrestrial_radiation(self.clm["time"].dt.dayofyear, lat) / 2.43
        self.clm["pet"] = (
            0.0023
            * (self.clm["tmean"] + 17.8)
            * np.sqrt(self.clm["tmax"] - self.clm["tmin"])
            * rad_a
        )

        self.clm = self.clm.drop_vars("tmean")

        return self.set_new_attrs(self.clm)


def potential_et(
    clm: DF,
    coords: Optional[Tuple[float, float]] = None,
    crs: Union[str, pyproj.CRS] = DEF_CRS,
    method: str = "hargreaves_samani",
    params: Optional[Dict[str, float]] = None,
) -> DF:
    """Compute Potential EvapoTranspiration for both gridded and a single location.

    Parameters
    ----------
    clm : pandas.DataFrame or xarray.Dataset
        The dataset must include at least the following variables:

        * Minimum temperature in degree celsius
        * Maximum temperature in degree celsius
        * Solar radiation in in W/m2
        * Daylight duration in seconds

        Optionally, relative humidity and wind speed at 2-m level will be used if available.

        Table below shows the variable names that the function looks for in the input data.

        ==================== ========
        DataFrame            Dataset
        ==================== ========
        ``tmin (degrees C)`` ``tmin``
        ``tmax (degrees C)`` ``tmax``
        ``srad (W/m2)``      ``srad``
        ``dayl (s)``         ``dayl``
        ``rh (-)``           ``rh``
        ``u2 (m/s)``         ``u2``
        ==================== ========

        If relative humidity and wind speed at 2-m level are not available,
        actual vapour pressure is assumed to be saturation vapour pressure at daily minimum
        temperature and 2-m wind speed is considered to be 2 m/s.
    coords : tuple of floats, optional
        Coordinates of the daymet data location as a tuple, (x, y). This is required when ``clm``
        is a ``DataFrame``.
    crs : str, optional
        The spatial reference of the input coordinate, defaults to ``epsg:4326``. This is only used
        when ``clm`` is a ``DataFrame``.
    method : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, ``hargreaves_samani``, and
        None (don't compute PET). The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``hargreaves_samani``.
    params : dict, optional
        Model-specific parameters as a dictionary, defaults to ``None``.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        The input DataFrame/Dataset with an additional variable named ``pet (mm/day)`` for
        DataFrame and ``pet`` for Dataset.

    References
    ----------
    .. footbibliography::
    """  # noqa: DAR203
    valid_methods = ["penman_monteith", "hargreaves_samani", "priestley_taylor"]
    if method not in valid_methods:
        raise InvalidInputValue("method", valid_methods)

    if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
        raise InvalidInputType("clm", "pd.DataFrame or xr.Dataset")

    pet: Union[PETCoords, PETGridded]
    if isinstance(clm, pd.DataFrame):
        if coords is None:
            raise MissingItems(["coords"])
        crs = ogc.utils.validate_crs(crs)
        pet = PETCoords(clm, coords, crs, params)
    else:
        pet = PETGridded(clm, params)
    with xr.set_options(keep_attrs=True):  # type: ignore
        return getattr(pet, method)()  # type: ignore
