=======
History
=======

0.15.2 (2023-09-22)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Remove dependency on ``dask``.

0.15.1 (2023-09-02)
-------------------

Bug Fixes
~~~~~~~~~
- Fix HyRiver libraries requirements by specifying a range instead
  of exact version so ``conda-forge`` can resolve the dependencies.

0.15.0 (2023-05-07)
-------------------
From release 0.15 onward, all minor versions of HyRiver packages
will be pinned. This ensures that previous minor versions of HyRiver
packages cannot be installed with later minor releases. For example,
if you have ``py3dep==0.14.x`` installed, you cannot install
``pydaymet==0.15.x``. This is to ensure that the API is
consistent across all minor versions.

New Features
~~~~~~~~~~~~
- For now, retain compatibility with ``shapely<2`` while supporting
  ``shapley>=2``.

0.14.0 (2023-03-05)
-------------------

New Features
~~~~~~~~~~~~
- Change missing value of both single-pixel and gridded
  versions to ``numpy.nan`` from -9999.
- Add a new model parameter for computing PET using ``priestlet_taylor``
  and ``penman_monteith`` models called ``arid_correction``. For arid
  regions, FAO 56 suggests subtracting the min temperature by 2 degrees.
  This parameter can be passed via ``pet_params`` in ``daymet_by*`` functions,
  or ``params`` in ``potential_pet`` function.
- Refactor ``get_bycoords`` to reduce memory usage by using a combination
  of ``itertools`` and ``Generator`` objects.
- Refactor the ``pet`` module to improve performance and readability, and
  reduce code duplication.

Documentation
~~~~~~~~~~~~~
- Add more information about parameters that ``pet`` functions accept.

Breaking Changes
~~~~~~~~~~~~~~~~
- Bump the minimum required version of ``shapely`` to 2.0,
  and use its new API.

Internal Changes
~~~~~~~~~~~~~~~~
- Sync all minor versions of HyRiver packages to 0.14.0.

0.13.12 (2023-02-10)
--------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Fully migrate ``setup.cfg`` and ``setup.py`` to ``pyproject.toml``.
- Convert relative imports to absolute with ``absolufy-imports``.
- Sync all patch versions of HyRiver packages to x.x.12.

0.13.10 (2023-01-08)
--------------------

New Features
~~~~~~~~~~~~
- Refactor the ``show_versions`` function to improve performance and
  print the output in a nicer table-like format.

Bug Fixes
~~~~~~~~~
- Fix a bug in ``get_bygeom`` where for small requests that lead to
  a single download URL, the function failed.

Internal Changes
~~~~~~~~~~~~~~~~
- Skip 0.13.9 version so the minor version of all HyRiver packages become
  the same.

0.13.8 (2022-12-09)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- More robust handling of getting large gridded data. Instead of caching the requests/
  responses, directly store the responses as NetCDF files to a cache folder using
  ``pygeoogc.streaming_download`` and ultimately read them using ``xarray.open_mfdataset``.
  This should make the ``bygeom`` function even faster than before and also make it
  possible to make large requests without having to worry about running out of memory
  (:issue_day:`59`).
- Modify the codebase based on `Refurb <https://github.com/dosisod/refurb>`__
  suggestions.

0.13.7 (2022-11-04)
-------------------

**Since the release of Daymet v4 R1 on November 2022, the URL of Daymet's server has
been changed. Therefore, only PyDaymet v0.13.7+ is going to work, and previous
versions will not work anymore.**

New Features
~~~~~~~~~~~~
- Add support for passing a list of coordinates to the ``get_bycoords`` function.
  Also, optionally, you can pass a list of IDs for the input coordinates that
  will be used as ``keys`` for the returned ``pandas.DataFrame`` or a dimension
  called ``id`` in the returned ``xarray.Dataset`` if ``to_xarray`` is enabled.
- Add a new argument called ``to_xarray`` to the ``get_bycoords`` function for
  returning the results as a ``xarray.Dataset`` instead of a ``pandas.DataFrame``.
  When set to ``True``, the returned ``xarray.Dataset`` will have three attributes
  called ``units``, ``description``, and ``long_name``.
- The ``date`` argument of both ``get_bycoords`` and ``by_geom`` functions
  now accepts ``range``-type objects for passing years, e.g., ``range(2000-2005)``.

.. code-block:: python

    import pydaymet as daymet

    coords = [(-94.986, 29.973), (-95.478, 30.134)]
    idx = ["P1", "P2"]
    clm = daymet.get_bycoords(coords, range(2000, 2021), coords_id=idx, to_xarray=True)

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``pyupgrade`` package to update the type hinting annotations
  to Python 3.10 style.
- Fix the Daymet server URL.

0.13.6 (2022-08-30)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add the missing PyPi classifiers for the supported Python versions.

0.13.5 (2022-08-29)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Append "Error" to all exception classes for conforming to PEP-8 naming conventions.

Internal Changes
~~~~~~~~~~~~~~~~
- Bump the minimum versions of ``pygeoogc``, ``pygeoutils``, ``py3dep`` to 0.13.5 and
  that of ``async-retriever`` to 0.3.5.

0.13.3 (2022-07-31)
-------------------

Bug Fixes
~~~~~~~~~
- Fix a bug in ``PETGridded`` where the wrong data type was being set for
  ``pet`` and ``elevation`` variables.
- When initializing ``PETGridded``, only chunk the elevation if the input
  climate data is chunked.

0.13.2 (2022-06-14)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Set the minimum supported version of Python to 3.8 since many of the
  dependencies such as ``xarray``, ``pandas``, ``rioxarray`` have dropped support
  for Python 3.7.

Internal Changes
~~~~~~~~~~~~~~~~
- Use `micromamba <https://github.com/marketplace/actions/provision-with-micromamba>`__
  for running tests
  and use `nox <https://github.com/marketplace/actions/setup-nox>`__
  for linting in CI.

0.13.1 (2022-06-11)
-------------------

New Features
~~~~~~~~~~~~
- Adopt the default snow parameters' values from a new source
  https://doi.org/10.5194/gmd-11-1077-2018 and add the citation.

Bug Fixes
~~~~~~~~~
- Set the end year based on the current year since Daymet data get updated
  every year (:pull_day:`55`) by `Tim Cera <https://github.com/timcera>`__.
- Set the months for the annual timescale to correct values (:pull_day:`55`)
  by `Tim Cera <https://github.com/timcera>`__.

0.13.0 (2022-03-03)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Remove caching-related arguments from all functions since now they
  can be set globally via three environmental variables:

  * ``HYRIVER_CACHE_NAME``: Path to the caching SQLite database.
  * ``HYRIVER_CACHE_EXPIRE``: Expiration time for cached requests in seconds.
  * ``HYRIVER_CACHE_DISABLE``: Disable reading/writing from/to the cache file.

  You can do this like so:

.. code-block:: python

    import os

    os.environ["HYRIVER_CACHE_NAME"] = "path/to/file.sqlite"
    os.environ["HYRIVER_CACHE_EXPIRE"] = "3600"
    os.environ["HYRIVER_CACHE_DISABLE"] = "true"

0.12.3 (2022-02-04)
-------------------

New Features
~~~~~~~~~~~~
- Add a new flag to both ``get_bycoords`` and ``get_bygeom`` functions
  called ``snow`` which separates snow from the precipitation using
  the `Martinez and Gupta (2010) <https://doi.org/10.1029/2009WR008294>`__ method.

Internal Changes
~~~~~~~~~~~~~~~~
- Add elevation data when computing PET regardless of the ``pet`` method.
- Match the chunk size of ``elevation`` with that of the climate data.
- Drop ``time`` dimension from ``elevation``, ``lon``, and ``lat`` variables.

Bug Fixes
~~~~~~~~~
- Fix a bug in setting dates for monthly timescales. For monthly timescale
  Daymet calendar is at 15th or 16th of the month, so input dates need to be
  adjusted accordingly.

0.12.2 (2022-01-15)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Clean up the PET computation functions' output by removing temporary
  variables that are created during the computation.
- Add more attributes for ``elevation`` and ``pet`` variables.
- Add type checking with ``typeguard`` and fixed typing issues raised by
  ``typeguard``.
- Refactor ``show_versions`` to ensure getting correct versions of all
  dependencies.

0.12.1 (2021-12-31)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use the three new ``ar.retrieve_*`` functions instead of the old ``ar.retrieve``
  function to improve type hinting and to make the API more consistent.

0.12.0 (2021-12-27)
-------------------

New Features
~~~~~~~~~~~~
- Expose the ``ssl`` argument for disabling the SSL certification verification (:issue_day:`41`).
  Now, you can pass ``ssl=False`` to disable the SSL verification in both ``get_bygeom`` and
  ``get_bycoord`` functions. Moreover, you can pass ``--disable_ssl`` to PyDaymet's command line
  interface to disable the SSL verification.

Breaking Changes
~~~~~~~~~~~~~~~~
- Set the request caching's expiration time to never expire. Add two flags to all
  functions to control the caching: ``expire_after`` and ``disable_caching``.

Internal Changes
~~~~~~~~~~~~~~~~
- Add all the missing types so ``mypy --strict`` passes.

0.11.4 (2021-11-12)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``importlib-metadata`` for getting the version instead of ``pkg_resources``
  to decrease import time as discussed in this
  `issue <https://github.com/pydata/xarray/issues/5676>`__.

0.11.3 (2021-10-07)
-------------------

Bug Fixes
~~~~~~~~~
- There was an issue in the PET computation due to ``dayofyear`` being added as a new dimension.
  This version fixes it and even further simplifies the code by using ``xarray``'s ``dt`` accessor
  to gain access to the ``dayofyear`` method.

0.11.2 (2021-10-07)
-------------------

New Features
~~~~~~~~~~~~
- Add ``hargreaves_samani`` and ``priestley_taylor`` methods for computing PET.

Breaking Changes
~~~~~~~~~~~~~~~~
- Rewrite the command-line interface using ``click.group`` to improve UX.
  The command is now ``pydaymet [command] [args] [options]``. The two supported
  commands are ``coords`` for getting climate data for a dataframe of coordinates
  and ``geometry`` for getting gridded climate data for a geo-dataframe. Moreover,
  Each sub-command now has a separate help message and example.
- Deprecate ``get_byloc`` in favor of ``get_bycoords``.
- The ``pet`` argument in both ``get_bycoords`` and ``get_bygeom`` functions now
  accepts ``hargreaves_samani``, ``penman_monteith``, ``priestley_taylor``, and ``None``.

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor the ``pet`` module for reducing duplicate code and improving readability and
  maintainability. The code is smaller now and the functions for computing physical properties
  include references to equations from the respective original paper.

0.11.1 (2021-07-31)
-------------------

The highlight of this release is a major refactor of ``Daymet`` to allow for
extending PET computation function for using methods other than FAO-56.

New Features
~~~~~~~~~~~~
- Refactor ``Daymet`` class by removing ``pet_bycoords`` and ``pet_bygrid`` methods and
  creating a new public function called ``potential_et``. This function computes potential
  evapotranspiration (PET) and supports both gridded (``xarray.Dataset``) and single pixel
  (``pandas.DataFrame``) climate data. The long-term plan is to add support for methods
  other than FAO 56 for computing PET.

0.11.0 (2021-06-19)
-------------------

New Features
~~~~~~~~~~~~
- Add command-line interface (:issue_day:`7`).
- Use ``AsyncRetriever`` for sending requests asynchronously with persistent caching.
  A cache folder in the current directory is created.
- Check for validity of start/end dates based on Daymet V4 since Puerto Rico data
  starts from 1950 while North America and Hawaii start from 1980.
- Check for validity of input coordinate/geometry based on the Daymet V4 bounding boxes.
- Improve accuracy of computing Psychometric constant in PET calculations by using
  an equation in Allen et al. 1998.

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.6 since many of the dependencies such as ``xarray`` and ``pandas``
  have done so.
- Change ``loc_crs`` and ``geo_crs`` arguments to ``crs`` in ``get_bycoords`` and ``get_bygeom``.

Documentation
~~~~~~~~~~~~~
- Add examples to docstrings and improve writing.
- Add more notes regarding the underlying assumptions for ``pet_bycoords`` and ``pet_bygrid``.

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor ``Daymet`` class to use ``pydantic`` for validating the inputs.
- Increase test coverage.

0.10.2 (2021-03-27)
-------------------

- Add announcement regarding the new name for the software stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.0 (2021-03-06)
-------------------

- The first release after renaming hydrodata to PyGeoHydro.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible bugs.
- Speed up CI testing by using ``mamba`` and caching.


0.9.0 (2021-02-14)
------------------

- Bump version to the same version as PyGeoHydro.
- Update to version 4 of Daymet database. You can check the release information
  `here <https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html>`_
- Add a new function called ``get_bycoords`` that provides an alternative to ``get_byloc``
  for getting climate data at a single pixel. This new function uses THREDDS data server
  with NetCDF Subset Service (NCSS), and supports getting monthly and annual averages directly
  from the server. Note that this function will replace ``get_byloc`` in the future.
  So consider migrating your code by replacing ``get_byloc`` with ``get_bycoords``. The
  input arguments of ``get_bycoords`` is very similar to ``get_bygeom``. Another difference
  between ``get_byloc`` and ``get_bycoords`` is column names where ``get_bycoords`` uses
  the units that are return by NCSS server.
- Add support for downloading monthly and annual summaries in addition to the daily
  timescale. You can pass ``time_scale`` as ``daily``, ``monthly``, or ``annual``
  to ``get_bygeom`` or ``get_bycoords`` functions to download the respective summaries.
- Add support for getting climate data for Hawaii and Puerto Rico by passing ``region``
  to ``get_bygeom`` and ``get_bycoords`` functions. The acceptable values are ``na`` for
  CONUS, ``hi`` for Hawaii, and ``pr`` for Puerto Rico.

0.2.0 (2020-12-06)
------------------

- Add support for multipolygon.
- Remove the ``fill_hole`` argument.
- Improve masking by geometry.
- Use the newly added ``async_requests`` function from ``pygeoogc`` for getting
  Daymet data to increase the performance (almost 2x faster)

0.1.3 (2020-08-18)
------------------

- Replaced ``simplejson`` with ``orjson`` to speed-up JSON operations.

0.1.2 (2020-08-11)
------------------

- Add ``show_versions`` for showing versions of the installed deps.

0.1.1 (2020-08-03)
------------------

- Retained the compatibility with ``xarray`` 0.15 by removing the ``attrs`` flag.
- Replaced ``open_dataset`` with ``load_dataset`` for automatic handling of closing
  the input after reading the content.
- Removed ``years`` argument from both ``byloc`` and ``bygeom`` functions. The ``dates``
  argument now accepts both a tuple of start and end dates and a list of years.

0.1.0 (2020-07-27)
------------------

- Initial release on PyPI.
