=======
History
=======

0.11.4 (unreleased)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``importlib-metadata`` for getting the version insead of ``pkg_resources``
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

The highligth of this release is a major refactor of ``Daymet`` to allow for
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
- Use ``AsyncRetriever`` for sending requests asyncronosly with persistent caching.
  A cache folder in the current directory is created.
- Check for validity of start/end dates based on Daymet V4 since Puerto Rico data
  starts from 1950 while North America and Hwaii start from 1980.
- Check for validity of input coordinate/geometry based on the Daymet V4 bounding boxes.
- Improve accuracy of computing Psychrometric constant in PET calculations by using
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

- Add announcement regarding the new name for the softwate stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.0 (2021-03-06)
-------------------

- The first release after renaming hydrodata to pygeohydro.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible bugs.
- Speed up CI testing by using ``mamba`` and caching.


0.9.0 (2021-02-14)
------------------

- Bump version to the same version as pygeohydro.
- Update to version 4 of Daymet database. You can check the release information
  `here <https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html>`_
- Add a new function called ``get_bycoords`` that provides an alternative to ``get_byloc``
  for getting climate data at a single pixel. This new function uses THREDDS data server
  with NetCDF Subset Service (NCSS), and supports getting monthly and annual averages directly
  from the server. Note that this function will replace ``get_byloc`` in  the future.
  So consider migrating your code by replacing ``get_byloc`` with ``get_bycoords``. The
  input arguments of ``get_bycoords`` is very similar to ``get_bygeom``. Another difference
  between ``get_byloc`` and ``get_bycoords`` is column names where ``get_bycoords`` uses
  the units that are return by NCSS server.
- Add support for downloading mothly and annual summaries in addition to the daily
  time-scale. You can pass ``time_scale`` as ``daily``, ``monthly``, or ``annual``
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
  datymet data to increase the performance (almost 2x faster)

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
