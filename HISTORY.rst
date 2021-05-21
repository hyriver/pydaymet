=======
History
=======

0.11.0 (unreleased)
-------------------

New Features
~~~~~~~~~~~~

- Add command-line interface (:issue_day:`7`).
- All feature query functions automatically check if ``aiohttp-client-cache`` and
  ``aiosqlite`` are installed and if so, they use persistent caching. This significantly
  improves the performance.

Breaking Changes
~~~~~~~~~~~~~~~~
- Change ``loc_crs`` and ``geo_crs`` arguments to ``crs`` in ``get_bycoords`` and ``get_bygeom``.

Documentation
~~~~~~~~~~~~~
- Add examples to docstrings and improve writing.

0.10.2 (2021-03-27)
-------------------

- Add announcement regarding the new name for the softwate stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.0 (2021-03-06)
-------------------

- The first release after renaming hydrodata to pygeohydro.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible
  bugs.
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
