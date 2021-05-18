.. image:: https://raw.githubusercontent.com/cheginit/HyRiver-examples/main/notebooks/_static/pydaymet_logo.png
    :target: https://github.com/cheginit/HyRiver

|

.. image:: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77/status.svg
    :target: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77
    :alt: JOSS

|

.. |pygeohydro| image:: https://github.com/cheginit/pygeohydro/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeohydro/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/cheginit/pygeoogc/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeoogc/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/cheginit/pygeoutils/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeoutils/actions/workflows/test.yml
    :alt: Github Actions

.. |pynhd| image:: https://github.com/cheginit/pynhd/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pynhd/actions/workflows/test.yml
    :alt: Github Actions

.. |py3dep| image:: https://github.com/cheginit/py3dep/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/py3dep/actions/workflows/test.yml
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/cheginit/pydaymet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pydaymet/actions/workflows/test.yml
    :alt: Github Actions

=========== ==================================================================== ============
Package     Description                                                          Status
=========== ==================================================================== ============
PyNHD_      Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_     Access topographic data through National Map's 3DEP web service      |py3dep|
PyGeoHydro_ Access NWIS, NID, HCDN 2009, NLCD, and SSEBop databases              |pygeohydro|
PyDaymet_   Access Daymet for daily climate data both single pixel and gridded   |pydaymet|
PyGeoOGC_   Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets |pygeoutils|
=========== ==================================================================== ============

.. _PyGeoHydro: https://github.com/cheginit/pygeohydro
.. _PyGeoOGC: https://github.com/cheginit/pygeoogc
.. _PyGeoUtils: https://github.com/cheginit/pygeoutils
.. _PyNHD: https://github.com/cheginit/pynhd
.. _Py3DEP: https://github.com/cheginit/py3dep
.. _PyDaymet: https://github.com/cheginit/pydaymet

PyDaymet: Daily climate data through Daymet
-------------------------------------------

.. image:: https://img.shields.io/pypi/v/pydaymet.svg
    :target: https://pypi.python.org/pypi/pydaymet
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/pydaymet.svg
    :target: https://anaconda.org/conda-forge/pydaymet
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/pydaymet/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/pydaymet
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/pydaymet.svg
    :target: https://pypi.python.org/pypi/pydaymet
    :alt: Python Versions

.. image:: https://pepy.tech/badge/pydaymet
    :target: https://pepy.tech/project/pydaymet
    :alt: Downloads

|

.. image:: https://www.codefactor.io/repository/github/cheginit/pydaymet/badge
   :target: https://www.codefactor.io/repository/github/cheginit/pydaymet
   :alt: CodeFactor

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/HyRiver-examples/main?urlpath=lab/tree/notebooks
    :alt: Binder

|

Features
--------

PyDaymet is a part of `HyRiver <https://github.com/cheginit/HyRiver>`__ software stack that
is designed to aid in watershed analysis through web services. This package provides
an interface to access to daily climate data through the `Daymet <https://daymet.ornl.gov/>`__
RESTful service. Both single pixel and gridded data can be requested which are returned as
``pandas.DataFrame`` and ``xarray.Dataset``, respectively. Climate data is available for CONUS,
Hawaii, and Puerto Rico. Additionally, PyDaymet can compute Potential EvapoTranspiration (PET)
using `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__ method for both single
pixel and gridded data.

Note that starting from version ``0.9.0``, the recently released version of Daymet database
is used. You can check the release information `here <https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html>`_.
Moreover, there's a new function called ``get_bycoords`` that is an alternative to ``get_byloc``
for getting climate data at a single pixel. This new function uses THREDDS data server
with NetCDF Subset Service (NCSS), and supports getting monthly and annual summaries directly
from the server. You can pass ``time_scale`` as ``daily``, ``monthly``, or ``annual``
to ``get_bygeom`` or ``get_bycoords`` functions to download the respective summaries.
``get_bycoords`` will replace ``get_byloc`` in  the future.
So, please consider migrating your code by replacing ``get_byloc`` with ``get_bycoords``. The
input arguments of ``get_bycoords`` is identical to ``get_bygeom``. Another difference
between ``get_byloc`` and ``get_bycoords`` is column names where ``get_bycoords`` uses
the units that are returned by NCSS server. Moreover, both ``get_bygeom`` and ``get_bycoords``
accept an additional argument called ``region`` for passing the region of interest.

You can try using PyDaymet without installing it on you system by clicking on the binder badge
below the PyDaymet banner. A Jupyter notebook instance with the stack
pre-installed will be launched in your web browser and you can start coding!

Please note that since this project is in early development stages, while the provided
functionalities should be stable, changes in APIs are possible in new releases. But we
appreciate it if you give this project a try and provide feedback. Contributions are most welcome.

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pydaymet/issues>`__.

Installation
------------

You can install PyDaymet using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``). Moreover, PyDaymet has two optional
dependencies for using persistent caching, ``aiohttp-client-cache`` and ``aiosqlite``. We highly
recommend to install this package as it can significantly speedup send/receive queries. You don't
have to change anything in your code, since PyDaymet under-the-hood looks for them and if available,
it will automatically use persistent caching:

.. code-block:: console

    $ pip install pydaymet[cache]

Alternatively, PyDaymet can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pydaymet

Quick start
-----------

You can use PyDaymet using command-line or as a Python library. The commanda-line
provides access to two functionality:

- Getting climate data within a geometry: You must create a ``geopandas.GeoDataFrame`` that contains
  the geometries of the target locations. This dataframe must have at least four columns:
  ``id``, ``start``, ``end``, and ``geometry``. The ``id`` column is used as filenames for saving
  the obtained climate data to a NetCDF (``.nc``) file. The ``start`` and ``end`` columns are
  starting and ending dates. Then, you must save the dataframe to a file with extensions
  such as ``.shp`` or ``.gpkg`` (whatever that ``geopandas.read_file`` can read).
- Getting climate data for a list of coordinates: You must create a ``pandas.DataFrame`` that
  contains coordinates of the target locations. This dataframe must have at least six columns:
  ``id``, ``start``, ``end``, ``x``, and ``y``. The ``id`` column is used as filenames for saving
  the obtained climate data to a CSV (``.csv``) file.

``pydaymet`` has three required arguments and four optional:

.. code-block:: bash

    pydaymet --help
    Usage: pydaymet [OPTIONS] TARGET [geometry|coords] CRS

      Retrieve cliamte data within geometries or elevations for a list of coordinates.

      TARGET: Path to a geospatial file (any file that geopandas.read_file can
      open) or a csv file.

      The input files should have three columns:

          - id: Feature identifiers that daymet uses as the output netcdf/csv filenames.
          - start: Starting time.
          - end: Ending time.
          - region: Target region (na for CONUS, hi for Hawaii, and pr for Puerto Rico.

      If target_type is geometry, an additional geometry column is required.
      If it is coords, two additional columns are need: x and y.

      TARGET_TYPE: Type of input file: "coords" for csv and "geometry" for geospatial.

      CRS: CRS of the input data.

      Examples:

          $ pydaymet ny_coords.csv coords epsg:4326 -v prcp -v tmin -p -t monthly
          $ pydaymet ny_geom.gpkg geometry epsg:3857 -v prcp

    Options:
      -v, --variables TEXT            Target variables. You can pass this flag
                                      multiple times for multiple variables.

      -t, --time_scale [daily|monthly|annual]
                                      Target time scale.
      -p, --pet                       Compute PET.
      -s, --save_dir PATH             Path to a directory to save the requested
                                      files. Extension for the outputs is .nc for
                                      geometry and .csv for coords.

      -h, --help                      Show this message and exit.

Now, let's see how we can use PyDaymet as a library.

PyDaymet offers two functions for getting climate data; ``get_bycoords`` and ``get_bygeom``.
The arguments of these functions are identical except the first argument where the latter
should be polygon and the former should be a coordinate (a tuple of length two as in (x, y)).
The input geometry or coordinate can be in any valid CRS (defaults to EPSG:4326). The ``dates``
argument can be either a tuple of length two like ``(start_str, end_str)`` or a list of years
like ``[2000, 2005]``. It is noted that both functions have a ``pet`` flag for computing PET.
Additionally, we can pass ``time_scale`` to get daily, monthly or annual summaries. This flag
by default is set to daily.

.. code-block:: python

    from pynhd import NLDI
    import pydaymet as daymet

    geometry = NLDI().get_basins("01031500").geometry[0]

    var = ["prcp", "tmin"]
    dates = ("2000-01-01", "2000-06-30")

    daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
    monthly = daymet.get_bygeom(geometry, dates, variables=var, time_scale="monthly")

.. image:: https://raw.githubusercontent.com/cheginit/HyRiver-examples/main/notebooks/_static/daymet_grid.png
    :target: https://github.com/cheginit/HyRiver-examples/blob/main/notebooks/daymet.ipynb
    :width: 400

If the input geometry (or coordinate) is in a CRS other than EPSG:4326, we should pass
it to the functions.

.. code-block:: python

    coords = (-1431147.7928, 318483.4618)
    crs = "epsg:3542"
    dates = ("2000-01-01", "2006-12-31")
    annual = daymet.get_bycoords(
        coords, dates, variables=var, loc_crs=crs, time_scale="annual"
    )

.. image:: https://raw.githubusercontent.com/cheginit/HyRiver-examples/main/notebooks/_static/daymet_loc.png
    :target: https://github.com/cheginit/HyRiver-examples/blob/main/notebooks/daymet.ipynb
    :width: 400

Next, let's get annual total precipitation for Hawaii and Puerto Rico for 2010.

.. code-block:: python

    hi_ext = (-160.3055, 17.9539, -154.7715, 23.5186)
    pr_ext = (-67.9927, 16.8443, -64.1195, 19.9381)
    hi = daymet.get_bygeom(hi_ext, 2010, variables="prcp", region="hi", time_scale="annual")
    pr = daymet.get_bygeom(pr_ext, 2010, variables="prcp", region="pr", time_scale="annual")

Some example plots are shown below:

.. image:: https://raw.githubusercontent.com/cheginit/HyRiver-examples/main/notebooks/_static/hi.png
    :target: https://github.com/cheginit/HyRiver-examples/blob/main/notebooks/daymet.ipynb
    :width: 400

.. image:: https://raw.githubusercontent.com/cheginit/HyRiver-examples/main/notebooks/_static/pr.png
    :target: https://github.com/cheginit/HyRiver-examples/blob/main/notebooks/daymet.ipynb
    :width: 400

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pygeoogc/blob/main/CONTRIBUTING.rst>`__
file for instructions.

Credits
-------
Credits to `Koen Hufkens <https://github.com/khufkens>`__ for his implementation of
accessing the Daymet RESTful service, `daymetpy <https://github.com/bluegreen-labs/daymetpy>`__.
