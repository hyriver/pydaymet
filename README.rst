.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/pydaymet_logo.png
    :target: https://github.com/cheginit/pydaymet
    :align: center

|

.. |hydrodata| image:: https://github.com/cheginit/hydrodata/workflows/pytest/badge.svg
    :target: https://github.com/cheginit/hydrodata/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/cheginit/pygeoogc/workflows/pytest/badge.svg
    :target: https://github.com/cheginit/pygeoogc/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/cheginit/pygeoutils/workflows/pytest/badge.svg
    :target: https://github.com/cheginit/pygeoutils/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pynhd| image:: https://github.com/cheginit/pynhd/workflows/pytest/badge.svg
    :target: https://github.com/cheginit/pynhd/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |py3dep| image:: https://github.com/cheginit/py3dep/workflows/pytest/badge.svg
    :target: https://github.com/cheginit/py3dep/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/cheginit/pydaymet/workflows/pytest/badge.svg
    :target: https://github.com/cheginit/pydaymet/actions?query=workflow%3Apytest
    :alt: Github Actions

=========== ==================================================================== ============
Package     Description                                                          Status
=========== ==================================================================== ============
Hydrodata_  Access NWIS, HCDN 2009, NLCD, and SSEBop databases                   |hydrodata|
PyGeoOGC_   Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets |pygeoutils|
PyNHD_      Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_     Access topographic data through National Map's 3DEP web service      |py3dep|
PyDaymet_   Access Daymet for daily climate data both single pixel and gridded   |pydaymet|
=========== ==================================================================== ============

.. _Hydrodata: https://github.com/cheginit/hydrodata
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

.. image:: https://codecov.io/gh/cheginit/pydaymet/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/pydaymet
    :alt: CodeCov

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/hydrodata/master?filepath=docs%2Fexamples
    :alt: Binder

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

|

Features
--------

PyDaymet is a part of Hydrodata software stack and provides an interface to access to daily
climate data through the `Daymet <https://daymet.ornl.gov/>`__ RESTful service. Both single
pixel and gridded data can be requested which are returned as ``pandas.DataFrame`` for
single pixel requests and ``xarray.Dataset`` for gridded data requests. Additionally, it
can compute Potential EvapoTranspiration (PET) using
`UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__
method for both single pixel and gridded data.

Note that starting from version ``0.9.0``, the recently released version of Daymet database
is used. You can check the release information `here <https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html>`_
Moreover, there's a new function called ``get_bycoords`` that is an alternative to ``get_byloc``
for getting climate data at a single pixel. This new function uses THREDDS data server
with NetCDF Subset Service (NCSS), and supports getting monthly and annual averages directly
from the server. You can pass ``time_scale`` as ``daily``, ``monthly``, or ``annual``
to ``get_bygeom`` or ``get_bycoords`` functions to download the respective summaries.
``get_bycoords`` will replace ``get_byloc`` in  the future.
So, please consider migrating your code by replacing ``get_byloc`` with ``get_bycoords``. The
input arguments of ``get_bycoords`` is very similar to ``get_bygeom``. Another difference
between ``get_byloc`` and ``get_bycoords`` is column names where ``get_bycoords`` uses
the units that are return by NCSS server.

You can try using PyDaymet without installing it on you system by clicking on the binder badge
below the PyDaymet banner. A Jupyter notebook instance with the Hydrodata software stack
pre-installed will be launched in your web browser and you can start coding!

Please note that since Hydrodata is in early development stages, while the provided
functionaities should be stable, changes in APIs are possible in new releases. But we
appreciate it if you give this project a try and provide feedback. Contributions are most welcome.

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pydaymet/issues>`__.

Installation
------------

You can install PyDaymet using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``):

.. code-block:: console

    $ pip install pydaymet

Alternatively, PyDaymet can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pydaymet

Quick start
-----------

PyDaymet offers two functions for getting climate data; ``get_bycoords`` and ``get_bygeom``.
The arguments of these functions are identical except the first argument where the latter
should be polygon and the former should be a coordinate (a tuple of length two as in (x, y)).
The input geometry or coordinate can be in any valid CRS (defaults to EPSG:4326). The ``dates``
argument can be either a tuple of length two like ``(start_str, end_str)`` or a list of years
like ``[2000, 2005]``. It is noted that both functions have a ``pet`` flag for computing PET.
Additionally, we can pass ``time_scale`` to get daily, monthly or annual averages. This flag
by default is set to daily.

.. code-block:: python

    from pynhd import NLDI
    import pydaymet as daymet

    dates = ("2000-01-01", "2000-06-12")
    var = ["prcp", "tmin"]

    geometry = NLDI().get_basins("01031500").geometry[0]

    daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
    monthly = daymet.get_bygeom(geometry, 2000, variables=var, time_scale="monthly")
    annual = daymet.get_bygeom(geometry, 2000, variables=var, time_scale="annual")

    coords = (-1431147.7928, 318483.4618)
    crs = "epsg:3542"

    daily = daymet.get_bycoords(coords, dates, variables=var, loc_crs=crs, pet=True)
    monthly = daymet.get_bycoords(coords, 2000, variables=var, loc_crs=crs, time_scale="monthly")
    annual = daymet.get_bycoords(coords, 2000, variables=var, loc_crs=crs, time_scale="annual")

Some example plots are shown below:

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/example_plots_pydaymet.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/example_plots_pydaymet.png
    :width: 600
    :align: center

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pygeoogc/blob/master/CONTRIBUTING.rst>`__
file for instructions.

Credits
-------
Credits to `Koen Hufkens <https://github.com/khufkens>`__ for his implementation of
accessing the Daymet RESTful service, `daymetpy <https://github.com/bluegreen-labs/daymetpy>`__.
