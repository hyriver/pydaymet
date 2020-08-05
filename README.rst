.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/pydaymet_logo.png
    :target: https://github.com/cheginit/pydaymet
    :align: center

|

=========== ===========================================================================
Package     Description
=========== ===========================================================================
Hydrodata_  Access NWIS, HCDN 2009, NLCD, and SSEBop databases
PyGeoOGC_   Query data from any ArcGIS RESTful-, WMS-, and WFS-based services
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets
PyNHD_      Access NLDI and WaterData web services for navigating the NHDPlus database
Py3DEP_     Access topographic data through the 3D Elevation Program (3DEP) web service
PyDaymet_   Access the Daymet database for daily climate data
=========== ===========================================================================

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

.. image:: https://github.com/cheginit/pydaymet/workflows/build/badge.svg
    :target: https://github.com/cheginit/pydaymet/workflows/build
    :alt: Github Actions

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/hydrodata/develop
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

🚨 **This package is under heavy development and breaking changes are likely to happen.** 🚨

Features
--------

PyDaymet is a part of Hydrodata software stack and provides an interface to access to daily
climate data through the `Daymet <https://daymet.ornl.gov/>`__ RESTful service. Both single
pixel and gridded data can be requested which are returned as ``pandas.DataFrame`` for
single pixel requests and ``xarray.Dataset`` for gridded data requests. Additionally, it
can compute Potential EvapoTranspiration (PET) using
`UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__
method for both single pixel and gridded data.

You can try using PyDaymet without installing it on you system by clicking on the binder badge
below the PyDaymet banner. A Jupyter notebook instance with the Hydrodata software stack
pre-installed will be launched in your web browser and you can start coding!

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

PyDaymet offers two functions for getting climate data; ``get_byloc`` and ``get_bygeom``.
The arguments of these functions are identical except the first argument where the latter
should be polygon and the former should be a coordinate (a tuple of length two as in (x, y)).
The input geometry or coordinate can be in any valid CRS (defaults to EPSG:4326). The ``dates``
argument can be either a tuple of length two like ``(start_str, end_str)`` or a list of years
like ``[2000, 2005]``. It is noted that both functions have a ``pet`` flag for computing PET.

.. code-block:: python

    from pynhd import NLDI
    import pydaymet as daymet

    dates = ("2000-01-01", "2000-06-12")
    variables = ["prcp", "tmin"]

    geometry = NLDI.getfeature_byid("nwissite", "USGS-01031500", basin=True).geometry[0]
    clm_g = daymet.get_bygeom(geometry, dates, variables=variables, pet=True)

    coords = (-1431147.7928, 318483.4618)
    crs = "epsg:3542"
    clm_p = daymet.get_byloc(coords, dates, crs=crs, variables=variables, pet=True)

Some example plots are shown below:

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots_pydaymet.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots_pydaymet.png
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
accessing the Daymet in the `daymetpy <https://github.com/bluegreen-labs/daymetpy>`__ package.
