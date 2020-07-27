.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/pydaymet_logo.png
    :target: https://github.com/cheginit/pydaymet
    :align: center

|

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

PyDaymet is a part of `Hydrodata <https://github.com/cheginit/hydrodata>`__ software stack
and provides an interface to access daily climate data through the
`Daymet <https://daymet.ornl.gov/>`__ RESTful service. Both single pixel and gridded data
can be requested which are returned as ``pandas.DataFrame`` for single pixel and
``xarray.Dataset`` for gridded data. Additionally, it can compute Potential EvapoTranspiration
(PET) using `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__
method for both single pixel and gridded data.


You can try using PyDaymet without installing it on you system by clicking on the binder badge
below the PyDaymet banner. A Jupyter notebook instance with the Hydrodata software stack
pre-installed will be launched in your web browser and you can start coding!

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pydaymet/issues>`__.

Installation
------------

You can install pydaymet using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``):

.. code-block:: console

    $ pip install pydaymet

Alternatively, pydaymet can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pydaymet

Quickstart
----------

PyDaymet offers two functions for getting climate data; ``get_byloc`` and ``get_bygeom``.
The arguments of these function are identical except that the first argument of the latter
should be polygon and the former a coordinate (a tuple of length two as in (x, y)). The input
geometry or coordinates can be any valid CRS and by default EPSG:4326 is considered as the
input CRS. It is noted that both functions have a ``pet`` flag for computing PET.

.. code-block:: python

    from pynhd import NLDI
    import pydaymet as daymet

    dates = ("2000-01-01", "2000-06-12")
    variables = ["prcp", "tmin"]

    geometry = NLDI.getfeature_byid("nwissite", "USGS-01031500", basin=True).geometry[0]
    clm_g = daymet.get_bygeom(geometry, dates=dates, variables=variables, pet=True)

    coords = (-1431147.7928, 318483.4618)
    crs = "epsg:3542"
    clm_p = daymet.get_byloc(coords, crs=crs, dates=dates, variables=variables, pet=True)

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

- `daymetpy <https://github.com/bluegreen-labs/daymetpy>`__
