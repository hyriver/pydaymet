=======
History
=======

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
