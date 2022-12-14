.. _dpnp-extension:

DPNP extension for Numba
========================

Support DPNP types and functions in Numba functions.

DPNP array
----------

You should distinct 3 arrays. 2 arrys from dpctl and one from DPNP.

.. toctree::
    :maxdepth: 1

    arrays

Extending Numba
---------------

`Extending Numba <https://numba.readthedocs.io/en/stable/extending/index.html>`_
describes how to extend Numba to make it recognize and support additional
operations, functions or types.

Numba provides two categories of APIs:

- The high-level APIs
- The low-level APIs

Example:

.. code-block:: python
  :linenos:

  @numba.njit
  def sum(a, b):
      return a + b


  a = dpnp.ndarray([10])
  b = dpnp.ndarray([10])
  c = sum(a, b)

.. toctree::
    :maxdepth: 1

    numba_type
    typeof
    model
    boxing

Common
------

Main classes
````````````

`PythonAPI` contains useful functions for generating calls to Python API.
See `numba/core/pythonapi.py`.

It uses `IRBuilder` for generating code.
See llvmlite.

It also uses `BaseContext` for receiving context specific information.
See `numba/core/base.py`.

Topics to know
--------------

Python API:

- `NumPy C-API <https://numpy.org/doc/stable/reference/c-api/index.html>`_
__array_inteface__
