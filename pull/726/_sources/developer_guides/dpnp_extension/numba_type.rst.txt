Define Numba type
`````````````````

Each Python type should have corresponding Numba type.

Usually the new type is a subclassing the ``Type`` class.

Define Numba type for DPNP array (``dpnp.ndarray``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. assumption::
  If DPNP array Numba type will inherit ``types.Array`` then it will support all
  NumPy array functions.

.. code-block:: python

  # numba_dppy/dppy_array_type.py
  from numba.core import types

  class DPPYArray(types.Array):
    """Array with address space"""
    ...

  # numba_dppy/types/dpnp_types.py
  class dpnp_ndarray_Type(DPPYArray):
    ...

Design Topics
+++++++++++++

Create 3 classes for USM types:
  - dpnp_device_array
  - dpnp_shared_array
  - dpnp_host_array
