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

  # numba_dpex/dpex_array_type.py
  from numba.core import types


  class DPEXArray(types.Array):
      """Array with address space"""

      ...


  # numba_dpex/types/dpnp_types.py
  class dpnp_ndarray_Type(DPEXArray):
      ...

Design Topics
+++++++++++++

Create 3 classes for USM types:
  - dpnp_device_array
  - dpnp_shared_array
  - dpnp_host_array
