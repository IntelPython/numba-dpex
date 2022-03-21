Arrays
======

``dpnp.ndarray`` type
`````````````````````

DPNP provides array type ``ndarray``.

.. code-block:: python
  from dpnp import ndarray

  # dpnp/__init__.py
  from dpnp.dpnp_array import dpnp_array as ndarray

It is a container for ``dpctl.tensor.usm_ndarray``.

.. code-block:: python

  # dpnp/dpnp_array.py
  class dpnp_array:
    ...
    def __init__(self, shape, dtype=numpy.float64):
      self._array_obj = dpctl.tensor.usm_ndarray(shape, dtype=dtype)
    ...
    @property
    def __sycl_usm_array_interface__(self):
        return self._array_obj.__sycl_usm_array_interface__
    ...

.. warning::
  ``dpctl.tensor.usm_ndarray`` and ``dpnp.ndarray`` are not subclasses of
  ``numpy.ndarray``.

.. also::
  There are ``dpnp.dparray`` in :file:`dpnp/dparray.pyx`.

``dpctl.tensor.usm_ndarray`` type
`````````````````````````````````

dpctl provides ``usm_ndarray``:

.. code-block:: python

  # dpctl/tensor/__init__.py
  from dpctl.tensor._usmarray import usm_ndarray

  # dpctl/tensor/_usmarray.pyx
  import dpctl.memory as dpmem
  ...
  cdef class usm_ndarray:
    ...
    @property
    def __sycl_usm_array_interface__(self):
      ...
      assert isinstance(self.base_, dpmem._memory._Memory)
      ary_iface = self.base_.__sycl_usm_array_interface__
      ...
      ary_iface['data'] = (<size_t> mem_ptr, ro_flag)
      ary_iface['shape'] = self.shape
      ...
      ary_iface['strides'] = ...
      ...
      ary_iface['typestr'] =
      ...
      ary_iface['offset'] =
      ...
    def __cinit__(..., buffer='device', ...):
      if isinstance(buffer, dpmem._memory._Memory):
        _buffer = buffer
      elif isinstance(buffer, (str, bytes)):
        if (buffer == "shared"):
          _buffer = dpmem.MemoryUSMShared(...)
        elif (buffer == "device"):
          _buffer = dpmem.MemoryUSMDevice(...)
        elif (buffer == "host"):
          _buffer = dpmem.MemoryUSMHost(...)
        elif isinstance(buffer, usm_ndarray):
          _buffer = buffer.usm_data
      ...
      self.base_ = _buffer
      ...

  # dpctl/memory/_memory.pyx
  cdef class _Memory:
    ...
    property __sycl_usm_array_interface__:
    def __get__(self):
      cdef dict iface = {
        "data": (<size_t>(<void *>self.memory_ptr), True), # bool(self.writeable)),
        "shape": (self.nbytes,),
        "strides": None,
        "typestr": "|u1",
        "version": 1,
        "syclobj": self.queue
      }
      # no typedescr and offset
      return iface

``dpctl.tensor.numpy_usm_shared.ndarray`` type
``````````````````````````````````````````````

Existing implementation of array support is placed in
:file:`numba_dppy/numpy_usm_shared.py` and is based on
``dpctl.tensor.numpy_usm_shared.ndarray`` from
:file:`dpctl/tensor/numpy_usm_shared.py`.

.. code-block:: python

  # numba_dppy/numpy_usm_shared.py
  from dpctl.tensor.numpy_usm_shared import ndarray

  # dpctl/tensor/numpy_usm_shared.py
  import numpy as np

  class ndarray(np.ndarray):
    ...

.. warning:: ``dpctl.tensor.numpy_usm_shared.ndarray`` is not related to
  ``dpnp.ndarray``.
