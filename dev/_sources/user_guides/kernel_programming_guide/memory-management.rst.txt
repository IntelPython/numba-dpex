Memory Management
=================

``numba-dppy`` uses DPC++'s USM shared memory allocator (``memory_alloc``) to
enable host to device and *vice versa* data transfer. By using USM shared
memory allocator, ``numba-dppy`` allows seamless interoperability between
``numba-dppy`` and other SYCL-based Python extensions and across multiple
kernels written using ``numba_dppy.kernel`` decorator.

``numba-dppy`` uses the USM memory manager provided by ``dpctl`` and supports
the **SYCL USM Array Interface** protocol to enable zero-copy data
exchange across USM memory-backed Python objects.

.. note::

    USM pointers make sense within a SYCL context and can be of four allocation
    types: ``host``, ``device``, ``shared``, or ``unknown``. Host applications,
    including CPython interpreter, can work with USM pointers of type ``host``
    and ``shared`` as if they were ordinary host pointers. Accessing ``device``
    USM pointers by host applications is not allowed.

SYCL USM Array Interface
------------------------

A SYCL library may allocate USM memory for the result that needs to be passed to
Python. A native Python extension that makes use of such a library may expose
this memory as an instance of Python class that will implement memory management
logic (ensures that memory is freed when the instance is no longer needed).
The need to manage memory arises whenever a library uses a custom allocator.
For example, |dp4p|_ uses Python capsule to ensure that a native
library-allocated memory is freed using the appropriate deallocator.

To enable native extensions to pass the memory allocated by a native SYCL
library to Numba, or another SYCL-aware Python extension without making a copy,
the class must provide ``__sycl_usm_array_interface__`` attribute which
returns a Python dictionary with the following fields:

- ``shape``: tuple of ``int``
- ``typestr``: ``string``
- ``typedescr``: a list of tuples
- ``data``: (``int``, ``bool``)
- ``strides``: tuple of ``int``
- ``offset``: ``int``
- ``version``: ``int``
- ``syclobj``: ``dpctl.SyclQueue`` or ``dpctl.SyclContext`` object

The dictionary keys align with those of |npai|_ and |cai|_. For host accessible
USM pointers, the object may also implement CPython
`PEP-3118 <https://www.python.org/dev/peps/pep-3118/>`_
compliant buffer interface which will be used if a ``data`` key is not present
in the dictionary. Use of a buffer interface extends the interoperability to
other Python objects, such as ``bytes``, ``bytearray``, ``array.array``, or
``memoryview``. The type of the USM pointer stored in the object can be queried
using methods of the ``dpctl``.

.. |npai| replace:: ``numpy.ndarray.__array_interface__``
.. _npai: https://numpy.org/doc/stable/reference/arrays.interface.html

.. |cai| replace:: ``__cuda_array_interface__``
.. _cai: http://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html

.. |dp4p| replace:: ``daal4py``
.. _dp4p: https://intelpython.github.io/daal4py/


Device-only memory and explicit data transfer
---------------------------------------------

At the moment, there is no mechanism for the explicit transfer of arrays to
the device and back. Please use usm arrays.


Local memory
------------

In SYCL's memory model, local memory is a contiguous region of memory allocated
per work group and is visible to all the work items in that group. Local memory
is device-only and cannot be accessed from the host. From the perspective offers
the device, the local memory is exposed as a contiguous array of a specific
types. The maximum available local memory is hardware-specific. The SYCL local
memory concept is analogous to CUDA's shared memory concept.

``numba-dppy`` provides a special function ``dppy.local.array`` to
allocate local memory for a kernel.

.. literalinclude:: ../../../numba_dppy/examples/barrier.py
   :pyobject: local_memory

.. note::

  To go convert from ``numba.cuda`` to ``numba-dppy``, replace
  ``numba.cuda.shared.array`` with
  ``numba_dppy.local.array(shape=blocksize, dtype=float32)``.

.. todo::

  Add details about current limitations for local memory allocation in
  ``numba-dppy``.

Private and Constant memory
---------------------------

``numba-dppy`` does not yet support SYCL private and constant memory.
