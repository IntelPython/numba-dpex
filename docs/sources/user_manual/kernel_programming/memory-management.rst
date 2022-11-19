.. _memory_management:
.. include:: ./../../ext_links.txt

Memory Management
=================

**Data Parallel Extension for Numba** follows `SYCL*`_ *memory model* implemented in `DPC++`_;
``numba-dpex`` relies on DPC++ runtimes underneath, including USM memory management features.

It uses USM shared memory allocator to enable host to device and device to host data transfers.
It allows interoperability between compiled ``numba-dpex`` kernels and other SYCL-based Python extensions.

``numba-dpex`` uses the USM memory manager provided by `Data Parallel Control`_ library, ``dpctl``, and supports
the **SYCL USM Array Interface** protocol to enable zero-copy data exchange across USM memory-backed Python objects.

.. note::

    USM pointers can be of four allocation types:
    ``host``, ``device``, ``shared``, or ``unknown``.

    Host applications, including Python interpreter, can work with USM pointers of type ``host``
    and ``shared`` as if they were ordinary host pointers.

    Accessing ``device`` USM pointers by host applications is not allowed

    For more details please refer to `Data Parallel Extensions for Python*`_ documentation.

SYCL USM Array Interface
------------------------

A SYCL-based library may allocate USM memory for the result that needs to be passed to
Python. A native Python extension that makes use of such a library may expose
this memory as an instance of Python class that will implement memory management
logic (ensures that memory is freed when the instance is no longer needed).
The need to manage memory arises whenever a library uses a custom allocator.
For example, |sklext|_ uses Python capsule to ensure that a native
library-allocated memory is freed using the appropriate deallocator.

To enable native extensions to pass the memory allocated by a native SYCL
library to `Numba*`_, or another SYCL-aware Python extension without making a copy,
the class must provide ``__sycl_usm_array_interface__`` attribute which
returns a Python dictionary with the following fields:

.. list-table:: **SYCL USM Array Interface structure**
   :widths: 30 50 100
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``shape``
     - tuple of ``int``
     - Tuple whose elements are the array size in each dimension
   * - ``typestr``
     - ``string``
     - A string providing the basic type of the homogeneous array. String format is the same as for |npai|_
   * - ``typedescr`` (optional)
     - a list of tuples
     - Provides a more detailed description of the memory layout for each item in the homogeneous array. See |npai|_
       for details.
   * - ``data`` (optional)
     - (``int``, ``bool``)
     - A 2-tuple whose first argument points to the data-area storing the array contents.
       This pointer must point to the first element of data.
       The second entry in the tuple is a read-only flag (true means the data area is read-only).
   * - ``strides`` (optional)
     - tuple of ``int`` or ``None``
     - ``None`` indicates a C-style contiguous. Strides provide the number of bytes needed to jump
       to the next array element in the corresponding dimension.
   * - ``offset`` (optional)
     - ``int``
     - An integer offset into the array data region. See |npai|_
       for details.
   * - ``version``
     - ``int``
     - An integer showing the version of the interface
   * - ``syclobj``
     - ``dpctl.SyclQueue`` or ``dpctl.SyclContext``
     - The device queue (or alternatively the device context). For details please refer to
       `Data Parallel Control`_ documentation.

For host accessible USM pointers, the object may also implement CPython
`PEP-3118 <https://www.python.org/dev/peps/pep-3118/>`_
compliant buffer interface, which will be used if a ``data`` key is not present
in the dictionary. Use of a buffer interface extends the interoperability to
other Python objects, such as ``bytes``, ``bytearray``, ``array.array``, or
``memoryview``. The type of the USM pointer stored in the object can be queried
using methods of the ``dpctl`` library.

.. |npai| replace:: ``numpy.ndarray.__array_interface__``
.. _npai: https://numpy.org/doc/stable/reference/arrays.interface.html

.. |cai| replace:: ``__cuda_array_interface__``
.. _cai: http://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html

.. |sklext| replace:: ``scikit-learn-intelex``
.. _sklext: https://intel.github.io/scikit-learn-intelex/


Device-only memory and explicit data transfer
---------------------------------------------

All data transfers between the device and the host must be explicit. Please use ``dpnp`` arrays while
working with the device data in data parallel kernels. There are ``dpnp`` conventience functions
that allow copying data from and to NumPy arrays. Read `Data Parallel Extension for Numpy*`_ documentation
for further details.

Local memory
------------

`SYCL*`_ *memory model* defines local memory as a contiguous region of memory allocated
per work group and visible to all the work items in that group.

The local memory is device-only and cannot be accessed from the host. From the perspective of
the device, the local memory is exposed as a contiguous array of a specific
type. The maximum available local memory is hardware-specific.

The SYCL local memory concept is analogous to CUDA*'s shared memory concept.

``numba-dpex`` provides a special function ``numba_dpex.local.array()`` to
allocate local memory for a kernel.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/barrier.py
   :pyobject: local_memory

.. note::

  To go convert from ``numba.cuda`` to ``numba-dpex``, replace
  ``numba.cuda.shared.array`` with
  ``numba_dpex.local.array(shape=local_size, dtype=float32)``.

.. todo::

  Add details about current limitations for local memory allocation in
  ``numba-dpex``.

Private and Constant memory
---------------------------

``numba-dpex`` does not yet support SYCL private and constant memory.
