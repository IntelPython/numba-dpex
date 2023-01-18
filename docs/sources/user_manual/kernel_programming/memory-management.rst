.. _memory_management:
.. include:: ./../../ext_links.txt

Memory Management
=================

**Data Parallel Extension for Numba** uses USM shared memory allocator to enable host to device and device
to host data transfers. This allows interoperability between compiled ``numba-dpex`` kernels and other
SYCL-based Python extensions.

``numba-dpex`` relies on the USM memory manager provided by `Data Parallel Control`_ library, ``dpctl``, and supports
the **SYCL USM Array Interface** protocol to enable zero-copy data exchange across USM memory-backed Python objects.

.. note::

    USM pointers can be of four allocation types:
    ``host``, ``device``, ``shared``, and ``unknown``.

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
library-allocated memory is freed using the appropriate de-allocator.

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

Local memory is a contiguous region of memory allocated
per work group and visible to all work items in that group.

The local memory is device-only and cannot be accessed from the host. From the perspective of
the device, the local memory is exposed as a contiguous array of a specific
type. The maximum available local memory is hardware-specific.

The **Data Parallel Extensions for Python**'s local memory concept is analogous to CUDA*'s shared memory concept.

``numba-dpex`` provides a special function ``numba_dpex.local.array()`` to
allocate local memory for a kernel.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/scan.py
   :lines: 21-23

In this example two local arrays, ``b`` and ``c`` , of size ``ls`` are created. Their type is specified
in the parameter ``dtype``.

.. note::

  To go convert from ``numba.cuda`` to ``numba-dpex``, replace ``numba.cuda.shared.array`` with
  ``numba_dpex.local.array(shape=local_size)``.


Private memory
--------------

Private memory is a region of memory allocated per work item, visible only to that work item.
Private memory is used for kernel parameters and local stack variables. Private memory cannot be accessed from the host.

Private memory is typically mapped to hardware registers. There is no mechanism in
**Data Parallel Extensions for Python** to query the number of registers available to a particular device.
Developers must refer to the documentation of the hardware vendor to understand
the limits of private memory.

Similarly to local memory, ``numba-dpex`` provides built-in function for private memory allocation
``numba_dpex.private.array(shape, dtype)``

Constant memory
---------------

The constant memory is a read-only device memory. ``numba-dpex`` does not yet support this type of memory.

Barriers
--------

**Data Parallel Extension for Numba** has a built-in function ``numba_dpex.barrier(mem_fence_type)`` that
allows implementing memory fencing for global and local arrays.

.. list-table:: **Global and local memory fences**
   :widths: 70 200
   :header-rows: 1

   * - ``mem_fence_type``
     - Description
   * - ``LOCAL_MEM_FENCE``
     - The barrier function will either flush any variables stored in local memory or queue a memory fence
       to ensure correct ordering of memory operations to local memory.
   * - ``GLOBAL_MEM_FENCE``
     - The barrier function will queue a memory fence to ensure correct ordering of memory operations to global memory.
       This can be useful when work-items, for example, write to buffer or image objects
       and then want to read the updated data.

.. Note::
   Calling ``numba_dpex.barrier()`` with no argument is equivalent to ``GLOBAL_MEM_FENCE``

The following example implements Hillis-Steele algorithm for prefix sum, and illustrates the usage of
global and local memory along with global and local barriers:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/scan.py
   :lines: 4-
   :linenos:
   :emphasize-lines: 19-20, 24, 34, 44

Two local arrays of size equal to work-group size are allocated in lines 19-20. Local barrier on the line 24
is used to ensure all local work-group items are initialized prior to their use in the loop starting on the line 28.
Finally, prior to writing back to the global memory ``a[]`` the global barrier in the line 44 ensures
work completion among all work-group items.

Memory access types
-------------------

The access type declares how the programmer intends to use the memory associated with an array argument of a kernel.
It is used by the runtime to create an execution order for the kernels and perform data movement.
This will ensure that kernels are executed in an order intended by the programmer.
Depending on the capabilities of the underlying hardware, the runtime can execute kernels concurrently
if the dependencies do not give rise to dependency violations or race conditions.

The following table summarizes allows access types declarations for a kernel.

.. list-table:: **Memory access types**
   :widths: 70 200
   :header-rows: 1

   * - ``access_type``
     - Description
   * - ``read_only``
     - Declares that a given argument is readable only.
       It informs the runtime that the data needs to be available on the device before the kernel can begin executing,
       but the data need not be copied from the device to the host at the end of the computation.
   * - ``write_only``
     - Decalares that a given argument is writable only. It informs the runtime that the data does not need
       to be available on the device before the kernel can begin executing. However, the data need to be copied
       from the device to the host at the end of the computation.
   * - ``read_write``
     - Decalares that a given argument is both readable and writable.
       It informs the runtime that the data has to be available on the device before the kernel
       can begin executing. It also informs that the data need to be copied from the device back to the host
       at the end of the computation.

The following example shows how to specify access type for the kernel arguments:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/black_scholes.py
   :lines: 49-55
   :linenos:
   :emphasize-lines: 2-4

Lines 50-53 indicate that arguments ``price``, ``strike``, and ``t`` are read only. The need not be copied back to
the host after the kernel ``kernel_black_scholes()`` completes execution; arguments ``call`` and ``put`` are
write only, and hence these need not be ready by the time of the kernel invocation.

.. note::
  Please note that arguments ``rate`` and ``volatility`` do not have access type specificators, because these
  are scalar arguments

For better performance, make sure that the access types reflect the operations performed by the kernel.
The compiler will flag an error when a write is done into the array, which is declared as ``read_only``.
Also the compiler does not change the declaration of an accessor form ``read_write`` to ``read_only`` if no write
is done in the kernel.
