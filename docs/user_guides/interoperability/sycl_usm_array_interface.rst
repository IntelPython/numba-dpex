======================================
Zero copy data exchange using SYCL USM
======================================

.. note::
  This page originally posted in dpctl wiki
  `Zero copy data exchange using SYCL USM`_.

Dpctl includes a DPC++ USM memory manager to allow Python objects to be
allocated using USM memory allocators. Supporting USM memory allocators is
needed to achieve seamless interoperability between SYCL-based Python
extensions. USM pointers make sense within a SYCL context and can be of four
allocation types: `"host"`, `"device"`, `"shared"`, or `"unknown"`. Host
applications, including CPython interpreter, can work with USM pointers of type
`host` and `"shared"` as if they were ordinary host pointers. Accessing
`"device"` USM pointers by host applications is not allowed.

.. _sycl-usm-arrays-interface:

SYCL USM Array Interface
========================

A SYCL library may allocate USM memory for the result that needs to be passed to
Python. A native Python extension that makes use of such a library may expose
this memory as an instance of Python class that will implement memory management
logic (ensures that memory is freed when the instance is no longer needed). The
need to manage memory arises whenever a library uses a custom allocator. For
example, `daal4py` uses Python capsule to ensure that DAAL-allocated memory is
freed using the appropriate deallocator.

To enable native extensions to pass the memory allocated by a native SYCL
library to Numba, or another SYCL-aware Python extension without making a copy,
the class must provide `__sycl_usm_array_interface__` attribute which returns a
Python dictionary with the following fields:

.. data:: object.__sycl_usm_array_interface__

  A dictionary of items.

  The keys are:

  **shape**
    tuple of integers

  **typestr**
    string

  **typedescr**
    a list of tuples

  **data**
    (int, bool)

    - `data[0]` is to be understood as `static_cast<size_t>(usm_ptr)`
    - `data[1]` is a boolean value indicating whether the array is read-only, or
      not.

  **strides**
    tuple

  **offset**
    int

  **version**
    int

  **syclobj**
    A Python object indicating context to which USM pointer is bound

    - filter selector string: default context for this root device is used
    - `dpctl.SyclContext`: explicitly given context
    - named Python capsule with name "SyclContextRef" that carries pointer to
      `sycl::context` to use
    - `dpctl.SyclQueue`  : use context stored in the queue
    - named Python capsule with name "SyclQueueRef" that carries pointer to
      `sycl::queue` from which to use the context
    - Any Python object with method `_get_capsule()` that produces a named
      Python capsule as described above.

The dictionary keys align with those of `NumPy Array Interface`_ and
`CUDA Array Interface`_. For host accessible USM pointers, the object may also
implement CPython :pep:`3118` compliant buffer interface which will be used
if a `data` key is not present in the dictionary. Use of a buffer interface
extends the interoperability to other Python objects, such as `bytes`,
`bytearray`, `array.array`, or `memoryview`. The type of the USM pointer stored
in the object can be queried using methods of the `dpctl`.

.. _Zero copy data exchange using SYCL USM: https://github.com/IntelPython/dpctl/wiki/Zero-copy-data-exchange-using-SYCL-USM
.. _NumPy Array Interface: https://numpy.org/doc/stable/reference/arrays.interface.html#python-side
.. _CUDA Array Interface: http://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
