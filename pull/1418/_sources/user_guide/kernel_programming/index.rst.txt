.. _kernel-programming-guide:
.. include:: ./../../ext_links.txt

Kernel Programming
##################

The tutorial covers the numba-dpex kernel programming API (kapi) and introduces
the concepts needed to write data-parallel kernels in numba-dpex.


.. Preliminary concepts
.. --------------------

.. Data parallelism
.. ++++++++++++++++

.. Single Program Multiple Data
.. ++++++++++++++++++++++++++++

.. Range v/s Nd-Range Kernels
.. ++++++++++++++++++++++++++

.. Work items and Work groups
.. ++++++++++++++++++++++++++

Core concepts
*************

Writing a *range* kernel
========================

.. include:: ./writing-range-kernel.rst

Writing an *nd-range* kernel
============================

.. include:: ./writing-ndrange-kernel.rst

.. Launching a kernel
.. ==================

.. include:: ./call-kernel.rst

The ``device_func`` decorator
=============================

.. include:: ./device-functions.rst


Supported types of kernel argument
==================================

A kapi kernel function can have both array and scalar arguments. At least one of
the argument to every kernel function has to be an array. The requirement is
enforced so that a execution queue can be inferred at the kernel launch stage.
An array type argument is passed as a reference to the kernel and all scalar
arguments are passed by value.

Supported array types
---------------------
- `dpctl.tensor.usm_ndarray`_ : A SYCL-based Python Array API complaint tensor.
- `dpnp.ndarray`_ :  A ``numpy.ndarray``-like array container that supports SYCL USM memory allocation.

Scalar types
------------

Scalar values can be passed to a kernel function either using the default Python
scalar type or as explicit NumPy or dpnp data type objects.
:ref:`ex_scalar_kernel_arg_ty` shows the two possible ways of defining a scalar
type. In both scenarios, numba-dpex depends on the default Numba* type inferring
algorithm to determine the LLVM IR type of a Python object that represents a
scalar value. At the kernel submission stage the LLVM IR type is reinterpreted
as a C++11 type to interoperate with the underlying SYCL runtime.

.. code-block:: python
    :caption: **Example:** Ways of defining a scalar kernel argument
    :name: ex_scalar_kernel_arg_ty

    import dpnp

    a = 1
    b = dpnp.dtype("int32").type(1)

    print(type(a))
    print(type(b))

.. code-block:: bash
    :caption: **Output:** Ways of defining a scalar kernel argument
    :name: ex_scalar_kernel_arg_ty_output

    <class 'int'>
    <class 'numpy.int32'>

The following scalar types are currently supported as arguments of a numba-dpex
kernel function:

- ``int``
- ``float``
- ``complex``
- ``numpy.int32``
- ``numpy.uint32``
- ``numpy.int64``
- ``numpy.uint32``
- ``numpy.float32``
- ``numpy.float64``

.. important::

    The Numba* type inferring algorithm by default infers a native Python
    scalar type to be a 64-bit value. The algorithm is defined that way to be
    consistent with the default CPython behavior. The default inferred 64-bit
    type can cause compilation failures on platforms that do not have native
    64-bit floating point support. Another potential fallout of the default
    64-bit type inference can be when a narrower width type is required by a
    specific kernel. To avoid these issues, users are advised to always use a
    dpnp/numpy type object to explicitly define the type of a scalar value.

DLPack support
--------------
At this time direct support for the `DLPack`_ protocol is has not been added to
numba-dpex. To interoperate numba_dpex with other SYCL USM based libraries,
users should first convert their input tensor or ndarray object into either of
the two supported array types, both of which support DLPack.


Supported Python features
*************************

Mathematical operations
=======================

.. include:: ./math-functions.rst

Operators
=========

.. include:: ./operators.rst

General Python features
=======================

.. include:: ./supported-python-features.rst


Advanced concepts
*****************

Local memory allocation
=======================

Private memory allocation
=========================

Group barrier synchronization
=============================

Atomic operations
=================

.. Async kernel execution
.. ======================

.. include:: ./call-kernel-async.rst

Specializing a kernel or a device_func
======================================
