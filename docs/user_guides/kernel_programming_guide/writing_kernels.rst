Writing SYCL Kernels
====================

Introduction
-------------

Numba-dpex offers a way to write data-parallel kernels directly using Python.
The compiler extension to Numba has a programming model similar to the SYCL C++
domain-specific language. By providing similar abstractions as SYCL, Python
programmers can use the compiler to express data-parallelism using a
hierarchical syntax. Note that not all SYCL concepts are currently supported by
numba-dpex.

The explicit kernel programming mode of numba-dpex bears similarities with
Numba's other GPU backends: ``numba.cuda`` and ``numba.roc``. The documentation
should serves as a guide for using the current kernel programming features
available in numba-dpex.

Kernel declaration
------------------
A kernel function is a device function that is meant to be called from host
code, where a device can be any SYCL supported device such as a GPU, CPU, or an
FPGA. The present focus of development of numba-dpex is mainly on Intel's
GPU hardware. The main characteristics of a kernel function are:

- kernels cannot explicitly return a value; all result data must be written to
  an array passed to the function (if computing a scalar, you will probably pass
  a one-element array)
- kernels explicitly declare their thread hierarchy when called: i.e. the number
  of thread blocks and the number of threads per block (note that while a kernel
  is compiled once, it can be called multiple times with different block sizes
  or grid sizes).

Example
~~~~~~~~~

.. literalinclude:: ../../../numba_dpex/examples/sum.py

Kernel invocation
------------------

A kernel is typically launched in the following way:

.. literalinclude:: ../../../numba_dpex/examples/sum.py
   :pyobject: driver

Indexing functions
------------------

Numba-dpex provides the following indexing functions that have OpenCL-like
semantics:

- ``numba_dpex.get_local_id``
- ``numba_dpex.get_local_size``
- ``numba_dpex.get_group_id``
- ``numba_dpex.get_num_groups``
