Writing SYCL Kernels
====================

Introduction
-------------

``numba-dppy`` offers a way of programming SYCL supporting devices using Python.
Similar to SYCL's C++ programming model for heterogeneous computing,
``numba-dppy`` offers Python abstractions for expressing data-parallelism using
a hierarchical syntax. Note that not all SYCL concepts are currently supported
in ``numba-dppy``, and some of the concepts may not be a good fit for Python.

The explicit kernel programming mode of ``numba-dppy`` bears lots of
similarities with Numba's other GPU backends:``numba.cuda`` and ``numba.roc``.
Readers who are familiar with either of the existing backends of Numba, or in
general with OpenCL, CUDA, or SYCL programming should find writing kernels in
``numba-dppy`` extremely intuitive. Irrespective of the reader's level of
familiarity with GPU programming frameworks, this documentation should serves
as a guide for using the current features available in ``numba-dppy``.

Kernel declaration
------------------
A kernel function is a device function that is meant to be called from host
code, where a device can be any SYCL supported device such as a GPU, CPU, or an
FPGA. The present focus of development of ``numba-dppy`` is mainly on Intel's
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

.. literalinclude:: ../../../numba_dppy/examples/sum.py

Kernel invocation
------------------

A kernel is typically launched in the following way:

.. literalinclude:: ../../../numba_dppy/examples/sum.py
   :pyobject: driver

Indexing functions
------------------

Currently, ``numba-dppy`` supports the following indexing functions that have
the same semantics as OpenCL.

- ``numba_dppy.get_local_id``
- ``numba_dppy.get_local_size``
- ``numba_dppy.get_group_id``
- ``numba_dppy.get_num_groups``
