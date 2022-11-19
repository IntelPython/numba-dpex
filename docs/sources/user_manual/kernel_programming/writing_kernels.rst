Writing Numba-DPEx Kernels
==========================

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

.. literalinclude:: ./../../../../numba_dpex/examples/sum.py

Kernel invocation
------------------

A kernel is typically launched in the following way:

.. literalinclude:: ./../../../../numba_dpex/examples/sum.py
   :pyobject: driver

Indexing functions
------------------

Numba-dpex provides the following indexing functions that have OpenCL-like
semantics:

- ``numba_dpex.get_local_id``
- ``numba_dpex.get_local_size``
- ``numba_dpex.get_group_id``
- ``numba_dpex.get_num_groups``
