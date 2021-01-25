Writing DPPY kernels
====================

Introduction
-------------

Numba-dppy has an execution model unlike the traditional sequential model used for programming CPUs.
In Numba-dppy, the code you write will be executed by multiple threads at once (often hundreds or thousands).
Your solution will be modeled by defining a thread hierarchy of work-groups and work-items.

Numba-dppy  support exposes facilities to declare and manage this hierarchy of threads.
The facilities are largely similar to those exposed by `OpenCL language <https://www.khronos.org/opencl/>`_.

Kernel declaration
------------------
A kernel function is a GPU function that is meant to be called from CPU code. It gives it
two fundamental characteristics:

- kernels cannot explicitly return a value; all result data must be written to an array passed to the function
  (if computing a scalar, you will probably pass a one-element array)
- kernels explicitly declare their thread hierarchy when called: i.e. the number of thread blocks and the number
  of threads per block (note that while a kernel is compiled once, it can be called multiple times with different
  block sizes or grid sizes).

Example
~~~~~~~~~

.. literalinclude:: ../../numba_dppy/examples/sum.py

Kernel invocation
------------------

A kernel is typically launched in the following way:

.. literalinclude:: ../../numba_dppy/examples/sum.py
   :pyobject: driver

Positioning
------------

- ``dppy.get_local_id``
- ``dppy.get_local_size``
- ``dppy.get_group_id``
- ``dppy.get_num_groups``

