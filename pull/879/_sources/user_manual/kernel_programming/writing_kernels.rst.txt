.. _writing_kernels:
.. include:: ./../../ext_links.txt

Writing Data Parallel Kernels
=============================

Kernel declaration
------------------
A kernel function is a device function that is meant to be called from host
code, where a device can be any SYCL supported device such as a GPU, CPU, or an
FPGA. The main characteristics of a kernel function are:

- **Scalars must be passed as an array**. Kernels operate with ``dpnp`` array arguments only.
  If your want a scalar argument, then represent it as 0-dimensional ``dpnp`` array.

.. note::
   Please refer to `Data Parallel Extension for Numpy*`_ to learn more about ``dpnp``.

- **Kernels cannot explicitly return a value**. All result data must be written to
  ``dpnp`` array passed as a function's argument.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 14-18
   :caption: **EXAMPLE:** Data parallel kernel implementing the vector sum a+b
   :name: ex_kernel_declaration_vector_sum


Kernel invocation
------------------

When a kernel is launched you must specify the *global size* and the *local size*, which determine
the hierarchy of threads, that is the order in which kernels will be invoked.

The following syntax is used in ``numba-dpex`` for kernel invocation with specified global and local sizes:

``kernel_function_name[global_size, local_size](kernel arguments)``

In the following example we invoke kernel ``kernel_vector_sum`` with global size specified via variable
``global_size``, and use ``numba_dpex.DEFAULT_LOCAL_SIZE`` constant for setting local size to some
default value. Arguments are two input vectors ``a`` and ``b`` and one output vector ``c`` for storing the
result of vector summation:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 11-15
   :caption: **EXAMPLE:** Invocation of the vector sum kernel
   :name: ex_kernel_invocation_vector_sum

.. note::
  Each kernel is compiled once, but it can be called multiple times with different global and local sizes settings.


Kernel invocation (New Syntax)
------------------------------

Since the release 0.20.0 (Phoenix), we have introduced new kernel launch parameter
syntax for specifying ``global_size`` and ``local_size`` that similar to ``SYCL``'s
``range`` and ``ndrange`` classes. The ``global_size`` and ``local_size`` can now
be specified with ``numba_dpex``'s ``Range`` and ``NdRange`` classes.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 11-15
   :caption: **EXAMPLE:** Invocation of the vector sum kernel
   :name: ex_kernel_invocation_vector_sum



Kernel indexing functions
-------------------------

In *data parallel kernel programming* all work items are enumerated and accessed by their index.
You will use ``numba_dpex.get_global_id()`` function to get the index of a current work item from the kernel.
The total number of work items can be determined by calling ``numba_dpex.get_global_size()`` function.

The work group size can be determined by calling ``numba_dpex.get_local_size()`` function. Work items
in the current work group are accessed by calling ``numba_dpex.get_local_id()``.

The total number of work groups are determined by calling ``numba_dpex.get_num_groups()`` function.
The current work group index is obtained by calling ``numba_dpex.get_group_id()`` function.
