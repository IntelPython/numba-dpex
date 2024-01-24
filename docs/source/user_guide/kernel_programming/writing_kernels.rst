.. include:: ./../../ext_links.txt

Writing Data Parallel Kernels
=============================

Kernel Declaration
------------------
A kernel function is a device function that is meant to be called from host
code, where a device can be any SYCL supported device such as a GPU, CPU, or an
FPGA. The main characteristics of a kernel function are:

- **Scalars must be passed as an array**. Kernels operate with ``dpnp`` array
  arguments only. If your want a scalar argument, then represent it as
  0-dimensional ``dpnp`` array.

.. note::
   Please refer to `Data Parallel Extension for Numpy*`_ to learn more about ``dpnp``.

- **Kernels cannot explicitly return a value**. All result data must be written
  to ``dpnp`` array passed as a function's argument.

Here is an example of a kernel that computes sum of two vectors ``a`` and ``b``.
Arguments are two input vectors ``a`` and ``b`` and one output vector ``c`` for
storing the result of vector summation:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 8-9, 11-15
   :caption: **EXAMPLE:** Data parallel kernel implementing the vector sum a+b
   :name: ex_kernel_declaration_vector_sum


Kernel Invocation
------------------

The kernel launch parameter syntax for specifying global and local sizes are
similar to ``SYCL``'s ``range`` and ``ndrange`` classes. The global and local
sizes need to be specified with ``numba_dpex``'s ``Range`` and ``NdRange``
classes.

For example, below is a following kernel that computes a sum of two vectors:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 8-9, 11-15
   :caption: **EXAMPLE:** A vector sum kernel
   :name: vector_sum_kernel

If the ``global size`` parameter is needed to run, it could be like this (where
``global_size`` is an ``int``):

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 8-9, 18-24
   :emphasize-lines: 5
   :caption: **EXAMPLE:** A vector sum kernel with a global size/range
   :name: vector_sum_kernel_with_launch_param

If both local and global ranges are needed, they can be specified using two
instances of ``Range`` inside an ``NdRange`` object. For example, below is a
kernel to compute pair-wise Euclidean distances of n-dimensional data points:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/pairwise_distance.py
   :language: python
   :lines: 14-15, 36-51
   :caption: **EXAMPLE:** A kernel to compute pair-wise Euclidean distances
   :name: pairwise_distance_kernel

Now the local and global sizes can be specified as follows (here both ``args.n``
and ``args.l`` are ``int``):

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/pairwise_distance.py
   :language: python
   :lines: 14-15, 27-31, 54-67
   :emphasize-lines: 4,6,13
   :caption: **EXAMPLE:** A kernel to compute pair-wise Euclidean distances with
               a global and a local size/range
   :name: pairwise_distance_kernel_with_launch_param


Kernel Indexing Functions
-------------------------

In *data parallel kernel programming* all work items are enumerated and accessed
by their index. You will use ``numba_dpex.get_global_id()`` function to get the
index of a current work item from the kernel. The total number of work items can
be determined by calling ``numba_dpex.get_global_size()`` function.

The work group size can be determined by calling ``numba_dpex.get_local_size()``
function. Work items in the current work group are accessed by calling
``numba_dpex.get_local_id()``.

The total number of work groups are determined by calling
``numba_dpex.get_num_groups()`` function. The current work group index is
obtained by calling ``numba_dpex.get_group_id()`` function.

.. _Black Scholes: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
