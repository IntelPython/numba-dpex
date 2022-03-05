Reduction on SYCL-supported Devices
===================================

The is no a specific decorator to implement reduction kernels yet. However, a kernel reduction
can be written explicitly. This section provides two approaches for writing a kernel reduction using
``numba_dpex.kernel``.

Examples
--------

Example 1
~~~~~~~~~

This example demonstrates a summation reduction on a one-dimensional array.

Full example can be found at ``numba_dpex/examples/sum_reduction.py``.

In this example, to reduce the array we invoke the kernel multiple times.

.. literalinclude:: ../../../numba_dpex/examples/sum_reduction.py
   :pyobject: sum_reduction_kernel

.. literalinclude:: ../../../numba_dpex/examples/sum_reduction.py
   :pyobject: sum_reduce

Example 2
~~~~~~~~~

Full example can be found at
``numba_dpex/examples/sum_reduction_recursive_ocl.py``.

.. literalinclude:: ../../../numba_dpex/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_reduction_kernel

.. literalinclude:: ../../../numba_dpex/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_recursive_reduction

.. literalinclude:: ../../../numba_dpex/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_reduce

.. note::

    There is no an analogue to the ``numba.cuda.reduce`` decorator for writing reductions kernel yet.
    Such decorator will be added in future releases.

Full examples
-------------

- ``numba_dpex/examples/sum_reduction_recursive_ocl.py``
- ``numba_dpex/examples/sum_reduction_ocl.py``
- ``numba_dpex/examples/sum_reduction.py``
