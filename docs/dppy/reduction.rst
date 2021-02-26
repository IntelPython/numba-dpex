Reduction on SYCL-supported Devices
===================================

DPPY does not provide specific decorators for implementing reductions on
SYCL-supported devices.
Examples contain different approaches for calculating reductions using both
device and host.

Examples
--------

Example 1
~~~~~~~~~

This example demonstrates a sum reduction of 1d array.

Full example can be found at ``numba_dppy/examples/sum_reduction.py``.

In this example to sum the 1d array we invoke the Kernel multiple times.

.. literalinclude:: ../../numba_dppy/examples/sum_reduction.py
   :pyobject: sum_reduction_kernel

.. literalinclude:: ../../numba_dppy/examples/sum_reduction.py
   :pyobject: sum_reduce

Example 2
~~~~~~~~~

Full example can be found at ``numba_dppy/examples/sum_reduction_ocl.py``.

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_ocl.py
   :pyobject: sum_reduction_kernel

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_ocl.py
   :pyobject: sum_reduce

Example 3
~~~~~~~~~

Full example can be found at
``numba_dppy/examples/sum_reduction_recursive_ocl.py``.

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_reduction_kernel

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_recursive_reduction

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_reduce

Limitations
-----------

Examples show invocation of the kernel multiple times.
Reduction could be implemented by invoking the kernel once, but that requires
support for local device memory and barrier, which is a work in progress.

Transition from Numba CUDA
--------------------------

DPPY does not provide ``@reduce`` decorator for writing a reduction algorithm
for SYCL-supported devices.

See also
--------

Examples:

- ``numba_dppy/examples/sum_reduction.py``
- ``numba_dppy/examples/sum_reduction_ocl.py``
- ``numba_dppy/examples/sum_reduction_recursive_ocl.py``
