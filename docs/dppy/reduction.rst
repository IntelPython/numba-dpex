GPU Reduction
=============

DPPY does not provide specific features for implementing reductions on GPU.
Examples contain different approaches for calculating reductions using both
device and host.

Example 1:

.. literalinclude:: ../../numba_dppy/examples/sum_reduction.py
   :pyobject: reduction_kernel

.. literalinclude:: ../../numba_dppy/examples/sum_reduction.py
   :pyobject: sum_reduce

Example 2:

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_ocl.py
   :pyobject: reduction_kernel

.. literalinclude:: ../../numba_dppy/examples/sum_reduction_ocl.py
   :pyobject: sum_reduce


Transition from Numba CUDA
--------------------------

DPPY does not provide ``@reduce`` decorator for writing a reduction algorithm
for DPPY GPU.

See also
--------

Examples:

- ``numba_dppy/examples/sum_reduction.py``
- ``numba_dppy/examples/sum_reduction_ocl.py``
- ``numba_dppy/examples/sum_reduction_recursive_ocl.py``
