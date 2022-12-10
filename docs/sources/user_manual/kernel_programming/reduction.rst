Reductions
==========

``numba-dpex`` does not provide a decorator to implement reduction kernels.
However, it provides several essential examples how to write a reduction kernel explicitly. This
section provides two approaches for writing a reduction kernel as a
``numba_dpex.kernel`` function.

Example 2
---------

Full example can be found at
``numba_dpex/examples/sum_reduction_recursive_ocl.py``.

.. literalinclude:: ./../../../../numba_dpex/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_reduction_kernel

.. literalinclude:: ./../../../../numba_dpex/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_recursive_reduction

.. literalinclude:: ./../../../../numba_dpex/examples/sum_reduction_recursive_ocl.py
   :pyobject: sum_reduce

.. note::

    Numba-dpex does not yet provide any analogue to the ``numba.cuda.reduce``
    decorator for writing reductions kernel. Such a decorator will be added in
    future releases.

Full examples
-------------

- ``numba_dpex/examples/sum_reduction_recursive_ocl.py``
- ``numba_dpex/examples/sum_reduction_ocl.py``
- ``numba_dpex/examples/sum_reduction.py``
