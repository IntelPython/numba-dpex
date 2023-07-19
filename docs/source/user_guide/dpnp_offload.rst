.. include:: ./../ext_links.txt

Compiling and Offloading ``dpnp`` Functions
===========================================

Data Parallel Extension for NumPy* (``dpnp``) is a drop-in ``NumPy*``
replacement library. The library is developed using SYCL and oneMKL.
``numba-dpex`` relies on ``dpnp`` to support offloading ``NumPy`` library
functions to SYCL devices. For ``NumPy`` functions that are offloaded using
``dpnp``, ``numba-dpex`` generates library calls directly to ``dpnp``'s
`low-level API`_ inside the generated LLVM IR.

.. _low-level API: https://github.com/IntelPython/dpnp/tree/master/dpnp/backend

.. _integration-dpnp-backend:

During compiling a Python function decorated with the ``numba.njit`` decorator,
``numba-dpex`` substitutes ``NumPy`` function calls with corresponding ``dpnp``
low-level API function calls. The substitution happens transparent to an
end-user and is implemented as a renaming pass in ``numba-dpex``'s pass
pipeline.

.. code-block:: python

    import numpy as np
    from numba import njit
    import dpctl


    @njit
    def foo(a):
        return np.sum(a)  # this call will be replaced with the dpnp.sum function


    a = np.arange(42)

    with dpctl.device_context():
        result = foo(a)

    print(result)

:samp:`np.sum(a)` will be replaced with `dpnp_sum_c<int, int>(...)`_.

.. _`dpnp_sum_c<int, int>(...)`: https://github.com/IntelPython/dpnp/blob/ef404c0f284b0c508ed1e556e140f02f76ae5551/dpnp/backend/kernels/dpnp_krnl_reduction.cpp#L58

The following sections go over as aspects of the dpnp integration inside
numba-dpex.

.. _dpnp-integration-repository-map:

Repository map
--------------

- The code for numba-dpex's dpnp integration resides in the
  :file:`numba_dpex/dpnp_iface` sub-module.
- Tests resides in :file:`numba_dpex/tests/njit_tests/dpnp`.
- Helper pass resides in :file:`numba_dpex/rename_numpy_functions_pass.py`.

.. _dpnp-integration-architecture:

Design
------

The rewrite logic to substitute NumPy functions with dpnp function calls in the
Numba IR is implemented by the :class:`RewriteOverloadedNumPyFunctionsPass`
pass. The :mod:`numba_dpex.dpnp_iface.stubs` module defines a set of `stub`
classes for each of the NumPy functions calls that are currently substituted
out. The outline of a stub class is as follows:

.. code-block:: python

    # numba_dpex/dpnp_iface/stubs.py - imported in numba_dpex.__init__.py


    class dpnp(Stub):
        class sum(Stub):  # stub function
            pass

Each stub is provided with a concrete implementation to generates the actual
code using Numba's ``overload`` function API. E.g.,

.. code-block:: python

    @overload(stubs.dpnp.sum)
    def dpnp_sum_impl(a):
        ...

The complete implementation is in
:file:`numba_dpex/dpnp_iface/dpnp_transcendentalsimpl.py`.

Parallel Range
--------------

``numba-dpex`` implements the ability to run loops in parallel,
similar to OpenMP parallel for loops and Numba*’s ``prange``. The loop-
body is scheduled in seperate threads, and they execute in a ``nopython`` numba
context. ``prange`` automatically takes care of data privatization:



- prange, reduction prange
- blackscholes, math example
