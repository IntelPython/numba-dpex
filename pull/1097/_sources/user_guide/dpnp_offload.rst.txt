.. include:: ./../ext_links.txt

Compiling and Offloading ``dpnp`` Functions
===========================================

Data Parallel Extension for NumPy* (``dpnp``) is a drop-in ``NumPy*``
replacement library built on top of oneMKL. ``numba-dpex`` allows various
``dpnp`` library functions to be jit-compiled thorugh its ``dpjit`` decorator.

``numba-dpex`` implements its own runtime library to support offloading ``dpnp``
library functions to SYCL devices. For ``dpnp`` function signatures that are
offloaded, ``numba-dpex`` implements their corresponding function calls through
Numba*'s |numba.extending.overload|_ and |numba.extending.intrinsic|_
constructs.

During compiling a Python function decorated with the ``numba_dpex.dpjit``
decorator, ``numba-dpex`` generates ``dpnp`` function calls through its runtime
library and injects them into the LLVM IR through |numba.extending.intrinsic|_.

.. code-block:: python

    import dpnp
    from numba_dpex import dpjit


    @dpjit
    def foo():
        return dpnp.ones(10)  # the function call for this signature
        # will be generated through the runtime
        # library and inlined into the LLVM IR


    a = foo()
    print(a)
    print(type(a))

:samp:`dpnp.ones(10)` will be called through |ol_dpnp_ones(...)|_.

The following sections go over as aspects of the dpnp integration inside
numba-dpex.

Repository map
--------------

- The code for numba-dpex's dpnp integration runtime resides in the
  :file:`numba_dpex/core/runtime` sub-module.
- All the |numba.extending.overload|_ for ``dpnp`` function signatures are
  implemented in :file:`numba_dpex/dpnp_iface/arrayobj.py`
- Tests resides in :file:`numba_dpex/tests/dpjit_tests/dpnp`.

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

.. |numba.extending.overload| replace:: ``numba.extending.overload``
.. |numba.extending.intrinsic| replace:: ``numba.extending.intrinsic``
.. |ol_dpnp_ones(...)| replace:: ``ol_dpnp_ones(...)``

.. _low-level API: https://github.com/IntelPython/dpnp/tree/master/dpnp/backend
.. _`ol_dpnp_ones(...)`: https://github.com/IntelPython/numba-dpex/blob/main/numba_dpex/dpnp_iface/arrayobj.py#L358
.. _`numba.extending.overload`: https://numba.pydata.org/numba-doc/latest/extending/high-level.html#implementing-functions
.. _`numba.extending.intrinsic`: https://numba.pydata.org/numba-doc/latest/extending/high-level.html#implementing-intrinsics
