.. include:: ./../ext_links.txt

Compiling and Offloading ``dpnp`` Functions
===========================================

Data Parallel Extension for NumPy* (``dpnp``) is a drop-in ``NumPy*``
replacement library built on top of oneMKL. ``numba-dpex`` allows various
``dpnp`` library function calls to be jit-compiled thorugh its
``numba_dpex.dpjit`` decorator.

``numba-dpex`` implements its own runtime library to support offloading ``dpnp``
library functions to SYCL devices. For each ``dpnp`` function signature to be
offloaded, ``numba-dpex`` implements the corresponding direct SYCL function call
in the runtime and the function call is inlined in the generated LLVM IR.

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

- The code for numba-dpex's ``dpnp`` integration runtime resides in the
  :file:`numba_dpex/core/runtime` sub-module.
- All the |numba.extending.overload|_ for ``dpnp`` array creation/initialization
  function signatures are implemented in
  :file:`numba_dpex/dpnp_iface/arrayobj.py`
- Each overload's corresponding |numba.extending.intrinsic|_ is implemented in
  :file:`numba_dpex/dpnp_iface/_intrinsic.py`
- Tests resides in :file:`numba_dpex/tests/dpjit_tests/dpnp`.

Design
------

``numba_dpex`` uses the |numba.extending.overload| decorator to create a Numba*
implementation of a function that can be used in `nopython mode`_ functions.
This is done through translation of ``dpnp`` function signature so that they can
be called in ``numba_dpex.dpjit`` decorated code.

The specific SYCL operation for a certain ``dpnp`` function is performed by the
runtime interface. During compiling a function decorated with the ``@dpjit``
decorator, ``numba-dpex`` generates the corresponding SYCL function call through
its runtime library and injects it into the LLVM IR through
|numba.extending.intrinsic|_. The ``@intrinsic`` decorator is used for marking a
``dpnp`` function as typing and implementing the function in nopython mode using
the `llvmlite IRBuilder API`_. This is an escape hatch to build custom LLVM IR
that will be inlined into the caller.

The code injection logic to enable ``dpnp`` functions calls in the Numba IR is
implemented by :mod:`numba_dpex.core.dpnp_iface.arrayobj` module which replaces
Numba*'s :mod:`numba.np.arrayobj`. Each ``dpnp`` function signature is provided
with a concrete implementation to generates the actual code using Numba's
``overload`` function API. e.g.:

.. code-block:: python

    @overload(dpnp.ones, prefer_literal=True)
    def ol_dpnp_ones(
        shape, dtype=None, order="C", device=None, usm_type="device", sycl_queue=None
    ):
        ...

The corresponding intrinsic implementation is in :file:`numba_dpex/dpnp_iface/_intrinsic.py`.

.. code-block:: python

   @intrinsic
   def impl_dpnp_ones(
       ty_context,
       ty_shape,
       ty_dtype,
       ty_order,
       ty_device,
       ty_usm_type,
       ty_sycl_queue,
       ty_retty_ref,
   ):
       ...

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
.. |numba.np.arrayobj| replace:: ``numba.np.arrayobj``

.. _low-level API: https://github.com/IntelPython/dpnp/tree/master/dpnp/backend
.. _`ol_dpnp_ones(...)`: https://github.com/IntelPython/numba-dpex/blob/main/numba_dpex/dpnp_iface/arrayobj.py#L358
.. _`numba.extending.overload`: https://numba.pydata.org/numba-doc/latest/extending/high-level.html#implementing-functions
.. _`numba.extending.intrinsic`: https://numba.pydata.org/numba-doc/latest/extending/high-level.html#implementing-intrinsics
.. _nopython mode: https://numba.pydata.org/numba-doc/latest/glossary.html#term-nopython-mode
.. _`numba.np.arrayobj`: https://github.com/numba/numba/blob/main/numba/np/arrayobj.py
.. _`llvmlite IRBuilder API`: http://llvmlite.pydata.org/en/latest/user-guide/ir/ir-builder.html
