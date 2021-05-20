DPNP integration
================

Currently ``numba-dppy`` uses `DPNP backend library`_.

Integration with `DPNP backend library`_
----------------------------------------

``numba-dppy`` replaces ``NumPy`` function calls with ``DPNP`` function calls.

.. code::

    import numpy as np
    from numba import njit
    import dpctl

    @njit
    def foo(a):
      return np.sum(a)

    a = np.arange(42)

    with dpctl.device_context():
      result = foo(a)

    print(result)

``np.sum(a)`` will be replaced with `dpnp_sum_c<int, int>(...) <https://github.com/IntelPython/dpnp/blob/ef404c0f284b0c508ed1e556e140f02f76ae5551/dpnp/backend/kernels/dpnp_krnl_reduction.cpp#L58>`_.

Repository map
``````````````

- Code for integration is mostly resides in `numba_dppy/dpnp_glue`_.
- Tests resides in `numba_dppy/tests/njit_tests/dpnp`_.
- Helper pass resides in `numba_dppy/rename_numpy_functions_pass.py`_.

Architecture
````````````

``numba-dppy`` modifies default ``Numba`` compiler pipeline and extends it with
``DPPYRewriteOverloadedNumPyFunctions`` pass.

The main work is performed in ``RewriteNumPyOverloadedFunctions`` used by the pass.
It rewrites call for ``NumPy`` function in following way:

.. code::

    np.sum(a) -> numba_dppy.dpnp.sum(a)

``numba_dppy.dpnp`` contains stub classes like following:

.. code::

    # numba_dppy/dpnp_glue/stubs.py - imported in numba_dppy.__init__.py

    class dpnp(Stub):

      class sum(Stub):
        pass

For the stub function call to be lowered with ``Numba`` compiler pipeline there
is overload in `numba_dppy/dpnp_glue/dpnp_transcendentalsimpl.py`_:

.. code::

    @overload(stubs.dpnp.sum)
    def dpnp_sum_impl(a):
      ...

Overload implementation knows about DPNP functions.
It receives DPNP function pointer and uses known signature from DPNP headers.
The implementation calls DPNP function via ``ctypes`` supported by ``Numba``.

For more details about overloads implementation see `Writing overload for stub function`_.

Pleces to update
````````````````

1. `numba_dppy/dpnp_glue/stubs.py`_: Add new class to ``stubs.dpnp`` class.
2. `numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx`_: Update items in ``DPNPFuncName`` enum.
3. `numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx`_: Update if statements in ``get_DPNPFuncName_from_str()`` function.
4. `numba_dppy/rename_numpy_functions_pass.py`_: Update items in ``rewrite_function_name_map`` dict.
5. Add ``@overload(stubs.dpnp.YOUR_FUNCTION))`` in one of the `numba_dppy/dpnp_glue/*.py`_ modules or create new.
6. Add test in one of the `numba_dppy/tests/njit_tests/dpnp`_ test modules or create new.

Writing overload for stub function
``````````````````````````````````

``@overload(stubs.dpnp.YOUR_FUNCTION))``



Types matching for Numba and DPNP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- [const] T* -> types.voidptr
- size_t -> types.intp
- long -> types.int64

We are using void * in case of size_t * as Numba currently does not have
any type to represent size_t *. Since, both the types are pointers,
if the compiler allows there should not be any mismatch in the size of
the container to hold different types of pointer.

Troubleshooting
```````````````

1. Do not forget build ``numba-dppy`` with current installed version of ``DPNP``.
   There is headers dependency.
2. Do not forget add array to ``dpnp_ext._dummy_liveness_func([YOUR_ARRAY.size])``.
   Dead code elimination could delete temporary variables before they are used for DPNP function call.
   As a result wrong data could be passed to DPNP function.


.. _`DPNP backend library`: https://github.com/IntelPython/dpnp/tree/master/dpnp/backend
