.. _dpnp-integration:

DPNP integration
================

Currently `numba-dppy` uses `DPNP backend library`_.

.. _`DPNP backend library`: https://github.com/IntelPython/dpnp/tree/master/dpnp/backend

.. _integration-dpnp-backend:

Integration with `DPNP backend library`_
----------------------------------------

`numba-dppy` replaces `NumPy` function calls with `DPNP` function calls.

.. code-block:: python

    import numpy as np
    from numba import njit
    import dpctl

    @njit
    def foo(a):
      return np.sum(a)  # this call will be replaced with DPNP function

    a = np.arange(42)

    with dpctl.device_context():
      result = foo(a)

    print(result)

:samp:`np.sum(a)` will be replaced with `dpnp_sum_c<int, int>(...)`_.

.. _`dpnp_sum_c<int, int>(...)`: https://github.com/IntelPython/dpnp/blob/ef404c0f284b0c508ed1e556e140f02f76ae5551/dpnp/backend/kernels/dpnp_krnl_reduction.cpp#L58

.. _dpnp-integration-repository-map:

Repository map
``````````````

- Code for integration is mostly resides in :file:`numba_dppy/dpnp_glue`.
- Tests resides in :file:`numba_dppy/tests/njit_tests/dpnp`.
- Helper pass resides in :file:`numba_dppy/rename_numpy_functions_pass.py`.

.. _dpnp-integration-architecture:

Architecture
````````````

`numba-dppy` modifies default `Numba` compiler pipeline and extends it with
:class:`DPPYRewriteOverloadedNumPyFunctions` pass.

The main work is performed in :class:`RewriteNumPyOverloadedFunctions` used by the pass.
It rewrites call for `NumPy` function in following way:

    :samp:`np.sum(a)` -> :samp:`numba_dppy.dpnp.sum(a)`

:mod:`numba_dppy.dpnp` contains stub functions (defined as classes) like following:

.. code-block:: python

    # numba_dppy/dpnp_glue/stubs.py - imported in numba_dppy.__init__.py

    class dpnp(Stub):

      class sum(Stub):  # stub function
        pass

For the stub function call to be lowered with `Numba` compiler pipeline there
is overload in :file:`numba_dppy/dpnp_glue/dpnp_transcendentalsimpl.py`:

.. code-block:: python

    @overload(stubs.dpnp.sum)
    def dpnp_sum_impl(a):
      ...

Overload implementation knows about `DPNP` functions.
It receives `DPNP` function pointer from `DPNP` and uses known signature from `DPNP` headers.
The implementation calls `DPNP` function via creating `Numba` :class:`ExternalFunctionPointer`.

For more details about overloads implementation see :ref:`overload-for-stub`.

For more details about testing the integration see :ref:`dpnp-integration-tests`.

.. _dpnp-integration-places:

Places to update
````````````````

1. :file:`numba_dppy/dpnp_glue/stubs.py`: Add new class to :class:`stubs.dpnp` class.
2. :file:`numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx`: Update items in :class:`DPNPFuncName` enum.
3. :file:`numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx`: Update if statements in :func:`get_DPNPFuncName_from_str` function.
4. Add :samp:`@overload(stubs.dpnp.{YOUR_FUNCTION})` in one of the :file:`numba_dppy/dpnp_glue/{*}.py` modules or create new.
5. :file:`numba_dppy/rename_numpy_functions_pass.py`: Update items in :obj:`rewrite_function_name_map` dict.
6. :file:`numba_dppy/rename_numpy_functions_pass.py`: Update imported modules in :meth:`DPPYRewriteOverloadedNumPyFunctions.__init__`.
7. Add test in one of the :file:`numba_dppy/tests/njit_tests/dpnp` test modules or create new.

.. _overload-for-stub:

Writing overload for stub function
``````````````````````````````````

Overloads for stub functions resides in :file:`numba_dppy/dpnp_glue/{*}.py` modules.
If you need create new module try to name it corresponding to `DPNP` naming.
I.e. :file:`dpnp/backend/kernels/dpnp_krnl_indexing.cpp` -> :file:`numba_dppy/dpnp_glue/dpnp_indexing.py`.

.. code-block:: python

    from numba.core.extending import overload
    import numba_dppy.dpnp_glue as dpnp_lowering
    ...

    @overload(stubs.dpnp.sum)
    def dpnp_sum_impl(a):
      dpnp_lowering.ensure_dpnp("sum")

:func:`ensure_dpnp` checks that `DPNP` package is available and contains the function.

.. code-block:: python

    from numba import types
    from numba.core.typing import signature
    ...
    # continue of dpnp_sum_impl()
      """
      dpnp source:
      https://github.com/IntelPython/dpnp/blob/0.6.1dev/dpnp/backend/kernels/dpnp_krnl_reduction.cpp#L59

      Function declaration:
      void dpnp_sum_c(void* result_out,
                      const void* input_in,
                      const size_t* input_shape,
                      const size_t input_shape_ndim,
                      const long* axes,
                      const size_t axes_ndim,
                      const void* initial,
                      const long* where)

      """
      sig = signature(
          types.void,  # return type
          types.voidptr,  # void* result_out,
          types.voidptr,  # const void* input_in,
          types.voidptr,  # const size_t* input_shape,
          types.intp,  # const size_t input_shape_ndim,
          types.voidptr,  # const long* axes,
          types.intp,  # const size_t axes_ndim,
          types.voidptr,  # const void* initial,
          types.voidptr,  # const long* where)
      )

Signature :obj:`sig` is based on the `DPNP` function signature defined in header file.
It is recommended to provide link to signature in `DPNP` sources and copy it in comment
as shown above.

For mapping between `C` types and `Numba` types see :ref:`dpnp-integration-types-matching`.

.. code-block:: python

    import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
    ...
    # continue of dpnp_sum_impl()
      dpnp_func = dpnp_ext.dpnp_func("dpnp_sum", [a.dtype.name, "NONE"], sig)

:func:`dpnp_ext.dpnp_func` returns function pointer from `DPNP`.
It receives:

- Function name (i.e. :samp:`"dpnp_sum"`) which is converted to
  :class:`DPNPFuncName` enum in :func:`get_DPNPFuncName_from_str()`.
- List of input and output data types names
  (i.e. :samp:`[a.dtype.name, "NONE"]`, :samp:`"NONE"` means reusing previous type name)
  which is converted to :class:`DPNPFuncType` enum in :func:`get_DPNPFuncType_from_str()`.
- Signature which is used for creating `Numba` :class:`ExternalFunctionPointer`.

.. code-block:: python

    import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
    ...
    # continue of dpnp_sum_impl()
      PRINT_DEBUG = dpnp_lowering.DEBUG

      def dpnp_impl(a):
          out = np.empty(1, dtype=a.dtype)
          common_impl(a, out, dpnp_func, PRINT_DEBUG)

          return out[0]

      return dpnp_impl

This code created implementation function and returns it from the overload function.

:obj:`PRINT_DEBUG` used for printing debug information which is used in tests.
Tests rely on debug information to check that DPNP implementation was used.
See :ref:`dpnp-integration-tests`.

:func:`dpnp_impl` creates output array with size and data type corresponding
to `DPNP` function output array.

:func:`dpnp_impl` could call `NumPy` functions supported by `Numba` and
other stab functions (i.e. :func:`numba_dppy.dpnp.dot`).

The implementation function usually reuse a common function like :func:`common_impl`.
This approach eliminates code duplication.
You should consider all available common functions at the top of the file before
creating the new one.

.. code-block:: python

    from numba.core.extending import register_jitable
    from numba_dppy import dpctl_functions
    import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
    ...

    @register_jitable
    def common_impl(a, out, dpnp_func, print_debug):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()
        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)  # 1
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)  # 2

        out_usm = dpctl_functions.malloc_shared(a.itemsize, sycl_queue)  # 1

        axes, axes_ndim = 0, 0
        initial = 0
        where = 0

        dpnp_func(out_usm, a_usm, a.shapeptr, a.ndim, axes, axes_ndim, initial, where)  # 3

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )  # 4

        dpctl_functions.free_with_queue(a_usm, sycl_queue)  # 5
        dpctl_functions.free_with_queue(out_usm, sycl_queue)  # 5

        dpnp_ext._dummy_liveness_func([a.size, out.size])  # 6

        if print_debug:
            print("dpnp implementation")  # 7

Key parts of any common function are:

1. Allocate input and output USM arrays
2. Copy input array to input USM array
3. Call :func:`dpnp_func`
4. Copy output USM array to output array
5. Deallocate USM arrays
6. Disable dead code elimination for input and output arrays
7. Print debug information used for testing

.. _dpnp-integration-types-matching:

Types matching for Numba and DPNP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :samp:`[const] {T}*` -> :obj:`types.voidptr`
- `size_t` -> :obj:`types.intp`
- `long` -> :obj:`types.int64`

We are using `void *` in case of `size_t *` as `Numba` currently does not have
any type to represent `size_t *`.
Since, both the types are pointers, if the compiler allows there should not be
any mismatch in the size of the container to hold different types of pointer.

.. _dpnp-integration-tests:

Writing `DPNP` integration tests
````````````````````````````````

See all `DPNP` integration tests in :file:`numba_dppy/tests/njit_tests/dpnp`.

Usually adding new test is as easy as adding function name to the corresponding list of function names.
Each item in the list is used as a parameter for tests.
You should find tests for the category of functions similar to your function and
update a list with function names like :obj:`list_of_unary_ops`, :obj:`list_of_nan_ops`.

.. code-block:: python

    def test_unary_ops(filter_str, unary_op, input_array, get_shape, capfd):
      if skip_test(filter_str):
          pytest.skip()

      a = input_array  # 1
      a = np.reshape(a, get_shape)
      op, name = unary_op  # 2
      if (name == "cumprod" or name == "cumsum") and (
          filter_str == "opencl:cpu:0" or is_gen12(filter_str)
      ):
          pytest.skip()
      actual = np.empty(shape=a.shape, dtype=a.dtype)
      expected = np.empty(shape=a.shape, dtype=a.dtype)

      f = njit(op)  # 3
      with dpctl.device_context(filter_str), dpnp_debug():  # 7
          actual = f(a)  # 4
          captured = capfd.readouterr()
          assert "dpnp implementation" in captured.out  # 8

      expected = op(a)  # 5
      max_abs_err = np.sum(actual - expected)
      assert max_abs_err < 1e-4  # 6

Test functions starts from :samp:`test_` (see `pytest` docs) and
all input parameters are provided by fixtures.

In example above :obj:`unary_op` contains tuple :samp:`({FUNCTION}, {FUNCTION_NAME})`,
see fixture :func:`unary_op`.

Key parts of any test are:

1. Receive input array from the fixture :obj:`input_array`
2. Receive the tested function from fixture :obj:`unary_op`
3. Compile the tested function with :func:`njit`
4. Call the compiled tested function inside :func:`device_context` device_context
   and receive :obj:`actual` result
5. Call the original tested function and receive :obj:`expected` result
6. Compare :obj:`actual` and :obj:`expected` result
7. Run the compiled test function inside debug contex :func:`dpnp_debug`
8. Check that `DPNP` was usede as debug information was printed to output

.. _dpnp-troubleshooting:

Troubleshooting
```````````````

1. Do not forget build `numba-dppy` with current installed version of `DPNP`.
   There is headers dependency in `Cython` files (i.e. :file:`numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx`).
2. Do not forget add array to :samp:`dpnp_ext._dummy_liveness_func([{YOUR_ARRAY}.size])`.
   Dead code elimination could delete temporary variables before they are used for `DPNP` function call.
   As a result wrong data could be passed to `DPNP` function.
