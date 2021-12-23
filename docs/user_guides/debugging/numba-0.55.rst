Debugging Features in Numba 0.55
================================

Numba 0.55 enables following features:

Added ``info args``
-------------------

See :ref:`info-args`.

Extended ``info locals``
------------------------

See :ref:`info-locals`.

Breakpoint with condition by function argument
----------------------------------------------

Test :file:`numba_dppy/tests/debugging/test_breakpoints.py:test_breakpoint_with_condition_by_function_argument`.

When set breakpoint on the function or the first line of the function
than ``info locals`` and ``info args`` provide correct values.
It makes it posible to use breakpoint with condition by function argument.

Example
```````

Source code :file:`numba_dppy/examples/debug/side-by-side.py`:

.. literalinclude:: ../../../numba_dppy/examples/debug/side-by-side.py
    :pyobject: common_loop_body
    :linenos:
    :lineno-match:
    :emphasize-lines: 2

Set breakpoint with condition by function argument:

.. code-block:: shell-session
    :emphasize-lines: 3

    $ NUMBA_OPT=0 gdb-oneapi -q python
    ...
    (gdb) break side-by-side.py:25 if param_a == 3
    ...
    (gdb) run numba_dppy/examples/debug/side-by-side.py --api=numba-dppy-kernel
    ...
    Thread 2.1 hit Breakpoint 1, with SIMD lane 3, __main__::common_loop_body (param_a=3, param_b=3) at side-by-side.py:25
    25          param_c = param_a + 10  # Set breakpoint here
    (gdb) print param_a
    $1 = 3
