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
