Backtrace
==========

Let's consider the work of the command ``backtrace`` in the following example ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:

.. note::

    Known issues:
        - The first line of the kernel and functions is hit twice. See the :ref:`single_stepping`.

The section presents two examples of using GDB to generate backtrace from a numa_dppy.kernel function.
The first example presents the case where the kernel function does not invoke any other function.
The second example presents the case where the kernel function invokes a numba_dppy.func.

Example 1:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/backtrace_kernel

Example 2:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/backtrace

See also:

    - `Backtraces in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Backtrace.html#Backtrace>`_
