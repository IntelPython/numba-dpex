Backtrace
==========

The ``backtrace`` command displays a summary of how your program got where it
is. Consider the following example
``numba_dpex/examples/debug/simple_dpex_func.py``:

.. literalinclude:: ../../../numba_dpex/examples/debug/simple_dpex_func.py
    :lines: 15-
    :linenos:
    :lineno-match:


The section presents two examples of using Intel Distribution for GDB* to
generate backtrace from a numa_dpex.kernel function. The first example presents
the case where the kernel function does not invoke any other function. The
second example presents the case where the kernel function invokes a
numba_dpex.func.

Example 1:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/backtrace_kernel
    :language: shell-session
    :emphasize-lines: 8,9

Example 2:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/backtrace
    :language: shell-session
    :emphasize-lines: 8-10

See also:

    - `Backtraces in GDB*
      <https://sourceware.org/gdb/current/onlinedocs/gdb/Backtrace.html#Backtrace>`_
