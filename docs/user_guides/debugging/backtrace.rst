Backtrace
==========

Let's consider the work of the command ``backtrace`` in the following example ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:

.. note::

    Known issues:
        - The first line of the kernel and functions is hit twice. See the :ref:`single_stepping`.

Below are examples showing the backtrace for the kernel and for the nested function.
The call stack for the kernel consists of one function (``kernel_sum ()``), and the call colline for the func consists of two functions (``func_sum ()``, ``kernel_sum ()``).
Run debugger and do following commands:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/backtrace

See also:

    - `Backtraces in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Backtrace.html#Backtrace>`_
