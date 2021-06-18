Backtrace
==========

``backtrace``
-------------

Let's consider the work of the command ``backtrace`` in the following example ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:

.. note::

    Known issues:
        - Debug of the first line of the kernel and functions works out twice. See :ref:`single_stepping`.

Run debugger:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_dppy_func.py

Next we see the call stack from the kernel and the nested function:

.. code-block:: bash

    (gdb) break simple_dppy_func.py:28
    Breakpoint 1 (simple_dppy_func.py:28) pending.
    (gdb) run
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) backtrace
    #0  __main__::kernel_sum () at simple_dppy_func.py:28
    (gdb) step
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) step
    29          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) step
    __main__::func_sum () at simple_dppy_func.py:22
    22          result = a_in_func + b_in_func
    (gdb) backtrace
    #0  __main__::func_sum () at simple_dppy_func.py:22
    #1  __main__::kernel_sum () at simple_dppy_func.py:29
    (gdb) continue
    Continuing.

See also:

    - `Backtraces in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Backtrace.html#Backtrace>`_
