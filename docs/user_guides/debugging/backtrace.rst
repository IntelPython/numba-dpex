Backtrace
==========

Let's consider the work of the command ``backtrace`` in the following example ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:


Run GDB debugger:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_dppy_func.py

The call stack from the kernel and the nested function:

.. code-block:: bash

    (gdb) break simple_dppy_func.py:14
    Breakpoint 1 (simple_dppy_func.py:14) pending.
    (gdb) run
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:14
    14          i = dppy.get_global_id(0)
    (gdb) backtrace
    #0  __main__::kernel_sum () at simple_dppy_func.py:14
    (gdb) step
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at simple_dppy_func.py:14
    14          i = dppy.get_global_id(0)
    (gdb) step
    15          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) step
    __main__::func_sum () at simple_dppy_func.py:8
    8          result = a_in_func + b_in_func
    (gdb) backtrace
    #0  __main__::func_sum () at simple_dppy_func.py:8
    #1  __main__::kernel_sum () at simple_dppy_func.py:15
    (gdb) continue
    Continuing.

See also:

    - `Backtraces in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Backtrace.html#Backtrace>`_
