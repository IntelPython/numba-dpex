Local variables
===============

.. note::

    - :samp:`NUMBA_OPT=0` "no optimization" level - all local variables of the kernel function are available.
    - :samp:`NUMBA_OPT=1` or higher - some variables may be optimized out.

Consider Numba-dppy kernel code :file:`sum_local_vars.py`

.. literalinclude:: ../../../numba_dppy/examples/debug/sum_local_vars.py
    :lines: 15-
    :linenos:

``info locals``
---------------

Run GDB debugger:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python sum_local_vars.py
    Reading symbols from python...
    (gdb) break sum_local_vars.py:8
    No source file named sum_local_vars.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (sum_local_vars.py:8) pending.
    (gdb) run

GDB output on "no optimization" level ``NUMBA_OPT=0``:

.. code-block:: bash

    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at sum_local_vars.py:8
    8           i = dppy.get_global_id(0)
    (gdb) info locals

    a = '\000' <repeats 55 times>
    b = '\000' <repeats 55 times>
    c = '\000' <repeats 55 times>
    i = 0
    l1 = 0
    l2 = 0
    __ocl_dbg_gid0 = 0
    __ocl_dbg_gid1 = 0
    __ocl_dbg_gid2 = 0
    __ocl_dbg_lid0 = 0
    __ocl_dbg_lid1 = 0
    __ocl_dbg_lid2 = 0
    __ocl_dbg_grid0 = 0
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0

    (gdb) next
    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at sum_local_vars.py:8
    8           i = dppy.get_global_id(0)
    (gdb) next
    9           l1 = a[i] + 2.5
    (gdb) next
    10          l2 = b[i] * 0.3

    (gdb) info locals
    a = '\000' <repeats 55 times>
    b = '\000' <repeats 16 times>, "\n\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\240\016XUU\000\000\n\000\000\000\000\000\000\000\004\000\000\000\000\000\000"
    c = '\000' <repeats 16 times>, "\n\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000@\256WUU\000\000\n\000\000\000\000\000\000\000\004\000\000\000\000\000\000"
    i = 8
    l1 = 2.5931931659579277
    l2 = 0
    __ocl_dbg_gid0 = 0
    __ocl_dbg_gid1 = 0
    __ocl_dbg_gid2 = 0
    __ocl_dbg_lid0 = 42949672970
    __ocl_dbg_lid1 = 0
    __ocl_dbg_lid2 = 93825037590528
    __ocl_dbg_grid0 = 4612811918334230528
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0

Since GDB debugger does not hit a line with target variable, the value of this variable is equal to 0. The true value of the variable ``l1`` is shown after stepping to line 9.

.. code-block:: bash

    (gdb) next
    11          c[i] = l1 + l2

    (gdb) info locals
    a = '\000' <repeats 55 times>
    b = '\000' <repeats 55 times>
    c = '\000' <repeats 16 times>, "\n\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000@\256WUU\000\000\n\000\000\000\000\000\000\000\004\000\000\000\000\000\000"
    i = 8
    l1 = 2.5931931659579277
    l2 = 0.22954882979393004
    __ocl_dbg_gid0 = 0
    __ocl_dbg_gid1 = 8
    __ocl_dbg_gid2 = 8
    __ocl_dbg_lid0 = 93825034429928
    __ocl_dbg_lid1 = 0
    __ocl_dbg_lid2 = 93825034429936
    __ocl_dbg_grid0 = 4599075939470750515
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0

When GDB debugger hits the last line of the kernel, ``info locals`` command returns all the local variables with their values.

.. note::

    Known issues:
      - GDB debugger can show the variable values, but these values may be equal to 0 after the variable is explicitly deleted or the function scope is ended. For more information refer to `Numba variable policy <https://numba.pydata.org/numba-doc/latest/developer/live_variable_analysis.html?highlight=delete#live-variable-analysis>`_.

GDB output on "O1 optimization" level ``NUMBA_OPT=1``:

.. code-block:: bash

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at sum_local_vars.py:8
    8           i = dppy.get_global_id(0)
    (gdb) info locals
    __ocl_dbg_gid0 = 8
    __ocl_dbg_gid1 = 0
    __ocl_dbg_gid2 = 0
    __ocl_dbg_lid0 = 8
    __ocl_dbg_lid1 = 0
    __ocl_dbg_lid2 = 0
    __ocl_dbg_grid0 = 0
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0
    i = 0
    l1 = 0
    l2 = 0

.. note::

    The GDB debugger does not show the local variables ``a``, ``b`` and ``c``, they are optimized out on "O1" optimization level.

``print variable``
------------------

.. code-block:: bash

    (gdb) print a
    $1 = '\000' <repeats 55 times>

    (gdb) print l1
    $3 = 2.5931931659579277

    (gdb) print l2
    $4 = 0.22954882979393004

.. note::

    Known issues:
      - Kernel variables are shown in intermidiate representation view (with "$" sign). The actual values of the variables are currently not available.

``ptype variable``
------------------

Variable type may be printed by the command ``ptype variable`` and ``whatis variable``:

.. code-block:: bash

    (gdb) ptype a
    type = byte [56]

    (gdb) whatis a
    type = byte [56]

    (gdb) ptype l1
    type = double

    (gdb) whatis l1
    type = double

See also:

    - `Local variables in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Frame-Info.html#Frame-Info>`_
