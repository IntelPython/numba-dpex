Local variables
===============

.. note::

    - :samp:`NUMBA_OPT=0` "no optimization" level - all local variables of the kernel function are available.
    - :samp:`NUMBA_OPT=1` or higher - some variables may be optimized out.

Consider `numba-dppy` kernel code :file:`simple_sum.py`

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

``info locals``
---------------

Run debugger:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_sum.py
    (gdb) break simple_sum.py:8
    No source file named simple_sum.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (simple_sum.py:8) pending.
    (gdb) run
    (gdb) info locals

GDB output on "no optimization" level ``NUMBA_OPT=0``:

.. code-block:: bash

    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:8
    8           i = dppy.get_global_id(0)

    a = "@\001\000\000\000\000\000\000 \000\000\000\000\000\000\000@A\177YUU\000\000\030\a\200YUU\000\000\000\000\000\000U\003\000\000\241\000\000\000\000\000\000\000\240C\177YUU\000"
    b = "\000\003\000\000\000\000\000\000 \000\000\000\000\000\000\000\200B\177YUU\000\000\060\b\200YUU\000\000\000\000\000\000U\003\000\000!\000\000\000\000\000\000\000\060\304\357YUU\000"
    c = "\000d\356Y\001\000`\000(\n@&\240\001", '\000' <repeats 15 times>, "U\000\000p\025\000\000\000\000\000\000\000\000\356YUU\000\000\000i\356Y\001\000`"
    i = 0
    __ocl_dbg_gid0 = 4
    __ocl_dbg_gid1 = 2
    __ocl_dbg_gid2 = 2
    __ocl_dbg_lid0 = 0
    __ocl_dbg_lid1 = 7
    __ocl_dbg_lid2 = 0
    __ocl_dbg_grid0 = 5
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0

GDB output on "O1 optimization" level ``NUMBA_OPT=1``:

.. code-block:: bash

    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:8
    8           i = dppy.get_global_id(0)
    (gdb) info locals
    __ocl_dbg_gid0 = 0
    __ocl_dbg_gid1 = 0
    __ocl_dbg_gid2 = 0
    __ocl_dbg_lid0 = 0
    __ocl_dbg_lid1 = 0
    __ocl_dbg_lid2 = 0
    __ocl_dbg_grid0 = 0
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0
    i = 0

.. note::

    The debugger does not show the local variables ``a``, ``b`` and ``c``, they are optimized out on "O1" optimization level.

.. note::

    Known issues:
      - Debugger can show the variable values, but these values may not match the actual value of the referred variables.

``print variable``
------------------

.. code-block:: bash

    (gdb) print a
    $1 = "\000\f\335YUU\000\000\260\200\250YUU\000\000*.\367!\n\026\364?\000\000\000\000\000\000\000\000\330|\177YUU\000\000Q\002\000\000\000\000\000\000\000\217\250YUU\000"

    (gdb) print i
    $1 = 93823560581120

.. note::

    Known issues:
      - Kernel variables are shown in intermidiate representation view (with "$" sign). The actual values of the variables are currently not available.

``ptype variable``
------------------

Variable type may be printed by the command ``ptype variable`` and ``whatis variable``:

.. code-block:: bash

    (gdb) ptype i
    type = i64

    (gdb) whatis i
    type = i64

    (gdb) whatis a
    type = byte [56]

    (gdb) ptype a
    type = byte [56]

See also:

    - `Local variables in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Frame-Info.html#Frame-Info>`_
