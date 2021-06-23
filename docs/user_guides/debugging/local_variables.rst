Local variables
===============

.. note::

    - :samp:`NUMBA_OPT=0` "no optimization" level - all local variables of the kernel function are available.
    - :samp:`NUMBA_OPT=1` or higher - some variables may be optimized out.

Consider Numba-dppy kernel code :file:`simple_sum.py`

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

``info locals``
---------------

Run GDB debugger:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables
    :lines: 1-9

GDB output on "no optimization" level ``NUMBA_OPT=0``:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables
    :lines: 10-23

GDB output on "O1 optimization" level ``NUMBA_OPT=1``:

.. code-block:: bash

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

    The GDB debugger does not show the local variables ``a``, ``b`` and ``c``, they are optimized out on "O1" optimization level.

.. note::

    Known issues:
      - GDB debugger can show the variable values, but these values may not match the actual value of the referred variables.

``print variable``
------------------

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables
    :lines: 27-28

.. note::

    Known issues:
      - Kernel variables are shown in intermidiate representation view (with "$" sign). The actual values of the variables are currently not available.

``ptype variable``
------------------

Variable type may be printed by the command ``ptype variable`` and ``whatis variable``:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables
    :lines: 29-32

See also:

    - `Local variables in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Frame-Info.html#Frame-Info>`_
