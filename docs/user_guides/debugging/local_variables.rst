Local variables
===============

.. note::

    - :samp:`NUMBA_OPT=0` "no optimization" level - all local variables of the kernel function are available.
    - :samp:`NUMBA_OPT=1` or higher - some variables may be optimized out.

Consider `numba-dppy` kernel code from `sum.py` example:

.. code-block:: python
    :linenos:

    import numpy as np
    import numba_dppy as dppy
    import dpctl
        
        
    @dppy.kernel
    def data_parallel_sum(a_in_kernel, b_in_kernel, c_in_kernel):
        i = dppy.get_global_id(0)
        l1 = a_in_kernel[i]
        l2 = b_in_kernel[i]
        c_in_kernel[i] = l1 + l2
        
    global_size = 10
    N = global_size
    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty_like(a) 

    device = dpctl.select_default_device()
    with dpctl.device_context(device):
        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

``info locals``
---------------

Run debugger:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=0
    gdb-oneapi -q --args python sum.py
    (gdb) break sum.py:8
    No source file named sum.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (sum.py:8) pending.
    (gdb) run
    (gdb) info locals

GDB output ``NUMBA_OPT=0``:

.. code-block:: bash

    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at sum.py:8
    8         i = dppy.get_global_id(0)

    a_in_kernel = "\340\354\205YUU\000\000\340\354\205YUU\000\000\001", '\000' <repeats 11 times>, "\001\000\000\000\000\000\000\000UU\000\000 \355\205YUU\000\000 \355\205YUU\000"
    b_in_kernel = "\240\357\205YUU\000\000\001", '\000' <repeats 11 times>, "\001\000\000\000\000\000\000\000\377\177\000\000\340\357\205YUU\000\000\340\357\205YUU\000\000\001\000\000\000\000\000\000"
    c_in_kernel = "\001", '\000' <repeats 11 times>, "\001\000\000\000\000\000\000\000UU\000\000М\205YUU\000\000М\205YUU\000\000\001", '\000' <repeats 11 times>, "\001\000\000"
    i = 1
    l1 = 0
    l2 = 0
    __ocl_dbg_gid0 = 4
    __ocl_dbg_gid1 = 2
    __ocl_dbg_gid2 = 2
    __ocl_dbg_lid0 = 0
    __ocl_dbg_lid1 = 7
    __ocl_dbg_lid2 = 0
    __ocl_dbg_grid0 = 5
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0

.. note::

    Known issues:
      - Debugger can show the variable values, but these values may not match the actual value of the referred variables.


``print variable``
------------------

.. code-block:: bash

    (gdb) print a_in_kernel
    $1 = "\340\354\205YUU\000\000\340\354\205YUU\000\000\001", '\000' <repeats 11 times>, "\001\000\000\000\000\000\000\000UU\000\000 \355\205YUU\000\000 \355\205YUU\000"

    (gdb) print i
    $2 = 1


.. note::

    Known issues:
      - Kernel variables names are shown in IR representation.
