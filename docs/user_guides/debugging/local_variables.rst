Local variables
===========================

.. note::
    - ``NUMBA_OPT=0`` "no optimization" level - all local variables of the kernel function are available.

    - ``NUMBA_OPT=1`` or higher - some variables can be optimized out.
    
.. note::
    - Known issues:  
    - ``NUMBA_OPT=0`` "no optimization" level may not work due to llvm issues.

Consider ``numba-dppy`` kernel code:
    
.. code-block:: python
    :linenos:

    import numpy as np
    import numba_dppy as dppy
    import dpctl


    @dppy.kernel
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 10
    N = global_size
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    with dpctl.device_context("opencl:gpu") as gpu_queue:
        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

``info locals``
---------------

.. code-block:: bash

    export NUMBA_DPPY_DEBUG=True  
    export NUMBA_OPT=0  
    gdb-oneapi -q --args python local_vars.py  
    (gdb) break local_vars.py:9  
    No source file named local_vars_ex.py.  
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (local_vars_ex.py:9) pending.
    (gdb) run
    (gdb) info locals

**GDB output**

.. code-block:: bash

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], dppy_py_devfn__5F__5F_main_5F__5F__2E_data_5F_parallel_5F_sum_24_1_2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29_ () at local_vars.py:9
    9           c[i] = a[i] + b[i]
    (gdb) info locals
    a = '\000' <repeats 55 times>
    b = '\000' <repeats 55 times>
    c = '\000' <repeats 55 times>
    __ocl_dbg_gid0 = 0
    __ocl_dbg_gid1 = 0
    __ocl_dbg_gid2 = 0
    __ocl_dbg_lid0 = 0
    __ocl_dbg_lid1 = 0
    __ocl_dbg_lid2 = 93825017857072
    __ocl_dbg_grid0 = 0
    __ocl_dbg_grid1 = 0
    __ocl_dbg_grid2 = 0

.. note::

    - Known issues:  
    - Representation of local variables values is currently not available.

``print variable``
------------------

.. code-block:: bash

    (gdb) print a

**GDB output**

.. code-block:: bash

    $1 = '\000' <repeats 55 times>

.. note::

    - Known issues:  
    - Kernel variables are shown in IR representation.
