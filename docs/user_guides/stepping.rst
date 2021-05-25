GDB stepping
===========================

.. note::
    - Known issues:  
    - Stepping works correctly only for kernels. Nested functions can cause problems.

Consider ``numba-dppy`` kernel code:

.. code-block:: python
    :linenos:

    import numpy as np
    import numba_dppy as dppy
    import dpctl
        
    @dppy.kernel
    def data_parallel_sum (a, b, c):
        i = dppy.get_global_id (0)  # numba-kernel-breakpoint
        l1 = a[i]                   # second-line
        l2 = b[i]                   # third-line
        c[i] = l1 + l2              # fourth-line
    
    global_size = 10
    N = global_size
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)
    with dpctl.device_context("opencl:gpu") as gpu_queue:
        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
        
``step``
---------------

.. code-block:: bash

    export NUMBA_DPPY_DEBUG=1  
    export NUMBA_OPT=1  
    gdb-oneapi -q --args python stepping.py
    (gdb) b stepping.py:7
    No source file named stepping.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (stepping.py:7) pending.
    (gdb) r
    Starting program: /localdisk/work/etotmeni/miniconda3/envs/numba-dppy-docs/bin/python stepping.py
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    [Detaching after fork from child process 10505]
    [New Thread 0x7fffd601f700 (LWP 10513)]
    intelgt: gdbserver-gt started for process 10491.
    intelgt: attached to device 1 of 1; id 0x5927 (Gen9)
    [New Thread 0x7fffc593d700 (LWP 10530)]
    [Detaching after fork from child process 10531]
    compile_kernel (array(float32, 1d, C), array(float32, 1d, C), array(float32, 1d, C))
    [Detaching after fork from child process 10532]
    [Detaching after fork from child process 10533]
    [New Thread 1.1073741824]
    [New Thread 1.1073742080]
    [Switching to Thread 1.1073741824 lane 0]

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], dppy_py_devfn__5F__5F_main_5F__5F__2E_data_5F_parallel_5F_sum_24_1_2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29_ () at stepping.py:7
    7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    (gdb) s

**GDB output**

.. code-block:: bash

    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], dppy_py_devfn__5F__5F_main_5F__5F__2E_data_5F_parallel_5F_sum_24_1_2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29_ () at stepping.py:7
    7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    (gdb) s
    8           l1 = a[i]                   # second-line
    (gdb) s
    9           l2 = b[i]                   # third-line
    (gdb) s
    10          c[i] = l1 + l2              # fourth-line
    (gdb) s
    [Thread 0x7fffc593d700 (LWP 10530) exited]
    [Thread 0x7ffff7fd1740 (LWP 10491) exited]
    [Inferior 2 (process 1) exited normally]
    intelgt: inferior 2 (gdbserver-gt) has been removed.

.. note::

    - Known issues:  
    - Debug of the first line of the kernel works out twice.

``next``
------------------

Stepping-like behavior.
