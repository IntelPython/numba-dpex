Stepping
========

Consider the following two examples. ``sum.py``:

.. code-block:: python
    :linenos:

    import numpy as np
    import numba_dppy as dppy
    import dpctl

    @dppy.kernel
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id (0)  # numba-kernel-breakpoint
        l1 = a[i]                   # second-line
        l2 = b[i]                   # third-line
        c[i] = l1 + l2              # fourth-line

    global_size = 10
    N = global_size
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

    print("Done...")

Example with a nested function ``dppy_func.py``:

.. code-block:: python
    :linenos:

    import numpy as np
    import numba_dppy as dppy
    import dpctl
    
    @dppy.func
    def func_sum(a_in_func, b_in_func):
        result = a_in_func + b_in_func
        return result
    
    @dppy.kernel
    def kernel_sum(a_in_kernel, b_in_kernel, c_in_kernel):
        i = dppy.get_global_id(0)
        c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    
    global_size = 10
    a = np.arange(global_size, dtype=np.float32)
    b = np.arange(global_size, dtype=np.float32)
    c = np.empty_like(a)
    
    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()
    
    with dpctl.device_context(device):
        kernel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
    
    print("Done...")    

``step``
--------

Run debugger:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python sum.py

GDB output:

.. code-block:: bash

    (gdb) b sum.py:7
    No source file named sum.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (sum.py:7) pending.
    (gdb) r
    Starting program: /localdisk/work/etotmeni/miniconda3/envs/numba-dppy-new/bin/python sum.py
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    [Detaching after fork from child process 1691392]
    [Detaching after fork from child process 1691393]
    [Detaching after fork from child process 1691400]
    [New Thread 0x7fffd6e9f700 (LWP 1691402)]
    intelgt: gdbserver-gt started for process 1691381.
    intelgt: attached to device 1 of 1; id 0x9bca (Gen9)
    [New Thread 0x7fffc6027700 (LWP 1691414)]
    [Detaching after fork from child process 1691415]
    [Detaching after fork from child process 1691416]
    [Detaching after fork from child process 1691417]
    [Detaching after fork from child process 1691418]
    [Detaching after fork from child process 1691419]
    Using device ...
        Name            Intel(R) UHD Graphics [0x9bca]
        Driver version  1.1.19883
        Vendor          Intel(R) Corporation
        Profile         FULL_PROFILE
        Filter string   level_zero:gpu:0
    [New Thread 0x7fffb6792700 (LWP 1691426)]
    [New Thread 0x7fffb6391700 (LWP 1691427)]
    [New Thread 0x7fffb5f90700 (LWP 1691428)]
    [New Thread 0x7fffb578e700 (LWP 1691429)]
    [New Thread 0x7fffb538d700 (LWP 1691431)]
    [New Thread 0x7fffb5b8f700 (LWP 1691430)]
    [New Thread 0x7fffb4f8c700 (LWP 1691432)]
    [New Thread 0x7fffb4b8b700 (LWP 1691433)]
    [New Thread 0x7fffb478a700 (LWP 1691434)]
    [New Thread 0x7fffaffff700 (LWP 1691435)]
    [New Thread 0x7fffafbfe700 (LWP 1691436)]
    [Detaching after fork from child process 1691437]
    [Detaching after fork from child process 1691438]
    [New Thread 1.1073741824]
    [New Thread 1.1073742080]
    [Switching to Thread 1.1073741824 lane 0]

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at sum.py:7
    7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    (gdb) s
    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at sum.py:7
    7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    (gdb) s
    8           l1 = a[i]                   # second-line
    (gdb) s
    9           l2 = b[i]                   # third-line
    (gdb) s
    10          c[i] = l1 + l2              # fourth-line
    (gdb) c
    Continuing.
    Done...

Another use of stepping when there is a nested function. Below example:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) b dppy_func.py:12
    No source file named dppy_func.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (dppy_func.py:12) pending.
    (gdb) r
    Starting program: /localdisk/work/etotmeni/miniconda3/envs/numba-dppy-new/bin/python dppy_func.py
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    [Detaching after fork from child process 1707272]
    [Detaching after fork from child process 1707273]
    [Detaching after fork from child process 1707280]
    [New Thread 0x7fffd6e5f700 (LWP 1707282)]
    intelgt: gdbserver-gt started for process 1707262.
    intelgt: attached to device 1 of 1; id 0x9bca (Gen9)
    [New Thread 0x7fffc6027700 (LWP 1707294)]
    [Detaching after fork from child process 1707295]
    [Detaching after fork from child process 1707296]
    [Detaching after fork from child process 1707297]
    [Detaching after fork from child process 1707298]
    [Detaching after fork from child process 1707299]
    Using device ...
        Name            Intel(R) UHD Graphics [0x9bca]
        Driver version  1.1.19883
        Vendor          Intel(R) Corporation
        Profile         FULL_PROFILE
        Filter string   level_zero:gpu:0
    [New Thread 0x7fffb6792700 (LWP 1707306)]
    [New Thread 0x7fffb6391700 (LWP 1707307)]
    [New Thread 0x7fffb5f90700 (LWP 1707308)]
    [New Thread 0x7fffb5b8f700 (LWP 1707309)]
    [New Thread 0x7fffb538d700 (LWP 1707311)]
    [New Thread 0x7fffb578e700 (LWP 1707310)]
    [New Thread 0x7fffb4f8c700 (LWP 1707312)]
    [New Thread 0x7fffb4b8b700 (LWP 1707313)]
    [New Thread 0x7fffb478a700 (LWP 1707314)]
    [New Thread 0x7fff9ffff700 (LWP 1707315)]
    [New Thread 0x7fff9fbfe700 (LWP 1707316)]
    [Detaching after fork from child process 1707317]
    [Detaching after fork from child process 1707318]
    [New Thread 1.1073741824]
    [New Thread 1.1073742080]
    [Switching to Thread 1.1073741824 lane 0]

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at dppy_func.py:12
    12          i = dppy.get_global_id(0)
    (gdb) s
    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at dppy_func.py:12
    12          i = dppy.get_global_id(0)
    (gdb) s
    13          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) s
    __main__::func_sum () at dppy_func.py:7
    7           result = a_in_func + b_in_func
    (gdb) s
    8           return result
    (gdb) c
    Continuing.
    Done...

.. note::

    Known issues:
      - Debug of the first line of the kernel and functions works out twice.

``stepi``
--------

The command allows you to move forward in machine instructions. The example uses an additional command ``x/i $pc``, which print the instruction to be executed.

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python sum.py

GDB output:

.. code-block:: bash

    (gdb) b sum.py:7
    No source file named sum.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (sum.py:7) pending.
    (gdb) r
    Starting program: /localdisk/work/etotmeni/miniconda3/envs/numba-dppy-new/bin/python sum.py
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    [Detaching after fork from child process 1712759]
    [Detaching after fork from child process 1712760]
    [Detaching after fork from child process 1712765]
    [New Thread 0x7fffd6e9f700 (LWP 1712783)]
    intelgt: gdbserver-gt started for process 1712747.
    intelgt: attached to device 1 of 1; id 0x9bca (Gen9)
    [New Thread 0x7fffc6027700 (LWP 1712796)]
    [Detaching after fork from child process 1712798]
    [Detaching after fork from child process 1712799]
    [Detaching after fork from child process 1712800]
    [Detaching after fork from child process 1712801]
    [Detaching after fork from child process 1712802]
    Using device ...
        Name            Intel(R) UHD Graphics [0x9bca]
        Driver version  1.1.19883
        Vendor          Intel(R) Corporation
        Profile         FULL_PROFILE
        Filter string   level_zero:gpu:0
    [New Thread 0x7fffb6792700 (LWP 1712807)]
    [New Thread 0x7fffae391700 (LWP 1712808)]
    [New Thread 0x7fffb6391700 (LWP 1712809)]
    [New Thread 0x7fffb5f90700 (LWP 1712810)]
    [New Thread 0x7fffb5b8f700 (LWP 1712811)]
    [New Thread 0x7fffb578e700 (LWP 1712812)]
    [New Thread 0x7fffb538d700 (LWP 1712813)]
    [New Thread 0x7fffb4b8b700 (LWP 1712815)]
    [New Thread 0x7fffb478a700 (LWP 1712816)]
    [New Thread 0x7fffb4f8c700 (LWP 1712814)]
    [New Thread 0x7fffaffff700 (LWP 1712817)]
    [Detaching after fork from child process 1712830]
    [Detaching after fork from child process 1712831]
    [New Thread 1.1073741824]
    [New Thread 1.1073742080]
    [Switching to Thread 1.1073741824 lane 0]

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at sum.py:7
    7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    (gdb) display/i $pc
    1: x/i $pc
    => 0xfffdb5d0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1488>:
        (W)     mov (1|M0)               r53.1<1>:w    0:w                              
    (gdb) stepi
    0x00000000fffdb5e0      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb5e0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1504>: (W)     cmp (1|M0)    (eq)f0.0   null<1>:w     r53.1<0;1,0>:w    0:w              
    (gdb) stepi
    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at sum.py:7
    7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb5d0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1488>: (W)     mov (1|M0)               r53.1<1>:w    0:w                              
    (gdb) stepi
    0x00000000fffdb5e0      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb5e0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1504>: (W)     cmp (1|M0)    (eq)f0.0   null<1>:w     r53.1<0;1,0>:w    0:w              
    (gdb) stepi
    0x00000000fffdb5f0      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb5f0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1520>: (W)     mov (1|M0)               r53.2<1>:w    1:w                              
    (gdb) stepi
    0x00000000fffdb600      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb600 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1536>: (W&f0.0) sel (1|M0)              r36.8<2>:b    r53.2<0;1,0>:w    5:w              
    (gdb) stepi
    0x00000000fffdb610      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb610 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1552>: (W)     mov (1|M0)               r36.3<1>:d    r36.8<0;1,0>:ub                 
    (gdb) stepi
    0x00000000fffdb620      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb620 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1568>: (W)     add (1|M0)               r36.4<1>:d    r36.3<0;1,0>:d    0:w              
    (gdb) stepi
    0x00000000fffdb630      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb630 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1584>: (W)     mul (1|M0)               r36.5<1>:uw   r36.8<0;1,0>:uw   0x4:uw             
    (gdb) stepi
    0x00000000fffdb640      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb640 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1600>: (W)     add (1|M0)               a0.0<1>:uw    r36.5<0;1,0>:uw   0x160:uw             
    (gdb) stepi
    0x00000000fffdb650      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb650 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1616>: (W)     mov (1|M0)               r36.5<1>:d    r[a0.0]<0;1,0>:d                
    (gdb) stepi
    0x00000000fffdb660      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb660 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1632>: (W)     mul (1|M0)               r36.3<1>:q    r10.4<0;1,0>:ud   r36.5<0;1,0>:ud 
    (gdb) stepi
    0x00000000fffdb670      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb670 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1648>:         mov (8|M0)               r37.0<1>:d    r1.0<8;8,1>:uw                  
    (gdb) stepi
    0x00000000fffdb680      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb680 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1664>:         mov (8|M0)               r38.0<1>:q    r37.0<8;8,1>:ud                 
    (gdb) stepi
    0x00000000fffdb690      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb690 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1680>:         add (8|M0)               r38.0<1>:q    r38.0<4;4,1>:q    r36.3<0;1,0>:q  
    (gdb) stepi
    0x00000000fffdb6a0      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb6a0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1696>: (W)     mov (1|M0)               r40.0<1>:q    r4.0<0;1,0>:ud                  
    (gdb) stepi
    0x00000000fffdb6b0      7           i = dppy.get_global_id (0)  # numba-kernel-breakpoint
    1: x/i $pc
    => 0xfffdb6b0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1712>:         add (8|M0)               r41.0<1>:q    r38.0<4;4,1>:q    r40.0<0;1,0>:q  
    (gdb) stepi
    8           l1 = a[i]                   # second-line
    1: x/i $pc
    => 0xfffdb6c0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1728>:         cmp (8|M0)    (lt)f0.0   null<1>:q     r41.0<4;4,1>:q    0:w              
    (gdb) stepi
    0x00000000fffdb6d0      8           l1 = a[i]                   # second-line
    1: x/i $pc
    => 0xfffdb6d0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1744>: (f0.0)  sel (8|M0)               r43.0<1>:q    r7.3<0;1,0>:q     0:w              
    (gdb) stepi
    0x00000000fffdb6e0      8           l1 = a[i]                   # second-line
    1: x/i $pc
    => 0xfffdb6e0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1760>:         add (8|M0)               r43.0<1>:q    r43.0<4;4,1>:q    r41.0<4;4,1>:q  
    (gdb) stepi
    0x00000000fffdb6f0      8           l1 = a[i]                   # second-line
    1: x/i $pc
    => 0xfffdb6f0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1776>:         shl (8|M0)               r43.0<1>:q    r43.0<4;4,1>:q    2:w              
    (gdb) stepi
    0x00000000fffdb700      8           l1 = a[i]                   # second-line
    1: x/i $pc
    => 0xfffdb700 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1792>:         add (8|M0)               r43.0<1>:q    r5.2<0;1,0>:q     r43.0<4;4,1>:q  
    (gdb) stepi
    0x00000000fffdb710      8           l1 = a[i]                   # second-line
    1: x/i $pc
    => 0xfffdb710 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1808>:         send (8|M0)              r45:f    r43:uq  0xC            0x041401FF           // wr:2+0, rd:1; hdc.dc1; a64 dword gathering read
    (gdb) stepi
    9           l2 = b[i]                   # third-line
    1: x/i $pc
    => 0xfffdb720 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1824>: (f0.0)  sel (8|M0)               r46.0<1>:q    r8.3<0;1,0>:q     0:w              
    (gdb) stepi
    0x00000000fffdb730      9           l2 = b[i]                   # third-line
    1: x/i $pc
    => 0xfffdb730 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1840>:         add (8|M0)               r46.0<1>:q    r46.0<4;4,1>:q    r41.0<4;4,1>:q  
    (gdb) stepi
    0x00000000fffdb740      9           l2 = b[i]                   # third-line
    1: x/i $pc
    => 0xfffdb740 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1856>:         shl (8|M0)               r46.0<1>:q    r46.0<4;4,1>:q    2:w              
    (gdb) stepi
    0x00000000fffdb750      9           l2 = b[i]                   # third-line
    1: x/i $pc
    => 0xfffdb750 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1872>:         add (8|M0)               r46.0<1>:q    r6.1<0;1,0>:q     r46.0<4;4,1>:q  
    (gdb) stepi
    0x00000000fffdb760      9           l2 = b[i]                   # third-line
    1: x/i $pc
    => 0xfffdb760 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1888>:         send (8|M0)              r48:f    r46:uq  0xC            0x041401FF           // wr:2+0, rd:1; hdc.dc1; a64 dword gathering read
    (gdb) stepi
    10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb770 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1904>:         add (8|M0)               r45.0<1>:f    r45.0<8;8,1>:f    r48.0<8;8,1>:f  
    (gdb) stepi
    0x00000000fffdb780      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb780 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1920>: (f0.0)  sel (8|M0)               r49.0<1>:q    r9.3<0;1,0>:q     0:w              
    (gdb) stepi
    0x00000000fffdb790      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb790 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1936>:         add (8|M0)               r49.0<1>:q    r49.0<4;4,1>:q    r41.0<4;4,1>:q  
    (gdb) stepi
    0x00000000fffdb7a0      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb7a0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1952>:         shl (8|M0)               r49.0<1>:q    r49.0<4;4,1>:q    2:w              
    (gdb) stepi
    0x00000000fffdb7b0      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb7b0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1968>:         add (8|M0)               r49.0<1>:q    r7.0<0;1,0>:q     r49.0<4;4,1>:q  
    (gdb) stepi
    0x00000000fffdb7c0      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb7c0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1984>:         sends (8|M0)             null:ud  r49     r45     0x4C            0x040681FF           // wr:2+1, rd:0; hdc.dc1; a64 dword scattering write
    (gdb) stepi
    0x00000000fffdb7d0      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb7d0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+2000>:         mov (8|M0)               r51.0<1>:uq   0x0:uw                             
    (gdb) stepi
    0x00000000fffdb7e0      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb7e0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+2016>:         sends (8|M0)             null:ud  r13     r51     0x8C            0x040682FF           // wr:2+2, rd:0; hdc.dc1; a64 qword scattering write
    (gdb) stepi
    0x00000000fffdb7f0      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb7f0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+2032>: (W)     mov (8|M0)               r127.0<1>:ud  r11.0<8;8,1>:ud                 
    (gdb) stepi
    0x00000000fffdb800      10          c[i] = l1 + l2              # fourth-line
    1: x/i $pc
    => 0xfffdb800 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+2048>: (W)     send (8|M0)              null     r127    0x27            0x02000010           {EOT} // wr:1+0, rd:0; spawner; end of thread
    (gdb) stepi
    Done...

``next``
--------

Stepping-like behavior, but the command does not go into nested functions.

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) b dppy_func.py:12
    No source file named dppy_func.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (dppy_func.py:12) pending.
    (gdb) r
    Starting program: /localdisk/work/etotmeni/miniconda3/envs/numba-dppy-new/bin/python /localdisk/work/etotmeni/stepping/dppy_func.py
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    [Detaching after fork from child process 1721691]
    [Detaching after fork from child process 1721692]
    [Detaching after fork from child process 1721693]
    [New Thread 0x7fffd6e9f700 (LWP 1721701)]
    intelgt: gdbserver-gt started for process 1721674.
    intelgt: attached to device 1 of 1; id 0x9bca (Gen9)
    [New Thread 0x7fffc6027700 (LWP 1721713)]
    [Detaching after fork from child process 1721714]
    [Detaching after fork from child process 1721715]
    [Detaching after fork from child process 1721716]
    [Detaching after fork from child process 1721717]
    [Detaching after fork from child process 1721718]
    Using device ...
        Name            Intel(R) UHD Graphics [0x9bca]
        Driver version  1.1.19883
        Vendor          Intel(R) Corporation
        Profile         FULL_PROFILE
        Filter string   level_zero:gpu:0
    [New Thread 0x7fffb6792700 (LWP 1721719)]
    [New Thread 0x7fffb6391700 (LWP 1721720)]
    [New Thread 0x7fffb5f90700 (LWP 1721721)]
    [New Thread 0x7fffb5b8f700 (LWP 1721722)]
    [New Thread 0x7fffb578e700 (LWP 1721723)]
    [New Thread 0x7fffb538d700 (LWP 1721724)]
    [New Thread 0x7fffb4f8c700 (LWP 1721725)]
    [New Thread 0x7fffb478a700 (LWP 1721727)]
    [New Thread 0x7fffb4b8b700 (LWP 1721726)]
    [New Thread 0x7fff9ffff700 (LWP 1721728)]
    [New Thread 0x7fff9fbfe700 (LWP 1721729)]
    [Detaching after fork from child process 1721736]
    [Detaching after fork from child process 1721737]
    [New Thread 1.1073741824]
    [New Thread 1.1073742080]
    [Switching to Thread 1.1073741824 lane 0]

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at dppy_func.py:12
    12          i = dppy.get_global_id(0)
    (gdb) n
    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at dppy_func.py:12
    12          i = dppy.get_global_id(0)
    (gdb) n
    13          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) n
    Done...
