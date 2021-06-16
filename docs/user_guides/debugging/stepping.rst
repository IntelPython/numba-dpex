Stepping
========

Consider the following two examples. ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

Example with a nested function ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:   

``step``
--------

Run debugger:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python simple_sum.py

GDB output:

.. code-block:: bash

    (gdb) b simple_sum.py:22
    Breakpoint 1 (simple_sum.py:22) pending.
    (gdb) r
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:22
    22          i = dppy.get_global_id(0)
    (gdb) s
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at simple_sum.py:22
    22          i = dppy.get_global_id(0)
    (gdb) s
    23          c[i] = a[i] + b[i]
    (gdb) continue
    Continuing.
    Done...

Another use of stepping when there is a nested function. Below example:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python simple_dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) b simple_dppy_func.py:28
    Breakpoint 1 (simple_dppy_func.py:28) pending.
    (gdb) r
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) s
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) s
    29          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) s
    __main__::func_sum () at simple_dppy_func.py:22
    22          result = a_in_func + b_in_func
    (gdb) continue
    Continuing.

.. note::

    Known issues:
      - Debug of the first line of the kernel and functions works out twice.

``stepi``
--------

The command allows you to move forward in machine instructions. The example uses an additional command ``x/i $pc``, which print the instruction to be executed.

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python simple_sum.py

GDB output:

.. code-block:: bash

    (gdb) b simple_sum.py:22
    Breakpoint 1 (simple_sum.py:22) pending.
    (gdb) r
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:22
    22          i = dppy.get_global_id(0)
    (gdb) display/i $pc
    1: x/i $pc
    => 0xfffeb5d0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1488>:
        (W)     mov (1|M0)               r53.1<1>:w    0:w                              
    (gdb) stepi
    0x00000000fffeb5e0      22          i = dppy.get_global_id(0)
    1: x/i $pc
    => 0xfffeb5e0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1504>: (W)     cmp (1|M0)    (eq)f0.0   null<1>:w     r53.1<0;1,0>:w    0:w              
    (gdb) stepi
    [Switching to Thread 1.1073742080 lane 0]

    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at simple_sum.py:22
    22          i = dppy.get_global_id(0)
    1: x/i $pc
    => 0xfffeb5d0 <_ZN8__main__17data_parallel_sumE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE9DPPYArrayIfLi1E1C7mutable7alignedE+1488>: (W)     mov (1|M0)               r53.1<1>:w    0:w                              
    (gdb) continue
    Continuing.
    Done...

``next``
--------

Stepping-like behavior, but the command does not go into nested functions.

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python simple_dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) b simple_dppy_func.py:28
    Breakpoint 1 (simple_dppy_func.py:28) pending.
    (gdb) r
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) n
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) n
    29          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) n
    Done...
