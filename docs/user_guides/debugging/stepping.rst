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

.. note::

    Known issues:
        - Debug of the first line of the kernel and functions works out twice. See :ref:`single_stepping`.

``step``
--------

Run debugger:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_sum.py

GDB output:

.. code-block:: bash

    (gdb) break simple_sum.py:22
    Breakpoint 1 (simple_sum.py:22) pending.
    (gdb) run
    ...
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:22
    22          i = dppy.get_global_id(0)
    (gdb) step
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::data_parallel_sum () at simple_sum.py:22
    22          i = dppy.get_global_id(0)
    (gdb) step
    23          c[i] = a[i] + b[i]
    (gdb) continue
    Continuing.
    Done...

Another use of stepping when there is a nested function. Below example:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) break simple_dppy_func.py:28
    Breakpoint 1 (simple_dppy_func.py:28) pending.
    (gdb) run
    ...
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) step
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) step
    29          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) step
    __main__::func_sum () at simple_dppy_func.py:22
    22          result = a_in_func + b_in_func
    (gdb) continue
    Continuing.

``stepi``
--------

The command allows you to move forward in machine instructions. The example uses an additional command ``x/i $pc``, which print the instruction to be executed.

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_sum.py

GDB output:

.. code-block:: bash

    (gdb) break simple_sum.py:22
    Breakpoint 1 (simple_sum.py:22) pending.
    (gdb) run
    ...
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

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) break simple_dppy_func.py:28
    Breakpoint 1 (simple_dppy_func.py:28) pending.
    (gdb) run
    ...
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) next
    Thread 2.3 hit Breakpoint 1, with SIMD lanes [0-1], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) next
    29          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) next
    Done...

.. _single_stepping:

``set scheduler-locking step``
-------------------------------

Debug of the first line of the kernel and functions works out twice.
This happens because you are debugging a multi-threaded program and multiple events may be received from different threads.
This is the default behavior, but you can configure it for more efficient debugging.
To ensure the current thread executes a single line without interference, set the scheduler-locking setting to on or step:

.. code-block:: bash

    export NUMBA_OPT=0
    gdb-oneapi -q --args python simple_dppy_func.py

GDB output:

.. code-block:: bash

    (gdb) break simple_dppy_func.py:28
    Breakpoint 1 (simple_dppy_func.py:28) pending.
    (gdb) run
    ...
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::kernel_sum () at simple_dppy_func.py:28
    28          i = dppy.get_global_id(0)
    (gdb) set scheduler-locking step
    (gdb) step
    29          c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])
    (gdb) step
    __main__::func_sum () at simple_dppy_func.py:22
    22          result = a_in_func + b_in_func
    (gdb) continue
    Continuing.
    Done...

See also:

- `Single Stepping <https://software.intel.com/content/www/us/en/develop/documentation/debugging-dpcpp-linux/top/debug-a-dpc-application-on-a-cpu/single-stepping.html>`_
- `Continuing and Stepping in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Continuing-and-Stepping.html#Continuing-and-Stepping>`_
  