Breakpoints
===========

A `breakpoint` makes your program stop whenever a certain point in the program is reached.

You can set breakpoints with the ``break`` command to specify the place where your program should stop in the kernel by line number or function name.

You have several ways to set breakpoints:
  - break function
  - break filename:function
  - break filename:linenumber
  
See also:
  - `GDB documentation of breakpoints`_.

.. _GDB documentation of breakpoints: https://sourceware.org/gdb/current/onlinedocs/gdb/Set-Breaks.html#Set-Breaks

Consider ``numba-dppy`` kernel code. See source file ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

Run debugger:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    gdb-oneapi -q --args python simple_sum.py

``break function``
------------------

GDB output:

.. code-block:: bash

  (gdb) break data_parallel_sum
  Breakpoint 1 (data_parallel_sum) pending.
  (gdb) run

  Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:20
  20      @dppy.kernel

``break filename: linenumber``
------------------------------

GDB output:

.. code-block:: bash

  (gdb) break simple_sum.py:20
  Breakpoint 1 (simple_sum.py:20) pending.
  (gdb) run
  
  Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:20
  20      @dppy.kernel

``break filename: function``
----------------------------

GDB output:

.. code-block:: bash

  (gdb) break simple_sum.py:data_parallel_sum
  Breakpoint 1 (simple_sum.py:data_parallel_sum) pending.
  (gdb) run
  
  Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:20
  20      @dppy.kernel

Breakpoints with nested functions
-------------------------------------

Consider ``numba-dppy`` kernel code. See source file ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:

GDB output:

.. code-block:: bash

  export NUMBA_DPPY_DEBUGINFO=1
  gdb-oneapi -q --args python simple_sum.py
  (gdb) break func_sum
  Breakpoint 1 (func_sum) pending.
  (gdb) run

  Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::func_sum () at simple_dppy_func.py:22
  22      result = a_in_func + b_in_func
