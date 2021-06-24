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

Consider Numba-dppy kernel code. See the source file ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

Run GDB debugger:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    gdb-oneapi -q --args python simple_sum.py

``break function``
------------------

GDB output:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/break_func
    :language: shell-session

``break filename: linenumber``
------------------------------

GDB output:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/break_line_number
    :language: shell-session

``break filename: function``
----------------------------

GDB output:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/break_file_func
    :language: shell-session

Breakpoints with nested functions
-------------------------------------

Consider Numba-dppy kernel code. See source file ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:
    :lineno-match:

GDB output:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/break_nested_func
    :language: shell-session
