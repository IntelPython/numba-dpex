Breakpoints
===========

A `breakpoint` makes your program stop whenever a certain point in the program
is reached.

You can set breakpoints with the ``break`` command to specify the place where
your program should stop in the kernel. Define breakpoints by line numbers or
function names.

You have several ways to set breakpoints:
  - break <function>
  - break <filename>:<linenumber>
  - break <filename>:<function>
  - break … if <condition>

See also:
  - `Breakpoints in GDB*`_.

.. _Breakpoints in GDB*: https://sourceware.org/gdb/current/onlinedocs/gdb/Set-Breaks.html#Set-Breaks

Consider the following numba-dpex kernel code (refer
``numba_dpex/examples/debug/simple_sum.py`` for full example):

.. literalinclude:: ../../../numba_dpex/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

``break function``
------------------

The debugger output:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/break_func
    :language: shell-session
    :emphasize-lines: 3

``break filename:linenumber``
-----------------------------

The debugger output:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/break_line_number
    :language: shell-session
    :emphasize-lines: 3

``break filename:function``
---------------------------

The debugger output:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/break_file_func
    :language: shell-session
    :emphasize-lines: 3

``break … if cond``
-------------------

The debugger output:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/break_conditional
    :language: shell-session
    :emphasize-lines: 3

Breakpoints with nested functions
---------------------------------

Consider numba-dpex kernel code. See the source file
``numba_dpex/examples/debug/simple_dpex_func.py``:

.. literalinclude:: ../../../numba_dpex/examples/debug/simple_dpex_func.py
    :lines: 15-
    :linenos:
    :lineno-match:

The debugger output:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/break_nested_func
    :language: shell-session
    :emphasize-lines: 3
