Stepping
========

Consider the following two examples. ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

Example with a nested function ``numba_dppy/examples/debug/simple_dppy_func.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_dppy_func.py
    :lines: 15-
    :linenos:
    :lineno-match:


``step``
--------

Run debugger and do following commands:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/step_sum
    :language: shell-session
    :emphasize-lines: 8-13

Another use of stepping when there is a nested function. Below example:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/step_dppy_func
    :language: shell-session
    :emphasize-lines: 8-14

``stepi``
---------

The command allows you to move forward in machine instructions. The example uses an additional command ``x/i $pc``, which prints the instruction to be executed.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/stepi
    :language: shell-session
    :emphasize-lines: 8-13

``next``
--------

Stepping-like behavior, but the command does not go into nested functions.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/next
    :language: shell-session
    :emphasize-lines: 8-14

.. _single_stepping:

``set scheduler-locking step``
------------------------------

The first line of the kernel and functions is debugged twice.
This happens because you are debugging a multi-threaded program, so multiple events may be received from different threads.
This is the default behavior, but you can configure it for more efficient debugging.
To ensure the current thread executes a single line without interference, set the scheduler-locking setting to on or step:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/sheduler_locking
    :language: shell-session
    :emphasize-lines: 8-13

See also:

- `Single Stepping <https://software.intel.com/content/www/us/en/develop/documentation/debugging-dpcpp-linux/top/debug-a-dpc-application-on-a-cpu/single-stepping.html>`_
- `Continuing and Stepping in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Continuing-and-Stepping.html#Continuing-and-Stepping>`_
