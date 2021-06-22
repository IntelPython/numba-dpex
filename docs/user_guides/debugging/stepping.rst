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

Run debugger and do following commands:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/step_sum

Another use of stepping when there is a nested function. Below example:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/step_dppy_func

``stepi``
---------

The command allows you to move forward in machine instructions. The example uses an additional command ``x/i $pc``, which print the instruction to be executed.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/stepi

``next``
--------

Stepping-like behavior, but the command does not go into nested functions.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/next

.. _single_stepping:

``set scheduler-locking step``
-------------------------------

Debug of the first line of the kernel and functions works out twice.
This happens because you are debugging a multi-threaded program and multiple events may be received from different threads.
This is the default behavior, but you can configure it for more efficient debugging.
To ensure the current thread executes a single line without interference, set the scheduler-locking setting to on or step:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/sheduler_locking

See also:

- `Single Stepping <https://software.intel.com/content/www/us/en/develop/documentation/debugging-dpcpp-linux/top/debug-a-dpc-application-on-a-cpu/single-stepping.html>`_
- `Continuing and Stepping in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Continuing-and-Stepping.html#Continuing-and-Stepping>`_
