Local variables
===============

.. note::

    - :samp:`NUMBA_OPT=0` "no optimization" level - all local variables of the kernel function are available.
    - :samp:`NUMBA_OPT=1` or higher - some variables may be optimized out.

Consider Numba-dppy kernel code :file:`sum_local_vars.py`

.. literalinclude:: ../../../numba_dppy/examples/debug/sum_local_vars.py
    :lines: 15-
    :linenos:
    :lineno-match:

``info locals``
---------------

Run the debugger:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 1-6

Run the ``info locals`` command. The sample output on "no optimization" level ``NUMBA_OPT=0`` is as follows:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 8-48
    :emphasize-lines: 1-16, 24-39

Since the debugger does not hit a line with the target variable ``l1``, the value equals 0. The true value of the variable ``l1`` is shown after stepping to line 22.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 49-66
    :emphasize-lines: 1-16

When the debugger hits the last line of the kernel, ``info locals`` command returns all the local variables with their values.

.. note::

    The debugger can show the variable values, but these values may be equal to 0 after the variable is explicitly deleted or the function scope is ended. For more information, refer to `Numba variable policy <https://numba.pydata.org/numba-doc/latest/developer/live_variable_analysis.html?highlight=delete#live-variable-analysis>`_.

When you use "O1 optimization" level ``NUMBA_OPT=1`` and run the ``info locals`` command, the output is as follows:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_1
    :language: shell-session
    :lines: 8-23
    :emphasize-lines: 1-14

.. note::

    The debugger does not show the local variables ``a``, ``b`` and ``c``, they are optimized out on "O1 optimization" level.

``print <variable>``
------------------

To print the value of a variable, run the ``print <variable>`` command.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 67-72
    :emphasize-lines: 1-6

.. note::

    Kernel variables are shown in intermidiate representation view (with "$" sign). The actual values of the arrays are currently not available.

``ptype <variable>``
------------------

To print the type of a variable, run the ``ptype <variable>`` or ``whatis <variable>`` commands:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 73-81
    :emphasize-lines: 1-6

See also:

    - `Local variables in GDB* <https://sourceware.org/gdb/current/onlinedocs/gdb/Frame-Info.html#Frame-Info>`_
