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

Run GDB debugger:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 1-6

GDB output on "no optimization" level ``NUMBA_OPT=0``:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 8-48

Since GDB debugger does not hit a line with target variable, the value of this variable is equal to 0. The true value of the variable ``l1`` is shown after stepping to line 22.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 49-66

When GDB debugger hits the last line of the kernel, ``info locals`` command returns all the local variables with their values.

.. note::

    Known issues:
      - GDB debugger can show the variable values, but these values may be equal to 0 after the variable is explicitly deleted or the function scope is ended. For more information refer to `Numba variable policy <https://numba.pydata.org/numba-doc/latest/developer/live_variable_analysis.html?highlight=delete#live-variable-analysis>`_.

GDB output on "O1 optimization" level ``NUMBA_OPT=1``:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_1
    :language: shell-session
    :lines: 8-23

.. note::

    The GDB debugger does not show the local variables ``a``, ``b`` and ``c``, they are optimized out on "O1" optimization level.

``print variable``
------------------

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 67-72

.. note::

    Known issues:
      - Kernel variables are shown in intermidiate representation view (with "$" sign). The actual values of the arrays are currently not available.

``ptype variable``
------------------

Variable type may be printed by the command ``ptype variable`` and ``whatis variable``:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 73-81

See also:

    - `Local variables in GDB <https://sourceware.org/gdb/current/onlinedocs/gdb/Frame-Info.html#Frame-Info>`_
