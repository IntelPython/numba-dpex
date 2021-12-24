Examining Data
==============

See `GDB* documentation <https://www.sourceware.org/gdb/onlinedocs/gdb/Data.html>`_.

``print expr``
--------------

To print the value of a variable, run the ``print <variable>`` command.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 67-72
    :emphasize-lines: 1-6

.. note::

    Kernel variables are shown in intermidiate representation view (with "$" sign).
    The actual values of the arrays are currently not available.
