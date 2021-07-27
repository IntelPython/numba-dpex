Info commands
=============

The ``info functions`` command displays the list of functions in the debugged program.

**Syntax**
``````````

.. code-block:: bash

    info functions
    info functions [Regex]

.. note::

    Running the ``info functions`` command without arguments may produce a lot of output
    as the list of all functions in all loaded shared libraries is typically very long.

Parameters
``````````

**Regex**

If the regex is specified, the ``info functions`` command lists the functions matching the regex.
If omitted, the command lists all functions in all loaded modules (main program and shared libraries).

Consider Numba-dppy kernel code. See the source file ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

Run the debugger and use the ``info functions`` command. The output is as follows:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/info_func
    :language: shell-session
    :emphasize-lines: 5-9

See also:

- `Info functions in GDB* <https://sourceware.org/gdb/onlinedocs/gdb/Symbols.html>`_
