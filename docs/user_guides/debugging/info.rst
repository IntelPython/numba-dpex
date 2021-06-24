Info commands
=============

``info functions``
------------------

Displays the list of functions in the debugged program.

**Syntax**
``````````

.. code-block:: bash

    info functions
    info functions [Regex]

Parameters
``````````

`Regex`

If specified, the info functions command lists the functions matching the regex.
If omitted, the command lists all functions in all loaded modules (main program and shared libraries).

.. note::

    Running the ``info functions`` command without arguments may produce a lot of output
    as the list of all functions in all loaded shared libraries is typically very long.

Consider Numba-dppy kernel code. See the source file ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

Run GDB debugger:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/info_func

See also:

- `Info functions <https://sourceware.org/gdb/onlinedocs/gdb/Symbols.html>`_
