Examining the Symbol Table
==========================

See `GDB* documentation <https://sourceware.org/gdb/onlinedocs/gdb/Symbols.html>`_.

``info functions``
------------------

At least following syntax is supported:

.. code-block:: bash

    info functions
    info functions [regexp]

.. note::

    Running the ``info functions`` command without arguments may produce a lot of output
    as the list of all functions in all loaded shared libraries is typically very long.

Example
```````

Source file ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

Output of the debug session:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/info_func
    :language: shell-session
    :emphasize-lines: 5-9

``whatis [arg]`` and ``ptype [arg]``
----------------------------------------------

To print the type of a variable, run the ``ptype <variable>`` or ``whatis <variable>`` commands:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 73-81
    :emphasize-lines: 1-6
