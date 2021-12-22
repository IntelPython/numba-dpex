Information About a Frame
=========================

See `GDB* documentation <https://www.sourceware.org/gdb/onlinedocs/gdb/Frame-Info.html>`_.

``info args``
-------------

Test :file:`numba_dppy/tests/debugging/test_info.py:test_info_args`.

.. note::

   Requires Numba 0.55 or higher.

``info locals``
---------------

Test :file:`numba_dppy/tests/debugging/test_info.py:test_local_variables`.

.. note::

   Requires Numba 0.55 or higher.

Example
```````

Source code :file:`sum_local_vars.py`:

.. literalinclude:: ../../../numba_dppy/examples/debug/sum_local_vars.py
    :lines: 15-
    :linenos:
    :lineno-match:

Run the debugger with ``NUMBA_OPT=0``:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 1-6

Use ``info locals``.
Note that uninitialized variables are zeros:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 8-48
    :emphasize-lines: 1-16, 24-39
