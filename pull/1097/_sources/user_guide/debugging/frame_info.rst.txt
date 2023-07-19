.. include:: ./../../ext_links.txt

Information About a Frame
=========================

See `GDB* documentation <https://www.sourceware.org/gdb/onlinedocs/gdb/Frame-Info.html>`_.

.. _info-args:

``info args``
-------------

Test :file:`numba_dpex/tests/debugging/test_info.py:test_info_args`.

.. note::

   Requires Numba 0.55 or higher.
   In previous versions ``info args`` always returns ``No arguments.``.

Example
```````

Source code :file:`numba_dpex/examples/debug/side-by-side.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/side-by-side.py
   :pyobject: common_loop_body
   :linenos:
   :lineno-match:
   :emphasize-lines: 2

Debug session:

.. code-block:: shell-session
   :emphasize-lines: 9-11

   $ NUMBA_OPT=0 gdb-oneapi -q python
   ...
   (gdb) break side-by-side.py:25
   ...
   (gdb) run numba_dpex/examples/debug/side-by-side.py --api=numba-dpex-kernel
   ...
   Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::common_loop_body (param_a=0, param_b=0) at side-by-side.py:25
   25          param_c = param_a + 10  # Set breakpoint here
   (gdb) info args
   param_a = 0
   param_b = 0

.. _info-locals:

``info locals``
---------------

Test :file:`numba_dpex/tests/debugging/test_info.py:test_info_locals`.

.. note::

   Requires Numba 0.55 or higher.

Example
```````

Source code :file:`sum_local_vars.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/sum_local_vars.py
    :lines: 5-
    :linenos:
    :lineno-match:

Run the debugger with ``NUMBA_OPT=0``:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 1-6

Use ``info locals``.
Note that uninitialized variables are zeros:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 8-48
    :emphasize-lines: 1-16, 24-39
