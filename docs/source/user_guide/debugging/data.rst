.. include:: ./../../ext_links.txt

Examining Data
==============

See `GDB* documentation <https://www.sourceware.org/gdb/onlinedocs/gdb/Data.html>`_.

.. _print:

``print expr``
--------------

To print the value of a variable, run the ``print <variable>`` command.

.. literalinclude:: ./../../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 67-72
    :emphasize-lines: 1-6

.. note::

    Displaying complex data types requires Numba 0.55 or higher.

Example - Complex Data Types
````````````````````````````

Source code :file:`numba_dpex/examples/debug/side-by-side-2.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/side-by-side-2.py
   :pyobject: common_loop_body
   :linenos:
   :lineno-match:
   :emphasize-lines: 6

Debug session:

.. code-block:: shell-session
   :emphasize-lines: 9-

   $ gdb-oneapi -q python
   ...
   (gdb) set environment NUMBA_OPT 0
   (gdb) set environment NUMBA_EXTEND_VARIABLE_LIFETIMES 1
   (gdb) break side-by-side-2.py:29 if param_a == 5
   ...
   (gdb) run numba_dpex/examples/debug/side-by-side-2.py --api=numba-dpex-kernel
   ...
   Thread 2.1 hit Breakpoint 1, with SIMD lane 5, __main__::common_loop_body (i=5, a=..., b=...) at side-by-side-2.py:29
   29          result = param_c + param_d
   (gdb) print a
   $1 = {meminfo = 0x0, parent = 0x0, nitems = 10, itemsize = 4,
     data = 0x555558461000, shape = {10}, strides = {4}}
   (gdb) x/10f a.data
   0x555558461000: 0       1       2       3
   0x555558461010: 4       5       6       7
   0x555558461020: 8       9
   (gdb) print a.data[5]
   $2 = 5

This example prints array and its element.
