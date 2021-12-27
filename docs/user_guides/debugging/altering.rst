Altering Execution
==================

See `GDB* documentation <https://sourceware.org/gdb/onlinedocs/gdb/Altering.html>`_.

Assignment to Variables
-----------------------

To alter the value of a variable, evaluate an assignment expression.

Example
```````

Source code :file:`numba_dppy/examples/debug/side-by-side-2.py`:

.. literalinclude:: ../../../numba_dppy/examples/debug/side-by-side-2.py
   :pyobject: common_loop_body
   :linenos:
   :lineno-match:
   :emphasize-lines: 6

Debug session:

.. code-block:: shell-session
   :emphasize-lines: 11-

   $ gdb-oneapi -q python
   ...
   (gdb) set environment NUMBA_OPT 0
   (gdb) set environment NUMBA_EXTEND_VARIABLE_LIFETIMES 1
   (gdb) break side-by-side-2.py:29 if param_a == 5
   ...
   (gdb) run numba_dppy/examples/debug/side-by-side-2.py --api=numba-dppy-kernel
   ...
   Thread 2.1 hit Breakpoint 1, with SIMD lane 5, __main__::common_loop_body (i=5, a=..., b=...) at side-by-side-2.py:29
   29          result = param_c + param_d
   (gdb) print param_c
   $1 = 15
   (gdb) print param_c=200
   $2 = 200
   (gdb) print param_c
   $3 = 200
