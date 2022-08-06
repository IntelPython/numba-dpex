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

Source file ``numba_dpex/examples/debug/simple_sum.py``:

.. literalinclude:: ../../../numba_dpex/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

Output of the debug session:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/info_func
    :language: shell-session
    :emphasize-lines: 5-9

.. _whatis:

``whatis [arg]`` and ``ptype [arg]``
------------------------------------

To print the type of a variable, run the ``ptype <variable>`` or ``whatis <variable>`` commands:

.. literalinclude:: ../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 73-81
    :emphasize-lines: 1-6

Example - Complex Data Types
````````````````````````````

Source code :file:`numba_dpex/examples/debug/side-by-side-2.py`:

.. literalinclude:: ../../../numba_dpex/examples/debug/side-by-side-2.py
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
   (gdb) ptype a
   type = struct array(float32, 1d, C) ({float addrspace(1)*, float addrspace(1)*, i64, i64, float addrspace(1)*, [1 x i64], [1 x i64]}) {
       float *meminfo;
       float *parent;
       int64 nitems;
       int64 itemsize;
       float *data;
       i64 shape[1];
       i64 strides[1];
   }
   (gdb) whatis a
   type = array(float32, 1d, C) ({float addrspace(1)*, float addrspace(1)*, i64, i64, float addrspace(1)*, [1 x i64], [1 x i64]})
