Debugging with GDB
==================

Numba-dppy allows SYCL kernels to be debugged with the GDB debugger.
Setting the debug environment variable :envvar:`NUMBA_DPPY_DEBUGINFO`
(e.g. :samp:`export NUMBA_DPPY_DEBUGINFO=1`) enables the emission of debug information.
To disable debugging, unset the variable (e.g. :samp:`unset NUMBA_DPPY_DEBUGINFO`).

.. note::

    Beware that enabling debug info significantly increases the memory consumption for each compiled kernel.
    For large application, this may cause out-of-memory error.

Not all GDB features supported by Numba on CPUs are yet supported in Numba-dppy.
See :ref:`debugging-features-and-limitations`.


Requirements
------------

`Intel® Distribution for GDB` is required for Numba-dppy's debugging features to work.
`Intel® Distribution for GDB` is part of `Intel oneAPI` and
the relevant documentation can be found at `Intel® Distribution for GDB documentation`_.

.. _`Intel® Distribution for GDB documentation`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-gdb.html

.. toctree::
    :maxdepth: 2

    set_up_machine
    debugging_environment


Example of GDB usage
--------------------

For example, given the following Numba-dppy kernel code (:file:`simple_sum.py`):

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

Running GDB and creating breakpoint in kernel:

.. code-block:: bash

    $ gdb-oneapi -q --args python simple_sum.py
    (gdb) break simple_sum.py:8  ### Set breakpoint in kernel
    (gdb) run
    ...
    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:8
    8           i = dppy.get_global_id(0)
    (gdb) next
    9           c[i] = a[i] + b[i]
    (gdb) continue
    Done...
    ...

If the breakpoint is not hit, you will see the following output:

.. code-block:: bash

    ...
    intelgt: gdbserver-gt failed to start.  Check if igfxdcd is installed, or use
    env variable INTELGT_AUTO_ATTACH_DISABLE=1 to disable auto-attach.
    ...

See the :ref:`debugging-machine-dcd-driver`.

.. _debugging-features-and-limitations:

Features and Limitations
------------------------

.. toctree::
    :maxdepth: 2

    features
    limitations
