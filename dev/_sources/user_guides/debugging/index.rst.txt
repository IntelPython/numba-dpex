Debugging with Intel® Distribution for GDB*
===========================================

Numba-dppy allows you to debug SYCL* kernels with Intel® Distribution for GDB*.
To enable the emission of debug information, set the debug environment variable :envvar:`NUMBA_DPPY_DEBUGINFO`, for example:
:samp:`export NUMBA_DPPY_DEBUGINFO=1`
To disable debugging, unset the variable:
:samp:`unset NUMBA_DPPY_DEBUGINFO`

.. note::

    Enabling debug information significantly increases the memory consumption for each compiled kernel.
    For a large application, this may cause out-of-memory error.

Not all debugging features supported by Numba on CPUs are yet supported by Numba-dppy.
See :ref:`debugging-features-and-limitations`.

Requirements
------------

`Intel® Distribution for GDB*` is required for Numba-dppy debugging features to work.
`Intel® Distribution for GDB*` is part of `Intel oneAPI`. For relevant documentation, refer to the `Intel® Distribution for GDB* product page`_.

.. _`Intel® Distribution for GDB* documentation`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-gdb.html

.. toctree::
    :maxdepth: 2

    set_up_machine
    debugging_environment


Example of Intel® Distribution for GDB* usage
--------------------------------------------

You can use a sample Numba-dppy kernel code, :file:`simple_sum.py`, for basic debugging:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:
    :lineno-match:

Use the following commands to create a breakpoint inside the kernel and run the debugger:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/simple_sum
    :language: shell-session

.. _debugging-features-and-limitations:

Features and Limitations
------------------------

.. toctree::
    :maxdepth: 2

    features
    limitations
    common_issues
