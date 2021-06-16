Supported GDB Features
======================

Currently, the following debugging features are available:

- Source location (filename and line number).
- Setting break points by the line number.
- Stepping over break points.

.. note::

    Debug features depend heavily on optimization level.
    At full optimization (equivalent to O3), most of the variables are optimized out.
    It is recommended to debug at "no optimization" level via :envvar:`NUMBA_OPT` (e.g. :samp:`export NUMBA_OPT=0`).
    For more information refer to the Numba documentation `Debugging JIT compiled code with GDB`_.

.. _`Debugging JIT compiled code with GDB`: https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html?highlight=numba_opt#debugging-jit-compiled-code-with-gdb

`numba-dppy` supports at least following GDB commands:

.. toctree::
    :maxdepth: 2

    local_variables
    stepping
    info
    backtrace
