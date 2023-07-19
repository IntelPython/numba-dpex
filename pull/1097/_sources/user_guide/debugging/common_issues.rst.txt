.. include:: ./../../ext_links.txt

Common issues and tips
======================

Breakpoints are not hit
-----------------------
If the breakpoint is not hit, you will see the following output:

.. code-block:: bash

    ... intelgt: gdbserver-gt failed to start.  Check if igfxdcd is installed,
    or use env variable INTELGT_AUTO_ATTACH_DISABLE=1 to disable auto-attach.
    ...

To install the debug companion driver (igfxdcd), refer to the
:ref:`debugging-machine-dcd-driver` section.

Debugging is not stable
-----------------------

Debug features depend heavily on optimization level. At full optimization
(equivalent to O3), most of the variables are optimized out. It is recommended
to debug at "no optimization" level via :envvar:`NUMBA_OPT` (e.g. :samp:`export
NUMBA_OPT=0`). For more information, refer to the Numba documentation `Debugging
JIT compiled code with GDB*`_.

It is possible to enable debug mode for the full application by setting the
environment variable ``NUMBA_DPEX_DEBUGINFO=1`` instead of ``debug`` option
inside the ``numba_dpex.kernel`` decorator. This sets the default value of the
debug option in ``numba_dpex.kernel``. If ``NUMBA_DPEX_DEBUGINFO`` is set to a
non-zero value, the debug data is emitted for the full application. Debug mode
can be turned off on individual functions by setting ``debug=False`` in
``numba_dpex.kernel``.

See also:

    - `Debugging JIT compiled code with GDB*
      <http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#debugging-jit-compiled-code-with-gdb>`_
    - `NUMBA_DEBUGINFO
      <https://numba.pydata.org/numba-doc/dev/reference/envvars.html#envvar-NUMBA_DEBUGINFO>`_

Breakpoint is hit twice
-----------------------

The first line of the kernel and functions is hit twice. This happens because
you are debugging a multi-threaded program, so multiple events may be received
from different threads. This is the default behavior, but you can configure it
for more efficient debugging. To ensure that the current thread executes a
single line without interference, activate the scheduler-locking setting.

To activate the scheduler-locking setting, refer to the :ref:`single_stepping`
section.
