.. _debugging-environment:

Configuring debugging environment
=================================

**Activate GDB debugger and compiler**

.. code-block:: bash

    export ONEAPI_ROOT=/path/to/oneapi
    source $ONEAPI_ROOT/debugger/latest/env/vars.sh
    source $ONEAPI_ROOT/compiler/latest/env/vars.sh

Create and activate conda environment with installed Numba-dppy:

.. code-block:: bash

    conda create numba-dppy-dev numba-dppy
    conda activate numba-dppy-dev

.. note::

    - Debugging features were tested with following packages: ``numba-dppy=0.14``, ``dpctl=0.8``, ``numba=0.53``.

**Activate environment variables**

Debugging on "no optimization" level is more stable. Local variable are not optimized out.
You need to set the following variable for debugging:

.. code-block:: bash

    export NUMBA_OPT=0

It is possible to enable debug mode for the full application by setting the environment variable ``NUMBA_DPPY_DEBUGINFO=1``
instead of ``debug`` option inside the ``dppy.kernel`` decorator. This sets the default value of the debug option in
``dppy.kernel``. If ``NUMBA_DPPY_DEBUGINFO`` is set to a non-zero value, the debug data is emitted for the full application.
Debug mode can be turned off on individual functions by setting ``debug=False`` in ``dppy.kernel``.

See also:

    - `Debugging JIT compiled code with GDB <http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#debugging-jit-compiled-code-with-gdb>`_
    - `NUMBA_DEBUGINFO <https://numba.pydata.org/numba-doc/dev/reference/envvars.html#envvar-NUMBA_DEBUGINFO>`_

**Activate NEO drivers**

Further, if you want to use local NEO driver, you need to activate the variables for it. See :ref:`NEO-driver`.

**Check debugging environment**

You can check the correctness of the work with the following example:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

Launch gdb and set a breakpoint in the kernel:

.. code-block:: bash

    $ gdb-oneapi -q --args python simple_sum.py
    (gdb) break simple_sum.py:22
    No source file named simple_sum.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (simple_sum.py:22) pending.
    (gdb) run

In the output you can see that the breakpoint was hit successfully:

.. code-block:: bash

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:22
    22           i = dppy.get_global_id(0)
    (gdb) continue
    Done...
    ...
