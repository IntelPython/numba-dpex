.. _debugging-environment:

Configuring debugging environment
=================================

Activate debugger and compiler
------------------------------

First you need to activate the debugger that you will be using
and dpcpp compiler for numba-dppy:

.. code-block:: bash

    export ONEAPI_ROOT=/path/to/oneapi
    source $ONEAPI_ROOT/debugger/latest/env/vars.sh
    source $ONEAPI_ROOT/compiler/latest/env/vars.sh

Activate conda environment
--------------------------

You will also need to create and activate conda environment with installed `numba-dppy`:

.. code-block:: bash

    conda create numba-dppy-dev numba-dppy
    conda activate numba-dppy-dev

.. note::

    Known issues:
      - Debugging tested with following packages: ``numba-dppy=0.13.1``, ``dpctl=0.6``, ``numba=0.52``.

Activate environment variables
------------------------------

You need to set the following variables for debugging:

.. code-block:: bash

    export NUMBA_OPT=1
    export NUMBA_DPPY_DEBUGINFO=1

Activate NEO drivers
--------------------

Further, if you want to use local NEO driver, you need to activate the variables for it.

Checking debugging environment
------------------------------

You can check the correctness of the work with the following example:

.. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
    :linenos:

Launch gdb and set a breakpoint in the kernel:

.. code-block:: bash

    $ gdb-oneapi -q --args python simple_sum.py
    (gdb) break simple_sum.py:8
    No source file named simple_sum.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (simple_sum.py:8) pending.
    (gdb) run

In the output you can see that the breakpoint was set successfully:

.. code-block:: bash

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:8
    8           i = dppy.get_global_id(0)
    (gdb) continue
    Done...
    ...
