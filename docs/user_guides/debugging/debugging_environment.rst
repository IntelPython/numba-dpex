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

You will also need to create and acrivate conda environment with installed `numba-dppy`:

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
    export NUMBA_DPPY_DEBUG=1

Activate NEO drivers
--------------------

Further, if you want to use local NEO driver, you need to activate the variables for it.

Checking debugging environment
------------------------------

You can check the correctness of the work with the following example:

.. code-block:: python
    :linenos:

    import numpy as np
    import numba_dppy as dppy
    import dpctl

    @dppy.kernel
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 10
    N = global_size
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    with dpctl.device_context("opencl:gpu") as gpu_queue:
        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

Launch gdb and set a breakpoint in the kernel:

.. code-block:: bash

    gdb-oneapi -q --args python example.py
    (gdb) break example.py:7
    No source file named example.py.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (example.py:7) pending.
    (gdb) run

In the output you can see that the breakpoint was set successfully:

.. code-block:: bash

    Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], dppy_py_devfn__5F__5F_main_5F__5F__2E_data_5F_parallel_5F_sum_24_1_2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29_ () at example.py:7
    7           i = dppy.get_global_id(0)
