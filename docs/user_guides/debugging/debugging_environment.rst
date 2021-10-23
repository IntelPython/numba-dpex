.. _debugging-environment:

Configure debugging environment
=================================

1) Activate the debugger and compiler:

    .. code-block:: bash

        export ONEAPI_ROOT=/path/to/oneapi
        source $ONEAPI_ROOT/debugger/latest/env/vars.sh
        source $ONEAPI_ROOT/compiler/latest/env/vars.sh

2) Create and activate conda environment with the installed Numba-dppy:

    .. code-block:: bash

        conda create numba-dppy-dev numba-dppy
        conda activate numba-dppy-dev

3) Activate NEO drivers (optional).

    If you want to use the local NEO driver, activate the variables for it. See the :ref:`NEO-driver`.

4) Check debugging environment.

    You can check the correctness of the work with the following example:

    .. literalinclude:: ../../../numba_dppy/examples/debug/simple_sum.py
        :lines: 15-
        :linenos:
        :lineno-match:

    Launch the Intel® Distribution for GDB* and set a breakpoint in the kernel:

    .. code-block:: shell-session

        $ gdb-oneapi -q --args python simple_sum.py
        (gdb) break simple_sum.py:22
        No source file named simple_sum.py.
        Make breakpoint pending on future shared library load? (y or [n]) y
        Breakpoint 1 (simple_sum.py:22) pending.
        (gdb) run

    In the output you can see that the breakpoint was hit successfully:

    .. code-block:: shell-session

        Thread 2.2 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum () at simple_sum.py:22
        22           i = dppy.get_global_id(0)
        (gdb) continue
        Done...
        ...
