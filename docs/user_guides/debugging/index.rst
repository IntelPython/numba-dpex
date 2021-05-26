Debugging with GDB
==================

`numba-dppy` allows SYCL kernels to be debugged with the GDB debugger.
Setting the debug environment variable :envvar:`NUMBA_DPPY_DEBUG`
(e.g. :samp:`export NUMBA_DPPY_DEBUG=1`) enables the emission of debug information.
To disable debugging, unset the variable (e.g. :samp:`unset NUMBA_DPPY_DEBUG`).

.. note::

    Beware that enabling debug info significantly increases the memory consumption for each compiled kernel.
    For large application, this may cause out-of-memory error.

Not all GDB features supported by `Numba` on CPUs are yet supported in `numba-dppy`.
See :ref:`debugging-features-and-limitations`.


Requirements
------------

`Intel® Distribution for GDB` is needed for `numba-dppy`'s debugging features to work.
`Intel® Distribution for GDB` is part of `Intel oneAPI` and
the relevant documentation can be found at `Intel® Distribution for GDB documentation`_.

.. _`Intel® Distribution for GDB documentation`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-gdb.html


Example of GDB usage
--------------------

For example, given the following `numba-dppy` kernel code:

.. code-block:: python
    :caption: sum.py
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

Running GDB and creating breakpoint in kernel:

.. code-block:: bash

    $ export NUMBA_DPPY_DEBUG=1
    $ gdb-oneapi -q --args python sum.py
    (gdb) break sum.py:7  # Set breakpoint in kernel
    (gdb) run
    Thread 2.2 hit Breakpoint 1,  with SIMD lanes [0-7], dppl_py_devfn__5F__5F_main_5F__5F__2E_data_5F_parallel_5F_sum_24_1_2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29_ () at sum.py:7
    7          i = dppy.get_global_id(0)
    (gdb) next
    8          c[i] = a[i] + b[i]


.. _debugging-features-and-limitations:

Features and Limitations
------------------------

.. toctree::
    :maxdepth: 2

    features
    limitations
